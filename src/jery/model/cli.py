"""Train on train, select on validation, report test once, in one command.

PYTHONPATH=src python3 -m jery.model.cli --config configs/model/mcp-routing-v2-baseline.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time

from jery.evaluation.metrics import evaluate
from jery.model.features import index_catalog, model_input
from jery.model.predict import save_model
from jery.model.routers import LinearRouter, RetrievalRouter

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ('train', 'validation', 'test')
_data = {}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_data(data_dir):
    """Load splits after checking every file against the manifest hashes (data is read-only)."""
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    hashes = {}
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit('%s hash mismatch: dataset differs from manifest' % name)
        hashes[name] = actual
    splits = {s: read_jsonl(data_dir / (s + '.jsonl')) for s in SPLITS}
    for s, rows in splits.items():
        if any(r['split'] != s for r in rows):
            raise SystemExit('%s.jsonl contains rows from another split' % s)
    catalog = json.loads((data_dir / 'tools.json').read_text())
    return manifest, hashes, splits, catalog


def selection_key(metrics):
    return (metrics['joint_accuracy'], metrics['decision_macro_f1'], metrics['tool_accuracy'])


def _init_worker(data_dir, threads):
    os.environ.update({k: str(threads) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    _, _, splits, catalog = load_data(Path(data_dir))
    _data.update(splits=splits, catalog=catalog)


def _linear_trial(cfg):
    start = time.perf_counter()
    model = LinearRouter(_data['catalog'], **cfg).fit(_data['splits']['train'])
    fit_s = time.perf_counter() - start
    rows = _data['splits']['validation']
    metrics = evaluate(rows, [model.predict(model_input(r)) for r in rows], index_catalog(_data['catalog']))
    return {'config': model.config(), 'fit_seconds': round(fit_s, 2), 'validation': metrics}


def retrieval_sweep(catalog, rows, grid):
    """Thresholds and fallback chosen on validation only; ranking scores computed once."""
    base = RetrievalRouter(catalog)
    ranked = [base.rank(model_input(r)) for r in rows]
    trials = []
    for threshold, fallback in itertools.product(grid['threshold'], grid['fallback']):
        preds = []
        for rk in ranked:
            decision = 'select_tools' if rk and rk[0]['score'] >= threshold else fallback
            preds.append({'decision': decision, 'ranked_tools': rk,
                          'acceptable_choices': [{'tool': rk[0]['tool']}] if decision == 'select_tools' else []})
        trials.append({'config': {'kind': 'retrieval', 'threshold': threshold, 'fallback': fallback},
                       'validation': evaluate(rows, preds, base.index.catalog)})
    return trials


def timed_predictions(model, rows):
    preds, latencies = [], []
    for r in rows:
        start = time.perf_counter()
        preds.append(model.predict(model_input(r)))
        latencies.append((time.perf_counter() - start) * 1000)
    lat = sorted(latencies)
    return preds, {'mean_ms': round(statistics.mean(lat), 3), 'p50_ms': round(lat[len(lat) // 2], 3),
                   'p95_ms': round(lat[int(len(lat) * 0.95) - 1], 3), 'max_ms': round(lat[-1], 3), 'n': len(lat)}


def git_sha():
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'configs'], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except OSError:
        return None


def markdown(report):
    lines = ['# %s 탐색 실험 %s' % (report['data']['dataset_version'], report['run_id']), '',
             '> 미검토 합성 데이터(`needs_human_review`)의 탐색 실험이다. 실제 사용 정확도나 검토된 정답 성능이 아니다.', '',
             '선택(validation 기준): `%s`' % json.dumps(report['selected']['config'], ensure_ascii=False), '',
             '| 모델 | split | joint | decision acc | macro F1 | tool acc | server acc | 불필요 호출률 | R@1 | R@3 | R@5 | p50 ms |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, fam in report['families'].items():
        for split in ('validation', 'test'):
            m = fam[split]
            r = m['candidate_recall_at_k']
            lat = fam['test_latency']['p50_ms'] if split == 'test' else ''
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                name, split, m['joint_accuracy'], m['decision_accuracy'], m['decision_macro_f1'], m['tool_accuracy'],
                m['server_accuracy'], m['unnecessary_call_rate'], r['1'], r['3'], r['5'], lat))
    lines += ['', '## test decision별 precision / recall (support)', '',
              '| 모델 | ' + ' | '.join(report['families']['linear']['test']['per_decision']) + ' |',
              '|---|' + '---|' * 4]
    for name, fam in report['families'].items():
        lines.append('| %s | ' % name + ' | '.join('%s / %s (%s)' % (v['precision'], v['recall'], v['support'])
                                                  for v in fam['test']['per_decision'].values()) + ' |')
    lines += ['', '명령: `%s`' % report['command'], '', 'git: `%s`, seed: `%s`' % (report['git_sha'], report['seed'])]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(ROOT / 'configs/model/mcp-routing-v2-baseline.json'))
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--workers', type=int, help='override config max_workers (CPU processes)')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    cpu_start = time.process_time()
    config = json.loads(Path(args.config).read_text())
    workers = max(1, min(args.workers or config.get('max_workers', 4), os.cpu_count() or 1))
    data_dir = (ROOT / config['data_dir']).resolve()
    out_dir = ROOT / config['output_dir'] / args.run_id
    model_dir = ROOT / config['model_dir'] / args.run_id
    if out_dir.exists() or model_dir.exists():
        raise SystemExit('run %s already exists; test is reported once per run' % args.run_id)
    manifest, hashes, splits, catalog = load_data(data_dir)
    cat_index = index_catalog(catalog)

    # 1) Validation-only selection. Test rows are not touched here.
    retrieval_trials = retrieval_sweep(catalog, splits['validation'], config['retrieval_grid'])
    grid = config['linear_grid']
    linear_cfgs = [dict(zip(grid, values), seed=config['seed']) for values in itertools.product(*grid.values())]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(data_dir), config.get('threads_per_worker', 1))) as pool:
        linear_trials = pool.map(_linear_trial, linear_cfgs)
    best = {'retrieval': max(retrieval_trials, key=lambda t: selection_key(t['validation'])),
            'linear': max(linear_trials, key=lambda t: selection_key(t['validation']))}
    selected_name = max(best, key=lambda k: selection_key(best[k]['validation']))

    # 2) Refit the chosen linear config on train only (deterministic seed), then test once.
    fit_start = time.perf_counter()
    lin_cfg = {k: v for k, v in best['linear']['config'].items() if k != 'kind'}
    models = {'retrieval': RetrievalRouter(catalog, **{k: v for k, v in best['retrieval']['config'].items() if k != 'kind'}),
              'linear': LinearRouter(catalog, **lin_cfg).fit(splits['train'])}
    final_fit_s = time.perf_counter() - fit_start
    families, test_preds = {}, {}
    for name, model in models.items():
        preds, latency = timed_predictions(model, splits['test'])
        test_preds[name] = preds
        families[name] = {'config': model.config(), 'validation': best[name]['validation'],
                          'test': evaluate(splits['test'], preds, cat_index), 'test_latency': latency}

    meta = {'dataset_version': manifest['dataset_version'], 'catalog_version': manifest['catalog_version'],
            'sha256': hashes, 'run_id': args.run_id}
    model_dir.mkdir(parents=True)
    for name, model in models.items():
        save_model(model, model_dir / (name + '.json.gz'), dict(meta, config=model.config()))
    latest = ROOT / config['model_dir'] / 'latest'
    latest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_dir / (selected_name + '.json.gz'), latest / 'model.json.gz')

    usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == 'darwin' else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    report = {
        'run_id': args.run_id,
        'exploratory': True,
        'note': 'Unreviewed synthetic labels (needs_human_review). Not real-usage or reviewed-label accuracy.',
        'command': ' '.join([os.path.basename(sys.executable), '-m', 'jery.model.cli'] + (argv if argv is not None else sys.argv[1:])),
        'config_path': os.path.relpath(args.config, ROOT) if os.path.isabs(args.config) else args.config,
        'config': config,
        'seed': config['seed'],
        'git_sha': git_sha(),
        'data': dict(meta, split_counts=manifest['split_counts'], data_dir=os.path.relpath(data_dir, ROOT),
                     test_gold_tool_classes=families['linear']['test']['gold_tool_classes'],
                     catalog_tools=len(cat_index)),
        'protocol': 'fit on train only; hyperparameters and family chosen by validation joint_accuracy '
                    '(tie: macro F1, tool acc); test evaluated once after selection',
        'selected': {'family': selected_name, 'config': families[selected_name]['config'],
                     'model_path': os.path.relpath(model_dir / (selected_name + '.json.gz'), ROOT)},
        'families': families,
        'validation_trials': {'retrieval': len(retrieval_trials), 'linear': len(linear_trials)},
        'resources': {
            'wall_seconds': round(time.perf_counter() - started, 2),
            'final_fit_seconds': round(final_fit_s, 2),
            'cpu_seconds_main': round(time.process_time() - cpu_start, 2),
            'cpu_seconds_workers': round(child.ru_utime + child.ru_stime, 2),
            'peak_rss_mb_main': round(usage.ru_maxrss * scale / 2 ** 20, 1),
            'peak_rss_mb_worker_max': round(child.ru_maxrss * scale / 2 ** 20, 1),
            'workers': workers, 'threads_per_worker': config.get('threads_per_worker', 1),
            'linear_weights': len(models['linear'].w),
            'model_bytes': {n: (model_dir / (n + '.json.gz')).stat().st_size for n in models},
        },
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'dependencies': 'stdlib only'},
    }
    out_dir.mkdir(parents=True)
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out_dir / 'report.md').write_text(markdown(report))
    (out_dir / 'validation_sweep.json').write_text(json.dumps(
        {'retrieval': retrieval_trials, 'linear': linear_trials}, ensure_ascii=False, indent=1) + '\n')
    with (out_dir / 'test_predictions.jsonl').open('w') as f:
        for i, row in enumerate(splits['test']):
            line = {'id': row['id'], 'gold_decision': row['decision'], 'gold_choices': row['acceptable_choices']}
            for n in models:
                pred = test_preds[n][i]
                line[n] = {'decision': pred['decision'], 'acceptable_choices': pred['acceptable_choices'],
                           'top5': [r['tool'] for r in pred['ranked_tools'][:5]]}
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(markdown(report))
    print('report: %s' % os.path.relpath(out_dir / 'report.json', ROOT))
    print('model: %s (latest: %s)' % (report['selected']['model_path'], os.path.relpath(latest / 'model.json.gz', ROOT)))
    return report


if __name__ == '__main__':
    main()
