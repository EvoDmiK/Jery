"""Head-only Laya pilot on a decision-export-v1 export with the existing Unsloth MLX decision trainer.

Engineering check only (synthetic, unreviewed rows): token budget, frozen encoder, finite loss and
gradient, head updates, before/after predictions, save/reload identity, warm latency and peak memory.

PYTHONPATH=src tmp/venv-decision/bin/python -m jery.model.laya_pilot \
    --config configs/model/laya-pilot-20261017.json --assets-root <checkout with export and model> \
    --run-id <id> [--override '{"head_learning_rate": 0.001}']
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
QUESTION = 'route_in_window'
PACKAGES = ('unsloth', 'unsloth_zoo', 'mlx', 'mlx-metal', 'mlx-lm', 'torch', 'transformers',
            'tokenizers', 'safetensors', 'numpy', 'laya')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_export(folder, splits=('train', 'validation')):
    """Rows of each split after checking them against the export manifest (the export is read-only)."""
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('export_version') != 'decision-export-v1':
        raise ValueError(f'not a decision-export-v1 export: {folder}')
    rows, hashes = {}, {}
    for split in splits:
        path = folder / f'{split}.jsonl'
        hashes[split] = sha256(path)
        expected = manifest['splits'][split]['sha256']
        if hashes[split] != expected:
            raise ValueError(f'{path} sha256 {hashes[split]} != manifest {expected}')
        rows[split] = read_jsonl(path)
        if len(rows[split]) != manifest['splits'][split]['rows']:
            raise ValueError(f'{path} row count differs from manifest')
    return rows, {'manifest_sha256': sha256(folder / 'manifest.json'), 'sha256': hashes}


def accepted(row):
    """Internal options that count as correct: every option with gold mass."""
    gold = row['gold'][QUESTION]
    return {key for key, value in gold.get('probabilities', {gold['label']: 1.0}).items() if value > 0}


def choice_metrics(rows, predictions):
    """Top-1 any-of accuracy and soft-target cross-entropy of one prediction per row."""
    hits, losses, by_label = 0, [], {}
    for row, prediction in zip(rows, predictions):
        gold = row['gold'][QUESTION]
        target = gold.get('probabilities', {gold['label']: 1.0})
        total = sum(target.values())
        losses.append(-sum(v / total * math.log(max(prediction['probabilities'].get(k, 0.0), 1e-12))
                           for k, v in target.items() if v > 0))
        hit = prediction['choice'] in accepted(row)
        hits += hit
        kind = 'tool' if gold['label'] not in ('no_tool', 'clarify', 'window_none') else gold['label']
        seen = by_label.setdefault(kind, [0, 0])
        seen[0], seen[1] = seen[0] + hit, seen[1] + 1
    return {'rows': len(rows), 'accuracy': hits / len(rows), 'mean_soft_ce': statistics.fmean(losses),
            'by_gold_kind': {k: {'correct': c, 'rows': n} for k, (c, n) in sorted(by_label.items())}}


def sequence_budget(tokenizer, common, state, internal, max_len, head_max_len):
    """Replays laya common.build_sequence and reports what it cuts; ids must equal build_sequence's."""
    mask = tokenizer.mask_token
    options = common.render_options(internal)
    encode = lambda text: tokenizer(text, add_special_tokens=False)['input_ids']
    full = [[tokenizer.mask_token_id] + encode(' ' + text.replace(mask, ' ')) for text in options]
    kept = [ids[:49] for ids in full]
    per = None
    if head_max_len - sum(map(len, kept)) < 16:
        per = max(4, (head_max_len - 16) // max(1, len(kept)))
        kept = [ids[:per] for ids in kept]
    head = encode('%s question: %s' % (internal['t'], str(internal['ins']).replace(mask, ' ')))
    head_kept = head[:max(8, head_max_len - sum(map(len, kept)))]
    ids = [tokenizer.cls_token_id] + head_kept + [tokenizer.sep_token_id] + sum(kept, []) + [tokenizer.sep_token_id]
    room = max(0, max_len - len(ids) - 1)
    state_ids = encode(common.serialize_state(state).replace(mask, ' '))
    reference, _ = common.build_sequence(tokenizer, state, internal, max_len, head_max_len)
    if (ids + state_ids[:room] + [tokenizer.sep_token_id])[:max_len] != reference:
        raise AssertionError('budget replay differs from laya build_sequence')
    rendered = [tokenizer.decode(ids[1:]) for ids in kept]
    return {
        'sequence_tokens': len(reference),
        'options': len(options),
        'option_tokens_max': max(map(len, full)),
        'options_cut': sum(len(a) != len(b) for a, b in zip(full, kept)),
        'per_option_cap': per,
        'option_text_collisions': len(rendered) - len(set(rendered)),
        'instruction_tokens': len(head), 'instruction_cut': len(head) - len(head_kept),
        'state_tokens': len(state_ids), 'state_room': room, 'state_cut': max(0, len(state_ids) - room),
    }


def provenance(config, model_dir):
    """Environment, installed trainer source hashes and model revision, for the report."""
    import unsloth.models.decision_mlx as decision_mlx
    import unsloth.models._decision_common as decision_common
    import unsloth_zoo.mlx.decision as zoo_decision
    import unsloth_zoo.mlx.trainer as zoo_trainer
    import mlx.core as mx

    git = lambda *args: subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    revision = {}
    for meta in sorted((model_dir / '.cache/huggingface/download').glob('*.metadata')):
        lines = meta.read_text().splitlines()
        revision[meta.name[:-len('.metadata')]] = {'commit': lines[0], 'etag': lines[1]}
    commits = {value['commit'] for value in revision.values()}
    if commits != {config['model_revision']}:
        raise ValueError(f'model folder revision {commits} != configured {config["model_revision"]}')
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'python': sys.version.split()[0], 'executable': sys.executable, 'platform': platform.platform(),
        'machine': subprocess.run(['sysctl', '-n', 'hw.model'], capture_output=True, text=True).stdout.strip(),
        'memory_bytes': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
        'metal': mx.device_info() if mx.metal.is_available() else None,
        'packages': versions,
        'trainer_source_sha256': {module.__name__: sha256(module.__file__) for module in
                                  (decision_mlx, decision_common, zoo_decision, zoo_trainer)},
        'git': {'head': git('rev-parse', 'HEAD'), 'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(git('status', '--porcelain'))},
        'model': {'path': str(model_dir), 'revision': config['model_revision'], 'files': revision,
                  'model_safetensors_sha256': sha256(model_dir / 'model.safetensors')},
    }


class MemoryGuard(threading.Thread):
    """Stops the process before memory runs away; MLX allocations are counted as well as RSS."""

    def __init__(self, limit_gb, on_trip):
        super().__init__(daemon=True)
        self.limit, self.on_trip, self.peak = limit_gb * 2 ** 30, on_trip, 0

    def used(self):
        import mlx.core as mx
        import psutil
        return max(psutil.Process().memory_info().rss, mx.get_active_memory() + mx.get_cache_memory())

    def run(self):
        while True:
            used = self.used()
            self.peak = max(self.peak, used)
            if used > self.limit:
                self.on_trip(used)
                os._exit(3)
            time.sleep(0.25)


def predict_rows(model, tokenizer, rows):
    from unsloth import FastDecisionModel
    out, seconds = [], []
    for row in rows:
        started = time.perf_counter()
        answer = FastDecisionModel.predict(model, tokenizer, row['state'], row['questions'])[QUESTION]
        seconds.append(time.perf_counter() - started)
        alias = answer['choice']
        out.append({'source_id': row['metadata']['source_id'], 'choice': alias,
                    'tool': row['metadata']['alias_to_tool'].get(alias),
                    'confidence': answer['confidence'], 'probabilities': answer['probabilities']})
    return out, seconds


def tensors_sha256(tree):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    digest = hashlib.sha256()
    for name, value in sorted(tree_flatten(tree)):
        digest.update(name.encode())
        digest.update(np.array(value.astype(mx.float32)).tobytes())
    return digest.hexdigest()


def run(config, run_id, log):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from transformers import TrainerCallback
    from unsloth import FastDecisionModel
    from unsloth.models._decision_common import _internal, _laya
    from unsloth.models.decision_mlx import DecisionTrainer
    from unsloth_zoo.mlx.decision import load_trainable, save_trainable
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    report = {'run_id': run_id, 'config': config}
    model_dir = Path(config['model_dir'])
    rows, report['export'] = load_export(Path(config['export_dir']))
    report['export']['dir'] = config['export_dir']
    report['provenance'] = provenance(config, model_dir)
    log(f"export ok: train {len(rows['train'])} validation {len(rows['validation'])}; model {config['model_revision']}")

    mx.random.seed(config['seed'])
    started = time.perf_counter()
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(model_dir), local_files_only=True, max_seq_length=config['max_seq_length'],
        use_gradient_checkpointing=config['gradient_checkpointing'], random_state=config['seed'])
    FastDecisionModel.freeze_backbone(model)  # already frozen by the loader; explicit for the head-only stage
    report['cold_load_seconds'] = time.perf_counter() - started
    trainable = tree_flatten(model.trainable_parameters())
    report['parameters'] = {
        'trainable': sum(v.size for _, v in trainable), 'total': sum(v.size for _, v in tree_flatten(model.parameters())),
        'trainable_groups': sorted({name.split('.')[0] for name, _ in trainable}),
        'decision_config': model.decision_config, 'gradient_checkpointing': bool(model.gradient_checkpointing)}
    if any(name.startswith('encoder.') for name, _ in trainable):
        raise RuntimeError('encoder parameters are trainable in a head-only run')
    encoder_before = tensors_sha256(model.encoder.parameters())
    head_before = {name: np.array(value) for name, value in trainable}
    log(f"loaded in {report['cold_load_seconds']:.1f}s; trainable {report['parameters']['trainable']:,} "
        f"of {report['parameters']['total']:,} {report['parameters']['trainable_groups']}")

    # Exact-tokenizer budget: nothing the question or options need may be cut.
    common, cfg = _laya().common, model.decision_config
    budgets = {split: [sequence_budget(tokenizer, common, row['state'], _internal(row['questions'][QUESTION]),
                                       int(cfg['max_len']), int(cfg['head_max_len'])) for row in split_rows]
               for split, split_rows in rows.items()}
    report['token_budget'] = {split: {
        'sequence_tokens_max': max(b['sequence_tokens'] for b in items),
        'sequence_tokens_median': statistics.median(b['sequence_tokens'] for b in items),
        'option_tokens_max': max(b['option_tokens_max'] for b in items),
        'rows_with_options_cut': sum(b['options_cut'] > 0 for b in items),
        'rows_with_option_collisions': sum(b['option_text_collisions'] > 0 for b in items),
        'rows_with_instruction_cut': sum(b['instruction_cut'] > 0 for b in items),
        'rows_with_state_cut': sum(b['state_cut'] > 0 for b in items),
        'state_tokens_max': max(b['state_tokens'] for b in items),
        'state_room_min': min(b['state_room'] for b in items)} for split, items in budgets.items()}
    log(f"token budget: {json.dumps(report['token_budget'])}")
    cut = [(split, i) for split, items in budgets.items() for i, b in enumerate(items)
           if b['options_cut'] or b['option_text_collisions'] or b['instruction_cut'] or b['state_cut']]
    if cut:
        raise RuntimeError(f'input would be cut or options collide for {len(cut)} rows, first {cut[:5]}')

    datasets = {}
    for split, split_rows in rows.items():
        items, built = FastDecisionModel.build_dataset(split_rows, tokenizer, model)
        if built['skipped'] or len(items) != len(split_rows):
            raise RuntimeError(f'{split} dataset skipped rows: {built}')
        datasets[split] = items
    report['dataset'] = {split: len(items) for split, items in datasets.items()}

    FastDecisionModel.for_inference(model)
    before = {split: predict_rows(model, tokenizer, split_rows)[0] for split, split_rows in rows.items()}
    report['before'] = {split: choice_metrics(rows[split], before[split]) for split in rows}
    log(f"before: {json.dumps(report['before'])}")

    args = MLXTrainingConfig(
        output_dir=str(ROOT / config['models_dir'] / run_id / 'trainer'), seed=config['seed'],
        per_device_train_batch_size=config['micro_batch'], gradient_accumulation_steps=config['grad_accum'],
        max_steps=config['max_steps'], learning_rate=config['encoder_learning_rate'],
        lr_scheduler_type=config['lr_scheduler'], warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'], max_grad_norm=config['max_grad_norm'], logging_steps=1,
        gradient_checkpointing=config['gradient_checkpointing'],
        memory_limit_gb=config['mlx_memory_limit_gb'], wired_limit_gb=0)
    FastDecisionModel.for_training(model, use_gradient_checkpointing=config['gradient_checkpointing'])
    steps = []

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs and 'train_runtime' not in logs:
                steps.append({**logs, 'step': state.global_step, 'seconds': time.perf_counter() - train_started})
                if not math.isfinite(logs['loss']):
                    control.should_training_stop = True
                if state.global_step % 8 == 0 or state.global_step == 1:
                    log(f"step {state.global_step}: loss {logs['loss']:.4f} grad_norm {logs.get('grad_norm', 0):.4f} "
                        f"peak {logs['peak_memory_gb']:.2f}GB")

    trainer = DecisionTrainer(model=model, args=args, train_dataset=datasets['train'],
                              head_learning_rate=config['head_learning_rate'], tokenizer=tokenizer,
                              callbacks=[Progress()])
    # One explicit microbatch gradient before any update: finite and nonzero.
    step = trainer._trainer._step(False)
    loss, grads = step(step.collate(datasets['train'][:1]))
    mx.eval(loss, grads)
    flat = tree_flatten(grads)
    norm = math.sqrt(sum(float((g.astype(mx.float32) ** 2).sum().item()) for _, g in flat))
    report['first_gradient'] = {'loss': loss.item(), 'global_norm': norm, 'tensors': len(flat),
                                'encoder_tensors': sum(name.startswith('encoder.') for name, _ in flat)}
    if not (math.isfinite(loss.item()) and math.isfinite(norm) and norm > 0):
        raise RuntimeError(f"bad first gradient {report['first_gradient']}")
    mx.reset_peak_memory()
    train_started = time.perf_counter()
    output = trainer.train()
    report['train'] = {**output.metrics, 'wall_seconds': time.perf_counter() - train_started,
                       'mlx_peak_gb': mx.get_peak_memory() / 1e9, 'steps_logged': len(steps),
                       'optimizer_steps_per_epoch': math.ceil(len(datasets['train']) / config['micro_batch'] / config['grad_accum'])}
    if len(steps) != config['max_steps'] or not all(math.isfinite(s['loss']) for s in steps):
        raise RuntimeError(f'training stopped early or produced a non-finite loss at step {len(steps)}')
    report['loss_curve'] = {'first': steps[0]['loss'], 'last': steps[-1]['loss'],
                            'first8_mean': statistics.fmean(s['loss'] for s in steps[:8]),
                            'last8_mean': statistics.fmean(s['loss'] for s in steps[-8:]),
                            'grad_norm_first': steps[0].get('grad_norm'), 'grad_norm_last': steps[-1].get('grad_norm')}
    log(f"train: {json.dumps(report['train'])}")

    after_trainable = dict(tree_flatten(model.trainable_parameters()))
    deltas = {name: float(np.abs(np.array(after_trainable[name]) - value).max()) for name, value in head_before.items()}
    report['updates'] = {'tensors': len(deltas), 'changed': sum(d > 0 for d in deltas.values()),
                         'max_abs_delta': max(deltas.values()),
                         'encoder_unchanged': tensors_sha256(model.encoder.parameters()) == encoder_before,
                         'encoder_sha256': encoder_before}
    if not report['updates']['encoder_unchanged'] or not report['updates']['changed']:
        raise RuntimeError(f"update check failed {report['updates']}")

    FastDecisionModel.for_inference(model)
    after = {split: predict_rows(model, tokenizer, split_rows)[0] for split, split_rows in rows.items()}
    report['after'] = {split: choice_metrics(rows[split], after[split]) for split in rows}
    log(f"after: {json.dumps(report['after'])}")

    # Only the trained head is saved; the base weights stay in the read-only model folder.
    weights_dir = ROOT / config['models_dir'] / run_id
    save_trainable(model, weights_dir)
    artifact = weights_dir / 'trainable.safetensors'
    (weights_dir / 'pilot.json').write_text(json.dumps({
        'format': 'unsloth-mlx-laya-trainable-v1', 'base_model': str(model_dir), 'base_revision': config['model_revision'],
        'base_model_safetensors_sha256': report['provenance']['model']['model_safetensors_sha256'],
        'trainable_sha256': sha256(artifact), 'decision_config': model.decision_config,
        'loader': 'FastDecisionModel.from_pretrained(base) then unsloth_zoo.mlx.decision.load_trainable(model, dir)'},
        indent=2))
    del trainer, model
    started = time.perf_counter()
    restored, tokenizer = FastDecisionModel.from_pretrained(
        str(model_dir), local_files_only=True, max_seq_length=config['max_seq_length'])
    load_trainable(restored, weights_dir)
    FastDecisionModel.for_inference(restored)
    report['restore_load_seconds'] = time.perf_counter() - started
    reloaded = {split: predict_rows(restored, tokenizer, split_rows)[0] for split, split_rows in rows.items()}
    diff = max(abs(a['probabilities'][k] - b['probabilities'][k])
               for split in rows for a, b in zip(after[split], reloaded[split]) for k in a['probabilities'])
    same = all(a['choice'] == b['choice'] for split in rows for a, b in zip(after[split], reloaded[split]))
    report['restore'] = {'artifact': str(artifact.relative_to(ROOT)), 'artifact_bytes': artifact.stat().st_size,
                         'artifact_sha256': sha256(artifact), 'same_choices': same, 'max_abs_probability_diff': diff,
                         'encoder_unchanged': tensors_sha256(restored.encoder.parameters()) == encoder_before}
    if not same or diff > 1e-6 or not report['restore']['encoder_unchanged']:
        raise RuntimeError(f"restore mismatch {report['restore']}")

    # Warm latency on the restored model: one decision per call, after one warm-up pass.
    predict_rows(restored, tokenizer, rows['validation'])
    seconds = []
    for _ in range(config['latency_repeats']):
        seconds += predict_rows(restored, tokenizer, rows['validation'])[1]
    seconds.sort()
    report['latency_ms'] = {'calls': len(seconds), 'p50': 1000 * statistics.median(seconds),
                            'p95': 1000 * seconds[int(0.95 * (len(seconds) - 1))], 'max': 1000 * seconds[-1]}
    report['memory'] = {'mlx_peak_gb_since_train': mx.get_peak_memory() / 1e9,
                        'max_rss_gb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9}
    return report, {'steps': steps, 'before': before, 'after': after, 'reloaded': reloaded, 'budgets': budgets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--assets-root', default=str(ROOT), help='checkout holding export_dir and model_dir')
    parser.add_argument('--override', default='{}', help='JSON object merged over the config, kept in the report')
    options = parser.parse_args(argv)
    config = {**json.loads(Path(options.config).read_text()), **json.loads(options.override)}
    # Shared read-only inputs (export, base model) resolve against --assets-root; outputs stay in this worktree.
    config['assets_root'] = str(Path(options.assets_root).resolve())
    for key in ('export_dir', 'model_dir'):
        config[key] = str(Path(config['assets_root']) / config[key])
    out = ROOT / config['outputs_dir'] / options.run_id
    if out.exists() or (ROOT / config['models_dir'] / options.run_id).exists():
        parser.error(f'run {options.run_id} already exists')
    out.mkdir(parents=True)
    log_file = (out / 'run.log').open('a')

    def log(message):
        line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    def status(state, **extra):
        (out / 'status.json').write_text(json.dumps({'run_id': options.run_id, 'status': state,
                                                     'updated': datetime.now(timezone.utc).isoformat(), **extra}, indent=2))

    status('running', command=sys.argv)
    guard = MemoryGuard(config['memory_guard_gib'], lambda used: (
        log(f'memory guard tripped at {used / 2 ** 30:.2f} GiB'),
        status('failed', error=f'memory guard {used / 2 ** 30:.2f} GiB > {config["memory_guard_gib"]} GiB')))
    guard.start()
    started = time.perf_counter()
    try:
        report, details = run(config, options.run_id, log)
    except Exception as exc:
        log(traceback.format_exc())
        status('failed', error=f'{type(exc).__name__}: {exc}', seconds=time.perf_counter() - started)
        return 1
    report['memory']['guard_peak_gib'] = guard.peak / 2 ** 30
    report['wall_seconds'] = time.perf_counter() - started
    report['command'] = sys.argv
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for name in ('steps', 'budgets'):
        (out / f'{name}.json').write_text(json.dumps(details[name], indent=2))
    for name in ('before', 'after', 'reloaded'):
        with (out / f'predictions_{name}.jsonl').open('w') as handle:
            for split, predictions in details[name].items():
                for prediction in predictions:
                    handle.write(json.dumps({'split': split, **prediction}, ensure_ascii=False) + '\n')
    status('succeeded', seconds=report['wall_seconds'], report=str((out / 'report.json').relative_to(ROOT)))
    log(f"done in {report['wall_seconds']:.1f}s -> {out.relative_to(ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
