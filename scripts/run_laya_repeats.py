"""Run independent Laya head-only trials and record their models and learning curves."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.parse
import zipfile

from run_local_training import run
from sync_laya_training_metrics import request, sync_metrics

ROOT = Path(__file__).resolve().parents[1]
BASE_ARTIFACT = 'mlflow-artifacts:/46/cb04fedef6fc41859e37ae8aa6c37cc7/artifacts/model/base'
PUBLIC = 'https://mlflow.dove-nest.com'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload(base, artifact_uri, name, path):
    import urllib.request
    prefix = urllib.parse.urlsplit(artifact_uri).path.strip('/')
    endpoint = base.rstrip('/') + '/api/2.0/mlflow-artifacts/artifacts/' + urllib.parse.quote(prefix + '/' + name, safe='/')
    data = path.read_bytes()
    req = urllib.request.Request(endpoint, data=data, method='PUT', headers={
        'User-Agent': 'Jery-MLflow/1.0', 'Content-Type': 'application/octet-stream'})
    with urllib.request.urlopen(req, timeout=180) as response:
        response.read()
    with urllib.request.urlopen(urllib.request.Request(endpoint, headers={'User-Agent': 'Jery-MLflow/1.0'}), timeout=180) as response:
        actual = hashlib.sha256(response.read()).hexdigest()
    if actual != hashlib.sha256(data).hexdigest():
        raise RuntimeError('Artifact read-back mismatch: ' + name)
    return {'sha256': actual, 'bytes': len(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--tracking-uri', default=PUBLIC)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--experiment-id', help='Use an existing experiment, verifying its name')
    parser.add_argument('--plan', type=Path, help='JSON list of config overrides, each with a unique trial_id')
    options = parser.parse_args()
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=options.resume)
    os.environ.update(PYTHONPATH=str(ROOT / 'src'), HF_HUB_OFFLINE='1', USE_TF='0', TOKENIZERS_PARALLELISM='false')
    base = options.tracking_uri
    previous = json.loads((output / 'batch.json').read_text()) if options.resume else None
    experiment = previous['experiment_id'] if previous else options.experiment_id
    if experiment:
        if request(base, 'mlflow/experiments/get?experiment_id=' + experiment)['experiment']['name'] != options.experiment_name:
            raise ValueError('Experiment ID/name mismatch')
    else:
        experiment = request(base, 'mlflow/experiments/create', {'name': options.experiment_name,
        'tags': [{'key': 'purpose', 'value': 'five independent engineering pilots; synthetic unreviewed 32/12 rows'}]})['experiment_id']
    base_config = json.loads((ROOT / 'configs/model/laya-pilot-20261017.json').read_text())
    trials = json.loads(options.plan.read_text()) if options.plan else [
        {'trial_id': 's' + str(seed), 'seed': seed} for seed in range(3407, 3412)]
    if not trials or len({t['trial_id'] for t in trials}) != len(trials):
        raise ValueError('Trial IDs must be unique and nonempty')
    for trial in trials:
        if set(trial) - (set(base_config) | {'trial_id'}):
            raise ValueError('Unknown trial configuration key')
    (output / 'trials.json').write_text(json.dumps(trials, indent=2) + '\n')
    manifest = {'experiment_id': experiment, 'experiment_name': options.experiment_name,
        'experiment_url': PUBLIC + '/#/experiments/' + experiment,
        'status': 'running', 'seeds': [t.get('seed', base_config['seed']) for t in trials],
        'config': base_config, 'trials': trials, 'runs': [], 'base_artifact_uri': BASE_ARTIFACT}
    if previous:
        if previous['experiment_name'] != options.experiment_name:
            raise ValueError('Resume experiment name differs')
        manifest = previous
        if manifest.get('trials', trials) != trials:
            raise ValueError('Resume plan differs')
        manifest['status'] = 'running'

    def save():
        (output / 'batch.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')

    save()
    source = output / 'source.zip'
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as archive:
        files = list((ROOT / 'src/jery/model').glob('*.py'))
        files += list((ROOT / 'src/jery/runtime').glob('*.py'))
        files += [ROOT / p for p in ['src/jery/__init__.py', 'scripts/run_laya_repeats.py',
            'scripts/sync_laya_training_metrics.py', 'scripts/run_local_training.py', 'scripts/build_laya_pilot.py',
            'requirements-decision.txt', 'configs/model/laya-pilot-20261017.json', 'docs/contracts/decision-export-v1.md']]
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.write(output / 'trials.json', 'configs/trials.json')
    print('EXPERIMENT ' + manifest['experiment_url'], flush=True)
    total = len(trials)
    for index, trial in enumerate(trials, 1):
        trial_id = trial['trial_id']
        config = {**base_config, **{k: v for k, v in trial.items() if k != 'trial_id'}}
        seed = config['seed']
        export_manifest = json.loads((ROOT / config['export_dir'] / 'manifest.json').read_text())
        train_rows = export_manifest['splits']['train']['rows']
        val_rows = export_manifest['splits']['validation']['rows']
        steps_per_epoch = math.ceil(train_rows / config['micro_batch'] / config['grad_accum'])
        attempts = [r for r in manifest['runs'] if r.get('trial_id', 's' + str(r['seed'])) == trial_id]
        if any(r['status'] == 'succeeded' for r in attempts):
            continue
        if any(r['status'] == 'running' for r in attempts):
            raise ValueError('An unsettled run must be inspected before resuming')
        local_id = output.name + '-' + trial_id + '-a' + str(len(attempts) + 1)
        tags = {'mlflow.runName': 'laya-' + trial_id, 'evaluation.scope': 'engineering_smoke_only',
            'dataset.review_status': 'needs_human_review', 'jery.base_artifact_uri': BASE_ARTIFACT,
            'jery.repeat_index': str(index), 'jery.training_state': 'running',
            'mlflow.note.content': '32 unreviewed synthetic train / 12 validation rows. Fresh pinned base for every seed; no cumulative training. Native FP32 head with frozen FP16/FP32 encoder. Shared base weights are referenced by immutable artifact URI. Validation measured before/after only.'}
        remote = request(base, 'mlflow/runs/create', {'experiment_id': experiment,
            'start_time': int(time.time() * 1000), 'tags': [{'key': k, 'value': v} for k,v in tags.items()]})['run']
        rid, artifact_uri = remote['info']['run_id'], remote['info']['artifact_uri']
        item = {'index': index, 'trial_id': trial_id, 'seed': seed, 'config': config, 'local_id': local_id, 'run_id': rid,
            'artifact_uri': artifact_uri, 'status': 'running',
            'run_url': PUBLIC + '/#/experiments/' + experiment + '/runs/' + rid}
        manifest['runs'].append(item); save()
        params = {**config, 'seed': seed, 'num_epochs': config['max_steps'] / steps_per_epoch,
            'train_rows': train_rows, 'validation_rows': val_rows,
            'base_artifact_uri': BASE_ARTIFACT, 'repeat_index': index}
        request(base, 'mlflow/runs/log-batch', {'run_id': rid,
            'params': [{'key': k, 'value': str(v)} for k,v in params.items()]})
        print(f'START {index}/{total} seed={seed} lr={config["head_learning_rate"]} steps={config["max_steps"]} {item["run_url"]}', flush=True)
        # Keep the virtualenv symlink path: resolve() would launch the bare interpreter.
        command = [str(options.python.absolute()), '-m', 'jery.model.laya_pilot', '--config',
            str(ROOT / 'configs/model/laya-pilot-20261017.json'), '--assets-root', str(ROOT),
            '--run-id', local_id, '--override', json.dumps({k: v for k, v in trial.items() if k != 'trial_id'})]
        try:
            runner_dir = output / ('runner-' + local_id)
            code = run(command, runner_dir, cwd=ROOT, timeout_seconds=1200)
            if code:
                raise RuntimeError(f'Training exit {code}; see {runner_dir}/training.log')
            folder = ROOT / config['outputs_dir'] / local_id
            report = json.loads((folder / 'report.json').read_text())
            assert report['restore']['same_choices'] and report['updates']['encoder_unchanged']
            assert report['restore']['max_abs_probability_diff'] == 0
            request(base, 'mlflow/runs/set-tag', {'run_id': rid, 'key': 'jery.training_state', 'value': 'completed_uploading'})
            item['history'] = sync_metrics(folder, rid, base)
            metrics = {'train/runtime_seconds': report['train']['wall_seconds'],
                'train/mlx_peak_gb': report['train']['mlx_peak_gb'],
                'memory/max_rss_gb': report['memory']['max_rss_gb'],
                'latency/p50_ms': report['latency_ms']['p50'], 'latency/p95_ms': report['latency_ms']['p95'],
                'restore/max_probability_diff': report['restore']['max_abs_probability_diff']}
            request(base, 'mlflow/runs/log-batch', {'run_id': rid, 'metrics': [
                {'key': k, 'value': v, 'timestamp': int(time.time()*1000), 'step': config['max_steps']} for k,v in metrics.items()]})
            head = ROOT / report['restore']['artifact']
            assert sha(head) == report['restore']['artifact_sha256']
            dependencies = output / ('model-dependencies-' + trial_id + '.json')
            dependencies.write_text(json.dumps({'format': 'unsloth-mlx-laya-trainable-v1',
                'base_artifact_uri': BASE_ARTIFACT, 'base_hf_repo': 'convaiinnovations/laya-multilingual',
                'base_revision': report['provenance']['model']['revision'],
                'base_sha256': report['provenance']['model']['model_safetensors_sha256'],
                'head_sha256': sha(head), 'loader': 'FastDecisionModel.from_pretrained(downloaded_base); load_trainable(model, downloaded_head_directory)'}, indent=2))
            artifacts = [('source/source.zip', source), ('model/dependencies.json', dependencies)]
            artifacts += [('model/head/' + p.name, p) for p in head.parent.iterdir() if p.is_file()]
            artifacts += [('evaluation/' + p.name, p) for p in folder.iterdir() if p.is_file()]
            artifacts += [('execution/' + p.name, p) for p in runner_dir.iterdir() if p.is_file()]
            export = ROOT / config['export_dir']
            artifacts += [('datasets/decision-export-v1/' + p.name, p) for p in export.iterdir() if p.is_file()]
            dataset = ROOT / 'datasets/mcp-routing-v2'
            artifacts += [('datasets/mcp-routing-v2/' + p.name, p) for p in dataset.iterdir() if p.is_file()]
            item['verified_artifacts'] = {}
            for name, path in artifacts:
                item['verified_artifacts'][name] = upload(base, artifact_uri, name, path)
            request(base, 'mlflow/runs/set-tag', {'run_id': rid, 'key': 'jery.training_state', 'value': 'completed'})
            request(base, 'mlflow/runs/update', {'run_id': rid, 'status': 'FINISHED', 'end_time': int(time.time()*1000)})
            assert request(base, 'mlflow/runs/get?run_id=' + rid)['run']['info']['status'] == 'FINISHED'
            item.update(status='succeeded', train_accuracy=report['after']['train']['accuracy'],
                validation_accuracy=report['after']['validation']['accuracy'], train_seconds=report['train']['wall_seconds'],
                train_mlx_peak_gb=report['train']['mlx_peak_gb'], head_sha256=sha(head))
            print(f'DONE {index}/{total} train={item["train_accuracy"]:.4f} val={item["validation_accuracy"]:.4f}', flush=True)
        except Exception as exc:
            item.update(status='failed', error=str(exc)); manifest['status'] = 'failed'; save()
            request(base, 'mlflow/runs/set-tag', {'run_id': rid, 'key': 'jery.training_state', 'value': 'failed'})
            request(base, 'mlflow/runs/update', {'run_id': rid, 'status': 'FAILED', 'end_time': int(time.time()*1000)})
            raise
        save()
    manifest['status'] = 'succeeded'
    manifest['validation_accuracy_mean'] = statistics.fmean(r['validation_accuracy'] for r in manifest['runs'] if r['status'] == 'succeeded')
    save()
    print(f'ALL {total} COMPLETED ' + manifest['experiment_url'], flush=True)


if __name__ == '__main__':
    main()
