"""LR-only Hyperopt over fixed stratified folds; reuse the full training/upload supervisor for each trial."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import urllib.error

from prepare_laya_cuda_experiment import validate_explicit_plan
from run_laya_full_experiment import (ROOT, DEFAULT_TRACKING_URI, hyperopt_candidate, trial_settings,
                                     comparison_macro_f1, write_json, transient_tracking_error)
from run_laya_repeats import upload
from sync_laya_training_metrics import request


FIXED_KEYS = ('seed', 'micro_batch', 'grad_accum', 'warmup_steps', 'max_grad_norm',
              'lr_scheduler', 'weight_decay', 'label_smoothing', 'epochs')


def aggregate_folds(reports):
    if len(reports) < 2:
        raise ValueError('Multiple completed folds are required')
    first = reports[0]['config']
    for report in reports:
        for key in (*FIXED_KEYS, 'head_learning_rate'):
            if report['config'][key] != first[key]:
                raise ValueError('Fold parameters differ: ' + key)
        if report['selection']['completed_epochs'] < 1:
            raise ValueError('Fold was not trained')
    values = {
        'val_accuracy': [r['validation']['case_accuracy'] for r in reports],
        'val_macro_f1': [comparison_macro_f1(r['validation']) for r in reports],
        'test_accuracy': [r['test']['case_accuracy'] for r in reports],
        'test_macro_f1': [comparison_macro_f1(r['test']) for r in reports],
    }
    metrics = {}
    for key, rows in values.items():
        metrics[key] = statistics.fmean(rows)
        metrics[key + '_std'] = statistics.stdev(rows)
    metrics['cv_search_accuracy'] = statistics.fmean(r['selection']['best_trained_score'] for r in reports)
    return metrics


def validate_cv_plan(plan, config):
    validate_explicit_plan(plan, config)
    reference = plan.get('reference_model_config')
    if reference is not None:
        excluded = {'head_learning_rate', 'models_dir', 'outputs_dir'}
        if ({k: v for k, v in config.items() if k not in excluded}
                != {k: v for k, v in reference.items() if k not in excluded}):
            raise ValueError('Non-LR model parameters differ from the reference child')
    search = plan['hyperopt']
    if plan['cv'].get('search_mode') == 'fixed_lr_parameters':
        keys = {'weight_decay', 'label_smoothing', 'warmup_steps', 'grad_accum', 'max_grad_norm', 'lr_scheduler'}
        if (not reference or config['head_learning_rate'] != reference['head_learning_rate']
                or search['head_lr'] != [config['head_learning_rate']] * 2):
            raise ValueError('Reference LR must remain fixed')
        if plan.get('first_trial_parameters') != {k: config[k] for k in keys | {'head_learning_rate'}}:
            raise ValueError('First trial must exactly match the reference')
        if any(config[k] not in search[k] for k in keys):
            raise ValueError('Reference parameters must be in the search space')
        if set(search) - keys - {'algorithm', 'max_evals', 'startup_trials', 'head_lr', 'objective', 'test_used_for_search'}:
            raise ValueError('Unsupported search parameter')
    elif (set(search) - {'algorithm', 'max_evals', 'startup_trials', 'head_lr', 'weight_decay',
                      'label_smoothing', 'objective', 'test_used_for_search'}
            or search['weight_decay'] != [config['weight_decay']]
            or search['label_smoothing'] != [config['label_smoothing']]):
        raise ValueError('Only LR may vary in this CV experiment')
    if type(plan['cv']['n_splits']) is not int or plan['cv']['n_splits'] < 2:
        raise ValueError('Invalid fold count')
    for key in FIXED_KEYS:
        if key in plan['fixed_conditions'] and plan['fixed_conditions'][key] != config[key]:
            raise ValueError('Fixed condition mismatch: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    folder = args.state_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve())] + [v for v in sys.argv[1:] if v != '--detach']
        with (folder / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'supervisor_pid': process.pid, 'log': str(folder / 'supervisor.log')}))
        return
    lock = (folder / 'supervisor.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    config = json.loads(args.model_config.read_text())
    validate_cv_plan(plan, config)
    folds = ROOT / plan['cv']['folds_dir']
    fold_summary = json.loads((folds / 'folds.json').read_text())
    fold_summary_sha = hashlib.sha256((folds / 'folds.json').read_bytes()).hexdigest()
    if fold_summary['n_splits'] != plan['cv']['n_splits']:
        raise ValueError('Plan/folds mismatch')
    for info in fold_summary['folds']:
        directory = folds / f"fold-{info['fold']}"
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest['splits'] != info['splits']:
            raise ValueError('Fold manifest differs from frozen assignments')
        for split in ('train', 'validation', 'test'):
            if hashlib.sha256((directory / (split + '.jsonl')).read_bytes()).hexdigest() != info['splits'][split]['sha256']:
                raise ValueError('Fold input changed: ' + str(directory / split))
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state and (state['plan'] != plan or state['config'] != config or state['fold_summary_sha256'] != fold_summary_sha):
        raise ValueError('Prepared CV configuration changed')
    stopped, child = False, None

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def retry(function, *values):
        while True:
            try:
                return function(*values)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if stopped or not transient_tracking_error(error):
                    raise
                print('Tracking retry: ' + str(error), flush=True)
                time.sleep(15)

    def api(path, payload=None):
        return retry(request, args.tracking_uri, path, payload)

    def tag(rid, key, value):
        api('mlflow/runs/set-tag', {'run_id': rid, 'key': key, 'value': str(value)})

    def store(rid, uri, name, path):
        return retry(upload, args.tracking_uri, uri, name, path)

    def command(values, logname):
        nonlocal child
        with (folder / logname).open('a') as log:
            child = subprocess.Popen(values, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        if code:
            raise RuntimeError(f'Command failed ({code}); see {logname}')

    if state is None:
        existing = api('mlflow/runs/search', {'experiment_ids': [plan['experiment_id']],
            'filter': "tags.`jery.cv_job` = '" + plan['parent_name'] + "'", 'max_results': 2}).get('runs', [])
        if len(existing) > 1:
            raise ValueError('Duplicate CV parent')
        remote = existing[0] if existing else api('mlflow/runs/create', {
            'experiment_id': plan['experiment_id'], 'start_time': int(time.time() * 1000),
            'tags': [{'key': 'mlflow.runName', 'value': plan['parent_name']},
                     {'key': 'jery.cv_job', 'value': plan['parent_name']}, {'key': 'jery.role', 'value': 'cv_parent'}]})['run']
        state = {'parent_run_id': remote['info']['run_id'], 'parent_artifact_uri': remote['info']['artifact_uri'],
                 'plan': plan, 'config': config, 'fold_summary_sha256': fold_summary_sha,
                 'children': [], 'uploaded': {}, 'status': 'prepared'}
        write_json(state_path, state)
    parent, uri = state['parent_run_id'], state['parent_artifact_uri']
    state.update(status='running', supervisor_pid=os.getpid())
    state.pop('error', None)
    write_json(state_path, state)
    api('mlflow/runs/update', {'run_id': parent, 'status': 'RUNNING'})
    try:
        sources = {'plan.json': args.plan, 'model-config.json': args.model_config,
                   'data/folds.json': folds / 'folds.json',
                   'source/run_laya_cv_experiment.py': Path(__file__),
                   'source/build_laya_stratified_folds.py': ROOT / 'scripts/build_laya_stratified_folds.py'}
        for number in range(1, plan['cv']['n_splits'] + 1):
            directory = folds / f'fold-{number}'
            sources[f'data/fold-{number}/manifest.json'] = directory / 'manifest.json'
            for split in ('train', 'validation', 'test'):
                target = folder / 'data' / f'fold-{number}' / (split + '.json')
                write_json(target, [json.loads(line) for line in (directory / (split + '.jsonl')).read_text().splitlines()])
                sources[f'data/fold-{number}/{split}.json'] = target
        for name, path in sources.items():
            if name not in state['uploaded']:
                state['uploaded'][name] = store(parent, uri, name, path)
                write_json(state_path, state)
        for index in range(1, plan['hyperopt']['max_evals'] + 1):
            if stopped:
                break
            prior = [c for c in state['children'] if c['index'] == index]
            if prior and prior[0]['status'] == 'finished':
                continue
            item = prior[0] if prior else {'index': index, 'status': 'prepared',
                                           'hyperopt': hyperopt_candidate(plan, state['children'])}
            if not prior:
                state['children'].append(item)
                write_json(state_path, state)
            sampled = item['hyperopt']['parameters']
            item['head_lr'] = sampled['head_learning_rate']
            overrides, params = trial_settings(plan, config, sampled)
            trial_config = {**config, **overrides}
            trial = folder / f'trial-{index:02d}'
            trial.mkdir(exist_ok=True)
            nested_plan = {**plan, 'grad_accum': trial_config['grad_accum'], 'parent_name': f"lr-cv-trial{index:02d}-lr{item['head_lr']:g}",
                           'export_dir': str(folds / 'fold-1'),
                           'export_artifact_uris': {str(folds / f'fold-{n}'): uri + f'/data/fold-{n}'
                                                    for n in range(1, plan['cv']['n_splits'] + 1)},
                           'child_name_prefix': f'lr-cv-trial{index:02d}', 'trials': [
                               {'trial_id': f'fold{fold}', 'head_learning_rate': item['head_lr'],
                                'weight_decay': trial_config['weight_decay'], 'label_smoothing': trial_config['label_smoothing'],
                                'window_none_loss_weight': config['window_none_loss_weight'],
                                'export_dir': str(folds / f'fold-{fold}')}
                               for fold in range(1, plan['cv']['n_splits'] + 1)]}
            nested_plan.pop('hyperopt')
            model = {**trial_config, 'export_dir': nested_plan['export_dir']}
            write_json(trial / 'plan.json', nested_plan, overwrite=False)
            write_json(trial / 'model-config.json', model, overwrite=False)
            tracking = trial / 'folds'
            command([str(args.python), str(ROOT / 'scripts/prepare_laya_cuda_experiment.py'),
                     '--plan', str(trial / 'plan.json'), '--model-config', str(trial / 'model-config.json'),
                     '--state-dir', str(tracking), '--tracking-uri', args.tracking_uri], f'trial-{index:02d}-prepare.log')
            nested_state = json.loads((tracking / 'state.json').read_text())
            item.update(run_id=nested_state['parent_run_id'], artifact_uri=nested_state['parent_artifact_uri'], status='running')
            write_json(state_path, state)
            tag(item['run_id'], 'mlflow.parentRunId', parent)
            tag(item['run_id'], 'jery.role', 'cv_trial')
            tag(item['run_id'], 'jery.cv_folds', plan['cv']['n_splits'])
            tag(item['run_id'], 'jery.primary_metric', 'cv_search_accuracy')
            _, params = trial_settings(plan, config, sampled)
            api('mlflow/runs/log-batch', {'run_id': item['run_id'],
                'params': [{'key': k, 'value': str(v)} for k, v in params.items()]})
            print('START CV TRIAL', index, item['run_id'], item['head_lr'], flush=True)
            command([str(args.python), str(ROOT / 'scripts/run_laya_full_experiment.py'),
                     '--python', str(args.python), '--model-config', str(trial / 'model-config.json'),
                     '--state-dir', str(tracking), '--tracking-uri', args.tracking_uri], f'trial-{index:02d}-folds.log')
            nested_state = json.loads((tracking / 'state.json').read_text())
            if nested_state['status'] != 'finished' or len(nested_state['children']) != plan['cv']['n_splits']:
                raise ValueError('Incomplete fold trial')
            reports = [json.loads((ROOT / config['outputs_dir'] / c['local_id'] / 'report.json').read_text())
                       for c in nested_state['children']]
            metrics = aggregate_folds(reports)
            api('mlflow/runs/log-batch', {'run_id': item['run_id'], 'metrics': [
                {'key': k, 'value': v, 'step': index, 'timestamp': int(time.time() * 1000)} for k, v in metrics.items()]})
            detail = {'metrics': metrics, 'fold_runs': nested_state['children'],
                      'std': 'sample standard deviation, ddof=1', 'test_role': 'mean over five fold-trained models; excluded from TPE'}
            write_json(trial / 'cv-summary.json', detail)
            store(item['run_id'], item['artifact_uri'], 'cv-summary.json', trial / 'cv-summary.json')
            item.update(status='finished', search_validation=metrics['cv_search_accuracy'], metrics=metrics,
                        fold_run_ids=[c['run_id'] for c in nested_state['children']])
            write_json(state_path, state)
            print('FINISHED CV TRIAL', index, metrics, flush=True)
        state['status'] = 'interrupted' if stopped else 'finished'
        if not stopped:
            winner = max(state['children'], key=lambda c: (c['search_validation'], -c['head_lr']))
            state['selected_trial_run_id'] = winner['run_id']
            tag(parent, 'jery.selected_trial', winner['run_id'])
        write_json(state_path, state)
        store(parent, uri, 'summary.json', state_path)
        api('mlflow/runs/update', {'run_id': parent, 'status': 'KILLED' if stopped else 'FINISHED',
                                 'end_time': int(time.time() * 1000)})
    except Exception as error:
        state.update(status='interrupted' if stopped else 'supervisor_failed', error=repr(error))
        write_json(state_path, state)
        tag(parent, 'jery.phase', state['status'])
        api('mlflow/runs/update', {'run_id': parent, 'status': 'KILLED' if stopped else 'FAILED',
                                 'end_time': int(time.time() * 1000)})
        raise


if __name__ == '__main__':
    main()
