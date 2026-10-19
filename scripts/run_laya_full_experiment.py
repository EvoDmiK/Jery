"""Supervise full-data Laya runs with compact MLflow parent/child records and JSON detail."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.error

from run_laya_repeats import upload
from sync_laya_training_metrics import request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = 'https://mlflow.dove-nest.com'
PARAMS = {'head_lr', 'epochs', 'seed', 'effective_batch_size'}
METRICS = {'train_loss', 'val_accuracy', 'val_macro_f1', 'val_coverage',
           'test_accuracy', 'test_macro_f1', 'test_coverage'}
POLICY_V2 = 'laya-routing-v2'


def child_param_keys(plan):
    keys = set(plan['child_params'])
    if keys == PARAMS or (plan.get('training_policy_version') == POLICY_V2
                         and keys == PARAMS | {'window_none_loss_weight'}):
        return keys
    raise ValueError('Unexpected child tracking parameters')


def write_json(path, data, *, overwrite=True):
    if not overwrite and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        temporary = handle.name
    os.replace(temporary, path)


def json_lines(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    # A writer may be in the middle of its final line; never silently ignore complete corrupt records.
    return [json.loads(line) for line in lines if line.endswith('\n') and line.strip()]


def compact_epoch(record):
    epoch = record['epoch']
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError('Expected an integer completed epoch')
    validation = record['validation']
    values = {'val_accuracy': validation['case_accuracy'], 'val_macro_f1': validation['decision_macro_f1'],
              'val_coverage': validation['coverage']}
    if record['train_loss'] is not None:
        values['train_loss'] = record['train_loss']
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite epoch metric')
    return values


def final_epoch_bounds(report, max_epochs, best_validation=False):
    selection = report.get('selection', {})
    completed = selection['completed_epochs'] if best_validation else max_epochs
    selected = selection['selected_epoch'] if best_validation else max_epochs
    if not (isinstance(completed, int) and isinstance(selected, int) and 0 <= selected <= completed <= max_epochs):
        raise ValueError('Invalid selected/completed epoch')
    if report['validation']['epoch'] != selected or report['test']['epoch'] != selected:
        raise ValueError('Validation/test must use the same selected model')
    if best_validation and abs(report['validation']['case_accuracy'] - selection['best_score']) > 1e-9:
        raise ValueError('Restored model does not match the selected validation score')
    return selected, completed


def comparison_score(child, plan):
    """Never floor a v2 search result at the serving baseline."""
    key = 'search_validation' if plan.get('training_policy_version') == POLICY_V2 else 'final_validation'
    value = child[key]
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid {key}')
    return value


def validate_prepared_config(config, plan, prepared):
    if config != prepared or config.get('training_policy_version') != plan.get('training_policy_version'):
        raise ValueError('Model configuration differs from the prepared experiment')


def trained_search_score(report, epochs):
    selection = report['selection']
    if selection.get('policy_version') != POLICY_V2:
        raise ValueError('Expected the v2 selection contract')
    best = selection['best_trained_epoch']
    score = selection['best_trained_score']
    if not isinstance(best, int) or not 1 <= best <= selection['completed_epochs']:
        raise ValueError('Search must select a trained epoch')
    final_epoch_bounds(report, epochs, True)
    policy = {'training_policy_version': POLICY_V2}
    value = comparison_score({'search_validation': score}, policy)
    searched = comparison_score({'search_validation': selection['search_score']}, policy)
    if abs(searched - value) > 1e-9:
        raise ValueError('Search score differs from the trained checkpoint')
    descriptor = report['best_trained']
    described = comparison_score({'search_validation': descriptor['validation_case_accuracy']}, policy)
    if descriptor['epoch'] != best or abs(described - value) > 1e-9:
        raise ValueError('Best trained artifact does not match its selection record')
    baseline = report['before_validation']['case_accuracy']
    if selection['selected_epoch'] != (best if value > baseline else 0):
        raise ValueError('Serving selection does not match the baseline protection policy')
    return value


def hyperopt_candidate(plan, children):
    """Hyperopt ask/tell: only finished validation results enter TPE; persist the actual trial document as JSON."""
    import numpy as np
    from hyperopt import Trials, base, hp, space_eval, tpe, STATUS_OK
    from hyperopt.base import JOB_STATE_DONE
    search = plan['hyperopt']
    space = {'head_learning_rate': hp.loguniform('head_learning_rate', math.log(search['head_lr'][0]), math.log(search['head_lr'][1])),
             'weight_decay': hp.choice('weight_decay', search['weight_decay']),
             'label_smoothing': hp.choice('label_smoothing', search['label_smoothing'])}
    if 'window_none_loss_weight' in search:
        space['window_none_loss_weight'] = hp.choice('window_none_loss_weight', search['window_none_loss_weight'])
    trials = Trials()
    for child in children:
        if child['status'] != 'finished':
            continue
        document = json.loads(json.dumps(child['hyperopt']['document']))
        document.update(state=JOB_STATE_DONE, result={'status': STATUS_OK, 'loss': 1 - comparison_score(child, plan)})
        trials.insert_trial_docs([document])
    trials.refresh()
    tid = trials.new_trial_ids(1)[0]
    documents = tpe.suggest([tid], base.Domain(lambda value: value, space), trials,
                            seed=int(plan['seed'] + tid), n_startup_jobs=search['startup_trials'])
    values = {key: value[0] for key, value in documents[0]['misc']['vals'].items() if value}
    parameters = space_eval(space, values)
    # Hyperopt uses NumPy scalars; its trial dates are None until the objective is scheduled.
    document = json.loads(json.dumps(documents[0], default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    return {'parameters': {k: float(v) for k, v in parameters.items()}, 'document': document,
            'algorithm': 'hyperopt.tpe.suggest',
            'objective': '1 - best_trained_validation_case_accuracy' if plan.get('training_policy_version') == POLICY_V2
                         else '1 - best_validation_case_accuracy'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--state-dir', type=Path, default=ROOT / 'outputs/laya-full-20261017/tracking')
    parser.add_argument('--model-config', type=Path, default=ROOT / 'configs/model/laya-full-20261017.json')
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    parser.add_argument('--wait-for-run', help='Start only after this existing MLflow parent finishes successfully')
    parser.add_argument('--detach', action='store_true')
    options = parser.parse_args()
    folder = options.state_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if options.detach:
        args = [sys.executable, str(Path(__file__).absolute())] + [a for a in sys.argv[1:] if a != '--detach']
        with (folder / 'supervisor.log').open('a') as log:
            process = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'supervisor_pid': process.pid, 'log': str(folder / 'supervisor.log')}))
        return
    lock = (folder / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    state_path = folder / 'state.json'
    state = json.loads(state_path.read_text())
    plan = state['plan']
    expected_params = child_param_keys(plan)
    if set(plan['child_metrics']) != METRICS:
        raise ValueError('Tracking allowlist differs from the declared plan')
    config = json.loads(options.model_config.read_text())
    if plan.get('training_policy_version') == POLICY_V2:
        validate_prepared_config(config, plan, json.loads((folder / 'configuration.json').read_text()))
    base, parent = options.tracking_uri, state['parent_run_id']
    expected_export = json.loads((ROOT / plan['export_dir'] / 'manifest.json').read_text())
    for split in ('train', 'validation', 'test'):
        path = ROOT / plan['export_dir'] / (split + '.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_export['splits'][split]['sha256']:
            raise ValueError('Export changed: ' + split)
    os.environ.update(PYTHONPATH=str(ROOT / 'src'), HF_HUB_OFFLINE='1', USE_TF='0', TOKENIZERS_PARALLELISM='false')
    state.update(status='running', supervisor_pid=os.getpid(), tracking_uri=base,
                 started_at=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
    write_json(state_path, state)
    stopped, child_process = False, None

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if child_process is not None and child_process.poll() is None:
            child_process.terminate()  # The local-training runner forwards termination to its training group.

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def transient(error):
        return (isinstance(error, urllib.error.HTTPError) and error.code in (429, 500, 502, 503, 504)
                or isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))
                and not isinstance(error, urllib.error.HTTPError))

    def network_call(function, *args):
        while True:
            try:
                result = function(*args)
                if state.pop('tracking_error', None) is not None:
                    state['tracking_recovered_at'] = datetime.now(timezone.utc).isoformat()
                    write_json(state_path, state)
                return result
            except Exception as error:
                if not transient(error) or stopped:
                    raise
                state['tracking_error'] = repr(error)
                write_json(state_path, state)
                print('Tracking connection retry; local training is preserved:', repr(error), flush=True)
                time.sleep(15)

    def api(path, payload=None):
        return network_call(request, base, path, payload)

    def tag(run_id, key, value):
        api('mlflow/runs/set-tag', {'run_id': run_id, 'key': key, 'value': str(value)})

    def store_json(run_id, uri, name, path):
        proof = network_call(upload, base, uri, name, path)
        return proof

    def snapshot(item, out):
        # Raw histories and diagnostics stay in JSON artifacts, never flattened into MLflow keys.
        bundle = folder / item['local_id'] / 'json'
        proofs = item.setdefault('verified_artifacts', {})
        for path in sorted(out.iterdir()):
            if not path.is_file():
                continue
            if path.suffix == '.json':
                target = path
            elif path.suffix == '.jsonl':
                target = bundle / (path.stem + '.json')
                write_json(target, json_lines(path))
            elif path.suffix == '.log':
                target = bundle / (path.stem + '-log.json')
                write_json(target, {'text': path.read_text(), 'source_file': path.name})
            else:
                continue
            name = 'details/' + target.name
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if proofs.get(name, {}).get('sha256') != digest:
                proofs[name] = store_json(item['run_id'], item['artifact_uri'], name, target)

    def sync_epochs(item, out):
        rows = json_lines(out / 'metrics.jsonl')
        # Recover remote acknowledgment after a supervisor restart without adding duplicate points.
        if 'logged_epochs' not in item:
            query = urllib.parse.urlencode({'run_id': item['run_id'], 'metric_key': 'val_accuracy'})
            item['logged_epochs'] = sorted({p['step'] for p in api('mlflow/metrics/get-history?' + query).get('metrics', [])})
        for row in rows:
            epoch = row['epoch']
            if epoch in item['logged_epochs']:
                continue
            values = compact_epoch(row)
            if row['train_loss'] is None:
                # A checkpoint can precede its validation; recover the actual logged losses if available.
                steps = row['steps']; per_epoch = steps // epoch
                losses = [s['loss'] for s in json_lines(out / 'steps.jsonl') if steps - per_epoch < s['step'] <= steps]
                if len(losses) == per_epoch:
                    values['train_loss'] = sum(losses) / len(losses)
            timestamp = int(datetime.fromisoformat(row['time']).timestamp() * 1000)
            api('mlflow/runs/log-batch', {'run_id': item['run_id'], 'metrics': [
                {'key': k, 'value': v, 'timestamp': timestamp, 'step': epoch} for k, v in values.items()]})
            item['logged_epochs'].append(epoch)
            item['epoch'] = epoch
            print(f"{item['local_id']} epoch {epoch}/{plan['epochs']} val={values['val_accuracy']:.4f}", flush=True)
            # Compact but detailed epoch history remains available as JSON throughout the run.
            history = folder / item['local_id'] / 'epochs.json'
            write_json(history, rows)
            item.setdefault('verified_artifacts', {})['details/epochs.json'] = store_json(
                item['run_id'], item['artifact_uri'], 'details/epochs.json', history)
            write_json(state_path, state)

    if options.wait_for_run:
        if options.wait_for_run == parent:
            raise ValueError('Cannot wait for the current parent itself')
        state['status'] = 'waiting_for_upstream'
        state['wait_for_run_id'] = options.wait_for_run
        write_json(state_path, state)
        tag(parent, 'jery.phase', 'waiting_for_upstream')
        while not stopped:
            upstream = api('mlflow/runs/get?run_id=' + options.wait_for_run)['run']['info']['status']
            if upstream == 'FINISHED':
                break
            if upstream != 'RUNNING':
                raise RuntimeError(f'Upstream run ended without success: {upstream}')
            time.sleep(15)
        if stopped:
            state['status'] = 'interrupted'
            write_json(state_path, state)
            api('mlflow/runs/update', {'run_id': parent, 'status': 'KILLED', 'end_time': int(time.time()*1000)})
            return
        state['status'] = 'running'
        write_json(state_path, state)
        tag(parent, 'jery.phase', 'running')
    api('mlflow/runs/update', {'run_id': parent, 'status': 'RUNNING'})
    tag(parent, 'jery.phase', 'running')
    budget = (plan['hyperopt']['max_evals'] if plan.get('hyperopt') else
              len(plan['trials']) if plan.get('trials') else len(plan['head_learning_rates']))
    for index in range(1, budget + 1):
        prior = [c for c in state['children'] if c['index'] == index]
        if prior and prior[-1]['status'] == 'finished':
            continue
        if stopped:
            break
        search_trial = (prior[-1]['hyperopt'] if prior else hyperopt_candidate(plan, state['children'])) if plan.get('hyperopt') else None
        fixed_trial = plan['trials'][index - 1] if plan.get('trials') else None
        sampled = (search_trial['parameters'] if search_trial else
                   {k: v for k, v in fixed_trial.items() if k != 'trial_id'} if fixed_trial else
                   {'head_learning_rate': plan['head_learning_rates'][index - 1]})
        lr = sampled['head_learning_rate']
        overrides = {'epochs': plan['epochs'], 'seed': plan['seed'], 'micro_batch': plan['micro_batch'],
                     'grad_accum': plan['grad_accum'], **sampled, 'export_dir': plan['export_dir']}
        item = prior[-1] if prior else None
        if item is None:
            trial_id = parent + ':' + str(index)
            # Run creation is not idempotent: search the stable trial tag after an uncertain response.
            while True:
                found = api('mlflow/runs/search', {'experiment_ids': [plan['experiment_id']],
                    'filter': "tags.`jery.trial_id` = '" + trial_id + "'", 'max_results': 2}).get('runs', [])
                if len(found) > 1:
                    raise ValueError('Multiple child runs share the same trial ID')
                if found:
                    remote = found[0]
                    break
                try:
                    remote = request(base, 'mlflow/runs/create', {'experiment_id': plan['experiment_id'], 'start_time': int(time.time()*1000),
                        'tags': [{'key': 'mlflow.runName', 'value': f"{plan.get('child_name_prefix', 'laya-full-64ep')}-" +
                                  (fixed_trial['trial_id'] if fixed_trial else f'lr{lr:g}')},
                                 {'key': 'mlflow.parentRunId', 'value': parent}, {'key': 'jery.role', 'value': 'child'},
                                 {'key': 'jery.backend', 'value': config.get('backend', 'mlx')},
                                 {'key': 'jery.trial_id', 'value': trial_id}]})['run']
                    break
                except Exception as error:
                    if not transient(error) or stopped:
                        raise
                    time.sleep(10)
            item = {'head_lr': lr, 'index': index, 'run_id': remote['info']['run_id'],
                    'artifact_uri': remote['info']['artifact_uri'], 'local_id': f'trial{index}-lr{lr:g}-{parent[:8]}',
                    'status': 'initializing'}
            if search_trial:
                item['hyperopt'] = search_trial
            if fixed_trial:
                item['condition'] = fixed_trial
            state['children'].append(item); write_json(state_path, state)
        params = {'head_lr': lr, 'epochs': plan['epochs'], 'seed': plan['seed'],
                  'effective_batch_size': plan['micro_batch'] * plan['grad_accum']}
        if 'window_none_loss_weight' in expected_params:
            params['window_none_loss_weight'] = sampled.get('window_none_loss_weight', config.get('window_none_loss_weight', 1.0))
        if plan.get('training_policy_version') == POLICY_V2:
            tag(item['run_id'], 'jery.training_policy', POLICY_V2)
            tag(item['run_id'], 'jery.window_none_loss_weight', sampled.get('window_none_loss_weight', config.get('window_none_loss_weight', 1.0)))
        api('mlflow/runs/update', {'run_id': item['run_id'], 'status': 'RUNNING'})
        api('mlflow/runs/log-batch', {'run_id': item['run_id'], 'params': [{'key': k, 'value': str(v)} for k,v in params.items()]})
        local = folder / item['local_id']; local.mkdir(parents=True, exist_ok=True)
        configuration = local / 'configuration.json'
        write_json(configuration, {'config': {**config, **overrides}, 'plan': plan,
                                  'data_parent_artifact_uri': state['parent_artifact_uri'] + '/data'}, overwrite=False)
        item.setdefault('verified_artifacts', {})['configuration.json'] = store_json(
            item['run_id'], item['artifact_uri'], 'configuration.json', configuration)
        sources = {}
        for directory in ['src/jery/model', 'src/jery/evaluation', 'src/jery/runtime']:
            for path in (ROOT / directory).glob('*.py'):
                sources[str(path.relative_to(ROOT))] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'content': path.read_text()}
        for name in ['src/jery/__init__.py', 'scripts/run_laya_full_experiment.py', 'scripts/run_local_training.py',
                     'scripts/run_laya_repeats.py', 'scripts/sync_laya_training_metrics.py',
                     'scripts/build_laya_full_dataset.py', 'scripts/build_laya_pilot.py',
                     'scripts/prepare_laya_cuda_experiment.py', 'scripts/check_laya_cuda_resume.py', 'requirements-decision.txt',
                     'requirements-decision-cuda.txt']:
            path = ROOT / name
            sources[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'content': path.read_text()}
        write_json(local / 'source.json', sources, overwrite=False)
        item['verified_artifacts']['source.json'] = store_json(item['run_id'], item['artifact_uri'], 'source.json', local / 'source.json')
        out = ROOT / config['outputs_dir'] / item['local_id']
        command = [str(options.python.absolute()), '-m', 'jery.model.laya_full', '--config', str(options.model_config.absolute()),
                   '--assets-root', str(ROOT), '--run-id', item['local_id'], '--override', json.dumps(overrides)]
        if out.exists():
            if (out / 'status.json').exists() and json.loads((out / 'status.json').read_text())['status'] == 'succeeded':
                child_process = None
            else:
                command.append('--resume')
        if not (out / 'report.json').exists():
            attempt = 1
            while (local / f'runner-{attempt}').exists():
                attempt += 1
            wrapper = [sys.executable, str(ROOT / 'scripts/run_local_training.py'), '--output', str(local / f'runner-{attempt}'),
                       '--timeout-hours', str(plan['timeout_hours_per_child']), '--'] + command
            with (local / 'runner-stdout.json.log').open('a') as wrapper_log:
                child_process = subprocess.Popen(wrapper, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=wrapper_log, stderr=subprocess.STDOUT, start_new_session=True)
            item.update(status='running', runner_pid=child_process.pid); write_json(state_path, state)
            print('START', index, item['local_id'], item['run_id'], flush=True)
            next_snapshot = time.monotonic()
            while child_process.poll() is None:
                try:
                    sync_epochs(item, out)
                    if out.exists() and time.monotonic() >= next_snapshot:
                        # Keep a small live JSON progress record; full step arrays upload after completion.
                        progress = {'epoch': item.get('epoch', 0), 'status': 'running', 'runner_pid': child_process.pid,
                                    'updated_at': datetime.now(timezone.utc).isoformat()}
                        step_rows = json_lines(out / 'steps.jsonl')
                        progress['last_optimizer_step'] = step_rows[-1] if step_rows else None
                        write_json(local / 'progress.json', progress)
                        item['verified_artifacts']['progress.json'] = store_json(item['run_id'], item['artifact_uri'], 'progress.json', local / 'progress.json')
                        parent_progress = {'status': state['status'], 'updated_at': progress['updated_at'],
                            'planned_learning_rates': plan.get('head_learning_rates', []), 'hyperopt': plan.get('hyperopt'),
                            'planned_conditions': plan.get('trials'), 'training_policy_version': plan.get('training_policy_version'),
                            'children': [{k: c.get(k) for k in ('run_id', 'head_lr', 'status', 'epoch', 'runner_pid')}
                                         for c in state['children']]}
                        write_json(folder / 'progress.json', parent_progress)
                        state.setdefault('verified_artifacts', {})['progress.json'] = store_json(
                            parent, state['parent_artifact_uri'], 'progress.json', folder / 'progress.json')
                        next_snapshot = time.monotonic() + 300
                except Exception as exc:
                    item['tracking_error'] = repr(exc); write_json(state_path, state)
                    print('Tracking will retry; training continues:', repr(exc), flush=True)
                time.sleep(10)
            code = child_process.returncode
            if code:
                item.update(status='interrupted' if stopped else 'failed', returncode=code)
                state['status'] = item['status']; write_json(state_path, state)
                try:
                    if out.exists(): snapshot(item, out)
                    api('mlflow/runs/update', {'run_id': item['run_id'], 'status': 'KILLED' if stopped else 'FAILED', 'end_time': int(time.time()*1000)})
                    api('mlflow/runs/update', {'run_id': parent, 'status': 'KILLED' if stopped else 'FAILED', 'end_time': int(time.time()*1000)})
                finally:
                    write_json(state_path, state)
                return
        item['status'] = 'uploading'; write_json(state_path, state)
        # Upload failures preserve the trained head; never retrain a completed report.
        sync_epochs(item, out)
        report = json.loads((out / 'report.json').read_text())
        final = report['validation']; test = report['test']
        selected_epoch, completed_epochs = final_epoch_bounds(report, plan['epochs'], bool(config.get('validation_callback')))
        search_score = (trained_search_score(report, plan['epochs'])
                        if plan.get('training_policy_version') == POLICY_V2 else final['case_accuracy'])
        if plan.get('training_policy_version') == POLICY_V2:
            epoch_rows = json_lines(out / 'metrics.jsonl')
            if abs(max(r['validation']['case_accuracy'] for r in epoch_rows) - search_score) > 1e-9:
                raise ValueError('Search score differs from the actual trained epoch history')
            for key, value in {'jery.selected_epoch': selected_epoch,
                               'jery.best_trained_epoch': report['selection']['best_trained_epoch'],
                               'jery.best_trained_score': search_score,
                               'jery.selection_outcome': 'baseline_retained' if selected_epoch == 0 else 'trained_selected'}.items():
                tag(item['run_id'], key, value)
        if not report['updates']['encoder_unchanged'] or not report['restore']['same_validation_case_predictions']:
            raise ValueError('Frozen encoder/restore verification failed')
        api('mlflow/runs/log-batch', {'run_id': item['run_id'], 'metrics': [
            {'key': 'test_accuracy', 'value': test['case_accuracy'], 'step': selected_epoch, 'timestamp': int(time.time()*1000)},
            {'key': 'test_macro_f1', 'value': test['decision_macro_f1'], 'step': selected_epoch, 'timestamp': int(time.time()*1000)},
            {'key': 'test_coverage', 'value': test['coverage'], 'step': selected_epoch, 'timestamp': int(time.time()*1000)}]})
        snapshot(item, out)
        for runner in sorted(local.glob('runner-*')):
            if runner.is_dir():
                execution = {'runner': json.loads((runner / 'run.json').read_text()),
                             'log': (runner / 'training.log').read_text()}
                target = local / (runner.name + '.json')
                write_json(target, execution)
                name = 'execution/' + target.name
                item['verified_artifacts'][name] = store_json(item['run_id'], item['artifact_uri'], name, target)
        head = ROOT / report['restore']['artifact']
        if hashlib.sha256(head.read_bytes()).hexdigest() != report['restore']['artifact_sha256']:
            raise ValueError('Saved head checksum mismatch')
        for path in head.parent.iterdir():
            if path.is_file():
                name = 'model/' + path.name
                item['verified_artifacts'][name] = store_json(item['run_id'], item['artifact_uri'], name, path)
        if report.get('last_trained'):
            trained = ROOT / report['last_trained']['artifact']
            if hashlib.sha256(trained.read_bytes()).hexdigest() != report['last_trained']['artifact_sha256']:
                raise ValueError('Last trained head checksum mismatch')
            for path in trained.parent.iterdir():
                if path.is_file():
                    name = 'model/last-trained/' + path.name
                    item['verified_artifacts'][name] = store_json(item['run_id'], item['artifact_uri'], name, path)
        if report.get('best_trained'):
            trained = ROOT / report['best_trained']['artifact']
            if hashlib.sha256(trained.read_bytes()).hexdigest() != report['best_trained']['artifact_sha256']:
                raise ValueError('Best trained head checksum mismatch')
            for path in trained.parent.iterdir():
                if path.is_file():
                    name = 'model/best-trained/' + path.name
                    item['verified_artifacts'][name] = store_json(item['run_id'], item['artifact_uri'], name, path)
        dependencies = local / 'model-dependencies.json'
        write_json(dependencies, {'base_artifact_uri': plan['base_artifact_uri'], 'head_sha256': report['restore']['artifact_sha256'],
                                 'base_revision': report['provenance']['model']['revision'],
                                 'format': report['restore'].get('format', 'unsloth-mlx-laya-trainable-v1')})
        item['verified_artifacts']['model/dependencies.json'] = store_json(item['run_id'], item['artifact_uri'], 'model/dependencies.json', dependencies)
        saved = api('mlflow/runs/get?run_id=' + item['run_id'])['run']
        if {p['key'] for p in saved['data']['params']} != expected_params or {m['key'] for m in saved['data']['metrics']} != METRICS:
            raise ValueError('Nonessential MLflow keys were logged, or required keys are missing')
        for key in ['train_loss', 'val_accuracy', 'val_macro_f1', 'val_coverage']:
            query = urllib.parse.urlencode({'run_id': item['run_id'], 'metric_key': key})
            history = api('mlflow/metrics/get-history?' + query)['metrics']
            if {m['step'] for m in history} != set(range(1, completed_epochs + 1)):
                raise ValueError('Incomplete epoch curve: ' + key)
        api('mlflow/runs/update', {'run_id': item['run_id'], 'status': 'FINISHED', 'end_time': int(time.time()*1000)})
        item.update(status='finished', final_validation=final['case_accuracy'], final_test=test['case_accuracy'],
                    selected_epoch=selected_epoch, completed_epochs=completed_epochs, search_validation=search_score)
        write_json(state_path, state)
        if plan.get('hyperopt'):
            trials_path = folder / 'hyperopt-trials.json'
            write_json(trials_path, {'search': plan['hyperopt'], 'trials': [
                {'index': c['index'], **c['hyperopt'], 'status': c['status'],
                 'objective_loss': 1 - comparison_score(c, plan) if c['status'] == 'finished' else None}
                for c in state['children']]})
            store_json(parent, state['parent_artifact_uri'], 'hyperopt-trials.json', trials_path)
        print('FINISHED', item['local_id'], flush=True)
    if stopped:
        state['status'] = 'interrupted'; write_json(state_path, state)
        api('mlflow/runs/update', {'run_id': parent, 'status': 'KILLED', 'end_time': int(time.time()*1000)})
        return
    selected = max(state['children'], key=lambda c: (comparison_score(c, plan), -c['head_lr']))
    state.update(status='finished', selected_child_run_id=selected['run_id'])
    if plan.get('training_policy_version') == POLICY_V2:
        state['selected_search_score'] = comparison_score(selected, plan)
        tag(parent, 'jery.training_outcome', 'trained_improvement' if any(c['selected_epoch'] > 0 for c in state['children'])
            else 'all_baseline_retained')
    write_json(state_path, state)
    api('mlflow/runs/log-batch', {'run_id': parent, 'metrics': [
        {'key': 'best_val_accuracy', 'value': comparison_score(selected, plan), 'step': plan['epochs'], 'timestamp': int(time.time()*1000)},
        {'key': 'selected_test_accuracy', 'value': selected['final_test'], 'step': plan['epochs'], 'timestamp': int(time.time()*1000)}]})
    summary = folder / 'summary.json'; write_json(summary, state)
    network_call(upload, base, state['parent_artifact_uri'], 'summary.json', summary)
    tag(parent, 'jery.phase', 'finished')
    api('mlflow/runs/update', {'run_id': parent, 'status': 'FINISHED', 'end_time': int(time.time()*1000)})
    print('ALL CHILDREN FINISHED', parent, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        index = sys.argv.index('--state-dir') + 1 if '--state-dir' in sys.argv else None
        directory = Path(sys.argv[index]) if index is not None else ROOT / 'outputs/laya-full-20261017/tracking'
        record = {'pid': os.getpid(), 'error': repr(error), 'traceback': traceback.format_exc(),
                  'time': datetime.now(timezone.utc).isoformat()}
        write_json(directory / ('supervisor-error-' + str(os.getpid()) + '.json'), record)
        path = directory / 'state.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state.get('supervisor_pid') == os.getpid():
                state.update(status='supervisor_failed', error=repr(error))
                write_json(path, state)
        raise
