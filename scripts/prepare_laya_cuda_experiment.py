"""Prepare a distinct Jery CUDA parent; readiness uses one fixed one-epoch child, search uses five TPE trials."""
import argparse
import json
import hashlib
import math
from pathlib import Path
import time

from run_laya_full_experiment import DEFAULT_TRACKING_URI, ROOT, PARAMS, METRICS, POLICY_V2, write_json
from run_laya_repeats import upload
from sync_laya_training_metrics import request


def validate_explicit_plan(plan, config):
    if plan.get('backend') != 'cuda' or config.get('backend') != 'cuda':
        raise ValueError('CUDA backend required')
    if plan.get('training_policy_version') != config.get('training_policy_version'):
        raise ValueError('Plan/model policy versions differ')
    if set(plan['child_params']) != PARAMS or set(plan['child_metrics']) != METRICS:
        raise ValueError('Tracking allowlist changed')
    for key in ('seed', 'micro_batch', 'grad_accum'):
        if plan[key] != config[key]:
            raise ValueError(f'Plan/model {key} differs')
    if not isinstance(plan['epochs'], int) or not 1 <= plan['epochs'] <= config['epochs']:
        raise ValueError('Invalid epoch budget')
    if sum(bool(plan.get(k)) for k in ('trials', 'hyperopt', 'head_learning_rates')) != 1:
        raise ValueError('Exactly one trial schedule is required')
    if plan.get('training_policy_version') == POLICY_V2:
        if plan['validation_callback'] != config['validation_callback']:
            raise ValueError('Plan/model callback policies differ')
        if plan['epochs'] < config['validation_callback']['min_epochs']:
            raise ValueError('Diagnostic budget is shorter than minimum observation')
    if plan.get('trials'):
        if plan.get('hyperopt') or plan.get('head_learning_rates'):
            raise ValueError('Fixed and searched trials cannot be mixed')
        names = set()
        for trial in plan['trials']:
            if set(trial) != {'trial_id', 'head_learning_rate', 'weight_decay', 'label_smoothing', 'window_none_loss_weight'}:
                raise ValueError('Unexpected fixed-trial fields')
            if not isinstance(trial['trial_id'], str) or not trial['trial_id'] or trial['trial_id'] in names:
                raise ValueError('Unique nonempty fixed trial IDs required')
            names.add(trial['trial_id'])
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for k, v in trial.items() if k != 'trial_id'):
                raise ValueError('Nonfinite trial parameter')
            if (trial['head_learning_rate'] <= 0 or trial['weight_decay'] < 0
                    or not 0 <= trial['label_smoothing'] < 1 or not 0 < trial['window_none_loss_weight'] <= 1):
                raise ValueError('Invalid fixed-trial parameter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--plan', type=Path, help='Explicit versioned experiment plan; cannot combine with --readiness')
    parser.add_argument('--model-config', type=Path, default=ROOT / 'configs/model/laya-cuda-20261019.json')
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()
    folder = args.state_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'state.json'
    if args.plan and args.readiness:
        parser.error('Use either an explicit plan or --readiness')
    plan = json.loads((args.plan or ROOT / 'configs/training/laya-balanced-20261018.json').read_text())
    config = json.loads(args.model_config.read_text())
    if not args.plan:
        plan.update(parent_name='laya-cuda-readiness-20261019' if args.readiness else 'laya-cuda-hyperopt-20261019',
                micro_batch=config['micro_batch'], grad_accum=config['grad_accum'], backend='cuda',
                previous_parent_run_id='9325d1ca84944fe281f94b519c59f305',
                child_name_prefix='laya-cuda-readiness' if args.readiness else 'laya-cuda-max64ep')
    else:
        validate_explicit_plan(plan, config)
    if args.readiness:
        plan.pop('hyperopt')
        plan.update(epochs=1, head_learning_rates=[config['head_learning_rate']],
                    purpose='one full epoch, validation, checkpoint, persisted-head reload and final test; speed readiness only')
    manifest = json.loads((ROOT / plan['export_dir'] / 'manifest.json').read_text())
    for split in ('train', 'validation', 'test'):
        source = ROOT / plan['export_dir'] / (split + '.jsonl')
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['splits'][split]['sha256']:
            raise ValueError('Export changed before experiment preparation: ' + split)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['plan'] != plan:
            raise ValueError('Existing experiment has a different plan')
        if (folder / 'configuration.json').exists() and json.loads((folder / 'configuration.json').read_text()) != config:
            raise ValueError('Existing experiment has a different model configuration')
    else:
        experiment = request(args.tracking_uri, 'mlflow/experiments/get?experiment_id=' + plan['experiment_id'])['experiment']
        if experiment['name'] != 'Jery':
            raise ValueError('Expected Jery experiment')
        remote = request(args.tracking_uri, 'mlflow/runs/create', {
            'experiment_id': plan['experiment_id'], 'start_time': int(time.time() * 1000),
            'tags': [{'key': 'mlflow.runName', 'value': plan['parent_name']},
                     {'key': 'jery.role', 'value': 'parent'}, {'key': 'jery.backend', 'value': 'cuda'},
                     {'key': 'jery.purpose', 'value': 'controlled_ablation' if plan.get('trials') else
                      'readiness' if args.readiness else 'hyperopt'}]})['run']
        state = {'parent_run_id': remote['info']['run_id'], 'parent_artifact_uri': remote['info']['artifact_uri'],
                 'status': 'prepared', 'children': [], 'plan': plan, 'tracking_uri': args.tracking_uri}
        write_json(state_path, state)
    proofs = state.setdefault('verified_artifacts', {})
    for name, data in [('plan.json', plan), ('configuration.json', config),
                       ('data/dependencies.json', {'base_artifact_uri': plan['base_artifact_uri'],
                                                   'previous_parent': plan['previous_parent_run_id']})]:
        target = folder / name
        write_json(target, data)
        proofs[name] = upload(args.tracking_uri, state['parent_artifact_uri'], name, target)
    for split in ('train', 'validation', 'test'):
        path = ROOT / plan['export_dir'] / (split + '.jsonl')
        target = folder / 'data' / (split + '.json')
        write_json(target, [json.loads(line) for line in path.read_text().splitlines()])
        proofs['data/' + target.name] = upload(args.tracking_uri, state['parent_artifact_uri'], 'data/' + target.name, target)
    for name, path in [('data/export-manifest.json', ROOT / plan['export_dir'] / 'manifest.json'),
                        ('data/source-manifest.json', ROOT / 'datasets/mcp-routing-v2/manifest.json'),
                        ('data/tool-catalog.json', ROOT / 'datasets/mcp-routing-v2/tools.json')]:
        proofs[name] = upload(args.tracking_uri, state['parent_artifact_uri'], name, path)
    write_json(state_path, state)
    print(json.dumps({'parent_run_id': state['parent_run_id'], 'state_dir': str(folder), 'status': state['status']}))


if __name__ == '__main__':
    main()
