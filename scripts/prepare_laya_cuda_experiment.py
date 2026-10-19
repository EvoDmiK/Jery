"""Prepare a distinct Jery CUDA parent; readiness uses one fixed one-epoch child, search uses five TPE trials."""
import argparse
import json
from pathlib import Path
import time

from run_laya_full_experiment import DEFAULT_TRACKING_URI, ROOT, write_json
from run_laya_repeats import upload
from sync_laya_training_metrics import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()
    folder = args.state_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'state.json'
    plan = json.loads((ROOT / 'configs/training/laya-balanced-20261018.json').read_text())
    config = json.loads((ROOT / 'configs/model/laya-cuda-20261019.json').read_text())
    plan.update(parent_name='laya-cuda-readiness-20261019' if args.readiness else 'laya-cuda-hyperopt-20261019',
                micro_batch=config['micro_batch'], grad_accum=config['grad_accum'], backend='cuda',
                previous_parent_run_id='9325d1ca84944fe281f94b519c59f305',
                child_name_prefix='laya-cuda-readiness' if args.readiness else 'laya-cuda-max64ep')
    if args.readiness:
        plan.pop('hyperopt')
        plan.update(epochs=1, head_learning_rates=[config['head_learning_rate']],
                    purpose='one full epoch, validation, checkpoint, persisted-head reload and final test; speed readiness only')
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
                     {'key': 'jery.purpose', 'value': 'readiness' if args.readiness else 'hyperopt'}]})['run']
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
