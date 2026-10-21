"""Backfill optimizer comparison params from immutable run artifacts; never retrain or replace params."""
import argparse
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from run_laya_full_experiment import COMPARISON_PARAMS, DEFAULT_TRACKING_URI, write_json
from sync_laya_training_metrics import request


def comparison_params(config):
    values = {}
    for key in sorted(COMPARISON_PARAMS & config.keys()):
        value = config[key]
        if key == 'lr_scheduler':
            valid = isinstance(value, str) and bool(value)
        elif key == 'warmup_steps':
            valid = type(value) is int and value >= 0
        else:
            valid = type(value) in (int, float) and math.isfinite(value)
            if valid:
                valid = (value >= 0 if key == 'weight_decay' else
                         0 <= value < 1 if key == 'label_smoothing' else value > 0)
        if not valid:
            raise ValueError('Invalid artifact parameter: ' + key)
        values[key] = str(value)
    return values


def missing_params(expected, existing):
    missing = {}
    for key, value in expected.items():
        if key not in existing:
            missing[key] = value
        elif key == 'lr_scheduler':
            if existing[key] != value:
                raise ValueError('Existing parameter conflicts with artifact: ' + key)
        elif float(existing[key]) != float(value):
            raise ValueError('Existing parameter conflicts with artifact: ' + key)
    return missing


def artifact_config(base, run):
    prefix = urllib.parse.urlsplit(run['info']['artifact_uri']).path.strip('/')
    for name in ('configuration.json', 'details/report.json'):
        url = base.rstrip('/') + '/api/2.0/mlflow-artifacts/artifacts/' + urllib.parse.quote(prefix + '/' + name, safe='/')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Jery-MLflow/1.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.load(response)
            config = data.get('config', data)
            if COMPARISON_PARAMS & config.keys():
                return config, name
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    return None, None


def sync_once(base, experiment, parent=None):
    query = ("tags.`mlflow.parentRunId` = '" + parent + "'" if parent else "tags.`jery.role` = 'child'")
    payload = {'experiment_ids': [experiment], 'filter': query, 'max_results': 1000}
    runs = []
    while True:
        page = request(base, 'mlflow/runs/search', payload)
        runs.extend(page.get('runs', []))
        if not page.get('next_page_token'):
            break
        payload['page_token'] = page['next_page_token']
    results = []
    for run in runs:
        rid = run['info']['run_id']
        existing = {p['key']: p['value'] for p in run['data'].get('params', [])}
        try:
            config, source = artifact_config(base, run)
            if config is None:
                results.append({'run_id': rid, 'status': 'unavailable', 'reason': 'No configuration artifact yet'})
                continue
            expected = comparison_params(config)
            missing = missing_params(expected, existing)
            if missing:
                request(base, 'mlflow/runs/log-batch', {'run_id': rid,
                    'params': [{'key': k, 'value': v} for k, v in missing.items()]})
            current = request(base, 'mlflow/runs/get?run_id=' + rid)['run']
            actual = {p['key']: p['value'] for p in current['data'].get('params', [])}
            if missing_params(expected, actual):
                raise RuntimeError('Parameter read-back failed')
            results.append({'run_id': rid, 'status': 'verified', 'source': source,
                            'expected': expected, 'added': missing,
                            'unavailable_fields': sorted(COMPARISON_PARAMS - config.keys())})
        except Exception as error:
            results.append({'run_id': rid, 'status': 'error', 'error': str(error)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking-uri', default=DEFAULT_TRACKING_URI)
    parser.add_argument('--experiment-id', default='47')
    parser.add_argument('--watch-parent', help='Keep filling newly created children until this parent ends')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    while True:
        try:
            results = sync_once(options.tracking_uri, options.experiment_id, options.watch_parent)
            parent_status = None
            if options.watch_parent:
                parent_status = request(options.tracking_uri, 'mlflow/runs/get?run_id=' + options.watch_parent)['run']['info']['status']
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if not options.watch_parent:
                raise
            print('Tracking connection retry: ' + str(error), flush=True)
            time.sleep(15)
            continue
        summary = {'runs': len(results), 'verified': sum(r['status'] == 'verified' for r in results),
                   'added_params': sum(len(r.get('added', {})) for r in results),
                   'errors': sum(r['status'] == 'error' for r in results),
                   'parent_status': parent_status}
        write_json(options.output, {'summary': summary, 'results': results})
        print(json.dumps(summary), flush=True)
        if not options.watch_parent or parent_status != 'RUNNING':
            raise SystemExit(bool(summary['errors']))
        time.sleep(15)


if __name__ == '__main__':
    main()
