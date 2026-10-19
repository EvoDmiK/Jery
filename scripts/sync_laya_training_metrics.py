"""Backfill completed Laya pilot logs into an existing MLflow run, without retraining."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
import urllib.parse
import urllib.request


def request(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {'User-Agent': 'Jery-MLflow/1.0', 'Content-Type': 'application/json'}
    req = urllib.request.Request(base.rstrip('/') + '/api/2.0/' + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read()
    return json.loads(body) if body else {}


def training_metrics(directory, prefix=''):
    """Use original step values; head LR is derived from the shared logged LR scale.

    Timestamps are reconstructed from the first step's second-resolution run.log
    timestamp and monotonic elapsed times. Epoch loss is the mean logged optimizer
    step loss, not a new full-dataset evaluation.
    """
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    steps = json.loads((directory / 'steps.json').read_text())
    if [s['step'] for s in steps] != list(range(1, report['config']['max_steps'] + 1)):
        raise ValueError('Missing, duplicate or unordered optimizer steps')
    match = re.search(r'^(\S+) step 1:', (directory / 'run.log').read_text(), re.M)
    if not match:
        raise ValueError('Missing original timestamp anchor')
    start = datetime.fromisoformat(match[1]).timestamp() - steps[0]['seconds']
    encoder_lr = report['config']['encoder_learning_rate']
    if encoder_lr <= 0:
        raise ValueError('Cannot derive head LR from a nonpositive logged encoder LR')
    points, epochs = [], defaultdict(list)

    def append(key, value, step, seconds):
        if not math.isfinite(value):
            raise ValueError('Nonfinite metric: ' + key)
        points.append({'key': prefix + key, 'value': value, 'step': step,
                       'timestamp': round((start + seconds) * 1000)})

    for row in steps:
        step, seconds = row['step'], row['seconds']
        for key in ('loss', 'epoch', 'grad_norm', 'peak_memory_gb'):
            append('train/' + key, row[key], step, seconds)
        append('train/encoder_learning_rate', row['learning_rate'], step, seconds)
        append('train/head_learning_rate', row['learning_rate'] / encoder_lr *
               report['config']['head_learning_rate'], step, seconds)
        epochs[math.ceil(row['epoch'])].append(row)
    for epoch, rows in sorted(epochs.items()):
        append('epoch/mean_logged_loss', statistics.fmean(r['loss'] for r in rows),
               epoch, rows[-1]['seconds'])
    # Evaluation exists only at the two measured endpoints.
    for stage, step, seconds in [('before', 0, 0), ('after', steps[-1]['step'], steps[-1]['seconds'])]:
        for split in ('train', 'validation'):
            for key in ('accuracy', 'mean_soft_ce'):
                append('eval/' + split + '/' + key, report[stage][split][key], step, seconds)
    return points


def sync_metrics(directory, run_id, base, prefix=''):
    points = training_metrics(directory, prefix)
    grouped = defaultdict(list)
    for point in points:
        grouped[point['key']].append(point)
    missing = []
    for key, expected in grouped.items():
        query = urllib.parse.urlencode({'run_id': run_id, 'metric_key': key})
        history = request(base, 'mlflow/metrics/get-history?' + query).get('metrics', [])
        present = {(p['step'], p['timestamp'], p['value']) for p in history}
        for p in expected:
            if any(old['step'] == p['step'] and old['value'] != p['value'] for old in history):
                raise ValueError('Conflicting existing value for ' + key)
            if (p['step'], p['timestamp'], p['value']) not in present:
                missing.append(p)
    for offset in range(0, len(missing), 500):
        request(base, 'mlflow/runs/log-batch', {'run_id': run_id, 'metrics': missing[offset:offset + 500]})
    for key, expected in grouped.items():
        query = urllib.parse.urlencode({'run_id': run_id, 'metric_key': key})
        history = request(base, 'mlflow/metrics/get-history?' + query).get('metrics', [])
        actual = {(p['step'], p['timestamp'], p['value']) for p in history}
        if not all((p['step'], p['timestamp'], p['value']) in actual for p in expected):
            raise RuntimeError('Read-back verification failed: ' + key)
    request(base, 'mlflow/runs/set-tag', {'run_id': run_id, 'key': prefix + 'training_history.source',
        'value': 'Backfilled from original steps.json; timestamps reconstructed from second-resolution run.log anchor. Epoch loss is mean logged optimizer-step loss. Eval measured only before/after training. Head LR derived from shared scheduler scale; encoder frozen.'})
    return {'run_id': run_id, 'prefix': prefix, 'metric_keys': len(grouped),
            'verified_points': len(points), 'new_points': len(missing)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--tracking-uri', default='https://mlflow.dove-nest.com')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    print(json.dumps(sync_metrics(args.report_dir, args.run_id, args.tracking_uri, args.prefix), indent=2))
