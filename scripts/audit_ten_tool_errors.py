"""Audit selected-head validation errors across runs; never inspect test predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(export, predictions):
    rows = [json.loads(line) for line in export.read_text().splitlines()]
    by_id = {r['metadata']['source_id']: r for r in rows}
    runs = json.loads(predictions.read_text())
    if len(by_id) != len(rows) or not runs or len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Duplicate validation cases/runs or no runs')
    wrong, pairs, choices = Counter(), Counter(), {sid: Counter() for sid in by_id}
    total = 0
    for run in runs:
        seen = set()
        for pred in run['validation_predictions']:
            sid = pred['source_id']
            if sid not in by_id or sid in seen:
                raise ValueError('Unknown or duplicate validation prediction')
            seen.add(sid)
            row = by_id[sid]
            if row['metadata']['source_split'] != 'validation':
                raise ValueError('Only validation evidence is allowed')
            aliases = row['metadata']['alias_to_tool']
            target = aliases[row['gold']['route_in_window']['label']]
            allowed = {c['tool'] for c in row['metadata']['acceptable_choices']}
            if allowed != {target} or pred['tool'] not in aliases.values() or pred['decision'] != 'select_tools':
                raise ValueError('Invalid label, prediction or candidate contract')
            choices[sid][pred['tool']] += 1
            if pred['tool'] not in allowed:
                wrong[sid] += 1
                pairs[(target, pred['tool'])] += 1
            total += 1
        if seen != set(by_id):
            raise ValueError('Missing validation predictions')
    return {'scope': 'selected-head validation only; no test predictions', 'runs': len(runs),
            'cases': len(rows), 'prediction_events': total, 'wrong_events': sum(wrong.values()),
            'cases_wrong_in_any_run': len(wrong), 'cases_wrong_in_every_run': sum(n == len(runs) for n in wrong.values()),
            'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (export, predictions)},
            'confusions': [{'gold': a, 'predicted': b, 'count': n} for (a, b), n in pairs.most_common()],
            'cases_review': [{'source_id': sid, 'gold': r['metadata']['alias_to_tool'][r['gold']['route_in_window']['label']],
                              'wrong_runs': wrong[sid], 'predictions': dict(choices[sid]), 'state': r['state'],
                              'human_review_status': 'needs_human_review', 'execution_verified': False}
                             for sid, r in by_id.items()]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.validation, args.predictions)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('runs', 'cases', 'prediction_events', 'wrong_events', 'cases_wrong_in_every_run')}))
