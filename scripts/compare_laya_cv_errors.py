"""Compare saved selected-head OOF predictions; never load a model or use test scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(cases, prediction):
    labels = sorted({r['gold'] for r in cases})
    confusion = Counter((r['gold'], r[prediction]) for r in cases)
    per_tool = {}
    for label in labels:
        tp = confusion[label, label]
        support = sum(n for (gold, _), n in confusion.items() if gold == label)
        predicted = sum(n for (_, pred), n in confusion.items() if pred == label)
        per_tool[label] = {'support': support, 'correct': tp,
                          'precision': tp / predicted if predicted else 0,
                          'recall': tp / support, 'f1': 2 * tp / (support + predicted)}
    return {'accuracy': sum(r['gold'] == r[prediction] for r in cases) / len(cases),
            'macro_f1': sum(t['f1'] for t in per_tool.values()) / len(labels),
            'per_tool': per_tool,
            'confusions': [{'gold': g, 'predicted': p, 'count': n}
                           for (g, p), n in confusion.most_common() if g != p]}


def compare(root, state_dir, baseline_index=1, best_index=9):
    state = json.loads((state_dir / 'state.json').read_text())
    fold_dir = root / state['plan']['cv']['folds_dir']
    if hashlib.sha256((fold_dir / 'folds.json').read_bytes()).hexdigest() != state['fold_summary_sha256']:
        raise ValueError('Fold assignments changed')
    fold_info = json.loads((fold_dir / 'folds.json').read_text())
    joined, provenance = {}, {}
    for name, index in [('baseline', baseline_index), ('best', best_index)]:
        item = next(c for c in state['children'] if c['index'] == index)
        nested = json.loads((state_dir / f'trial-{index:02d}/folds/state.json').read_text())
        if item['status'] != 'finished' or nested['status'] != 'finished':
            raise ValueError('Only completed trials may be compared')
        if len(nested['children']) != fold_info['n_splits']:
            raise ValueError('Incomplete fold coverage')
        provenance[name] = {'run_id': item['run_id'], 'files': []}
        seen = set()
        for child in nested['children']:
            fold = child['index']
            path = fold_dir / f'fold-{fold}/validation.jsonl'
            info = next(i for i in fold_info['folds'] if i['fold'] == fold)
            if hashlib.sha256(path.read_bytes()).hexdigest() != info['splits']['validation']['sha256']:
                raise ValueError('Validation data changed')
            validation = rows(path)
            golds = {r['metadata']['source_id']: r for r in validation}
            if len(golds) != len(validation):
                raise ValueError('Duplicate validation IDs')
            output = root / state['config']['outputs_dir'] / child['local_id']
            report = json.loads((output / 'report.json').read_text())
            predictions = rows(output / 'validation_cases.jsonl')
            if len(predictions) != len(golds) or {r['source_id'] for r in predictions} != set(golds):
                raise ValueError('Prediction coverage differs from validation')
            correct = 0
            for pred in predictions:
                source_id = pred['source_id']
                if source_id in seen:
                    raise ValueError('OOF source appears more than once')
                seen.add(source_id)
                row = golds[source_id]; meta = row['metadata']
                choices = meta['acceptable_choices']
                if len(choices) != 1 or pred['decision'] != 'select_tools' or pred['tool'] not in meta['original_candidate_tools']:
                    raise ValueError('Expected one valid candidate tool')
                gold = choices[0]['tool']
                if meta['alias_to_tool'][row['gold']['route_in_window']['label']] != gold:
                    raise ValueError('Gold alias disagrees with tool ID')
                identity = {'source_id': source_id, 'fold': fold, 'scenario_id': meta['scenario_id'],
                            'gold': gold, 'request': row['state']['request'], 'review_status': meta['review_status']}
                if source_id in joined and any(joined[source_id][k] != v for k, v in identity.items()):
                    raise ValueError('Paired case or fold mismatch')
                joined.setdefault(source_id, identity)[name] = pred['tool']
                correct += pred['tool'] == gold
            if abs(correct / len(golds) - report['validation']['case_accuracy']) > .000051:
                raise ValueError('Predictions do not match selected-head report')
            provenance[name]['files'].append({'fold': fold, 'run_id': child['run_id'],
                'selected_epoch': report['selection']['selected_epoch'],
                'prediction_sha256': hashlib.sha256((output / 'validation_cases.jsonl').read_bytes()).hexdigest()})
    cases = list(joined.values())
    if not cases or any('baseline' not in r or 'best' not in r for r in cases):
        raise ValueError('Unpaired OOF cases')
    for row in cases:
        a, b = row['baseline'] == row['gold'], row['best'] == row['gold']
        row['outcome'] = 'both_correct' if a and b else 'improved' if b else 'regressed' if a else 'both_wrong'
    return {'source_parent_run_id': state['parent_run_id'], 'cases': len(cases),
            'scope': 'selected-head OOF on synthetic train pool; validation was used for selection, not independent test',
            'provenance': provenance, 'paired': dict(Counter(r['outcome'] for r in cases)),
            'baseline': metrics(cases, 'baseline'), 'best': metrics(cases, 'best'),
            'by_fold': {str(f): {n: metrics([r for r in cases if r['fold'] == f], n)
                               for n in ('baseline', 'best')} for f in sorted({r['fold'] for r in cases})}}, cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-index', type=int, default=1)
    parser.add_argument('--best-index', type=int, default=9)
    args = parser.parse_args()
    summary, cases = compare(args.root, args.state_dir, args.baseline_index, args.best_index)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'cases.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in cases))
    print(json.dumps({k: summary[k] for k in ('cases', 'paired')}))
