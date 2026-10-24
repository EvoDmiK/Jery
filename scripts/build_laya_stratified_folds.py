"""Build deterministic stratified folds while keeping scenario groups together."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


def label(row):
    return row['metadata']['alias_to_tool'][row['gold']['route_in_window']['label']]


def stratified_group_folds(rows, n_splits=5, seed=3407):
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError('At least two folds are required')
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['metadata']['scenario_id']].append(index)
    by_class = defaultdict(list)
    for group, indices in groups.items():
        labels = Counter(label(rows[i]) for i in indices)
        # ponytail: mixed groups need >= n_splits repetitions of each label histogram.
        # Reject sparse histograms rather than silently split a contrast pair.
        signature = tuple(sorted(labels)) if len(labels) == 1 else tuple(sorted(labels.elements()))
        by_class[signature].append(indices)
    folds = [[] for _ in range(n_splits)]
    rng = random.Random(seed)
    for tool, buckets in sorted(by_class.items()):
        if len(buckets) < n_splits:
            raise ValueError('Not enough independent scenario groups: ' + str(tool))
        rng.shuffle(buckets)
        buckets.sort(key=len, reverse=True)
        counts = [0] * n_splits
        for indices in buckets:
            order = list(range(n_splits))
            rng.shuffle(order)
            fold = min(order, key=lambda k: (counts[k], len(folds[k])))
            folds[fold].extend(indices)
            counts[fold] += len(indices)
        if max(counts) - min(counts) > max(map(len, buckets)):
            raise ValueError('Unbalanced class folds: ' + str(tool))
    return [sorted(indices) for indices in folds]


def build(source, output, n_splits=5, seed=3407):
    source, output = Path(source), Path(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    original = {}
    for split in ('train', 'validation', 'test'):
        data = (source / (split + '.jsonl')).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest['splits'][split]['sha256']:
            raise ValueError('Source hash mismatch: ' + split)
        original[split] = data
    lines = original['train'].decode().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    ids = [r['metadata']['source_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate source cases')
    for split in ('validation', 'test'):
        external = [json.loads(line) for line in original[split].decode().splitlines()]
        if ({r['metadata']['scenario_id'] for r in rows} & {r['metadata']['scenario_id'] for r in external}
                or set(ids) & {r['metadata']['source_id'] for r in external}):
            raise ValueError('Original heldout overlaps the CV pool')
    assignments = stratified_group_folds(rows, n_splits, seed)
    if sorted(i for fold in assignments for i in fold) != list(range(len(rows))):
        raise ValueError('Incomplete or duplicate validation coverage')
    all_indices = set(range(len(rows)))
    prepared = []
    mixed_groups = defaultdict(set)
    for row in rows:
        mixed_groups[row['metadata']['scenario_id']].add(label(row))
    algorithm = ('scenario-label-histogram-stratified-v2' if any(len(v) > 1 for v in mixed_groups.values())
                 else 'homogeneous-scenario-stratified-largest-first-v1')
    summary = {'algorithm': algorithm, 'seed': seed,
               'n_splits': n_splits, 'source_manifest_sha256': hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
               'source_train_sha256': manifest['splits']['train']['sha256'],
               'heldout_test_sha256': manifest['splits']['test']['sha256'], 'folds': []}
    for number, validation in enumerate(assignments, 1):
        train = sorted(all_indices - set(validation))
        if {rows[i]['metadata']['scenario_id'] for i in train} & {rows[i]['metadata']['scenario_id'] for i in validation}:
            raise ValueError('Scenario leakage across train/validation')
        contents = {'train': ''.join(lines[i] for i in train).encode(),
                    'validation': ''.join(lines[i] for i in validation).encode(), 'test': original['test']}
        folded = {**manifest, 'profile': f'stratified-group-fold-{number}', 'splits': {},
                  'cross_validation': {'fold': number, 'n_splits': n_splits, 'pool': 'original train only',
                                       'original_validation_used': False, 'algorithm': summary['algorithm']}}
        for split, data in contents.items():
            items = [json.loads(line) for line in data.decode().splitlines()]
            folded['splits'][split] = {'rows': len(items), 'source_rows': len(items),
                'scenario_groups': len({r['metadata']['scenario_id'] for r in items}),
                'tool_counts': dict(Counter(label(r) for r in items)), 'sha256': hashlib.sha256(data).hexdigest()}
        summary['folds'].append({'fold': number, 'validation_source_ids': [ids[i] for i in validation],
                                 'splits': folded['splits']})
        prepared.append((contents, folded))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for number, (contents, folded) in enumerate(prepared, 1):
        folder = output / f'fold-{number}'
        folder.mkdir()
        for split, data in contents.items():
            (folder / (split + '.jsonl')).write_bytes(data)
        (folder / 'manifest.json').write_text(json.dumps(folded, ensure_ascii=False, indent=2) + '\n')
    (output / 'folds.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    result = build(args.source, args.output, args.n_splits, args.seed)
    print(json.dumps({'n_splits': result['n_splits'], 'folds': [r['splits'] for r in result['folds']]}))
