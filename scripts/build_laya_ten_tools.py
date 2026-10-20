"""Create a new, balanced ten-tool diagnostic export without modifying any original dataset."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

from build_laya_pilot import ROOT, QUESTION
from jery.runtime.prediction import model_input


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def diverse_rows(rows, count, rng):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['id']):
        groups[row['scenario_id']].append(row)
    pools = list(groups.values())
    rng.shuffle(pools)
    for pool in pools:
        rng.shuffle(pool)
    ordered = [pool[i] for i in range(max(map(len, pools), default=0)) for pool in pools if i < len(pool)]
    if len(ordered) < count:
        raise ValueError(f'Need {count} cases; only {len(ordered)} available')
    return ordered[:count]


def build(source, selection_path, dataset_dir, output):
    if dataset_dir.exists() or output.exists():
        raise ValueError('Choose new dataset/export directories; never overwrite a previous dataset')
    selection = json.loads(selection_path.read_text())
    catalog = json.loads((source / 'tools.json').read_text())
    manifest = json.loads((source / 'manifest.json').read_text())
    descriptions = {t['id']: t['description'] for t in selection['tools']}
    tool_ids = list(descriptions)
    if len(tool_ids) != 10 or len(selection['tools']) != 10 or any(not s.strip() for s in descriptions.values()):
        raise ValueError('Exactly ten unique, described tools are required')
    by_id = {t['id']: t for t in catalog['tools']}
    if not set(tool_ids).issubset(by_id):
        raise ValueError('Selection contains a tool outside the pinned catalog')
    selected, hashes, groups = {}, {}, {}
    rng = random.Random(selection['seed'])
    for name in ('tools.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'):
        hashes[name] = sha(source / name)
        if hashes[name] != manifest['sha256'][name]:
            raise ValueError('Original source hash mismatch: ' + name)
    for split in ('train', 'validation', 'test'):
        rows = [json.loads(line) for line in (source / (split + '.jsonl')).read_text().splitlines()]
        eligible = defaultdict(list)
        seen = {}
        for row in sorted(rows, key=lambda r: r['id']):
            if row['decision'] != 'select_tools':
                continue
            allowed = {c['tool'] for c in row['acceptable_choices']} & set(tool_ids)
            if len(allowed) == 1:
                target = next(iter(allowed))
                key = json.dumps([row['request'], row['context']], ensure_ascii=False, sort_keys=True)
                if key in seen and seen[key] != target:
                    raise ValueError('Identical inputs have conflicting tool labels')
                if key not in seen:
                    eligible[target].append(row)
                seen[key] = target
        count = max(p[split] for p in selection['profiles'].values())
        selected[split] = {t: diverse_rows(eligible[t], count, rng) for t in tool_ids}
        groups[split] = {r['scenario_id'] for rows in selected[split].values() for r in rows}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if groups[a] & groups[b]:
            raise ValueError('Selected source scenario groups cross splits')
    catalog_version = 'hermes-ten-tools-' + sha(selection_path)[:12]
    subset_catalog = {'catalog_version': catalog_version, 'source_catalog_version': catalog['catalog_version'],
                      'tools': [copy.deepcopy(by_id[t]) for t in tool_ids]}
    exports, profiles = {}, {}
    for profile, counts in selection['profiles'].items():
        exports[profile], profiles[profile] = {}, {}
        for split in ('train', 'validation', 'test'):
            converted = []
            for target in tool_ids:
                for row in selected[split][target][:counts[split]]:
                    features = model_input(row, catalog)
                    candidates = tool_ids[:]
                    random.Random(f"{selection['seed']}:{row['id']}").shuffle(candidates)  # Label-blind, profile-independent.
                    aliases = {f'T{i}': t for i, t in enumerate(candidates)}
                    gold = next(a for a, t in aliases.items() if t == target)
                    converted.append({'state': {'request': features['request'], 'context': features['context']},
                        'questions': {QUESTION: {'type': 'choice', 'instructions': '요청을 처리할 도구 하나를 고르세요.',
                                                'criteria': {a: descriptions[t] for a, t in aliases.items()}}},
                        'gold': {QUESTION: {'label': gold, 'probabilities': {gold: 1.0}}},
                        'metadata': {'export_version': 'decision-export-v2', 'export_scope': selection['export_scope'],
                            'evaluation_task': selection['evaluation_task'], 'source_id': row['id'],
                            'scenario_id': row['scenario_id'], 'source_split': split,
                            'tool_catalog_version': catalog_version, 'window_index': 0, 'window_count': 1,
                            'export_id': row['id'] + ':ten-tools', 'window_tools': candidates, 'alias_to_tool': aliases,
                            'original_candidate_tools': tool_ids, 'original_decision': 'select_tools',
                            'acceptable_choices': [c for c in row['acceptable_choices'] if c['tool'] == target],
                            'source_candidate_tools': row['candidate_tools'], 'source_acceptable_choices': row['acceptable_choices'],
                            'source': row['source'], 'review_status': row['review_status'],
                            'execution_verified': row['execution_verified']}})
            random.Random(f"{selection['seed']}:{split}:order").shuffle(converted)
            if len({r['metadata']['source_id'] for r in converted}) != len(converted):
                raise ValueError('A source case occurs twice in one profile split')
            content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in converted)
            exports[profile][split] = content
            profiles[profile][split] = {'rows': len(converted), 'source_rows': len(converted),
                'scenario_groups': len({r['metadata']['scenario_id'] for r in converted}),
                'tool_counts': dict(Counter(next(c['tool'] for c in r['metadata']['acceptable_choices']) for r in converted)),
                'sha256': hashlib.sha256(content.encode()).hexdigest()}
    # Validate everything before creating output paths.
    dataset_dir.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    write_json(dataset_dir / 'tools.json', subset_catalog)
    write_json(dataset_dir / 'selection.json', selection)
    for profile, contents in exports.items():
        directory = output / profile
        directory.mkdir()
        for split, content in contents.items():
            (directory / (split + '.jsonl')).write_text(content)
        write_json(directory / 'manifest.json', {'export_version': 'decision-export-v2',
            'export_scope': selection['export_scope'], 'evaluation_task': selection['evaluation_task'],
            'dataset_version': selection['dataset_version'], 'catalog_version': catalog_version,
            'profile': profile, 'source_sha256': hashes, 'selection_sha256': sha(selection_path),
            'seed': selection['seed'], 'max_window_tools': 10, 'splits': profiles[profile],
            'limitations': ['Synthetic unreviewed labels; not production accuracy',
                'Fixed ten-candidate task changes original candidate sets',
                'Prior source test has been observed; regression evidence only', 'No action or null-decision prediction']})
    result = {'dataset_version': selection['dataset_version'], 'catalog_version': catalog_version,
              'source_dataset_version': selection['source_dataset_version'], 'source_sha256': hashes,
              'sha256': {'tools.json': sha(dataset_dir / 'tools.json'), 'selection.json': sha(dataset_dir / 'selection.json')},
              'profiles': profiles, 'selected_tools': tool_ids, 'review_status': 'needs_human_review'}
    write_json(dataset_dir / 'manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets/mcp-routing-v2')
    parser.add_argument('--selection', type=Path, default=ROOT / 'configs/training/laya-ten-tools-catalog-20261020.json')
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'datasets/hermes-ten-tools-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/laya-ten-tools-20261020')
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.selection, args.dataset_dir, args.output), ensure_ascii=False))
