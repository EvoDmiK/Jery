"""Build description-only and train-augmented controls; preserve original held-out cases and labels."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import unicodedata

QUESTION = 'route_in_window'
SPLITS = ('train', 'validation', 'test')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def signature(state):
    text = unicodedata.normalize('NFKC', dump(state)).casefold()
    return ''.join(c for c in text if c.isalnum())


def build(base, catalog_dir, authored, criteria_path, dataset, output):
    if dataset.exists() or output.exists():
        raise ValueError('Never overwrite dataset/export paths')
    manifest = json.loads((base / 'manifest.json').read_text())
    catalog = json.loads((catalog_dir / 'tools.json').read_text())
    source_manifest = json.loads((catalog_dir / 'manifest.json').read_text())
    if (sha(catalog_dir / 'tools.json') != source_manifest['sha256']['tools.json']
            or catalog['catalog_version'] != manifest['catalog_version']):
        raise ValueError('Pinned source catalog integrity mismatch')
    criteria = json.loads(criteria_path.read_text())
    descriptions = {t['id']: t['description'] for t in criteria['tools']}
    ids = list(descriptions)
    if (len(ids) != 10 or len(criteria['tools']) != 10 or any(not d.strip() for d in descriptions.values())
            or set(ids) != {t['id'] for t in catalog['tools']}):
        raise ValueError('Expected the same ten tools')
    by_tool = {t['id']: t for t in catalog['tools']}
    version = 'hermes-ten-tools-' + sha(criteria_path)[:12]
    old = {}
    source_hashes = {'tools.json': sha(catalog_dir / 'tools.json'), 'criteria.json': sha(criteria_path),
                     'authored.jsonl': sha(authored), 'export-manifest.json': sha(base / 'manifest.json')}
    for split in SPLITS:
        p = base / (split + '.jsonl')
        if sha(p) != manifest['splits'][split]['sha256']:
            raise ValueError('Base export checksum mismatch')
        source_hashes[split + '.jsonl'] = sha(p)
        old[split] = [json.loads(line) for line in p.read_text().splitlines()]
        for row in old[split]:
            meta = row['metadata']
            if meta['source_split'] != split or set(meta['alias_to_tool'].values()) != set(ids):
                raise ValueError('Invalid base split/candidate set')
            meta['previous_tool_catalog_version'] = meta['tool_catalog_version']
            meta['tool_catalog_version'] = version
            row['questions'][QUESTION]['criteria'] = {a: descriptions[t] for a, t in meta['alias_to_tool'].items()}
    extra, source_ids, normalized = [], set(), {signature(r['state']) for rows in old.values() for r in rows}
    for case in map(json.loads, authored.read_text().splitlines()):
        tool = case['tool']
        if (case['split'] != 'train' or tool not in ids or case['id'] in source_ids
                or case['review_status'] != 'needs_human_review' or case['execution_verified'] is not False):
            raise ValueError('Invalid authored case or unsupported review promotion')
        source_ids.add(case['id'])
        state = {'request': case['request'], 'context': case['context']}
        key = signature(state)
        if key in normalized:
            raise ValueError('Normalized duplicate authored or held-out input')
        normalized.add(key)
        if any(i in dump(state) for i in ids):
            raise ValueError('Tool ID leaked into model features')
        action = case['action']
        enum = by_tool[tool]['input_schema'].get('properties', {}).get('action', {}).get('enum')
        if (enum and action not in enum) or (not enum and action is not None):
            raise ValueError('Invalid catalog action')
        candidates = ids[:]
        random.Random(f"{criteria['seed']}:{case['id']}").shuffle(candidates)
        aliases = {f'T{i}': t for i, t in enumerate(candidates)}
        gold = next(a for a, t in aliases.items() if t == tool)
        choices = [{'tool': tool, 'action': action}]
        extra.append({'state': state,
            'questions': {QUESTION: {'type': 'choice', 'instructions': old['train'][0]['questions'][QUESTION]['instructions'],
                                    'criteria': {a: descriptions[t] for a, t in aliases.items()}}},
            'gold': {QUESTION: {'label': gold, 'probabilities': {gold: 1.0}}},
            'metadata': {'export_version': 'decision-export-v2', 'export_scope': 'fixed_ten_tools_select_only',
                'evaluation_task': 'ten_tool_choice', 'source_id': case['id'], 'scenario_id': case['scenario_id'],
                'source_split': 'train', 'tool_catalog_version': version, 'window_index': 0, 'window_count': 1,
                'export_id': case['id'] + ':ten-tools', 'window_tools': candidates, 'alias_to_tool': aliases,
                'original_candidate_tools': ids, 'original_decision': 'select_tools', 'acceptable_choices': choices,
                'source_candidate_tools': ids, 'source_acceptable_choices': choices, 'source': case['source'],
                'review_status': 'needs_human_review', 'execution_verified': False}})
    counts = Counter(r['metadata']['acceptable_choices'][0]['tool'] for r in extra)
    if set(counts) != set(ids) or len(set(counts.values())) != 1:
        raise ValueError('New cases must be balanced across all ten tools')
    profiles, contents = {}, {}
    for profile in ('description_control', 'augmented'):
        rows = {s: copy.deepcopy(old[s]) for s in SPLITS}
        if profile == 'augmented':
            rows['train'].extend(extra)
        groups = {s: {r['metadata']['scenario_id'] for r in rows[s]} for s in SPLITS}
        for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            if groups[a] & groups[b]:
                raise ValueError('Scenario group leakage')
            if {signature(r['state']) for r in rows[a]} & {signature(r['state']) for r in rows[b]}:
                raise ValueError('Normalized input leakage')
        profiles[profile], contents[profile] = {}, {}
        for split in SPLITS:
            text = ''.join(dump(r) + '\n' for r in rows[split])
            contents[profile][split] = text
            profiles[profile][split] = {'rows': len(rows[split]), 'source_rows': len(rows[split]),
                'scenario_groups': len(groups[split]), 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                'tool_counts': dict(Counter(r['metadata']['acceptable_choices'][0]['tool'] for r in rows[split]))}
    # All integrity checks complete before creating any output path.
    dataset.mkdir(parents=True)
    output.mkdir(parents=True)
    catalog['source_catalog_version'], catalog['catalog_version'] = catalog['catalog_version'], version
    (dataset / 'tools.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    result = {'dataset_version': criteria['dataset_version'], 'catalog_version': version,
              'source_sha256': source_hashes, 'profiles': profiles,
              'sha256': {'tools.json': sha(dataset / 'tools.json')}, 'review_status': 'needs_human_review',
              'limitations': ['Synthetic AI-authored/AI-inspected, not human-reviewed or executed',
                'Original validation/test cases observed previously: regression only',
                'Validation/test states, labels and aliases preserved; criteria changed explicitly',
                'Paraphrase and context siblings stay in train; no independent real-use evaluation']}
    (dataset / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for profile in contents:
        folder = output / profile
        folder.mkdir()
        for split, text in contents[profile].items():
            (folder / (split + '.jsonl')).write_text(text)
        exported = {'export_version': 'decision-export-v2', 'export_scope': 'fixed_ten_tools_select_only',
            'evaluation_task': 'ten_tool_choice', 'dataset_version': criteria['dataset_version'],
            'catalog_version': version, 'profile': profile, 'source_sha256': source_hashes,
            'max_window_tools': 10, 'seed': criteria['seed'], 'splits': profiles[profile],
            'limitations': result['limitations']}
        (folder / 'manifest.json').write_text(json.dumps(exported, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('outputs/laya-ten-tools-20261020/full'))
    p.add_argument('--catalog-dir', type=Path, default=Path('datasets/hermes-ten-tools-v1'))
    p.add_argument('--authored', type=Path, default=Path('datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl'))
    p.add_argument('--criteria', type=Path, default=Path('configs/training/laya-ten-tools-criteria-v2-20261020.json'))
    p.add_argument('--dataset', type=Path, default=Path('datasets/hermes-ten-tools-v2'))
    p.add_argument('--output', type=Path, default=Path('outputs/laya-ten-tools-contrast-20261020'))
    a = p.parse_args()
    r = build(a.base, a.catalog_dir, a.authored, a.criteria, a.dataset, a.output)
    print(json.dumps({'catalog_version': r['catalog_version'], 'profiles': r['profiles']}, ensure_ascii=False))
