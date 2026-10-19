"""Export a small unreviewed Laya engineering fixture without altering v2 data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from jery.runtime.prediction import model_input

QUESTION = 'route_in_window'
SENTINELS = {'no_tool': '도구 없이 답변', 'clarify': '필수 대상이나 맥락 확인 필요',
             'window_none': '이 창에 적합한 도구 없음'}


def export_case(row, catalog, rng, negative=False, window_tools=None):
    features = model_input(row, catalog)
    acceptable = {c['tool'] for c in row['acceptable_choices']}
    window = features['candidate_tools'][:] if window_tools is None else list(window_tools)
    if (len(window) != len(set(window)) or not set(window).issubset(features['candidate_tools'])
            or (negative and window_tools is not None)):
        raise ValueError('Explicit window must be unique original candidates; cannot combine with negative')
    if negative:
        if row['decision'] != 'select_tools':
            raise ValueError('Negative window requires a source select_tools case')
        window = [t for t in window if t not in acceptable]
    if len(window) > 8 or not window:
        raise ValueError('Pilot window must contain one to eight original candidates')
    rng.shuffle(window)
    aliases = {f'T{i}': tool for i, tool in enumerate(window)}
    by_id = {t['id']: t for t in features['catalog']['tools']}
    cards = []
    for alias, tool in aliases.items():
        spec = by_id[tool]
        cards.append({'alias': alias, 'server': spec['server'], 'name': spec['name'],
                      'summary': ' '.join(spec['description'].split())[:120]})
    criteria = {alias: '도구 ' + alias for alias in aliases}
    criteria.update(SENTINELS)
    if row['decision'] in ('no_tool', 'clarify'):
        targets = [row['decision']]
    else:
        targets = [alias for alias, tool in aliases.items() if tool in acceptable]
        if not targets:
            targets = ['window_none']
    probabilities = {key: 1 / len(targets) for key in targets}
    return {
        'state': {'request': features['request'], 'context': features['context'], 'candidates': cards},
        'questions': {QUESTION: {'type': 'choice',
            'instructions': '현재 창의 도구를 선택하세요. 도구 없이 답변은 no_tool, 필수 정보 부족은 clarify, 이 창에 적합한 도구가 없으면 window_none입니다.',
            'criteria': criteria}},
        'gold': {QUESTION: {'label': targets[0], 'probabilities': probabilities}},
        'metadata': {'export_version': 'decision-export-v1', 'source_id': row['id'],
            'scenario_id': row['scenario_id'], 'source_split': row['split'],
            'tool_catalog_version': row['tool_catalog_version'],
            'original_candidate_tools': features['candidate_tools'],
            'window_tools': window, 'alias_to_tool': aliases,
            'original_decision': row['decision'], 'acceptable_choices': row['acceptable_choices'],
            'derived_negative_window': negative, 'source': row['source'],
            'review_status': row['review_status'], 'execution_verified': row['execution_verified']}}


def choose(rows, decision, count, excluded):
    eligible = [r for r in rows if r['decision'] == decision and 0 < len(r['candidate_tools']) <= 8
                and r['id'] not in excluded]
    # Prefer distinct groups; limited no_tool/clarify groups require same-split variants.
    groups, chosen = set(), []
    for distinct in (True, False):
        for row in eligible:
            if row['id'] in excluded or (distinct and row['scenario_id'] in groups):
                continue
            chosen.append(row)
            excluded.add(row['id'])
            groups.add(row['scenario_id'])
            if len(chosen) == count:
                return chosen
    raise ValueError(f'Not enough {decision} rows for engineering fixture')


def build(directory, output, seed=3407):
    if output.exists():
        raise ValueError('Output already exists; preserve the previous export')
    catalog = json.loads((directory / 'tools.json').read_text())
    rng, exports, source_hashes = random.Random(seed), {}, {}
    quotas = {'train': [('select_tools', 12, False), ('select_tools', 4, True),
                        ('no_tool', 6, False), ('clarify', 5, False), ('unsupported', 5, False)],
              'validation': [('select_tools', 4, False), ('select_tools', 2, True),
                             ('no_tool', 2, False), ('clarify', 2, False), ('unsupported', 2, False)]}
    for split, selections in quotas.items():
        content = (directory / f'{split}.jsonl').read_bytes()
        source_hashes[f'{split}.jsonl'] = hashlib.sha256(content).hexdigest()
        rows = [json.loads(line) for line in content.splitlines()]
        selected, excluded = [], set()
        for decision, count, negative in selections:
            selected.extend(export_case(r, catalog, rng, negative)
                            for r in choose(rows, decision, count, excluded))
        rng.shuffle(selected)
        exports[split] = selected
    train_groups = {r['metadata']['scenario_id'] for r in exports['train']}
    validation_groups = {r['metadata']['scenario_id'] for r in exports['validation']}
    if train_groups & validation_groups:
        raise ValueError('Source scenario groups overlap')
    source_hashes['tools.json'] = hashlib.sha256((directory / 'tools.json').read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for split, rows in exports.items():
        content = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)
        (output / f'{split}.jsonl').write_text(content)
        manifests[split] = {'rows': len(rows), 'scenario_groups': len({r['metadata']['scenario_id'] for r in rows}),
            'original_decisions': dict(Counter(r['metadata']['original_decision'] for r in rows)),
            'internal_labels': dict(Counter(r['gold'][QUESTION]['label'] for r in rows)),
            'negative_windows': sum(r['metadata']['derived_negative_window'] for r in rows),
            'sha256': hashlib.sha256(content.encode()).hexdigest()}
    manifest = {'export_version': 'decision-export-v1', 'purpose': 'engineering_smoke_only',
        'seed': seed, 'catalog_version': catalog['catalog_version'], 'source_sha256': source_hashes,
        'splits': manifests, 'state_card_summary_characters': 120, 'max_window_tools': 8,
        'action_prediction': False,
        'limitations': ['Unreviewed synthetic labels; no quality or calibration claim',
            'Limited distinct nonselect scenario groups; candidate-count confounding remains',
            'Windows are source subsets, not an OOF learned retriever benchmark',
            'Token budgets must be measured with the exact Laya tokenizer before training',
            'Original unsupported is distinct from the internal window_none target']}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'datasets/mcp-routing-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/laya-pilot-20261017/export')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()
    print(json.dumps(build(args.directory, args.output, args.seed), ensure_ascii=False, indent=2))
