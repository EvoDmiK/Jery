"""Build synthetic MCP routing data from the versioned, offline catalog and scenarios."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from validate_mcp_dataset import validate_inputs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'datasets/mcp-routing-v2'


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'



def build(directory=DEFAULT):
    directory = Path(directory)
    catalog = json.loads((directory / 'tools.json').read_text())
    seeds = json.loads((directory / 'scenarios.json').read_text())
    tools = validate_inputs(catalog, seeds)
    rows = []
    for seed in seeds:
        choices = seed['acceptable_choices']
        targets = {c['tool'] for c in choices}
        # Candidate variants share the scenario split; answer presence never defines the split.
        full = sorted(tools)
        rng = random.Random(seed['scenario_id'])
        servers = {tools[t]['server'] for t in targets}
        siblings = [t for t in full if tools[t]['server'] in servers and t not in targets]
        reduced = sorted(targets | set(rng.sample(siblings, min(3, len(siblings)))) | set(rng.sample(full, min(4, len(full)))))
        candidates = [('full', full), ('subset', reduced)]
        if seed.get('missing_target_case'):
            candidates.append(('target_absent', [t for t in reduced if t not in targets]))
        for variant, candidate_ids in candidates:
            for index, prompt in enumerate(seed['requests']):
                uid = f"{seed['scenario_id']}.{seed.get('variant_id', 'base')}.{variant}.{index + 1}"
                candidate_ids = list(candidate_ids)
                random.Random(uid).shuffle(candidate_ids)
                missing = variant == 'target_absent'
                rows.append({
                    'id': uid, 'scenario_id': seed['scenario_id'], 'split': seed['split'],
                    'request': prompt, 'context': seed.get('context', []),
                    'candidate_tools': candidate_ids, 'tool_catalog_version': catalog['catalog_version'],
                    'decision': 'unsupported' if missing else seed['decision'],
                    'acceptable_choices': [] if missing else choices,
                    'source': 'synthetic', 'review_status': 'needs_human_review', 'execution_verified': False,
                })
    for split in ('train', 'validation', 'test'):
        selected = [r for r in rows if r['split'] == split]
        random.Random('mcp-v2-' + split).shuffle(selected)
        (directory / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    labels = [{'tool': t['id'], 'server': t['server'], 'kind': t['kind'],
               'actions': t['input_schema'].get('properties', {}).get('action', {}).get('enum', [])}
              for t in catalog['tools']]
    (directory / 'labels.json').write_text(dump(labels))
    files = ['tools.json', 'scenarios.json', 'labels.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl']
    manifest = {
        'dataset_version': 'mcp-routing-v2', 'catalog_version': catalog['catalog_version'],
        'task': 'next_step_tool_routing', 'language': 'ko', 'seed': 'mcp-v2',
        'synthetic': True, 'human_reviewed': False, 'servers': len({t['server'] for t in tools.values()}),
        'tools': len(tools), 'scenarios': len({s['scenario_id'] for s in seeds}), 'examples': len(rows),
        'split_counts': dict(Counter(r['split'] for r in rows)),
        'decision_counts': dict(Counter(r['decision'] for r in rows)),
        'server_tool_counts': dict(Counter(t['server'] for t in tools.values())),
        'split_tool_coverage': {split: len({c['tool'] for r in rows if r['split'] == split for c in r['acceptable_choices']}) for split in ('train', 'validation', 'test')},
        'sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files},
    }
    (directory / 'manifest.json').write_text(dump(manifest))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = build(args.directory)
    print(dump({k: v for k, v in result.items() if k != 'sha256'}))
