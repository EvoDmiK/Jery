"""Validate seed data integrity; does not certify semantic label correctness."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'datasets/hermes-routing-v1'


def validate():
    catalog = json.loads((ROOT / 'tools.json').read_text())
    tools = {t['id']: t for t in catalog['tools']}
    assert len(tools) == len(catalog['tools']), 'Duplicate catalog IDs'
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    seen_ids, seen_requests, scenarios, rows = set(), set(), {}, []
    for filename, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    for split in ('train', 'validation', 'test'):
        for line in (ROOT / f'{split}.jsonl').read_text().splitlines():
            r = json.loads(line)
            assert r['id'] not in seen_ids, 'Duplicate example'
            assert r['request'] not in seen_requests, 'Duplicate request'
            seen_ids.add(r['id']); seen_requests.add(r['request'])
            assert r['split'] == split
            assert scenarios.setdefault(r['scenario_id'], split) == split, 'Scenario leakage'
            assert r['tool_catalog_version'] == catalog['catalog_version']
            assert len(r['candidate_tools']) == len(set(r['candidate_tools']))
            assert set(r['candidate_tools']) <= tools.keys()
            assert r['source'] == 'synthetic' and not r['execution_verified']
            assert r['review_status'] == 'needs_human_review'
            assert r['decision'] in ('select_tools', 'no_tool', 'clarify', 'unsupported')
            assert bool(r['acceptable_choices']) == (r['decision'] == 'select_tools')
            for choice in r['acceptable_choices']:
                assert choice['tool'] in r['candidate_tools']
                if 'action' in choice:
                    schema = tools[choice['tool']]['input_schema']
                    assert choice['action'] in schema['properties']['action']['enum']
            rows.append(r)
    assert len(rows) == manifest['examples']
    assert len(tools) == manifest['tools']
    assert dict(Counter(r['split'] for r in rows)) == manifest['split_counts']
    assert dict(Counter(r['decision'] for r in rows)) == manifest['decision_counts']
    for split in ('train', 'validation', 'test'):
        covered = {c['tool'] for r in rows if r['split'] == split for c in r['acceptable_choices']}
        assert covered == tools.keys(), f'Missing tool coverage: {split}'
    print(f'PASS: {len(rows)} examples, {len(tools)} tools, {len(scenarios)} isolated scenarios')


if __name__ == '__main__':
    validate()
