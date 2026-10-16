"""Validate offline MCP routing v2 references, splits, coverage and checksums."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / 'datasets/mcp-routing-v2'
DECISIONS = {'select_tools', 'no_tool', 'clarify', 'unsupported'}


def validate_inputs(catalog, seeds):
    tools = {t['id']: t for t in catalog['tools']}
    assert len(tools) == len(catalog['tools']) and tools, 'Duplicate/empty tools'
    digest = hashlib.sha256(json.dumps(catalog['tools'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    assert catalog['catalog_version'] == 'hermes-mcp-' + digest, 'Catalog fingerprint mismatch'
    families, seen = {}, set()
    for t in tools.values():
        assert t['id'] == f"mcp__{t['server']}__{t['name']}", 'Noncanonical tool ID'
    for s in seeds:
        key = s['scenario_id'], s.get('variant_id', 'base')
        assert key not in seen, 'Duplicate scenario variant'
        seen.add(key)
        assert s['split'] in ('train', 'validation', 'test')
        assert families.setdefault(s['scenario_id'], s['split']) == s['split'], 'Scenario leakage'
        assert s['decision'] in DECISIONS
        assert s['requests'] and all(isinstance(p, str) and p.strip() for p in s['requests'])
        assert bool(s['acceptable_choices']) == (s['decision'] == 'select_tools')
        assert not s.get('missing_target_case') or s['decision'] == 'select_tools'
        for c in s['acceptable_choices']:
            assert c['tool'] in tools, 'Unknown tool'
            if 'action' in c:
                assert c['action'] in tools[c['tool']]['input_schema']['properties']['action']['enum'], 'Invalid action'
    return tools


def validate(directory=DEFAULT):
    root = Path(directory)
    catalog = json.loads((root / 'tools.json').read_text())
    seeds = json.loads((root / 'scenarios.json').read_text())
    tools = validate_inputs(catalog, seeds)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['dataset_version'] == 'mcp-routing-v2'
    assert manifest['catalog_version'] == catalog['catalog_version']
    assert manifest['synthetic'] is True and manifest['human_reviewed'] is False
    expected_files = {'tools.json', 'scenarios.json', 'labels.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl'}
    assert set(manifest['sha256']) == expected_files
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f'Checksum: {name}'
    labels = json.loads((root / 'labels.json').read_text())
    assert len(labels) == len(tools)
    assert {r['tool'] for r in labels} == set(tools)
    for label in labels:
        assert label['server'] == tools[label['tool']]['server'], 'Server label mismatch'
        assert label['actions'] == tools[label['tool']]['input_schema'].get('properties', {}).get('action', {}).get('enum', [])
    rows, seen_ids, families, inputs = [], set(), {}, {}
    candidate_sizes = set()
    for split in ('train', 'validation', 'test'):
        split_rows = []
        for line in (root / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            assert row['id'] not in seen_ids, 'Duplicate row ID'
            seen_ids.add(row['id'])
            assert row['split'] == split
            assert families.setdefault(row['scenario_id'], split) == split, 'Scenario leakage'
            input_key = json.dumps([row['request'], row['context'], sorted(row['candidate_tools'])], sort_keys=True, ensure_ascii=False)
            assert input_key not in inputs, 'Duplicate/conflicting model input'
            inputs[input_key] = split
            assert row['tool_catalog_version'] == catalog['catalog_version']
            assert row['source'] == 'synthetic' and row['review_status'] == 'needs_human_review'
            assert row['execution_verified'] is False
            assert row['decision'] in DECISIONS
            assert bool(row['acceptable_choices']) == (row['decision'] == 'select_tools')
            assert len(row['candidate_tools']) == len(set(row['candidate_tools']))
            assert set(row['candidate_tools']) <= tools.keys()
            candidate_sizes.add(len(row['candidate_tools']))
            for message in row['context']:
                assert set(message) == {'role', 'content'}
                assert message['role'] in ('user', 'assistant', 'tool') and isinstance(message['content'], str)
            for choice in row['acceptable_choices']:
                assert choice['tool'] in row['candidate_tools'], 'Target outside candidates'
                if 'action' in choice:
                    assert choice['action'] in tools[choice['tool']]['input_schema']['properties']['action']['enum']
            split_rows.append(row)
        assert {r['decision'] for r in split_rows} == DECISIONS, 'Missing decision coverage'
        coverage = {c['tool'] for r in split_rows for c in r['acceptable_choices']}
        assert manifest['split_tool_coverage'][split] == len(coverage), 'Tool coverage mismatch'
        if split == 'train':
            assert coverage == set(tools), 'Missing train tool coverage'
        rows.extend(split_rows)
    assert sorted(catalog['registered_servers']) == sorted({t['server'] for t in tools.values()}), 'Missing registered server'
    assert len(candidate_sizes) > 1 and any(r['context'] for r in rows)
    assert manifest['examples'] == len(rows) and manifest['tools'] == len(tools)
    assert manifest['scenarios'] == len(families)
    assert manifest['servers'] == len({t['server'] for t in tools.values()})
    assert manifest['server_tool_counts'] == dict(Counter(t['server'] for t in tools.values()))
    assert manifest['split_counts'] == dict(Counter(r['split'] for r in rows))
    assert manifest['decision_counts'] == dict(Counter(r['decision'] for r in rows))
    print(f"PASS: {len(rows)} examples, {len(tools)} tools, {manifest['servers']} servers, {len(families)} isolated scenario groups")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    validate(parser.parse_args().directory)
