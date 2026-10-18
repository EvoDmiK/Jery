"""Export every v2 case and every original candidate without changing source splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from build_laya_pilot import ROOT, export_case


def digest(content):
    return hashlib.sha256(content).hexdigest()


def build(directory, output, seed=3407):
    if output.exists():
        raise ValueError('Output already exists; preserve previous export')
    catalog = json.loads((directory / 'tools.json').read_text())
    source_manifest = json.loads((directory / 'manifest.json').read_text())
    if digest((directory / 'tools.json').read_bytes()) != source_manifest['sha256']['tools.json']:
        raise ValueError('Source catalog hash mismatch')
    rng = random.Random(seed)
    exports, hashes, groups = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        raw = (directory / (split + '.jsonl')).read_bytes()
        hashes[split + '.jsonl'] = digest(raw)
        if hashes[split + '.jsonl'] != source_manifest['sha256'][split + '.jsonl']:
            raise ValueError('Source split hash mismatch: ' + split)
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != source_manifest['split_counts'][split]:
            raise ValueError('Source row count mismatch: ' + split)
        groups[split] = {r['scenario_id'] for r in rows}
        converted = []
        for row in rows:
            # Partition before reading any target: every original tool occurs exactly once.
            candidates = list(row['candidate_tools'])
            rng.shuffle(candidates)
            windows = [candidates[i:i + 8] for i in range(0, len(candidates), 8)]
            if not windows:
                raise ValueError('Empty original candidates are outside this export contract')
            for index, window in enumerate(windows):
                item = export_case(row, catalog, rng, window_tools=window)
                item['metadata'].update(window_index=index, window_count=len(windows),
                    export_scope='exhaustive_original_candidates',
                    export_id=row['id'] + ':window:' + str(index))
                converted.append(item)
        exports[split] = (rows, converted)
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if groups[a] & groups[b]:
            raise ValueError('Source groups overlap')
    output.mkdir(parents=True, exist_ok=False)
    stats = {}
    for split, (source, rows) in exports.items():
        content = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()
        (output / (split + '.jsonl')).write_bytes(content)
        stats[split] = {'source_rows': len(source), 'rows': len(rows), 'scenario_groups': len(groups[split]),
            'original_decisions': dict(Counter(r['decision'] for r in source)),
            'window_labels': dict(Counter(r['gold']['route_in_window']['label'] for r in rows)),
            'sha256': digest(content), 'candidate_coverage': 1.0}
    hashes['tools.json'] = digest((directory / 'tools.json').read_bytes())
    manifest = {'export_version': 'decision-export-v1', 'purpose': 'full_synthetic_exploratory',
        'export_scope': 'exhaustive_original_candidates', 'seed': seed,
        'dataset_version': 'mcp-routing-v2', 'catalog_version': catalog['catalog_version'],
        'source_sha256': hashes, 'splits': stats, 'max_window_tools': 8,
        'state_card_summary_characters': 120, 'action_prediction': False,
        'limitations': ['Synthetic unreviewed source labels; not production accuracy',
            'Test was inspected in earlier baseline experiments; regression evaluation, not fresh held-out proof',
            'All windows are retained: negative-window imbalance and candidate-count confounding remain',
            'Tokenizer budget must pass on every row before training',
            'Cross-window softmax scores are not comparable; aggregation must be separately declared']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'datasets/mcp-routing-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/laya-full-20261017/export')
    args = parser.parse_args()
    print(json.dumps(build(args.directory, args.output), indent=2, ensure_ascii=False))
