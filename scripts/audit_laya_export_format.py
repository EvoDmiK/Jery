"""Audit the frozen export with the installed native Laya builder and tokenizer; never train or rewrite data."""
import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve existing audit outputs; choose a new path')
    import unsloth
    from transformers import AutoTokenizer
    from unsloth.models.decision import FastDecisionModel
    from jery.model.laya_cuda import CudaScorer
    from jery.model.laya_full import QUESTION, SPLITS
    from jery.model.laya_pilot import load_export, sha256

    config = json.loads(args.config.read_text())
    model_dir = Path(config['model_dir'])
    native_config = json.loads((model_dir / 'rl_agent_config.json').read_text())
    native_config['max_len'] = config['max_seq_length']
    # Only the native builder/encoder configuration is needed; no weights, forward pass or GPU allocation.
    model = SimpleNamespace(decision_config=native_config)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    scorer = CudaScorer(model, tokenizer, 1)
    rows, export = load_export(Path(config['export_dir']), SPLITS)
    report = {'scope': 'native schema, target mapping and tokenization; not data-quality or pretraining-corpus equivalence',
              'export': export, 'native_limits': {k: native_config[k] for k in ('max_len', 'head_max_len')},
              'packages': {p: importlib.metadata.version(p) for p in ('unsloth', 'laya', 'transformers')},
              'builder_source_sha256': sha256(Path(inspect.getfile(FastDecisionModel))), 'splits': {}}
    for split in SPLITS:
        items, built = FastDecisionModel.build_dataset(rows[split], tokenizer, model)
        if built['skipped'] or len(items) != len(rows[split]):
            raise ValueError(f'{split}: native builder rejected rows: {built}')
        max_tokens, kinds = 0, Counter()
        for row, item in zip(rows[split], items):
            blind, budget = scorer.encode(row)  # Raises on any truncation or option collision.
            if any(item[k] != blind[k] for k in ('input_ids', 'markers', 'qtype')):
                raise ValueError('Native training features differ from label-blind inference features')
            gold, meta = row['gold'][QUESTION], row['metadata']
            probs = gold['probabilities']
            expected = [probs.get(key, 0.) for key in blind['keys']]
            if (not set(probs).issubset(blind['keys']) or abs(sum(expected) - 1) > 1e-6
                    or len(item['target']) != len(expected)
                    or any(abs(a - b) > 1e-6 for a, b in zip(item['target'], expected))):
                raise ValueError('Gold probabilities were not mapped to the native option indices')
            decision = meta['original_decision']
            allowed = {c['tool'] for c in meta['acceptable_choices']}
            targets = ({decision} if decision in ('no_tool', 'clarify') else
                       {alias for alias, tool in meta['alias_to_tool'].items() if tool in allowed} or {'window_none'})
            if {key for key, value in probs.items() if value > 0} != targets or gold['label'] not in targets:
                raise ValueError('Export gold disagrees with the recorded original decision and acceptable tools')
            max_tokens = max(max_tokens, budget['sequence_tokens'])
            kinds.update([decision])
        report['splits'][split] = {'windows': len(items), 'native_builder': built, 'max_sequence_tokens': max_tokens,
                                  'original_decisions': dict(kinds), 'feature_mismatches': 0, 'target_mismatches': 0,
                                  'truncated_or_colliding_inputs': 0}
    report['status'] = 'passed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
