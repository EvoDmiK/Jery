"""Compare matched Korean/English requests without changing candidates or using the diagnostic for tuning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def paired_rows(source, specification):
    if hashlib.sha256(source.read_bytes()).hexdigest() != specification['source_sha256']:
        raise ValueError('Diagnostic source checksum changed')
    rows = {r['metadata']['source_id']: r for r in map(json.loads, source.read_text().splitlines())
            if r['metadata']['window_count'] == 1}
    korean, english = [], []
    for pair in specification['pairs']:
        row = rows[pair['source_id']]
        if row['state']['context']:
            raise ValueError('Request-only language probe requires empty conversation context')
        korean.append(copy.deepcopy(row))
        changed = copy.deepcopy(row)
        changed['state']['request'] = pair['english_request']
        english.append(changed)
    if len({r['metadata']['source_id'] for r in korean}) != len(korean):
        raise ValueError('Duplicate diagnostic cases')
    return korean, english


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=ROOT / 'datasets/diagnostics/laya-ko-en-request-v1.json')
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/laya-full-20261017/export/validation.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--head', type=Path, help='Optional head directory; base model is always evaluated first')
    options = parser.parse_args()
    if options.output.exists():
        parser.error('Output already exists; diagnostic evidence is not overwritten')
    specification = json.loads(options.spec.read_text())
    variants = dict(zip(('korean_request', 'english_request'), paired_rows(options.source, specification)))
    import mlx.core as mx
    from unsloth import FastDecisionModel
    from unsloth.models._decision_common import _internal, _option_keys
    from unsloth_zoo.mlx.decision import load_trainable
    from jery.model.laya_full import Scorer, case_gold, evaluate_split, group_cases, write_json
    from jery.model.laya_pilot import QUESTION, sequence_budget
    mx.set_cache_limit(2 * 2**30)
    model, tokenizer = FastDecisionModel.from_pretrained(str(ROOT / 'models/laya-multilingual'), local_files_only=True, max_seq_length=1024)
    results = {'specification': specification, 'pairs': len(variants['korean_request']), 'models': {}}
    for name, head in [('base', None)] + ([('previous_full64_head', options.head)] if options.head else []):
        if head:
            load_trainable(model, head)
        FastDecisionModel.for_inference(model)
        scorer = Scorer(model, tokenizer, 16)
        result = {}
        for language, rows in variants.items():
            for row in rows:
                budget = sequence_budget(tokenizer, scorer.common, row['state'], _internal(row['questions'][QUESTION]),
                                         scorer.max_len, scorer.head_max_len)
                if any(budget[k] for k in ('options_cut', 'option_text_collisions', 'instruction_cut', 'state_cut')):
                    raise ValueError('Language comparison would truncate an input')
            items, built = FastDecisionModel.build_dataset(rows, tokenizer, model)
            if built['skipped'] or len(items) != len(rows):
                raise ValueError('Language probe dropped cases')
            for item, row in zip(items, rows):
                item['keys'] = _option_keys(_internal(row['questions'][QUESTION]))
            metrics, predictions, windows = evaluate_split(scorer, rows, items, group_cases(rows), case_gold(rows))
            result[language] = {'metrics': metrics, 'cases': predictions, 'windows': windows}
        results['models'][name] = result
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(options.output, results)
    print(json.dumps({name: {lang: r['metrics']['case_accuracy'] for lang, r in result.items()}
                      for name, result in results['models'].items()}))


if __name__ == '__main__':
    main()
