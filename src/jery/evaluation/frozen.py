"""Read-only gate and runner for a frozen ten-tool evaluation set (frozen-eval-v1) on fixed Laya heads.

The gate refuses before any model call if the manifest, file hashes, catalog, candidates, labels or leakage checks
fail. Model input is built from request and context only; ids, labels, source and review fields never reach it.
Scoring reuses the native Laya sequence builder (CudaScorer) and the training-time ten_tool_choice metrics.

Gate only (no model, any status):
    PYTHONPATH=src python3 -m jery.evaluation.frozen --frozen <dir> --labels human --gate-only
Benchmark (status=frozen and the root-authorized manifest hash; CUDA host):
    PYTHONPATH=src python -m jery.evaluation.frozen --frozen <dir> --labels human \
        --authorize-manifest-sha256 <sha> --head <run_id>=<trainable.safetensors> ... --out <new dir>
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import unicodedata

from jery.model.laya_cuda import TEN_TOOLS, check_ten_tool_rows
from jery.model.laya_full import QUESTION, ROOT, blind, case_gold, evaluate_split, group_cases, write_json
from jery.model.laya_pilot import TEN_TOOL_SCOPE, read_jsonl, sha256
from jery.runtime.prediction import model_input, validate_prediction

SPEC = ROOT / 'configs/model/frozen-eval-ten-tools-20261020.json'
CASES, CATALOG, PROPOSALS = 'cases.jsonl', 'tools.json', 'ai-proposals.json'
LABEL_BASIS = {'human': 'human_reviewed',
               'provisional-ai': 'provisional_ai_agreement (not human reviewed; exploratory only)'}


class GateError(ValueError):
    """The frozen set must not be scored as it is."""


def signature(state):
    """Normalized request+context, the same leakage key as scripts/build_ten_tool_contrast.py."""
    text = unicodedata.normalize('NFKC', json.dumps(state, ensure_ascii=False, sort_keys=True)).casefold()
    return ''.join(c for c in text if c.isalnum())


def single_tool(choices, candidates, catalog):
    """The one distinct tool of select_tools choices (actions checked against the catalog), else None."""
    try:
        validate_prediction({'decision': 'select_tools', 'choices': choices}, candidates, catalog)
    except ValueError:
        return None
    tools = {c['tool'] for c in choices}
    return tools.pop() if len(tools) == 1 else None


def resolve_label(case, proposals, labels, catalog, min_proposers):
    """(tool, acceptable_choices) of a resolved label, or None; never promotes AI proposals to human review."""
    if labels == 'human':
        if case.get('review_status') != 'human_reviewed' or case.get('decision') != 'select_tools':
            return None
        tool = single_tool(case.get('acceptable_choices'), case['candidate_tools'], catalog)
        return tool and (tool, case['acceptable_choices'])
    votes = proposals.get(case['id'], [])
    if (len({p.get('proposer') for p in votes}) < min_proposers
            or any(p.get('disposition') != 'label' or p.get('decision') != 'select_tools' for p in votes)):
        return None
    tools = {single_tool(p.get('acceptable_choices'), case['candidate_tools'], catalog) for p in votes}
    if len(tools) != 1 or None in tools:
        return None
    tool = tools.pop()
    choices = list({json.dumps(c, sort_keys=True): c for p in votes for c in p['acceptable_choices']}.values())
    return tool, choices


def load_frozen(folder, labels, spec_path=SPEC, gate_only=False, authorized_sha=None, exclude_exports=()):
    """Gate the frozen folder and return (export-shaped rows, gate report); raises GateError on any refusal."""
    if labels not in LABEL_BASIS:
        raise GateError(f'labels must be one of {sorted(LABEL_BASIS)}')
    folder, spec = Path(folder), json.loads(Path(spec_path).read_text())
    criteria_path = ROOT / spec['criteria']
    if 'hermes-ten-tools-' + sha256(criteria_path)[:12] != spec['catalog_version']:
        raise GateError('criteria file differs from the pinned v2.1 catalog version')
    criteria = json.loads(criteria_path.read_text())
    tools = [t['id'] for t in criteria['tools']]
    descriptions = {t['id']: t['description'] for t in criteria['tools']}

    manifest_sha = sha256(folder / 'manifest.json')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest.get('schema_version') != spec['schema_version']:
        raise GateError(f"schema_version must be {spec['schema_version']}")
    if not gate_only and (manifest.get('status') != 'frozen' or authorized_sha != manifest_sha):
        raise GateError('scoring needs status=frozen and --authorize-manifest-sha256 equal to the manifest sha256')
    if manifest.get('catalog_version') != spec['catalog_version']:
        raise GateError('manifest catalog_version differs from the pinned catalog')
    seed = manifest.get('candidate_order_seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GateError('manifest candidate_order_seed must be an integer')
    hashes = manifest.get('sha256') or {}
    needed = {CASES, CATALOG} | ({PROPOSALS} if labels == 'provisional-ai' else set())
    if not needed <= set(hashes):
        raise GateError(f'manifest sha256 must cover {sorted(needed)}')
    for name, expected in hashes.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(folder / name) != expected:
            raise GateError(f'{name}: sha256 differs from the manifest')
    leak = manifest.get('leak_audit') or {}
    leak_ready = (leak.get('status') == 'passed' and leak.get('file') in hashes
                  and json.loads((folder / leak['file']).read_text()).get('status') == 'passed')
    if not gate_only and not leak_ready:
        raise GateError('scoring needs leak_audit.status=passed and a hashed leak_audit.file whose status is passed')
    if hashes[CATALOG] != spec['catalog_sha256']:
        raise GateError('tools.json differs from the pinned catalog')
    catalog = json.loads((folder / CATALOG).read_text())
    if catalog.get('catalog_version') != spec['catalog_version'] or [t['id'] for t in catalog['tools']] != tools:
        raise GateError('catalog version or tool ids differ from the pinned ten tools')
    proposals = json.loads((folder / PROPOSALS).read_text()) if labels == 'provisional-ai' else {}

    cases = read_jsonl(folder / CASES)
    if len({c.get('id') for c in cases}) != len(cases) or not cases:
        raise GateError('cases must be nonempty with unique ids')
    seen = {}
    for export in exclude_exports:
        for path in sorted(Path(export).glob('*.jsonl')):
            seen.update({signature(r['state']): f'{path.name}:{r["metadata"]["source_id"]}' for r in read_jsonl(path)})
    rows, unresolved, problems = [], [], []
    for case in cases:
        cid = case['id']
        try:
            features = model_input(case, catalog)
        except ValueError as error:
            problems.append(f'{cid}: {error}')
            continue
        state = {'request': features['request'], 'context': features['context']}
        if case.get('tool_catalog_version') != spec['catalog_version']:
            problems.append(f'{cid}: tool_catalog_version differs')
        if len(case['candidate_tools']) != 10 or set(case['candidate_tools']) != set(tools):
            problems.append(f'{cid}: candidate_tools must be exactly the pinned ten tools')
        if (case.get('source') or {}).get('kind') != 'real':
            problems.append(f'{cid}: source.kind must be real')
        if 'mcp__' in json.dumps(state, ensure_ascii=False):
            problems.append(f'{cid}: tool id leaked into request/context')
        key = signature(state)
        if key in seen:
            problems.append(f'{cid}: normalized input also in {seen[key]}')
        seen[key] = cid
        if problems:
            continue
        label = resolve_label(case, proposals, labels, catalog, spec['min_ai_proposers'])
        if label is None:
            unresolved.append(cid)
            continue
        order = tools[:]
        random.Random(f'{seed}:{cid}').shuffle(order)  # Label-blind, as in the training exports.
        aliases = {f'T{i}': t for i, t in enumerate(order)}
        gold = next(a for a, t in aliases.items() if t == label[0])
        rows.append({'state': state,
                     'questions': {QUESTION: {'type': 'choice', 'instructions': spec['instructions'],
                                              'criteria': {a: descriptions[t] for a, t in aliases.items()}}},
                     'gold': {QUESTION: {'label': gold, 'probabilities': {gold: 1.0}}},
                     'metadata': {'source_id': cid, 'export_id': cid, 'window_index': 0, 'window_count': 1,
                                  'window_tools': order, 'alias_to_tool': aliases, 'original_candidate_tools': tools,
                                  'original_decision': 'select_tools', 'acceptable_choices': label[1],
                                  'review_status': case.get('review_status')}})
    if problems:
        raise GateError('; '.join(problems))
    if unresolved:
        raise GateError(f'{len(unresolved)} unresolved labels ({labels}): {unresolved}')
    check_ten_tool_rows({'frozen': rows}, {'export_scope': TEN_TOOL_SCOPE, 'evaluation_task': TEN_TOOLS})
    counts = Counter(r['metadata']['alias_to_tool'][r['gold'][QUESTION]['label']] for r in rows)
    missing = {t: max(0, spec['target_per_tool'] - counts[t]) for t in tools}
    gate = {'schema_version': spec['schema_version'], 'status': manifest['status'], 'manifest_sha256': manifest_sha,
            'files_sha256': hashes, 'catalog_version': spec['catalog_version'], 'label_basis': LABEL_BASIS[labels],
            'leak_audit_ready': leak_ready, 'cases': len(rows), 'tool_counts': {t: counts[t] for t in tools},
            'target_per_tool': spec['target_per_tool'], 'missing_by_tool': missing, 'meets_target': not any(missing.values()),
            'review_status_counts': dict(Counter(c.get('review_status') for c in cases)),
            'excluded_exports': [str(e) for e in exclude_exports]}
    return rows, gate


def evaluate(rows, gate, scorer):
    """Per-case records and aggregate metrics of one head with the training-time ten_tool_choice metrics."""
    items = [scorer.encode(blind(row))[0] for row in rows]
    metrics, predictions, windows = evaluate_split(scorer, rows, items, group_cases(rows, 10), case_gold(rows), TEN_TOOLS)
    per_case = []
    for row, prediction, window in zip(rows, predictions, windows):
        aliases = row['metadata']['alias_to_tool']
        gold = aliases[row['gold'][QUESTION]['label']]
        per_case.append({'id': row['metadata']['source_id'], 'gold_tool': gold, 'predicted_tool': prediction['tool'],
                         'correct': prediction['tool'] == gold, 'review_status': row['metadata']['review_status'],
                         'probabilities': {aliases[a]: p for a, p in window['probabilities'].items()}})
    metrics.update({k: gate[k] for k in ('label_basis', 'meets_target', 'missing_by_tool', 'review_status_counts')})
    metrics['claim'] = ('real-request accuracy against human-reviewed labels' if gate['label_basis'] == 'human_reviewed'
                        else 'exploratory agreement with provisional AI labels; not reviewed real accuracy')
    if not gate['meets_target']:
        metrics['claim'] += f"; partial subset of {gate['cases']} cases below the per-tool target, not the full real set"
    return per_case, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--labels', choices=sorted(LABEL_BASIS), required=True)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--gate-only', action='store_true', help='validate without loading or running any model')
    parser.add_argument('--exclude-export', type=Path, action='append', default=[],
                        help='folder of training/validation/test JSONL whose inputs must not reappear')
    parser.add_argument('--authorize-manifest-sha256')
    parser.add_argument('--head', action='append', default=[], help='<run_id>=<trainable.safetensors>')
    parser.add_argument('--assets-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path)
    options = parser.parse_args()
    rows, gate = load_frozen(options.frozen, options.labels, options.spec, options.gate_only,
                             options.authorize_manifest_sha256, options.exclude_export)
    if options.gate_only:
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return
    spec = json.loads(options.spec.read_text())
    heads = dict(h.split('=', 1) for h in options.head)
    if set(heads) != set(spec['candidates'].values()) or options.out is None or options.out.exists():
        raise SystemExit('give --head for exactly the pinned candidate runs and a new --out directory')
    from jery.model.laya_cuda import CudaScorer, load_cuda_model, restore_head
    config = json.loads((ROOT / spec['model_config']).read_text())
    config['model_dir'] = str(options.assets_root / config['model_dir'])
    if sha256(Path(config['model_dir']) / 'model.safetensors') != config['base_model_sha256']:
        raise SystemExit('base model checksum differs')
    model, tokenizer = load_cuda_model(config)
    model.eval()
    options.out.mkdir(parents=True)
    write_json(options.out / 'gate.json', gate)
    for name, run_id in spec['candidates'].items():
        restore_head(model, heads[run_id])
        per_case, metrics = evaluate(rows, gate, CudaScorer(model, tokenizer, config['eval_batch_size']))
        metrics.update(run_id=run_id, head_sha256=sha256(heads[run_id]), base_model_sha256=config['base_model_sha256'])
        (options.out / f'{name}.cases.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in per_case))
        write_json(options.out / f'{name}.metrics.json', metrics)
        print(json.dumps({'candidate': name, 'case_accuracy': metrics['case_accuracy'],
                          'tool_macro_f1': metrics['tool_macro_f1'], 'label_basis': metrics['label_basis']}))


if __name__ == '__main__':
    main()
