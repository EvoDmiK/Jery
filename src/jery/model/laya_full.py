"""Head-only Laya training and case-level evaluation on the exhaustive decision-export-v1 windows.

Every original candidate set C is split into windows of at most 8 tools. Training retains every train window;
optional train-only oversampling balances decision buckets. Each epoch scores all validation windows and
aggregates them per source case without labels. The optional validation callback stops on a plateau and restores
the best head (including epoch zero); the legacy default uses the last epoch. Test is scored once after selection.

PYTHONPATH=src tmp/venv-decision/bin/python -m jery.model.laya_full \
    --config configs/model/laya-full-20261017.json --assets-root <checkout with export and model> \
    --run-id <id> [--override '{"head_learning_rate": 0.0003}'] [--probe] [--resume]
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
import traceback

from jery.model.features import DECISIONS
from jery.model.laya_pilot import (QUESTION, ROOT, MemoryGuard, load_export, provenance, sequence_budget,
                                   sha256, tensors_sha256)

SPLITS = ('train', 'validation', 'test')
NULLS = ('no_tool', 'clarify', 'window_none')
WINDOW = 8
RNG_FILE = 'mx_random_key.json'
# Internal hand-back to the calling agent; not a prediction-v1 decision.
DELEGATE = 'delegate'
# The only metadata aggregation may see: lineage and the alias map, never gold or original labels.
BLIND_METADATA = ('source_id', 'window_index', 'window_count', 'export_id', 'window_tools', 'alias_to_tool',
                  'original_candidate_tools')


def balanced_indices(rows, ratios, seed):
    """Train-only oversampling; retain every original window and repeat evenly within each bucket.

    A null bucket may have ratio 0 only when it has no source windows at all, so no window is ever dropped."""
    if set(ratios) != {'tool', *NULLS} or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in ratios.values()) \
            or ratios['tool'] <= 0:
        raise ValueError('Non-negative integer ratios required for all three null buckets, positive for tool')
    buckets = {key: [] for key in ratios}
    for index, row in enumerate(rows):
        label = row['gold'][QUESTION]['label']
        buckets[label if label in NULLS else 'tool'].append(index)
    if any(not values for key, values in buckets.items() if ratios[key]):
        raise ValueError('Cannot balance a training bucket with no examples')
    if any(values for key, values in buckets.items() if not ratios[key]):
        raise ValueError('Ratio 0 would discard source windows of a non-empty bucket')
    scale = max(math.ceil(len(buckets[key]) / ratio) for key, ratio in ratios.items() if ratio)
    rng, indices = random.Random(seed), []
    for key, ratio in ratios.items():
        if not ratio:
            continue
        pool = buckets[key][:]
        rng.shuffle(pool)
        target = scale * ratio
        indices.extend((pool * math.ceil(target / len(pool)))[:target])
    rng.shuffle(indices)
    return indices, {'source_counts': {k: len(v) for k, v in buckets.items()},
                     'sampled_counts': {k: scale * v for k, v in ratios.items()},
                     'unique_windows': len(set(indices)), 'sampled_windows': len(indices),
                     'ratios': ratios, 'seed': seed, 'indices': indices,
                     'policy': 'training-only deterministic oversampling; no source windows removed'}


class ValidationPolicy:
    """Only validation case accuracy selects a head; epoch zero is an eligible baseline."""
    def __init__(self, patience, min_delta):
        if not isinstance(patience, int) or patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('Invalid validation patience/min_delta')
        self.patience, self.min_delta = patience, min_delta
        self.best_epoch, self.best_score, self.bad_epochs, self.stopped = None, None, 0, False

    def observe(self, epoch, score):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Nonfinite or invalid validation accuracy')
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_epoch, self.best_score, self.bad_epochs = epoch, score, 0
        elif epoch > 0:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return improved

    def snapshot(self):
        return {'monitor': 'validation.case_accuracy', 'mode': 'max', 'patience': self.patience,
                'min_delta': self.min_delta, 'best_epoch': self.best_epoch, 'best_score': self.best_score,
                'bad_epochs': self.bad_epochs, 'stopped': self.stopped}


# ---- label-blind case aggregation (pure Python; reads only state, questions and the alias/window metadata) ----

def blind(row):
    """The window as aggregation sees it: model input plus lineage, without gold or original labels."""
    return {'state': row['state'], 'questions': row['questions'],
            'metadata': {key: row['metadata'][key] for key in BLIND_METADATA}}


def case_gold(rows):
    """Original v2 labels per source case, for evaluation only."""
    return {row['metadata']['source_id']: {'decision': row['metadata']['original_decision'],
                                           'candidate_tools': row['metadata']['original_candidate_tools'],
                                           'acceptable_choices': row['metadata']['acceptable_choices']} for row in rows}


def group_cases(rows, max_window_tools=WINDOW):
    """Blind windows of each source case in window order, after checking they cover C exactly once."""
    cases = defaultdict(list)
    for row in rows:
        cases[row['metadata']['source_id']].append(blind(row))
    for source_id, windows in cases.items():
        windows.sort(key=lambda w: w['metadata']['window_index'])
        meta = windows[0]['metadata']
        if [w['metadata']['window_index'] for w in windows] != list(range(meta['window_count'])):
            raise ValueError(f'{source_id}: windows missing or duplicated')
        tools = [t for w in windows for t in w['metadata']['window_tools']]
        if len(tools) != len(set(tools)) or set(tools) != set(meta['original_candidate_tools']):
            raise ValueError(f'{source_id}: windows do not cover the original candidates exactly once')
        for w in windows:
            if list(w['metadata']['alias_to_tool'].values()) != w['metadata']['window_tools'] or len(w['metadata']['window_tools']) > max_window_tools:
                raise ValueError(f'{source_id}: alias map does not match window tools')
    return dict(cases)


def final_window(windows, tools):
    """A new window over `tools` (original C order) built from the case's own cards, aliases T0..Tn."""
    first = windows[0]
    cards = {w['metadata']['alias_to_tool'][c['alias']]: c for w in windows for c in w['state']['candidates']}
    aliases = {f'T{i}': tool for i, tool in enumerate(tools)}
    question = first['questions'][QUESTION]
    sentinels = {k: v for k, v in question['criteria'].items() if k not in first['metadata']['alias_to_tool']}
    if set(sentinels) != set(NULLS):
        raise ValueError('window question must carry exactly no_tool, clarify and window_none')
    return {'state': {**first['state'], 'candidates': [{**cards[t], 'alias': a} for a, t in aliases.items()]},
            'questions': {QUESTION: {**question, 'criteria': {**{a: '도구 ' + a for a in aliases}, **sentinels}}},
            'alias_to_tool': aliases}


def aggregate(cases, first_choices, score):
    """Public-shaped case predictions from per-window top-1 choices, never comparing scores across windows.

    `first_choices[source_id]` holds the top-1 option of each window in window order; `score(windows)` returns
    the top-1 option of each new window. One window: its answer is final (window_none -> unsupported, C is fully
    seen). Several windows with tool winners: winners (C order) meet again in rounds of <= 8; a chunk answering
    no_tool/clarify/window_none just yields no winner, and a round with no winner delegates. The single last window
    decides (a tool -> select_tools, no_tool/clarify -> that, window_none -> delegate). No tool winner in the first
    pass: plurality of the window votes (window_none -> unsupported, every window of C seen), a tie -> delegate.
    """
    out, pending = {}, {}
    for source_id, windows in cases.items():
        choices = first_choices[source_id]
        record = {'source_id': source_id, 'window_choices': choices, 'rounds': 0, 'forwards': len(windows)}
        picked = {w['metadata']['alias_to_tool'][c] for w, c in zip(windows, choices) if c not in NULLS}
        if len(windows) == 1:
            out[source_id] = _resolve(record, 'single', choices[0], windows[0]['metadata']['alias_to_tool'])
        elif picked:
            pending[source_id] = [t for t in windows[0]['metadata']['original_candidate_tools'] if t in picked]
        else:
            votes = Counter(choices).most_common()
            tie = len(votes) > 1 and votes[0][1] == votes[1][1]
            out[source_id] = _resolve(record, 'vote', None if tie else votes[0][0], {})
            out[source_id]['votes'] = dict(votes)
        if source_id in pending:
            out[source_id] = record
    while pending:
        built = [(source_id, final_window(cases[source_id], winners[i:i + WINDOW]))
                 for source_id, winners in pending.items() for i in range(0, len(winners), WINDOW)]
        choices = score([window for _, window in built])
        by_case = defaultdict(list)
        for (source_id, window), choice in zip(built, choices):
            by_case[source_id].append((window, choice))
        pending = {}
        for source_id, results in by_case.items():
            record = out[source_id]
            record['rounds'] += 1
            record['forwards'] += len(results)
            record.setdefault('round_choices', []).append([c for _, c in results])
            if len(results) == 1:
                window, choice = results[0]
                _resolve(record, 'final', DELEGATE if choice == 'window_none' else choice, window['alias_to_tool'])
            else:
                winners = [w['alias_to_tool'][c] for w, c in results if c not in NULLS]
                if winners:
                    pending[source_id] = winners
                else:
                    _resolve(record, 'final', None, {})
    return [out[source_id] for source_id in cases]


def _resolve(record, path, choice, alias_to_tool):
    record['path'] = path
    if choice is None or choice == DELEGATE:
        record.update(decision=DELEGATE, tool=None)
    elif choice in ('no_tool', 'clarify'):
        record.update(decision=choice, tool=None)
    elif choice == 'window_none':
        record.update(decision='unsupported', tool=None)
    else:
        record.update(decision='select_tools', tool=alias_to_tool[choice])
    return record


def _ratio(n, d):
    return round(n / d, 4) if d else None


def case_metrics(gold, predictions):
    """Case-level metrics against original v2 labels; delegate is wrong for its gold decision and lost coverage,
    never a call and never a true or false positive of a public decision."""
    tp, fp, fn = Counter(), Counter(), Counter()
    correct = tool_ok = select_gold = calls_on_null = null_gold = outside = delegated = 0
    for prediction in predictions:
        truth, got, tool = gold[prediction['source_id']], prediction['decision'], prediction['tool']
        if got not in DECISIONS and got != DELEGATE:
            raise ValueError(f'unknown decision: {got!r}')
        good = {c['tool'] for c in truth['acceptable_choices']}
        delegated += got == DELEGATE
        outside += got == 'select_tools' and tool not in truth['candidate_tools']
        if got == truth['decision']:
            tp[got] += 1
        else:
            fn[truth['decision']] += 1
            fp[got] += got != DELEGATE
        if truth['decision'] == 'select_tools':
            select_gold += 1
            hit = got == 'select_tools' and tool in good
            tool_ok, correct = tool_ok + hit, correct + hit
        else:
            null_gold += 1
            correct += got == truth['decision']
            calls_on_null += got == 'select_tools'
    per_decision = {}
    for decision in DECISIONS:
        precision, recall = _ratio(tp[decision], tp[decision] + fp[decision]), _ratio(tp[decision], tp[decision] + fn[decision])
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
        per_decision[decision] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp[decision] + fn[decision]}
    f1s = [v['f1'] for v in per_decision.values() if v['support']]
    forwards = [p['forwards'] for p in predictions]
    return {'cases': len(predictions), 'case_accuracy': _ratio(correct, len(predictions)),
            'decision_accuracy': _ratio(sum(tp.values()), len(predictions)),
            'decision_macro_f1': round(sum(f1s) / len(f1s), 4) if f1s else None,
            'tool_accuracy': _ratio(tool_ok, select_gold), 'unnecessary_call_rate': _ratio(calls_on_null, null_gold),
            'out_of_candidate_selections': outside, 'delegated': delegated,
            'coverage': _ratio(len(predictions) - delegated, len(predictions)), 'per_decision': per_decision,
            'paths': dict(Counter(p['path'] for p in predictions)), 'forwards_mean': statistics.fmean(forwards),
            'forwards_max': max(forwards), 'rounds_max': max(p['rounds'] for p in predictions)}


def tool_macro_f1(gold, predictions):
    """ten_tool_choice: per-tool F1 of the predicted tool against the single acceptable gold tool, averaged
    over tools with gold support (the decision_macro_f1 convention). A wrong pick is a false positive of the
    picked tool and a false negative of the gold tool."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for prediction in predictions:
        truth = list({c['tool'] for c in gold[prediction['source_id']]['acceptable_choices']})
        if len(truth) != 1:
            raise ValueError('ten_tool_choice needs exactly one acceptable tool per case')
        if prediction['tool'] == truth[0]:
            tp[truth[0]] += 1
        else:
            fn[truth[0]] += 1
            fp[prediction['tool']] += prediction['tool'] is not None
    per_tool = {}
    for tool in sorted(set(tp) | set(fn)):
        precision, recall = _ratio(tp[tool], tp[tool] + fp[tool]), _ratio(tp[tool], tp[tool] + fn[tool])
        f1 = round(2 * precision * recall / (precision + recall), 4) if precision and recall else 0.0
        per_tool[tool] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': tp[tool] + fn[tool]}
    return round(statistics.fmean(v['f1'] for v in per_tool.values()), 4), per_tool


def window_metrics(rows, choices, probabilities):
    """Window-level any-of top-1 and soft CE (probabilities floored at 1e-12), by gold kind."""
    hits, losses, kinds = 0, [], defaultdict(lambda: [0, 0])
    for row, choice, probs in zip(rows, choices, probabilities):
        target = row['gold'][QUESTION]['probabilities']
        total = sum(target.values())
        losses.append(-sum(v / total * math.log(max(probs.get(k, 0.0), 1e-12)) for k, v in target.items() if v > 0))
        hit = target.get(choice, 0) > 0
        label = row['gold'][QUESTION]['label']
        kind = kinds[label if label in NULLS else 'tool']
        hits, kind[0], kind[1] = hits + hit, kind[0] + hit, kind[1] + 1
    return {'windows': len(rows), 'any_of_accuracy': hits / len(rows), 'soft_ce': statistics.fmean(losses),
            'by_gold_kind': {k: {'correct': c, 'windows': n, 'accuracy': c / n} for k, (c, n) in sorted(kinds.items())}}


# ---- MLX side ----

class Scorer:
    """Tokenizes windows with Laya's own sequence builder (refusing any cut) and returns eval-mode top-1s."""

    def __init__(self, model, tokenizer, batch_size):
        from unsloth.models._decision_common import _laya
        from unsloth.models.decision_mlx import _decision_pad_token_id
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size
        self.common, self.pad = _laya().common, _decision_pad_token_id(tokenizer)
        self.max_len, self.head_max_len = int(model.decision_config['max_len']), int(model.decision_config['head_max_len'])

    def encode(self, window):
        from unsloth.models._decision_common import _internal, _option_keys
        internal = _internal(window['questions'][QUESTION])
        budget = sequence_budget(self.tokenizer, self.common, window['state'], internal, self.max_len, self.head_max_len)
        if budget['options_cut'] or budget['option_text_collisions'] or budget['instruction_cut'] or budget['state_cut']:
            raise RuntimeError(f'window input would be cut: {budget}')
        ids, markers = self.common.build_sequence(self.tokenizer, window['state'], internal, self.max_len, self.head_max_len)
        keys = _option_keys(internal)
        return {'input_ids': ids, 'markers': markers, 'qtype': self.common.QTYPES[internal['t']],
                'target': [0.0] * len(keys), 'keys': keys}, budget

    def logits(self, items):
        from unsloth_zoo.mlx.decision import decision_logits
        return decision_logits(self.model, items, self.pad, batch_size=self.batch_size)

    def top(self, items):
        """Top-1 option and softmax probabilities (temperature 1, as served) of each item."""
        choices, probabilities = [], []
        for item, row in zip(items, self.logits(items)):
            shifted = [math.exp(v - max(row)) for v in row.tolist()]
            total = sum(shifted)
            probs = {k: v / total for k, v in zip(item['keys'], shifted)}
            choices.append(item['keys'][int(row.argmax())])
            probabilities.append(probs)
        return choices, probabilities

    def __call__(self, windows):
        return self.top([self.encode(w)[0] for w in windows])[0]


def evaluate_split(scorer, rows, items, cases, gold, task=None):
    """Window metrics, case metrics and raw predictions for one split; task=ten_tool_choice adds tool macro F1."""
    started = time.perf_counter()
    choices, probabilities = scorer.top(items)
    window_seconds = time.perf_counter() - started
    by_id = {row['metadata']['export_id']: choice for row, choice in zip(rows, choices)}
    first = {source_id: [by_id[w['metadata']['export_id']] for w in windows] for source_id, windows in cases.items()}
    predictions = aggregate(cases, first, scorer)
    case = case_metrics(gold, predictions)
    if case['out_of_candidate_selections']:
        raise RuntimeError(f"{case['out_of_candidate_selections']} selections outside the original candidates")
    metrics = {'case_accuracy': case['case_accuracy'], 'decision_macro_f1': case['decision_macro_f1'],
               'coverage': case['coverage'], 'case': case, 'window': window_metrics(rows, choices, probabilities),
               'seconds': {'windows': window_seconds, 'total': time.perf_counter() - started}}
    if task == 'ten_tool_choice':
        metrics['tool_macro_f1'], case['per_tool'] = tool_macro_f1(gold, predictions)
        metrics['macro_f1_scope'] = 'tools'
    return metrics, predictions, [{'export_id': r['metadata']['export_id'], 'choice': c, 'probabilities': p}
                                  for r, c, p in zip(rows, choices, probabilities)]


class JsonLines:
    def __init__(self, path):
        self.handle = path.open('a')

    def write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())


def write_json(path, data):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def repair_json_lines(path):
    """Discard only an interrupted trailing write; reject corrupt complete records."""
    if not path.exists():
        return
    content = path.read_bytes()
    complete = content[:content.rfind(b'\n') + 1]
    for line in complete.splitlines():
        if line.strip():
            json.loads(line)
    if complete != content:
        path.write_bytes(complete)


def run(config, run_id, out, log, probe=False, resume=False):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    import numpy as np
    from transformers import TrainerCallback
    from unsloth import FastDecisionModel
    from unsloth.models.decision_mlx import DecisionTrainer
    from unsloth_zoo.mlx.decision import load_trainable, save_trainable
    from unsloth_zoo.mlx.trainer import MLXTrainingConfig

    # Batched eval pads each batch to its own length and MLX keeps every shape's buffers in its cache (24.8 GB for
    # 320 validation windows uncapped, 2.2 GB capped at 2 GiB at the same speed). The trainer caps it while training.
    mx.set_cache_limit(int(config['mlx_cache_limit_gib'] * 2 ** 30))
    report = {'run_id': run_id, 'probe': probe, 'config': config}
    model_dir, weights = Path(config['model_dir']), ROOT / config['models_dir'] / run_id
    rows, report['export'] = load_export(Path(config['export_dir']), SPLITS)
    manifest = json.loads((Path(config['export_dir']) / 'manifest.json').read_text())
    report['export'].update(dir=config['export_dir'], purpose=manifest.get('purpose'), scope=manifest.get('export_scope'),
                            dataset_version=manifest.get('dataset_version'), catalog_version=manifest.get('catalog_version'))
    report['provenance'] = provenance(config, model_dir)
    cases = {split: group_cases(split_rows) for split, split_rows in rows.items()}
    gold = {split: case_gold(split_rows) for split, split_rows in rows.items()}
    report['data'] = {split: {
        'windows': len(rows[split]), 'cases': len(cases[split]),
        'window_labels': dict(Counter(r['gold'][QUESTION]['label'] if r['gold'][QUESTION]['label'] in NULLS else 'tool'
                                      for r in rows[split])),
        'case_decisions': dict(Counter(g['decision'] for g in gold[split].values()))} for split in SPLITS}
    log(f"export ok: {json.dumps({s: [d['windows'], d['cases']] for s, d in report['data'].items()})}; model {config['model_revision']}")

    mx.random.seed(config['seed'])
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(model_dir), local_files_only=True, max_seq_length=config['max_seq_length'],
        use_gradient_checkpointing=config['gradient_checkpointing'], random_state=config['seed'])
    FastDecisionModel.freeze_backbone(model)
    trainable = tree_flatten(model.trainable_parameters())
    if any(name.startswith('encoder.') for name, _ in trainable):
        raise RuntimeError('encoder parameters are trainable in a head-only run')
    encoder_sha = tensors_sha256(model.encoder.parameters())
    head_before = {name: np.array(value) for name, value in trainable}
    report['parameters'] = {'trainable': sum(v.size for _, v in trainable),
                            'total': sum(v.size for _, v in tree_flatten(model.parameters())),
                            'trainable_groups': sorted({n.split('.')[0] for n, _ in trainable}),
                            'dtypes': dict(Counter(f"{n.split('.')[0]}:{v.dtype}" for n, v in tree_flatten(model.parameters()))),
                            'encoder_sha256': encoder_sha, 'decision_config': model.decision_config}
    scorer = Scorer(model, tokenizer, config['eval_batch_size'])

    # Exact-tokenizer budget over every window of every split, before anything trains; any cut stops the run.
    started, budgets, cuts = time.perf_counter(), {}, []
    common = scorer.common
    from unsloth.models._decision_common import _internal, _option_keys
    for split in SPLITS:
        budgets[split] = [sequence_budget(tokenizer, common, row['state'], _internal(row['questions'][QUESTION]),
                                          scorer.max_len, scorer.head_max_len) for row in rows[split]]
        cuts += [{'split': split, 'export_id': row['metadata']['export_id'], **b} for row, b in zip(rows[split], budgets[split])
                 if b['options_cut'] or b['option_text_collisions'] or b['instruction_cut'] or b['state_cut']]
    report['token_budget'] = {split: {
        'sequence_tokens_max': max(b['sequence_tokens'] for b in bs),
        'sequence_tokens_median': statistics.median(b['sequence_tokens'] for b in bs),
        'option_tokens_max': max(b['option_tokens_max'] for b in bs), 'instruction_tokens': bs[0]['instruction_tokens'],
        'state_tokens_max': max(b['state_tokens'] for b in bs), 'state_room_min': min(b['state_room'] for b in bs),
        'rows_cut': sum(c['split'] == split for c in cuts)} for split, bs in budgets.items()}
    report['token_budget']['seconds'] = time.perf_counter() - started
    write_json(out / 'token_budget.json', {'summary': report['token_budget'], 'cuts': cuts})
    if cuts:
        worst = max(cuts, key=lambda c: c['state_cut'])
        raise RuntimeError(f'{len(cuts)} windows would be cut; worst {worst["export_id"]} state needs {worst["state_tokens"]} '
                           f'tokens with room {worst["state_room"]}; see token_budget.json')
    log(f"token budget ok, no cuts in any split: {json.dumps(report['token_budget'])}")

    items = {}
    for split in SPLITS:
        items[split], built = FastDecisionModel.build_dataset(rows[split], tokenizer, model)
        if built['skipped'] or len(items[split]) != len(rows[split]):
            raise RuntimeError(f'{split} dataset skipped rows: {built}')
        for item, row in zip(items[split], rows[split]):
            item['keys'] = _option_keys(_internal(row['questions'][QUESTION]))

    if config.get('train_bucket_ratios'):
        indices, balance = balanced_indices(rows['train'], config['train_bucket_ratios'], config['seed'])
        items['train'] = [items['train'][i] for i in indices]
        report['balance'] = {k: v for k, v in balance.items() if k != 'indices'}
        write_json(out / 'balance.json', balance)
        log(f"balanced training: {report['balance']}")

    if resume:
        for name in ('metrics.jsonl', 'steps.jsonl'):
            repair_json_lines(out / name)
    metrics_log, steps_log = JsonLines(out / 'metrics.jsonl'), JsonLines(out / 'steps.jsonl')
    history = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()] if resume else []
    micro, accumulation = config['micro_batch'], config['grad_accum']
    steps_per_epoch = math.ceil(math.ceil(len(items['train']) / micro) / accumulation)
    last_validation = {}
    callback_config = config.get('validation_callback')
    policy = ValidationPolicy(callback_config['patience'], callback_config['min_delta']) if callback_config else None

    def preserve_best(epoch):
        destination = weights / f'best-epoch-{epoch}'
        save_trainable(model, destination)
        write_json(destination / 'selection.json', {'epoch': epoch, **policy.snapshot()})

    def validate():
        FastDecisionModel.for_inference(model)
        try:
            return evaluate_split(scorer, rows['validation'], items['validation'], cases['validation'], gold['validation'])
        finally:
            FastDecisionModel.for_training(model, use_gradient_checkpointing=config['gradient_checkpointing'])

    def log_epoch(epoch, extra):
        metrics, predictions, windows = validate()
        last_validation.update(epoch=epoch, metrics=metrics, predictions=predictions, windows=windows)
        record = {'epoch': epoch, **extra, 'validation': metrics, 'time': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        if policy:
            if policy.observe(epoch, metrics['case_accuracy']):
                preserve_best(epoch)
            record['callback'] = policy.snapshot()
        metrics_log.write(record)
        history.append(record)
        if policy:
            write_json(out / 'callback.json', policy.snapshot())
        log(f"epoch {epoch}: train_loss {extra.get('train_loss')} val case_accuracy {metrics['case_accuracy']} "
            f"macroF1 {metrics['decision_macro_f1']} coverage {metrics['coverage']} window any-of "
            f"{metrics['window']['any_of_accuracy']:.4f} CE {metrics['window']['soft_ce']:.4f} ({metrics['seconds']['total']:.0f}s)")

    if resume:
        report['before_validation'] = json.loads((out / 'before_validation.json').read_text())
    else:
        before, _, _ = validate()
        report['before_validation'] = before
        write_json(out / 'before_validation.json', before)
        log(f"before training: val case_accuracy {before['case_accuracy']} macroF1 {before['decision_macro_f1']} "
            f"coverage {before['coverage']} ({before['seconds']['total']:.0f}s)")
    if policy:
        policy.observe(0, report['before_validation']['case_accuracy'])
        if not resume:
            preserve_best(0)

    max_steps = config['probe_steps'] if probe else config['epochs'] * steps_per_epoch
    args = MLXTrainingConfig(
        output_dir=str(weights / 'trainer'), seed=config['seed'], per_device_train_batch_size=micro,
        gradient_accumulation_steps=accumulation, max_steps=max_steps, learning_rate=config['encoder_learning_rate'],
        lr_scheduler_type=config['lr_scheduler'], warmup_steps=config['warmup_steps'], weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'], logging_steps=1, gradient_checkpointing=config['gradient_checkpointing'],
        save_steps=0 if probe else steps_per_epoch * config['checkpoint_every_epochs'], save_total_limit=2,
        memory_limit_gb=config['mlx_memory_limit_gb'], wired_limit_gb=0)
    epoch_losses, train_started = [], time.perf_counter()

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs and 'train_runtime' not in logs:
                if not math.isfinite(logs['loss']):
                    raise FloatingPointError('Nonfinite training loss; refusing to save a successful model')
                epoch_losses.append(logs['loss'])
                steps_log.write({'step': state.global_step, 'epoch': state.epoch, 'loss': logs['loss'],
                                 'grad_norm': logs.get('grad_norm'), 'encoder_lr': logs['learning_rate'],
                                 'head_lr': config['head_learning_rate'] * logs['learning_rate'] / config['encoder_learning_rate'],
                                 'peak_memory_gb': logs['peak_memory_gb'], 'seconds': time.perf_counter() - train_started})

        def on_train_begin(self, args, state, control, **kwargs):
            # Resumed at the checkpoint's step: steps and epochs logged after it are replayed, so they are dropped
            # (the replay's lines are the ones kept). A checkpoint is written before its epoch's validation;
            # validate a resumed one if that never happened.
            epoch = state.global_step // steps_per_epoch
            if resume:
                # The native checkpoint reseeds dropout from seed + step; restore the saved key to replay the run exactly.
                key = Path(args.output_dir) / f'checkpoint-{state.global_step}' / RNG_FILE
                if key.is_file():
                    high, low = json.loads(key.read_text())['key']
                    mx.random.seed((high << 32) | low)  # seed(n) sets the key to [n >> 32, n & 0xffffffff]
                log(f"dropout RNG {'restored from ' + key.parent.name if key.is_file() else 'not in checkpoint; native reseed'}")
                kept = [line for line in (out / 'steps.jsonl').read_text().splitlines() if json.loads(line)['step'] <= state.global_step]
                (out / 'steps.jsonl').write_text(''.join(line + '\n' for line in kept))
                history[:] = [h for h in history if h['epoch'] <= epoch]
                (out / 'metrics.jsonl').write_text(''.join(json.dumps(h, ensure_ascii=False) + '\n' for h in history))
                if policy:
                    for record in history:
                        policy.observe(record['epoch'], record['validation']['case_accuracy'])
                log(f'resumed at step {state.global_step} (epoch {epoch}); kept {len(kept)} step and {len(history)} epoch lines')
            if epoch and epoch not in {h['epoch'] for h in history}:
                log_epoch(epoch, {'train_loss': None, 'steps': state.global_step, 'resumed_validation': True})
            if policy:
                control.should_training_stop = policy.stopped
                write_json(out / 'callback.json', policy.snapshot())

        def on_save(self, args, state, control, **kwargs):
            # Called after checkpoint-<step> is in place and before any further random draw.
            key = [int(v) for v in mx.random.state[0].tolist()]
            (Path(args.output_dir) / f'checkpoint-{state.global_step}' / RNG_FILE).write_text(json.dumps({'key': key}))

        def on_epoch_end(self, args, state, control, **kwargs):
            if probe:
                return
            losses = epoch_losses[:]
            epoch_losses.clear()
            log_epoch(round(state.epoch), {'train_loss': statistics.fmean(losses) if losses else None,
                                           'train_loss_last': losses[-1] if losses else None, 'steps': state.global_step,
                                           'train_seconds': time.perf_counter() - train_started})
            if policy and policy.stopped:
                log(f'early stopping: {policy.snapshot()}')
                control.should_training_stop = True

    FastDecisionModel.for_training(model, use_gradient_checkpointing=config['gradient_checkpointing'])
    trainer = DecisionTrainer(model=model, args=args, train_dataset=items['train'],
                              head_learning_rate=config['head_learning_rate'], tokenizer=tokenizer,
                              label_smoothing=config.get('label_smoothing', 0.0), callbacks=[Progress()])
    mx.reset_peak_memory()
    output = trainer.train(resume_from_checkpoint=True if resume else None)
    steps = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
    report['train'] = {**output.metrics, 'wall_seconds': time.perf_counter() - train_started,
                       'mlx_peak_gb': mx.get_peak_memory() / 1e9, 'steps_per_epoch': steps_per_epoch, 'max_steps': max_steps,
                       'train_windows': len(items['train']), 'effective_batch_size': micro * accumulation,
                       'loss_first': steps[0]['loss'], 'loss_last': steps[-1]['loss']}
    if (max(s['step'] for s in steps) < max_steps and not (policy and policy.stopped)) or not all(math.isfinite(s['loss']) for s in steps):
        raise RuntimeError(f'training stopped early or produced a non-finite loss after {len(steps)} steps')
    after = dict(tree_flatten(model.trainable_parameters()))
    deltas = {name: float(np.abs(np.array(after[name]) - value).max()) for name, value in head_before.items()}
    report['updates'] = {'tensors': len(deltas), 'changed': sum(d > 0 for d in deltas.values()),
                         'max_abs_delta': max(deltas.values()),
                         'encoder_unchanged': tensors_sha256(model.encoder.parameters()) == encoder_sha}
    if not report['updates']['encoder_unchanged'] or not report['updates']['changed']:
        raise RuntimeError(f"update check failed {report['updates']}")

    if probe:
        # Speed: optimizer-step intervals after the first (compile) step; evaluation: one full validation pass.
        times = [b['seconds'] - a['seconds'] for a, b in zip(steps, steps[1:])]
        metrics, _, _ = validate()
        per_step, eval_seconds = statistics.median(times), metrics['seconds']['total']
        epoch_seconds = per_step * steps_per_epoch
        report['probe'] = {
            'optimizer_step_seconds': {'median': per_step, 'max': max(times), 'first_step_at': steps[0]['seconds']},
            'validation_eval_seconds': metrics['seconds'], 'steps_per_epoch': steps_per_epoch,
            'epoch_train_seconds': epoch_seconds, 'epoch_with_validation_seconds': epoch_seconds + eval_seconds,
            'epochs': config['epochs'], 'child_hours': config['epochs'] * (epoch_seconds + eval_seconds) / 3600,
            'test_eval_seconds_estimate': eval_seconds * len(rows['test']) / len(rows['validation']),
            'validation_after_probe_steps': metrics}
        log(f"probe: {json.dumps({k: v for k, v in report['probe'].items() if k != 'validation_after_probe_steps'})}")
        return report, {}

    completed_epochs = int(round(trainer.state.epoch))
    selected_epoch = policy.best_epoch if policy else config['epochs']
    if policy:
        load_trainable(model, weights / f'best-epoch-{selected_epoch}')
        report['selection'] = {**policy.snapshot(), 'completed_epochs': completed_epochs,
                               'selected_epoch': selected_epoch, 'max_epochs': config['epochs'],
                               'reason': 'early_stopping' if policy.stopped else 'max_epochs'}
        last_validation.clear()
    if last_validation.get('epoch') != selected_epoch:
        last_validation.update(zip(('metrics', 'predictions', 'windows'), validate()), epoch=selected_epoch)
    save_trainable(model, weights / 'final')
    artifact = weights / 'final' / 'trainable.safetensors'
    write_json(weights / 'final' / 'head.json', {
        'format': 'unsloth-mlx-laya-trainable-v1', 'base_model': str(model_dir), 'base_revision': config['model_revision'],
        'base_model_safetensors_sha256': report['provenance']['model']['model_safetensors_sha256'],
        'trainable_sha256': sha256(artifact), 'epochs': selected_epoch, 'completed_epochs': completed_epochs,
        'decision_config': model.decision_config,
        'loader': 'FastDecisionModel.from_pretrained(base) then unsloth_zoo.mlx.decision.load_trainable(model, dir)'})
    del trainer, model
    restored, tokenizer = FastDecisionModel.from_pretrained(str(model_dir), local_files_only=True,
                                                            max_seq_length=config['max_seq_length'])
    load_trainable(restored, weights / 'final')
    FastDecisionModel.for_inference(restored)
    scorer = Scorer(restored, tokenizer, config['eval_batch_size'])
    metrics, predictions, windows = evaluate_split(scorer, rows['validation'], items['validation'], cases['validation'], gold['validation'])
    diff = max(abs(a['probabilities'][k] - b['probabilities'][k])
               for a, b in zip(last_validation['windows'], windows) for k in a['probabilities'])
    same = [(p['decision'], p['tool']) for p in last_validation['predictions']] == [(p['decision'], p['tool']) for p in predictions]
    report['restore'] = {'artifact': str(artifact.relative_to(ROOT)), 'artifact_bytes': artifact.stat().st_size,
                         'artifact_sha256': sha256(artifact), 'validation_max_abs_probability_diff': diff,
                         'same_validation_case_predictions': same,
                         'encoder_unchanged': tensors_sha256(restored.encoder.parameters()) == encoder_sha}
    if diff > 1e-6 or not same or not report['restore']['encoder_unchanged']:
        raise RuntimeError(f"restore mismatch {report['restore']}")
    report['validation'] = {'epoch': selected_epoch, **metrics}
    log(f"restore ok: {json.dumps(report['restore'])}")

    # Test is scored once per run: persisted at once, and a resumed run reuses it instead of scoring again.
    if (out / 'test.json').is_file():
        saved = json.loads((out / 'test.json').read_text())
        if saved['artifact_sha256'] != report['restore']['artifact_sha256']:
            raise RuntimeError('test.json was scored with a different head; refusing to score test again')
        test, test_predictions, test_windows = saved['metrics'], saved['cases'], saved['windows']
        log('test already scored by this run; reusing test.json')
    else:
        test, test_predictions, test_windows = evaluate_split(scorer, rows['test'], items['test'], cases['test'], gold['test'])
        write_json(out / 'test.json', {'artifact_sha256': report['restore']['artifact_sha256'], 'metrics': test,
                                       'cases': test_predictions, 'windows': test_windows})
    report['test'] = {'epoch': selected_epoch, **test}
    log(f"test (epoch {selected_epoch}, once): case_accuracy {test['case_accuracy']} macroF1 {test['decision_macro_f1']} "
        f"coverage {test['coverage']}")
    report['memory'] = {'mlx_peak_gb': mx.get_peak_memory() / 1e9}
    return report, {'validation_cases': predictions, 'validation_windows': windows,
                    'test_cases': test_predictions, 'test_windows': test_windows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--assets-root', default=str(ROOT), help='checkout holding export_dir and model_dir')
    parser.add_argument('--override', default='{}', help='JSON object merged over the config, kept in the report')
    parser.add_argument('--probe', action='store_true', help='probe_steps optimizer steps and one validation pass; no test')
    parser.add_argument('--resume', action='store_true', help='continue an interrupted run from its newest checkpoint')
    options = parser.parse_args(argv)
    config = {**json.loads(Path(options.config).read_text()), **json.loads(options.override)}
    config['assets_root'] = str(Path(options.assets_root).resolve())
    for key in ('export_dir', 'model_dir'):
        config[key] = str(Path(config['assets_root']) / config[key])
    out = ROOT / config['outputs_dir'] / options.run_id
    if options.resume:
        if not (out / 'metrics.jsonl').is_file():
            parser.error(f'nothing to resume in {out}')
        if (out / 'report.json').is_file():
            parser.error(f'{out} already finished; its report is kept and is not rerun')
        previous = json.loads((out / 'config.json').read_text())
        if {k: v for k, v in previous.items() if k != 'assets_root'} != {k: v for k, v in config.items() if k != 'assets_root'}:
            parser.error('resume config differs from the original run')
    elif out.exists() or (ROOT / config['models_dir'] / options.run_id).exists():
        parser.error(f'run {options.run_id} already exists')
    out.mkdir(parents=True, exist_ok=options.resume)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    log_file = (out / 'run.log').open('a')

    def log(message):
        line = f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} {message}'
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    def status(state, **extra):
        (out / 'status.json').write_text(json.dumps({'run_id': options.run_id, 'status': state, 'pid': os.getpid(),
                                                     'updated': datetime.now(timezone.utc).isoformat(), **extra}, indent=2))

    status('running', command=sys.argv)
    backend = config.get('backend', 'mlx')
    if backend not in ('mlx', 'cuda'):
        parser.error(f'Unknown training backend: {backend}')
    if backend != 'cuda' and 'training_policy_version' in config:
        parser.error('training_policy_version is implemented only for backend=cuda')
    run_backend, guard_class = run, MemoryGuard
    if backend == 'cuda':
        from jery.model.laya_cuda import run as run_backend, CudaMemoryGuard as guard_class
    guard = guard_class(config['memory_guard_gib'], lambda used: (
        log(f'memory guard tripped at {used / 2 ** 30:.2f} GiB'),
        status('failed', error=f'memory guard {used / 2 ** 30:.2f} GiB > {config["memory_guard_gib"]} GiB')))
    guard.start()
    started = time.perf_counter()
    try:
        report, details = run_backend(config, options.run_id, out, log, probe=options.probe, resume=options.resume)
    except Exception as exc:
        log(traceback.format_exc())
        status('failed', error=f'{type(exc).__name__}: {exc}', seconds=time.perf_counter() - started)
        return 1
    report.update(wall_seconds=time.perf_counter() - started, command=sys.argv, guard_peak_gib=guard.peak / 2 ** 30)
    name = 'probe.json' if options.probe else 'report.json'
    for key, data in details.items():
        with (out / f'{key}.jsonl').open('w') as handle:
            handle.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in data)
    write_json(out / name, report)  # The supervisor treats this as the completed output marker.
    status('succeeded', seconds=report['wall_seconds'], report=str((out / name).relative_to(ROOT)))
    log(f"done in {report['wall_seconds']:.1f}s -> {out.relative_to(ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
