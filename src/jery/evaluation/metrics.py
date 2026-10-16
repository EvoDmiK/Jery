"""Routing metrics computed from gold rows and predictions in label shape."""
from collections import Counter

from jery.model.features import DECISIONS, server_of


def _ratio(n, d):
    return round(n / d, 4) if d else None


def evaluate(rows, predictions, catalog=None, ks=(1, 3, 5)):
    """Return decision, tool, server, unnecessary-call, candidate-restriction and recall@k metrics."""
    catalog = catalog or {}
    assert len(rows) == len(predictions)
    tp, fp, fn = Counter(), Counter(), Counter()
    joint = tool_ok = server_ok = select_gold = 0
    calls_on_null = null_gold = out_of_candidates = 0
    action_ok = action_total = 0
    hits = Counter()
    for row, pred in zip(rows, predictions):
        gold, got = row['decision'], pred['decision']
        if got not in DECISIONS:
            raise ValueError('unknown decision: %r' % got)
        if gold == got:
            tp[gold] += 1
        else:
            fp[got] += 1
            fn[gold] += 1
        candidates = set(row['candidate_tools'])
        chosen = [c['tool'] for c in pred['acceptable_choices']]
        out_of_candidates += any(t not in candidates for t in chosen)
        good = {c['tool'] for c in row['acceptable_choices']}
        if gold == 'select_tools':
            select_gold += 1
            ok = got == 'select_tools' and bool(chosen) and chosen[0] in good
            tool_ok += ok
            joint += ok
            servers = {server_of(t, catalog.get(t)) for t in good}
            server_ok += got == 'select_tools' and bool(chosen) and server_of(chosen[0], catalog.get(chosen[0])) in servers
            ranked = [r['tool'] for r in pred.get('ranked_tools') or []]
            for k in ks:
                hits[k] += bool(good & set(ranked[:k]))
            if ok:
                gold_actions = {c.get('action') for c in row['acceptable_choices'] if c['tool'] == chosen[0]} - {None}
                if gold_actions:
                    action_total += 1
                    action_ok += pred['acceptable_choices'][0].get('action') in gold_actions
        else:
            null_gold += 1
            joint += gold == got
            calls_on_null += got == 'select_tools'
    per_decision = {}
    for d in DECISIONS:
        p = _ratio(tp[d], tp[d] + fp[d])
        r = _ratio(tp[d], tp[d] + fn[d])
        f1 = round(2 * p * r / (p + r), 4) if p and r else 0.0
        per_decision[d] = {'precision': p, 'recall': r, 'f1': f1, 'support': tp[d] + fn[d]}
    f1s = [v['f1'] for v in per_decision.values() if v['support']]
    return {
        'examples': len(rows),
        'joint_accuracy': _ratio(joint, len(rows)),
        'decision_accuracy': _ratio(sum(tp.values()), len(rows)),
        'decision_macro_f1': round(sum(f1s) / len(f1s), 4) if f1s else None,
        'tool_accuracy': _ratio(tool_ok, select_gold),
        'server_accuracy': _ratio(server_ok, select_gold),
        'action_accuracy_given_tool': _ratio(action_ok, action_total),
        'unnecessary_call_rate': _ratio(calls_on_null, null_gold),
        'out_of_candidate_selections': out_of_candidates,
        'candidate_recall_at_k': {str(k): _ratio(hits[k], select_gold) for k in ks},
        'per_decision': per_decision,
        'gold_tool_classes': len({c['tool'] for r in rows for c in r['acceptable_choices']}),
    }
