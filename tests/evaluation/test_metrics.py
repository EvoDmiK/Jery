"""Hand-checked routing metrics."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from jery.evaluation.metrics import evaluate

A, B, C = 'mcp__s1__a', 'mcp__s1__b', 'mcp__s2__c'


def row(decision, choices=(), cands=(A, B, C)):
    return {'decision': decision, 'candidate_tools': list(cands), 'acceptable_choices': [{'tool': t} for t in choices]}


def pred(decision, ranked=(), pick=None):
    choices = [{'tool': pick or ranked[0]}] if decision == 'select_tools' else []
    return {'decision': decision, 'acceptable_choices': choices, 'ranked_tools': [{'tool': t} for t in ranked]}


class MetricsTest(unittest.TestCase):
    def test_counts(self):
        rows = [row('select_tools', [A]), row('select_tools', [C, B]), row('no_tool'), row('clarify'), row('unsupported')]
        preds = [pred('select_tools', [B, A, C]),       # wrong tool, right server, hit@2
                 pred('select_tools', [B, C, A]),       # alternative accepted
                 pred('select_tools', [A, B, C]),       # unnecessary call
                 pred('clarify'),
                 pred('no_tool')]
        m = evaluate(rows, preds, ks=(1, 2))
        self.assertEqual(m['tool_accuracy'], 0.5)
        self.assertEqual(m['server_accuracy'], 1.0)
        self.assertEqual(m['joint_accuracy'], 0.4)
        self.assertEqual(m['unnecessary_call_rate'], round(1 / 3, 4))
        self.assertEqual(m['candidate_recall_at_k'], {'1': 0.5, '2': 1.0})
        self.assertEqual(m['per_decision']['select_tools']['precision'], round(2 / 3, 4))
        self.assertEqual(m['per_decision']['unsupported']['recall'], 0.0)
        self.assertIsNone(m['per_decision']['unsupported']['precision'])
        self.assertEqual(m['out_of_candidate_selections'], 0)

    def test_out_of_candidate_counted(self):
        m = evaluate([row('select_tools', [A], cands=(A,))], [pred('select_tools', [A], pick=C)])
        self.assertEqual(m['out_of_candidate_selections'], 1)
        self.assertEqual(m['tool_accuracy'], 0.0)

    def test_rejects_unknown_decision(self):
        with self.assertRaises(ValueError):
            evaluate([row('no_tool')], [pred('maybe')])


if __name__ == '__main__':
    unittest.main()
