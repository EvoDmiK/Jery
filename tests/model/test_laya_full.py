import json
from pathlib import Path
import tempfile
import unittest

from jery.model.laya_full import (ValidationPolicy, aggregate, balanced_indices, case_gold, case_metrics,
                                  group_cases, repair_json_lines, window_metrics, write_json)

SENTINELS = {'no_tool': 'n', 'clarify': 'c', 'window_none': 'w'}


def case(source_id, windows, decision='select_tools', acceptable=()):
    """Export-shaped windows; `windows` is a list of tool lists in window order."""
    original = [t for tools in windows for t in tools]
    rows = []
    for index, tools in enumerate(windows):
        aliases = {f'T{i}': t for i, t in enumerate(tools)}
        rows.append({
            'state': {'request': 'r', 'context': [], 'candidates': [{'alias': a, 'server': 's', 'name': t, 'summary': t}
                                                                     for a, t in aliases.items()]},
            'questions': {'route_in_window': {'type': 'choice', 'instructions': 'i',
                                              'criteria': {**{a: a for a in aliases}, **SENTINELS}}},
            'gold': {'route_in_window': {'label': 'window_none', 'probabilities': {'window_none': 1.0}}},
            'metadata': {'source_id': source_id, 'window_index': index, 'window_count': len(windows),
                         'export_id': f'{source_id}:{index}', 'window_tools': tools, 'alias_to_tool': aliases,
                         'original_candidate_tools': original, 'original_decision': decision,
                         'acceptable_choices': [{'tool': t} for t in acceptable]}})
    return rows


def tools(prefix, n):
    return [f'mcp__{prefix}__t{i}' for i in range(n)]


class AggregateTest(unittest.TestCase):
    def test_train_balance_preserves_every_window_and_source(self):
        rows = [{'gold': {'route_in_window': {'label': label}}}
                for label, count in [('window_none', 18), ('T0', 2), ('no_tool', 1), ('clarify', 1)]
                for _ in range(count)]
        original = json.dumps(rows)
        indices, summary = balanced_indices(rows, {'tool': 4, 'window_none': 4, 'no_tool': 1, 'clarify': 1}, 42)
        self.assertEqual(set(indices), set(range(22)))
        self.assertEqual(summary['sampled_counts'], {'tool': 20, 'window_none': 20, 'no_tool': 5, 'clarify': 5})
        self.assertEqual(indices, balanced_indices(rows, summary['ratios'], 42)[0])
        self.assertEqual(json.dumps(rows), original)
        with self.assertRaises(ValueError):
            balanced_indices(rows[:18], summary['ratios'], 42)

    def test_validation_policy_restores_best_and_ignores_plateaus(self):
        policy = ValidationPolicy(patience=2, min_delta=.004)
        self.assertTrue(policy.observe(0, .16))
        self.assertFalse(policy.observe(1, .162))
        self.assertTrue(policy.observe(2, .18))
        self.assertFalse(policy.observe(3, .179))
        self.assertFalse(policy.observe(4, .18))
        self.assertTrue(policy.stopped)
        self.assertEqual((policy.best_epoch, policy.best_score), (2, .18))
        replay = ValidationPolicy(patience=2, min_delta=.004)
        for epoch, score in enumerate([.16,.162,.18,.179,.18]): replay.observe(epoch, score)
        self.assertEqual(replay.snapshot(), policy.snapshot())
        with self.assertRaises(ValueError): policy.observe(5, float('nan'))

    def test_interrupted_json_write_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            path.write_bytes(b'{"epoch": 1}\n{"epoch": "\xe3')
            repair_json_lines(path)
            self.assertEqual(path.read_text(), '{"epoch": 1}\n')
            path.write_text('invalid\n')
            with self.assertRaises(json.JSONDecodeError):
                repair_json_lines(path)
            self.assertEqual(path.read_text(), 'invalid\n')
            target = Path(folder) / 'report.json'
            write_json(target, {'complete': True})
            self.assertEqual(json.loads(target.read_text()), {'complete': True})

    def setUp(self):
        many = tools('a', 20)
        rows = (case('single', [tools('b', 3)], 'select_tools', ['mcp__b__t1'])
                + case('final', [many[:8], many[8:16], many[16:]], 'select_tools', [many[9]])
                + case('none', [many[:8], many[8:16], many[16:]], 'unsupported')
                + case('tie', [many[:8], many[8:16]], 'no_tool')
                + case('single_none', [tools('c', 2)], 'unsupported'))
        self.rows = rows
        self.cases = group_cases(rows)
        self.gold = case_gold(rows)
        self.asked = []

    def score(self, answer):
        def score(windows):
            self.assertFalse(any('gold' in w or 'original_decision' in w.get('metadata', {}) for w in windows))
            self.asked.extend(windows)
            return [answer(w) for w in windows]
        return score

    def test_rules(self):
        first = {'single': ['T1'], 'final': ['T2', 'T1', 'window_none'], 'none': ['window_none', 'no_tool', 'window_none'],
                 'tie': ['no_tool', 'clarify'], 'single_none': ['window_none']}
        out = {p['source_id']: p for p in aggregate(self.cases, first, self.score(lambda w: 'T1'))}
        self.assertEqual((out['single']['decision'], out['single']['tool']), ('select_tools', 'mcp__b__t1'))
        self.assertEqual(out['single_none']['decision'], 'unsupported')
        self.assertEqual(out['none']['decision'], 'unsupported')
        self.assertEqual(out['tie']['decision'], 'delegate')
        # Final round: winners a/t2 (window 0) and a/t9 (window 1) in C order; T1 of the final window is a/t9.
        self.assertEqual(len(self.asked), 1)
        final = self.asked[0]
        self.assertEqual(final['alias_to_tool'], {'T0': 'mcp__a__t2', 'T1': 'mcp__a__t9'})
        self.assertEqual([c['name'] for c in final['state']['candidates']], ['mcp__a__t2', 'mcp__a__t9'])
        self.assertEqual(set(final['questions']['route_in_window']['criteria']), {'T0', 'T1', *SENTINELS})
        self.assertEqual((out['final']['decision'], out['final']['tool'], out['final']['forwards']), ('select_tools', 'mcp__a__t9', 4))
        for windows in self.cases.values():  # aggregation sees no labels
            self.assertFalse(any('gold' in w or 'acceptable_choices' in w['metadata'] for w in windows))
        metrics = case_metrics(self.gold, [out[k] for k in self.cases])
        self.assertEqual(metrics['out_of_candidate_selections'], 0)
        self.assertEqual((metrics['delegated'], metrics['coverage']), (1, 0.8))
        self.assertEqual(metrics['tool_accuracy'], 1.0)
        # tie (gold no_tool) delegated: wrong, not a call, not a false positive of any decision.
        self.assertEqual(metrics['case_accuracy'], 0.8)
        self.assertEqual(metrics['unnecessary_call_rate'], 0.0)
        self.assertEqual(metrics['per_decision']['no_tool']['recall'], 0.0)
        self.assertEqual(metrics['per_decision']['unsupported']['precision'], 1.0)

    def test_case_metrics_rejects_unknown_and_counts_outside(self):
        gold = {'x': {'decision': 'select_tools', 'candidate_tools': ['a'], 'acceptable_choices': [{'tool': 'a'}]}}
        record = {'source_id': 'x', 'path': 'single', 'rounds': 0, 'forwards': 1}
        self.assertEqual(case_metrics(gold, [{**record, 'decision': 'select_tools', 'tool': 'z'}])['out_of_candidate_selections'], 1)
        with self.assertRaises(ValueError):
            case_metrics(gold, [{**record, 'decision': 'maybe', 'tool': None}])

    def test_final_window_none_delegates_and_never_unsupported(self):
        first = {k: ['window_none'] * len(v) for k, v in self.cases.items()}
        first['final'] = ['T0', 'window_none', 'window_none']
        out = {p['source_id']: p for p in aggregate(self.cases, first, self.score(lambda w: 'window_none'))}
        self.assertEqual(out['final']['decision'], 'delegate')

    def test_more_than_eight_winners_play_rounds(self):
        many = tools('a', 96)
        cases = group_cases(case('big', [many[i:i + 8] for i in range(0, 96, 8)]))
        out = aggregate(cases, {'big': ['T0'] * 12}, self.score(lambda w: 'T0'))[0]
        self.assertEqual((out['rounds'], out['forwards'], out['tool']), (2, 12 + 2 + 1, 'mcp__a__t0'))

    def test_coverage_is_checked(self):
        rows = case('x', [tools('a', 3), tools('a', 6)[3:]])
        rows[1]['metadata']['original_candidate_tools'] = tools('a', 7)
        rows[0]['metadata']['original_candidate_tools'] = tools('a', 7)
        with self.assertRaises(ValueError):
            group_cases(rows)
        with self.assertRaises(ValueError):
            group_cases(case('y', [tools('a', 2), tools('b', 2)])[:1] + case('y', [tools('a', 2), tools('b', 2)])[:1])

    def test_window_metrics(self):
        rows = [r for r in self.rows if r['metadata']['source_id'] == 'none']
        metrics = window_metrics(rows, ['window_none', 'T0', 'window_none'],
                                 [{'window_none': 1.0}, {'window_none': 0.5}, {'window_none': 0.25}])
        self.assertAlmostEqual(metrics['any_of_accuracy'], 2 / 3)
        self.assertEqual(metrics['by_gold_kind']['window_none']['windows'], 3)


if __name__ == '__main__':
    unittest.main()
