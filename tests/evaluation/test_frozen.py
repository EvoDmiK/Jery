"""Refusals and metadata-blind scoring of the frozen ten-tool evaluation gate (no model needed)."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from jery.evaluation.frozen import GateError, evaluate, load_frozen
from jery.model.laya_pilot import sha256

CATALOG = ROOT / 'datasets/hermes-ten-tools-v2/tools.json'
TOOLS = [t['id'] for t in json.loads(CATALOG.read_text())['tools']]


def case(i):
    return {'id': f'real-{i}', 'scenario_id': f's{i}', 'request': f'요청 {i}번 처리', 'context': [],
            'candidate_tools': TOOLS[:], 'tool_catalog_version': 'hermes-ten-tools-2289142ab57a',
            'decision': 'select_tools', 'acceptable_choices': [{'tool': TOOLS[i]}],
            'source': {'kind': 'real', 'ref': f'p{i}'}, 'review_status': 'human_reviewed', 'execution_verified': False}


class FakeScorer:
    """Records exactly what a native scorer would tokenize (state and question) and always picks T0."""

    def __init__(self):
        self.seen = []

    def encode(self, window):
        self.seen.append({'state': window['state'], 'questions': window['questions']})
        return {'keys': list(window['questions']['route_in_window']['criteria'])}, None

    def top(self, items):
        return ['T0'] * len(items), [{k: 1 / len(i['keys']) for k in i['keys']} for i in items]


class FrozenGateTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        shutil.copy(CATALOG, self.dir / 'tools.json')
        self.write([case(i) for i in range(10)])

    def write(self, cases, proposals=None, leak='passed', **manifest):
        (self.dir / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        (self.dir / 'leak-audit.json').write_text(json.dumps({'status': leak}))
        names = ['cases.jsonl', 'tools.json', 'leak-audit.json']
        if proposals is not None:
            (self.dir / 'ai-proposals.json').write_text(json.dumps(proposals))
            names.append('ai-proposals.json')
        body = {'schema_version': 'frozen-eval-v1', 'status': 'frozen', 'catalog_version': 'hermes-ten-tools-2289142ab57a',
                'candidate_order_seed': 3407, 'leak_audit': {'status': 'passed', 'file': 'leak-audit.json'},
                'sha256': {n: sha256(self.dir / n) for n in names}, **manifest}
        (self.dir / 'manifest.json').write_text(json.dumps(body))
        return sha256(self.dir / 'manifest.json')

    def load(self, labels='human', **kwargs):
        return load_frozen(self.dir, labels, gate_only=True, **kwargs)

    def refuse(self, pattern, labels='human', **kwargs):
        with self.assertRaisesRegex(GateError, pattern):
            self.load(labels, **kwargs)

    def test_scores_per_case_and_aggregate(self):
        sha = self.write([case(i) for i in range(10)])
        rows, gate = load_frozen(self.dir, 'human', authorized_sha=sha)
        per_case, metrics = evaluate(rows, gate, FakeScorer())
        self.assertEqual(len(per_case), 10)
        self.assertEqual(sum(r['correct'] for r in per_case), round(metrics['case_accuracy'] * 10))
        self.assertTrue(all(r['predicted_tool'] in TOOLS and abs(sum(r['probabilities'].values()) - 1) < 1e-9
                            for r in per_case))
        self.assertEqual(metrics['macro_f1_scope'], 'tools')
        self.assertEqual(metrics['label_basis'], 'human_reviewed')
        self.assertFalse(metrics['meets_target'])  # 10 cases, target 10 per tool: never reported as the full set.
        self.assertEqual(set(metrics['missing_by_tool'].values()), {9})
        self.assertIn('partial subset of 10 cases', metrics['claim'])

    def test_refuses_changed_gold_input_and_catalog_after_hashing(self):
        for field, value in (('acceptable_choices', [{'tool': TOOLS[1]}]), ('request', '바뀐 요청')):
            self.write([case(i) for i in range(10)])
            changed = [case(i) for i in range(10)]
            changed[0][field] = value
            (self.dir / 'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in changed))
            self.refuse('cases.jsonl: sha256 differs')
        catalog = json.loads(CATALOG.read_text())
        catalog['tools'][0]['description'] += ' 변경'
        (self.dir / 'tools.json').write_text(json.dumps(catalog, ensure_ascii=False))
        self.write([case(i) for i in range(10)])  # Even a re-hashed manifest cannot change the pinned catalog.
        self.refuse('pinned catalog')
        shutil.copy(CATALOG, self.dir / 'tools.json')
        self.write([case(i) for i in range(10)], catalog_version='hermes-ten-tools-b4648cc19a90')
        self.refuse('catalog_version')

    def test_refuses_unscored_states(self):
        sha = self.write([case(i) for i in range(10)], status='candidate')
        with self.assertRaisesRegex(GateError, 'status=frozen'):
            load_frozen(self.dir, 'human', authorized_sha=sha)
        sha = self.write([case(i) for i in range(10)])
        with self.assertRaisesRegex(GateError, 'status=frozen'):
            load_frozen(self.dir, 'human', authorized_sha='0' * 64)
        for leak, manifest in (('failed', {}), ('passed', {'leak_audit': {'status': 'passed'}}),
                               ('passed', {'leak_audit': {'status': 'failed', 'file': 'leak-audit.json'}})):
            sha = self.write([case(i) for i in range(10)], leak=leak, **manifest)
            self.assertFalse(self.load()[1]['leak_audit_ready'])  # Gate-only still reports readiness.
            with self.assertRaisesRegex(GateError, 'leak_audit'):
                load_frozen(self.dir, 'human', authorized_sha=sha)
        self.assertEqual(self.load(labels='human', exclude_exports=[])[1]['cases'], 10)
        self.write([])
        self.refuse('nonempty')

    def test_refuses_unresolved_labels(self):
        cases = [case(i) for i in range(10)]
        cases[3].update(decision=None, acceptable_choices=[], review_status='needs_human_review')
        cases[4]['acceptable_choices'] = [{'tool': TOOLS[4]}, {'tool': TOOLS[5]}]  # Ambiguous, never forced.
        self.write(cases)
        self.refuse(r"2 unresolved labels \(human\): \['real-3', 'real-4'\]")
        unlabeled = [dict(c, decision=None, acceptable_choices=[], review_status='needs_human_review') for c in cases]
        vote = lambda who, i: {'proposer': who, 'disposition': 'label', 'decision': 'select_tools',
                               'acceptable_choices': [{'tool': TOOLS[i]}], 'confidence': 0.9, 'rationale': '-'}
        agree = {f'real-{i}': [vote('claude', i), vote('antigravity', i)] for i in range(10)}
        self.write(unlabeled, agree)
        self.refuse('10 unresolved labels', 'human')  # AI agreement is never human review.
        rows, gate = self.load('provisional-ai')
        self.assertEqual(len(rows), 10)
        self.assertIn('not human reviewed', gate['label_basis'])
        per_case, metrics = evaluate(rows, gate, FakeScorer())
        self.assertEqual({r['review_status'] for r in per_case}, {'needs_human_review'})
        self.assertEqual(metrics['review_status_counts'], {'needs_human_review': 10})
        self.assertIn('not reviewed real accuracy', metrics['claim'])
        for broken in ([vote('claude', 1)], [vote('claude', 1), vote('claude', 1)], [vote('claude', 1), vote('antigravity', 2)],
                       [vote('claude', 1), dict(vote('antigravity', 1), disposition='needs_review')]):
            self.write(unlabeled, {**agree, 'real-1': broken})
            self.refuse(r"1 unresolved labels \(provisional-ai\): \['real-1'\]", 'provisional-ai')

    def test_metadata_never_reaches_model_input(self):
        sha = self.write([case(i) for i in range(10)])
        first = FakeScorer()
        evaluate(*load_frozen(self.dir, 'human', authorized_sha=sha), first)
        changed = [dict(case(i), acceptable_choices=[{'tool': TOOLS[9 - i]}], scenario_id='x', source={'kind': 'real', 'ref': 'y'})
                   for i in range(10)]
        sha = self.write(changed)
        second = FakeScorer()
        evaluate(*load_frozen(self.dir, 'human', authorized_sha=sha), second)
        self.assertEqual(first.seen, second.seen)
        text = json.dumps(first.seen, ensure_ascii=False)
        self.assertFalse(any(s in text for s in ['real-0', 'mcp__', 'human_reviewed', 'p0', 's0']))
        leaked = [case(i) for i in range(10)]
        leaked[2]['context'] = [{'role': 'assistant', 'content': 'mcp__obsidian_vault__vault 호출 예정'}]
        self.write(leaked)
        self.refuse('real-2: tool id leaked')

    def test_candidate_and_source_validation(self):
        for change, pattern in ((lambda c: c['candidate_tools'].pop(), 'exactly the pinned ten'),
                                (lambda c: c['candidate_tools'].__setitem__(0, 'mcp__x__y'), 'real-0: Candidate missing from catalog'),
                                (lambda c: c.update(source={'kind': 'synthetic'}), 'source.kind must be real'),
                                (lambda c: c.update(acceptable_choices=[{'tool': TOOLS[0], 'action': 'nope'}]), 'unresolved'),
                                (lambda c: c.update(request='요청 1번 처리'), 'normalized input also in')):
            cases = [case(i) for i in range(10)]
            change(cases[0])
            self.write(cases)
            self.refuse(pattern)
        export = self.dir / 'train-export'
        export.mkdir()
        (export / 'train.jsonl').write_text(json.dumps({'state': {'request': '요청 5번 처리!', 'context': []},
                                                        'metadata': {'source_id': 'old'}}, ensure_ascii=False) + '\n')
        self.write([case(i) for i in range(10)])
        self.refuse('real-5: normalized input also in train.jsonl:old', exclude_exports=[export])


if __name__ == '__main__':
    unittest.main()
