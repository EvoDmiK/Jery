import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from jery.runtime.prediction import model_input, validate_prediction


class PredictionContractTest(unittest.TestCase):
    def setUp(self):
        self.catalog = {'catalog_version': 'metadata-not-a-feature', 'source': 'private provenance', 'tools': [
            {'id': 'mcp__test__read', 'server': 'test', 'name': 'read', 'description': 'Read a document',
             'input_schema': {'properties': {'action': {'enum': ['read']}}}, 'review_status': 'hidden'},
            {'id': 'mcp__other__tool', 'server': 'other', 'name': 'tool', 'description': 'Not a candidate', 'input_schema': {}},
        ]}
        self.row = {'id': 'answer_hint', 'scenario_id': 'read', 'split': 'train', 'request': '문서를 읽어줘',
                    'context': [{'role': 'user', 'content': '문서는 test.md야', 'label': 'forbidden'}],
                    'candidate_tools': ['mcp__test__read'], 'decision': 'select_tools',
                    'acceptable_choices': [{'tool': 'mcp__test__read'}], 'review_status': 'needs_human_review'}

    def test_only_whitelisted_inputs_reach_model(self):
        features = model_input(self.row, self.catalog)
        self.assertEqual(set(features), {'request', 'context', 'candidate_tools', 'catalog'})
        self.assertEqual(features['context'], [{'role': 'user', 'content': '문서는 test.md야'}])
        self.assertEqual(len(features['catalog']['tools']), 1)
        self.assertNotIn('review_status', features['catalog']['tools'][0])
        features['catalog']['tools'][0]['input_schema'].clear()
        self.assertTrue(self.catalog['tools'][0]['input_schema'])

    def test_labels_cannot_affect_input(self):
        changed = dict(self.row, id='other', decision='unsupported', acceptable_choices=[], split='test')
        self.assertEqual(model_input(self.row, self.catalog), model_input(changed, self.catalog))

    def test_decisions_and_empty_candidates(self):
        for decision in ('no_tool', 'clarify', 'unsupported'):
            self.assertEqual(validate_prediction({'decision': decision, 'choices': []}, [], self.catalog)['decision'], decision)
        with self.assertRaises(ValueError):
            validate_prediction({'decision': 'select_tools', 'choices': []}, [], self.catalog)

    def test_candidate_action_and_confidence_validation(self):
        good = {'decision': 'select_tools', 'choices': [{'tool': 'mcp__test__read', 'action': 'read'}],
                'ranked_tools': ['mcp__test__read'], 'confidence': .5}
        self.assertEqual(validate_prediction(good, self.row['candidate_tools'], self.catalog), good)
        for mutate in (
            lambda p: p['choices'][0].update(tool='mcp__other__tool'),
            lambda p: p['choices'][0].update(action='delete'),
            lambda p: p.update(ranked_tools=['mcp__other__tool']),
            lambda p: p.update(confidence=float('nan')),
            lambda p: p.update(confidence=True),
        ):
            bad = copy.deepcopy(good)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_prediction(bad, self.row['candidate_tools'], self.catalog)

    def test_invalid_or_duplicate_candidate_rejected(self):
        for candidates in (['not-in-catalog'], ['mcp__test__read'] * 2):
            with self.assertRaises(ValueError):
                model_input(dict(self.row, candidate_tools=candidates), self.catalog)


if __name__ == '__main__':
    unittest.main()
