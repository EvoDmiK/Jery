"""Dependency-free checks for the optional MLX smoke runner boundary."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('llm_smoke', ROOT / 'scripts/run_llm_smoke.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class LLMSmokeTest(unittest.TestCase):
    def test_selection_keeps_distinct_scenarios_and_candidate_bound(self):
        rows = [dict(decision='select_tools', scenario_id=s, candidate_tools=['a'] * n)
                for s, n in [('large', 13), ('one', 1), ('one', 1), ('two', 2)]]
        self.assertEqual([r['scenario_id'] for r in smoke.select_cases(rows, 2, 12)], ['one', 'two'])

    def test_strict_json_and_candidate_validation(self):
        features = {'candidate_tools': ['a'], 'catalog': {'tools': [
            {'id': 'a', 'input_schema': {'properties': {'action': {'enum': ['read']}}}}]}}
        valid = '{"decision":"select_tools","choices":[{"tool":"a","action":"read"}]}'
        self.assertEqual(smoke.check_output(valid, features)['decision'], 'select_tools')
        for raw in ('```json\n{}\n```', valid.replace('"a"', '"missing"'),
                    valid.replace('"read"', '"write"'), valid.replace('select_tools', 'no_tool')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                smoke.check_output(raw, features)


if __name__ == '__main__':
    unittest.main()
