import copy
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('laya_export', ROOT / 'scripts/build_laya_pilot.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class LayaExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'datasets/mcp-routing-v2'
        cls.catalog = json.loads((cls.directory / 'tools.json').read_text())
        cls.rows = [json.loads(x) for x in (cls.directory / 'train.jsonl').read_text().splitlines()]

    def test_negative_window_preserves_global_gold_and_source(self):
        row = next(r for r in self.rows if r['decision'] == 'select_tools' and len(r['candidate_tools']) == 8)
        original = copy.deepcopy(row)
        result = export.export_case(row, self.catalog, random.Random(1), negative=True)
        self.assertEqual(result['metadata']['original_decision'], 'select_tools')
        self.assertEqual(result['gold']['route_in_window']['label'], 'window_none')
        self.assertTrue(set(result['metadata']['window_tools']) < set(row['candidate_tools']))
        self.assertEqual(row, original)
        self.assertEqual(set(result['state']), {'request', 'context', 'candidates'})
        self.assertNotIn('mcp__', json.dumps(result['questions']))

    def test_tool_alternatives_do_not_double_weight_action_variants(self):
        row = copy.deepcopy(next(r for r in self.rows if r['decision'] == 'select_tools' and len(r['candidate_tools']) == 8))
        a, b = row['candidate_tools'][:2]
        row['acceptable_choices'] = [{'tool': a, 'action': 'a'}, {'tool': a, 'action': 'b'}, {'tool': b}]
        result = export.export_case(row, self.catalog, random.Random(1))
        self.assertEqual(sorted(result['gold']['route_in_window']['probabilities'].values()), [0.5, 0.5])

    def test_export_split_isolation_reproducibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            first = export.build(self.directory, a)
            self.assertEqual(first, export.build(self.directory, b))
            self.assertEqual(first['splits']['train']['rows'], 32)
            self.assertEqual(first['splits']['validation']['rows'], 12)
            with self.assertRaises(ValueError):
                export.build(self.directory, a)
            train = [json.loads(x) for x in (a / 'train.jsonl').read_text().splitlines()]
            val = [json.loads(x) for x in (a / 'validation.jsonl').read_text().splitlines()]
            self.assertFalse({r['metadata']['scenario_id'] for r in train} &
                             {r['metadata']['scenario_id'] for r in val})


if __name__ == '__main__':
    unittest.main()
