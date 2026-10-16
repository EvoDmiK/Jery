"""Leakage, candidate restriction, empty candidates and all-decision support for the baseline routers."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from jery.model import predict
from jery.model.cli import load_data
from jery.model.features import DECISIONS, MODEL_INPUT_KEYS, model_input
from jery.model.predict import load_model, save_model
from jery.model.routers import LinearRouter, RetrievalRouter

DATA = ROOT / 'datasets/mcp-routing-v2'
CATALOG = json.loads((DATA / 'tools.json').read_text())
TRAIN = [json.loads(line) for line in (DATA / 'train.jsonl').read_text().splitlines()]
ALL_TOOLS = [t['id'] for t in CATALOG['tools']]


def sample():
    """Small train slice that still contains every decision type."""
    rows = [r for r in TRAIN if r['decision'] != 'select_tools']
    return rows + [r for r in TRAIN if r['decision'] == 'select_tools'][:150]


class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = sample()
        cls.linear = LinearRouter(CATALOG, epochs=8, lr=0.3, seed=1).fit(cls.rows)
        cls.retrieval = RetrievalRouter(CATALOG, threshold=0.05)

    def models(self):
        return (self.linear, self.retrieval)

    def test_model_input_drops_labels_ids_and_provenance(self):
        inp = model_input(TRAIN[0])
        self.assertEqual(set(inp), set(MODEL_INPUT_KEYS))
        blob = json.dumps(inp, ensure_ascii=False)
        for value in (TRAIN[0]['id'], TRAIN[0]['scenario_id'], 'needs_human_review', 'synthetic', 'train'):
            self.assertNotIn(value, blob)

    def test_prediction_ignores_leaky_fields(self):
        row = dict(TRAIN[0])
        poisoned = dict(row, id='mcp__grafana__get_doc', scenario_id='mcp__grafana__get_doc', split='test',
                        source='hermes', review_status='reviewed', decision='no_tool',
                        acceptable_choices=[{'tool': row['candidate_tools'][-1]}])
        for model in self.models():
            self.assertEqual(model.predict(model_input(row)), model.predict(model_input(poisoned)))

    def test_training_ignores_leaky_fields(self):
        rows = [dict(r, id='x%d' % i, scenario_id='s', split='z', source='q', review_status='r')
                for i, r in enumerate(self.rows[:60])]
        a = LinearRouter(CATALOG, epochs=2, seed=3).fit(self.rows[:60])
        b = LinearRouter(CATALOG, epochs=2, seed=3).fit(rows)
        self.assertEqual(a.w, b.w)

    def test_selection_stays_inside_candidates(self):
        for row in TRAIN[:200:7]:
            for model in self.models():
                # Shrink to tools that are not the gold answer: the model must not reach outside.
                gold = {c['tool'] for c in row['acceptable_choices']}
                cands = [t for t in row['candidate_tools'] if t not in gold][:3]
                out = predict(row['request'], row['context'], cands, CATALOG, model=model)
                self.assertTrue(all(c['tool'] in cands for c in out['acceptable_choices']))
                self.assertTrue({r['tool'] for r in out['ranked_tools']} <= set(cands))
                self.assertIn(out['decision'], DECISIONS)

    def test_empty_candidates_never_select(self):
        for model in self.models():
            out = predict('Grafana 대시보드 목록 보여줘', [], [], CATALOG, model=model)
            self.assertNotEqual(out['decision'], 'select_tools')
            self.assertEqual(out['acceptable_choices'], [])
            self.assertEqual(out['ranked_tools'], [])

    def test_unknown_and_duplicate_candidates(self):
        out = predict('무언가 해줘', [], ['mcp__nope__x', 'mcp__nope__x'], CATALOG, model=self.linear)
        self.assertLessEqual(len(out['ranked_tools']), 1)

    def test_linear_emits_all_four_decisions_on_train(self):
        seen = {self.linear.predict(model_input(r))['decision'] for r in self.rows}
        self.assertEqual(seen, set(DECISIONS))

    def test_unsupported_when_needed_tool_removed(self):
        row = next(r for r in self.rows if r['decision'] == 'unsupported')
        out = self.linear.predict(model_input(row))
        self.assertEqual(out['decision'], 'unsupported')

    def test_save_load_roundtrip_and_default_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model in self.models():
                path = Path(tmp) / (model.kind + '.json.gz')
                save_model(model, path)
                loaded = load_model(path, CATALOG)
                for row in self.rows[:20]:
                    self.assertEqual(loaded.predict(model_input(row)), model.predict(model_input(row)))
                out = predict(self.rows[0]['request'], [], ALL_TOOLS[:5], CATALOG['tools'], model=str(path))
                self.assertIn(out['decision'], DECISIONS)


class DataGuardTest(unittest.TestCase):
    def test_modified_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'd'
            shutil.copytree(DATA, copy)
            load_data(copy)
            with (copy / 'test.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaises(SystemExit):
                load_data(copy)


if __name__ == '__main__':
    unittest.main()
