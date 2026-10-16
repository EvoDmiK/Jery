"""Contract, leakage, candidate restriction, saved-state restoration and all-decision support."""
import copy
import gzip
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from jery.model import predict
from jery.model.cli import load_data
from jery.model.features import DECISIONS, MODEL_INPUT_KEYS, TfidfIndex, model_input
from jery.model.predict import load_model, save_model
from jery.model.routers import LinearRouter, RetrievalRouter
from jery.runtime.prediction import model_input as contract_input, validate_prediction

predict_module = importlib.import_module('jery.model.predict')

DATA = ROOT / 'datasets/mcp-routing-v2'
CATALOG = json.loads((DATA / 'tools.json').read_text())
TRAIN = [json.loads(line) for line in (DATA / 'train.jsonl').read_text().splitlines()]
VALIDATION = [json.loads(line) for line in (DATA / 'validation.jsonl').read_text().splitlines()]
SELECTED = ROOT / 'models/mcp-routing-v2/latest/model.json.gz'
BY_ID = {t['id']: t for t in CATALOG['tools']}


def sample():
    """Small train slice that still contains every decision type."""
    rows = [r for r in TRAIN if r['decision'] != 'select_tools']
    return rows + [r for r in TRAIN if r['decision'] == 'select_tools'][:150]


def subset_catalog(candidates):
    return {'tools': [copy.deepcopy(BY_ID[t]) for t in candidates]}


def public(row, model, catalog=CATALOG, candidates=None):
    cands = row['candidate_tools'] if candidates is None else candidates
    return predict(row['request'], row['context'], cands, catalog, model=model)


class RouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = sample()
        cls.linear = LinearRouter(CATALOG, epochs=8, lr=0.3, seed=1).fit(cls.rows)
        cls.retrieval = RetrievalRouter(CATALOG, threshold=0.05)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = {}
        for model in (cls.linear, cls.retrieval):
            cls.paths[model.kind] = Path(cls.tmp.name) / (model.kind + '.json.gz')
            save_model(model, cls.paths[model.kind])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def models(self):
        return (self.linear, self.retrieval)

    # ----- contract -----
    def test_public_output_matches_prediction_v1(self):
        for model in self.models():
            for row in VALIDATION[:40]:
                out = public(row, model)
                self.assertLessEqual(set(out), {'decision', 'choices', 'ranked_tools', 'confidence'})
                self.assertTrue(all(isinstance(t, str) for t in out['ranked_tools']))
                features = contract_input(row, CATALOG)
                self.assertEqual(validate_prediction(out, features['candidate_tools'], features['catalog']), out)

    def test_contract_helper_features_can_be_splatted(self):
        row = VALIDATION[0]
        features = contract_input(row, CATALOG)
        for model in self.models():
            self.assertEqual(predict(model=model, **features), public(row, model))

    def test_label_and_catalog_metadata_are_ignored(self):
        noisy = copy.deepcopy(CATALOG)
        noisy['catalog_version'] = 'leak'
        for tool in noisy['tools']:
            tool['review_status'] = 'reviewed'
            tool['label'] = 'select_tools'
        for row in VALIDATION[:30]:
            poisoned = dict(row, id='mcp__grafana__get_doc', scenario_id='x', split='train', source='hermes',
                            review_status='reviewed', decision='no_tool', acceptable_choices=[])
            for model in self.models():
                self.assertEqual(public(row, model), public(poisoned, model, noisy))
                self.assertEqual(model.predict(model_input(row), CATALOG), model.predict(model_input(poisoned), CATALOG))

    def test_model_input_drops_labels_ids_and_provenance(self):
        self.assertEqual(set(model_input(TRAIN[0])), set(MODEL_INPUT_KEYS))

    def test_training_ignores_leaky_fields(self):
        rows = [dict(r, id='x%d' % i, scenario_id='s', split='z', source='q', review_status='r')
                for i, r in enumerate(self.rows[:60])]
        a = LinearRouter(CATALOG, epochs=2, seed=3).fit(self.rows[:60])
        b = LinearRouter(CATALOG, epochs=2, seed=3).fit(rows)
        self.assertEqual(a.w, b.w)

    # ----- candidate restriction and actions -----
    def test_selection_stays_inside_candidates(self):
        for row in TRAIN[:200:7]:
            gold = {c['tool'] for c in row['acceptable_choices']}
            cands = [t for t in row['candidate_tools'] if t not in gold][:3]
            for model in self.models():
                out = public(row, model, candidates=cands)
                self.assertTrue(all(c['tool'] in cands for c in out['choices']))
                self.assertTrue(set(out['ranked_tools']) <= set(cands))
                self.assertIn(out['decision'], DECISIONS)

    def test_empty_candidates_never_select(self):
        for model in self.models():
            out = predict('Grafana 대시보드 목록 보여줘', [], [], CATALOG, model=model)
            self.assertNotEqual(out['decision'], 'select_tools')
            self.assertEqual(out['choices'], [])
            self.assertEqual(out['ranked_tools'], [])

    def test_invalid_candidates_rejected_at_boundary(self):
        for cands in (['mcp__nope__x'], ['mcp__grafana__get_doc'] * 2):
            with self.assertRaises(ValueError):
                predict('무언가 해줘', [], cands, CATALOG, model=self.linear)

    def test_bad_router_outputs_are_rejected(self):
        tool = 'mcp__obsidian_vault__vault'
        other = 'mcp__grafana__get_doc'

        class Fake:
            def __init__(self, choice):
                self.choice = choice

            def predict(self, inp, catalog):
                return {'decision': 'select_tools', 'acceptable_choices': [self.choice],
                        'ranked_tools': [{'tool': tool, 'score': 1.0}]}

        for choice in ({'tool': tool, 'action': 'drop_database'}, {'tool': other}):
            with self.assertRaises(ValueError):
                predict('볼트에서 검색해줘', [], [tool], CATALOG, model=Fake(choice))
        self.assertEqual(predict('볼트에서 검색해줘', [], [tool], CATALOG, model=Fake({'tool': tool, 'action': 'search'}))
                         ['choices'], [{'tool': tool, 'action': 'search'}])

    def test_learned_action_is_dropped_when_schema_disallows_it(self):
        row = next(r for r in self.rows if any(c.get('action') for c in r['acceptable_choices'])
                   and self.linear.predict(model_input(r), CATALOG)['acceptable_choices'][:1]
                   and self.linear.predict(model_input(r), CATALOG)['acceptable_choices'][0].get('action'))
        chosen = self.linear.predict(model_input(row), CATALOG)['acceptable_choices'][0]['tool']
        stripped = copy.deepcopy(CATALOG)
        for tool in stripped['tools']:
            if tool['id'] == chosen:
                tool['input_schema'].get('properties', {}).pop('action', None)
        out = public(row, self.linear, stripped)
        self.assertTrue(all('action' not in c for c in out['choices']))

    # ----- all decisions -----
    def test_linear_emits_all_four_decisions_on_train(self):
        seen = {self.linear.predict(model_input(r), CATALOG)['decision'] for r in self.rows}
        self.assertEqual(seen, set(DECISIONS))

    def test_unsupported_when_needed_tool_removed(self):
        row = next(r for r in self.rows if r['decision'] == 'unsupported')
        self.assertEqual(public(row, self.linear)['decision'], 'unsupported')

    # ----- saved-state restoration -----
    def test_cold_load_with_subset_catalog_matches_training_scores(self):
        for kind, model in (('linear', self.linear), ('retrieval', self.retrieval)):
            for row in VALIDATION[:40] + [r for r in VALIDATION if r['decision'] != 'select_tools']:
                inp = model_input(row)
                expected = model.predict(inp, CATALOG)
                full = load_model(self.paths[kind]).predict(inp, CATALOG)
                cold = load_model(self.paths[kind]).predict(inp, subset_catalog(row['candidate_tools']))
                self.assertEqual(expected, full)
                self.assertEqual(expected, cold)

    def test_refit_on_subset_would_change_scores(self):
        """Guards the regression: a subset-catalog refit changes IDF, the restored state does not."""
        row = next(r for r in VALIDATION if len(r['candidate_tools']) < 10)
        sub = subset_catalog(row['candidate_tools'])
        refit = RetrievalRouter(sub).rank(model_input(row), sub)
        restored = load_model(self.paths['retrieval']).rank(model_input(row), sub)
        self.assertNotEqual(refit, restored)
        self.assertEqual(restored, self.retrieval.rank(model_input(row), CATALOG))
        self.assertEqual(load_model(self.paths['retrieval']).index.state(), TfidfIndex(CATALOG).state())

    def test_artifact_without_index_state_is_refused(self):
        with gzip.open(self.paths['retrieval'], 'rt') as f:
            data = json.load(f)
        del data['index']
        path = Path(self.tmp.name) / 'legacy.json.gz'
        with gzip.open(path, 'wt') as f:
            json.dump(data, f)
        with self.assertRaises(ValueError):
            load_model(path)

    def test_cache_reloads_replaced_artifact(self):
        path = Path(self.tmp.name) / 'swap.json.gz'
        shutil.copyfile(self.paths['retrieval'], path)
        row = VALIDATION[0]
        first = public(row, str(path))
        self.assertEqual(first, public(row, self.retrieval))
        tmp = Path(self.tmp.name) / 'swap.tmp'
        shutil.copyfile(self.paths['linear'], tmp)
        os.replace(tmp, path)
        self.assertEqual(public(row, str(path)), public(row, self.linear))
        self.assertEqual(len([k for k in predict_module._cache if k[0] == str(path.resolve())]), 1)


@unittest.skipUnless(SELECTED.exists(), 'selected artifact not present (models/ is not versioned)')
class SelectedArtifactSmokeTest(unittest.TestCase):
    """New Korean requests against the saved selected model; checks shape only, never test labels."""

    REQUESTS = [
        ('Grafana에서 지난 한 시간 동안 에러 로그가 많은 서비스 알려줘', []),
        ('옵시디언 볼트에서 회의록 노트를 찾아줘', []),
        ('주택임대차보호법 최신 조문 확인해줘', []),
        ('오늘 저녁 메뉴 추천해줘', []),
        ('그거 다시 실행해줘', []),
    ]

    def test_full_partial_and_empty_candidates(self):
        all_ids = [t['id'] for t in CATALOG['tools']]
        partial = ['mcp__grafana__query_loki_logs', 'mcp__obsidian_vault__vault', 'mcp__korean_law__search_law']
        partial = [t for t in partial if t in BY_ID]
        for request, context in self.REQUESTS:
            for cands in (all_ids, partial, []):
                out = predict(request, context, cands, CATALOG, model=str(SELECTED))
                validate_prediction(out, cands, subset_catalog(cands))
                self.assertTrue(set(out['ranked_tools']) <= set(cands))
                if not cands:
                    self.assertNotEqual(out['decision'], 'select_tools')
                full_cold = load_model(SELECTED).predict(model_input({'request': request, 'context': context,
                                                                      'candidate_tools': cands}), CATALOG)
                self.assertEqual(predict_module.to_contract(full_cold), out)


class DataGuardTest(unittest.TestCase):
    def test_modified_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_dir = Path(tmp) / 'd'
            shutil.copytree(DATA, copy_dir)
            load_data(copy_dir)
            with (copy_dir / 'test.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaises(SystemExit):
                load_data(copy_dir)


if __name__ == '__main__':
    unittest.main()
