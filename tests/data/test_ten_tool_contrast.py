import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_laya_ten_tools import build as build_original
from build_ten_tool_contrast import build, sha, signature
from audit_ten_tool_errors import audit


class ContrastDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        build_original(ROOT / 'datasets/mcp-routing-v2', ROOT / 'configs/training/laya-ten-tools-catalog-20261020.json',
                       self.root / 'catalog', self.root / 'base')
        self.base = self.root / 'base/full'
        self.authored = ROOT / 'datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl'
        self.criteria = ROOT / 'configs/training/laya-ten-tools-criteria-v2-20261020.json'

    def build(self, authored=None):
        return build(self.base, self.root / 'catalog', authored or self.authored, self.criteria,
                     self.root / 'dataset', self.root / 'export')

    def test_deterministic_balanced_data_preserves_held_out_states_labels_and_aliases(self):
        before = {p.name: sha(p) for p in self.base.iterdir()}
        result = self.build()
        second = build(self.base, self.root / 'catalog', self.authored, self.criteria,
                       self.root / 'dataset2', self.root / 'export2')
        self.assertEqual(result, second)
        self.assertEqual(result['profiles']['augmented']['train']['rows'], 240)
        self.assertEqual(set(result['profiles']['augmented']['train']['tool_counts'].values()), {24})
        for split in ('validation', 'test'):
            original = list(map(json.loads, (self.base / (split + '.jsonl')).read_text().splitlines()))
            path = self.root / 'export/augmented' / (split + '.jsonl')
            self.assertEqual(path.read_bytes(), (self.root / 'export/description_control' / path.name).read_bytes())
            new = list(map(json.loads, path.read_text().splitlines()))
            for a, b in zip(original, new):
                self.assertEqual((a['state'], a['gold']), (b['state'], b['gold']))
                self.assertEqual(a['metadata']['alias_to_tool'], b['metadata']['alias_to_tool'])
                self.assertEqual(a['metadata']['scenario_id'], b['metadata']['scenario_id'])
                self.assertNotEqual(a['questions'], b['questions'])
        for profile in ('description_control', 'augmented'):
            for p in (self.root / 'export' / profile).iterdir():
                self.assertEqual(p.read_bytes(), (self.root / 'export2' / profile / p.name).read_bytes())
        self.assertEqual(before, {p.name: sha(p) for p in self.base.iterdir()})
        with self.assertRaises(ValueError): self.build()

    def test_rejects_normalized_holdout_copy_invalid_action_and_review_promotion(self):
        original = list(map(json.loads, self.authored.read_text().splitlines()))
        holdout = json.loads((self.base / 'validation.jsonl').read_text().splitlines()[0])
        for kind in ('copy', 'action', 'review', 'group'):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(original)
                if kind == 'copy':
                    rows[0].update(request=' ' + holdout['state']['request'] + ' ', context=holdout['state']['context'])
                elif kind == 'action': rows[0]['action'] = 'execute_shell'
                elif kind == 'review': rows[0]['review_status'] = 'human_reviewed'
                else: rows[0]['scenario_id'] = holdout['metadata']['scenario_id']
                path = self.root / (kind + '.jsonl')
                path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
                with self.assertRaises(ValueError): self.build(path)
                self.assertFalse((self.root / 'dataset').exists())
                self.assertFalse((self.root / 'export').exists())

    def test_catalog_tampering_is_rejected_before_writing(self):
        path = self.root / 'catalog/tools.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(ValueError): self.build()
        self.assertFalse((self.root / 'dataset').exists())

    def test_training_extension_preserves_pinned_catalog_and_existing_bytes(self):
        self.build()
        base, catalog = self.root / 'export/augmented', self.root / 'dataset'
        original = {p.name: p.read_bytes() for p in base.iterdir()}
        cases = list(map(json.loads, self.authored.read_text().splitlines()))
        for case in cases:
            case['id'] = 'new.' + case['id']
            case['scenario_id'] = 'new.' + case['scenario_id']
            case['request'] = '추가 학습 fixture: ' + case['request']
        source = self.root / 'new.jsonl'
        source.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        result = build(base, catalog, source, self.criteria, self.root / 'v3', self.root / 'v3-export', 'test-v3')
        self.assertEqual(result['dataset_version'], 'test-v3')
        self.assertEqual(result['catalog_version'], json.loads((catalog / 'manifest.json').read_text())['catalog_version'])
        self.assertEqual((catalog / 'tools.json').read_bytes(), (self.root / 'v3/tools.json').read_bytes())
        self.assertEqual(result['profiles']['augmented']['train']['rows'], 420)
        self.assertEqual(set(result['profiles']['augmented']['train']['tool_counts'].values()), {42})
        for name in ('validation.jsonl', 'test.jsonl'):
            self.assertEqual((self.root / 'v3-export/augmented' / name).read_bytes(), original[name])
        self.assertTrue((self.root / 'v3-export/augmented/train.jsonl').read_bytes().startswith(original['train.jsonl']))
        for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl'):
            self.assertEqual((self.root / 'v3-export/description_control' / name).read_bytes(), original[name])
        self.assertEqual(original, {p.name: p.read_bytes() for p in base.iterdir()})
        cases[0]['id'] = json.loads(original['train.jsonl'].splitlines()[0])['metadata']['source_id']
        source.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases))
        with self.assertRaises(ValueError):
            build(base, catalog, source, self.criteria, self.root / 'rejected', self.root / 'rejected-export', 'test-v3')
        self.assertFalse((self.root / 'rejected').exists())

    def test_audit_validates_join_grain_and_counts_selected_validation_errors(self):
        rows = list(map(json.loads, (self.base / 'validation.jsonl').read_text().splitlines()))
        predictions = [{'source_id': r['metadata']['source_id'], 'decision': 'select_tools',
                        'tool': r['metadata']['alias_to_tool'][r['gold']['route_in_window']['label']]} for r in rows]
        runs = [{'run_id': str(i), 'validation_predictions': copy.deepcopy(predictions)} for i in range(2)]
        aliases = rows[0]['metadata']['alias_to_tool'].values()
        runs[0]['validation_predictions'][0]['tool'] = next(t for t in aliases if t != predictions[0]['tool'])
        path = self.root / 'predictions.json'
        path.write_text(json.dumps(runs))
        result = audit(self.base / 'validation.jsonl', path)
        self.assertEqual((result['prediction_events'], result['wrong_events'], result['cases_wrong_in_every_run']), (40, 1, 0))
        runs[0]['validation_predictions'].append(runs[0]['validation_predictions'][0])
        path.write_text(json.dumps(runs))
        with self.assertRaises(ValueError): audit(self.base / 'validation.jsonl', path)


if __name__ == '__main__': unittest.main()
