import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('ten_export', ROOT / 'scripts/build_laya_ten_tools.py')
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class TenToolsExportTest(unittest.TestCase):
    def test_balanced_deduplicated_profiles_are_deterministic_and_preserve_source(self):
        source = ROOT / 'datasets/mcp-routing-v2'
        selection = ROOT / 'configs/training/laya-ten-tools-catalog-20261020.json'
        before = {name: export.sha(source / name) for name in ('tools.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl')}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = export.build(source, selection, root / 'catalog', root / 'exports')
            second = export.build(source, selection, root / 'catalog2', root / 'exports2')
            self.assertEqual(manifest, second)
            ids = set(manifest['selected_tools'])
            descriptions = {t['id']: t['description'] for t in json.loads(selection.read_text())['tools']}
            for profile in ('memorization', 'full'):
                groups, features, source_ids = {}, {}, {}
                for split in ('train', 'validation', 'test'):
                    path = root / 'exports' / profile / (split + '.jsonl')
                    self.assertEqual(path.read_bytes(), (root / 'exports2' / profile / path.name).read_bytes())
                    rows = list(map(json.loads, path.read_text().splitlines()))
                    expected = 6 if profile == 'full' and split == 'train' else 2
                    self.assertEqual(len(rows), expected * 10)
                    counts = Counter()
                    groups[split], features[split], source_ids[split] = set(), set(), set()
                    for row in rows:
                        meta = row['metadata']; alias = meta['alias_to_tool']; question = row['questions']['route_in_window']
                        self.assertEqual(set(row['state']), {'request', 'context'})
                        self.assertEqual(len(alias), 10)
                        self.assertEqual(set(alias.values()), ids)
                        self.assertEqual(meta['window_count'], 1)
                        self.assertEqual(set(question['criteria']), set(alias))
                        self.assertEqual(question['criteria'], {a: descriptions[t] for a, t in alias.items()})
                        gold = row['gold']['route_in_window']
                        self.assertEqual(gold['probabilities'], {gold['label']: 1.})
                        counts[alias[gold['label']]] += 1
                        self.assertEqual({c['tool'] for c in meta['acceptable_choices']}, {alias[gold['label']]})
                        self.assertEqual({c['tool'] for c in meta['source_acceptable_choices']} & ids, {alias[gold['label']]})
                        groups[split].add(meta['scenario_id'])
                        features[split].add(json.dumps(row['state'], ensure_ascii=False, sort_keys=True))
                        source_ids[split].add(meta['source_id'])
                    self.assertEqual(set(counts.values()), {expected})
                    self.assertEqual(len(features[split]), len(rows))
                for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
                    self.assertFalse(groups[a] & groups[b])
                    self.assertFalse(features[a] & features[b])
            for split in ('validation', 'test'):
                self.assertEqual((root / 'exports/memorization' / (split + '.jsonl')).read_bytes(),
                                 (root / 'exports/full' / (split + '.jsonl')).read_bytes())
            with self.assertRaises(ValueError): export.build(source, selection, root / 'catalog', root / 'exports')
        self.assertEqual(before, {name: export.sha(source / name) for name in before})

    def test_rejects_non_ten_selection_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = json.loads((ROOT / 'configs/training/laya-ten-tools-catalog-20261020.json').read_text())
            selection['tools'].pop()
            (root / 'selection.json').write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                export.build(ROOT / 'datasets/mcp-routing-v2', root / 'selection.json', root / 'catalog', root / 'exports')
            self.assertFalse((root / 'catalog').exists())
            self.assertFalse((root / 'exports').exists())


if __name__ == '__main__':
    unittest.main()
