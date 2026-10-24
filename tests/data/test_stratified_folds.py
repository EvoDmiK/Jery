from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from build_laya_stratified_folds import build, label, stratified_group_folds


def row(tool, group, number):
    alias = 'T0' if number % 2 else 'T1'
    return {'gold': {'route_in_window': {'label': alias}},
            'metadata': {'alias_to_tool': {alias: tool}, 'scenario_id': group,
                         'source_id': f'{group}-{number}'}}


class StratifiedFoldsTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row(tool, f'{tool}-g{g}', n) for tool in ('a', 'b') for g in range(6) for n in range(2)]

    def test_balanced_real_tool_labels_disjoint_groups_and_one_oof_prediction(self):
        folds = stratified_group_folds(self.rows, 3, 3407)
        self.assertEqual(folds, stratified_group_folds(self.rows, 3, 3407))
        self.assertEqual(sorted(i for f in folds for i in f), list(range(24)))
        for validation in folds:
            train = set(range(24)) - set(validation)
            self.assertEqual(Counter(label(self.rows[i]) for i in validation), {'a': 4, 'b': 4})
            self.assertFalse({self.rows[i]['metadata']['scenario_id'] for i in train} &
                             {self.rows[i]['metadata']['scenario_id'] for i in validation})

    def test_unsupported_group_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            stratified_group_folds([row('a', 'same', 1), row('b', 'same', 2)], 2)
        with self.assertRaises(ValueError):
            stratified_group_folds(self.rows, 7)

    def test_mixed_label_pairs_are_balanced_and_never_split(self):
        rows = self.rows + [row(tool, f'pair-{g}', n) for g in range(6)
                            for n, tool in enumerate(('a', 'b'))]
        folds = stratified_group_folds(rows, 3, 3407)
        self.assertEqual(folds, stratified_group_folds(rows, 3, 3407))
        self.assertEqual(sorted(i for fold in folds for i in fold), list(range(len(rows))))
        for fold in folds:
            self.assertEqual(Counter(label(rows[i]) for i in fold), {'a': 6, 'b': 6})
            train = set(range(len(rows))) - set(fold)
            self.assertFalse({rows[i]['metadata']['scenario_id'] for i in fold} &
                             {rows[i]['metadata']['scenario_id'] for i in train})

    def test_exports_preserve_source_rows_and_heldout_test_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / 'source', Path(temporary) / 'folds'
            source.mkdir()
            manifest = {'splits': {}}
            for split, items in [('train', self.rows), ('validation', [row('a', 'external-val', 0)]),
                                 ('test', [row('b', 'external-test', 0)])]:
                data = ''.join(json.dumps(r) + '\n' for r in items).encode()
                (source / (split + '.jsonl')).write_bytes(data)
                manifest['splits'][split] = {'sha256': hashlib.sha256(data).hexdigest(), 'rows': len(items)}
            (source / 'manifest.json').write_text(json.dumps(manifest))
            result = build(source, output, 3, 3407)
            self.assertEqual(len(result['folds']), 3)
            original = {json.dumps(r, sort_keys=True) for r in self.rows}
            for fold in range(1, 4):
                directory = output / f'fold-{fold}'
                self.assertEqual((directory / 'test.jsonl').read_bytes(), (source / 'test.jsonl').read_bytes())
                combined = [json.loads(line) for split in ('train', 'validation')
                            for line in (directory / (split + '.jsonl')).read_text().splitlines()]
                self.assertEqual({json.dumps(r, sort_keys=True) for r in combined}, original)
            with self.assertRaises(FileExistsError):
                build(source, output, 3, 3407)


if __name__ == '__main__':
    unittest.main()
