import json
import tempfile
import unittest
from pathlib import Path

from jery.model.laya_pilot import choice_metrics, load_export, sha256


def row(label, probabilities=None):
    gold = {'label': label, 'probabilities': probabilities or {label: 1.0}}
    return {'gold': {'route_in_window': gold}}


class LayaPilotTest(unittest.TestCase):
    def test_any_of_accuracy_and_soft_ce(self):
        rows = [row('T0', {'T0': 0.5, 'T2': 0.5}), row('window_none')]
        predictions = [{'choice': 'T2', 'probabilities': {'T0': 0.5, 'T2': 0.5, 'window_none': 0.0}},
                       {'choice': 'T0', 'probabilities': {'T0': 1.0, 'window_none': 0.0}}]
        metrics = choice_metrics(rows, predictions)
        self.assertEqual(metrics['accuracy'], 0.5)
        self.assertEqual(metrics['by_gold_kind'], {'tool': {'correct': 1, 'rows': 1},
                                                   'window_none': {'correct': 0, 'rows': 1}})
        self.assertGreater(metrics['mean_soft_ce'], 10)

    def test_export_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for split in ('train', 'validation'):
                (folder / f'{split}.jsonl').write_text(json.dumps(row('no_tool')) + '\n')
            splits = {split: {'rows': 1, 'sha256': sha256(folder / f'{split}.jsonl')} for split in ('train', 'validation')}
            manifest = {'export_version': 'decision-export-v1', 'splits': splits}
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            rows, _ = load_export(folder)
            self.assertEqual(len(rows['train']), 1)
            (folder / 'train.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                load_export(folder)


if __name__ == '__main__':
    unittest.main()
