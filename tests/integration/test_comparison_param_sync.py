import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from sync_laya_comparison_params import comparison_params, missing_params


class ComparisonParamSyncTest(unittest.TestCase):
    def test_missing_fields_are_not_guessed_and_repeat_is_idempotent(self):
        expected = comparison_params({'warmup_steps': 75, 'lr_scheduler': 'cosine',
                                      'weight_decay': .1, 'private': 123})
        self.assertEqual(set(expected), {'warmup_steps', 'lr_scheduler', 'weight_decay'})
        self.assertEqual(missing_params(expected, {}), expected)
        self.assertEqual(missing_params(expected, {**expected, 'warmup_steps': '75.0'}), {})

    def test_conflicting_existing_values_and_bad_source_are_rejected(self):
        with self.assertRaises(ValueError):
            missing_params({'warmup_steps': '75'}, {'warmup_steps': '10'})
        with self.assertRaises(ValueError):
            missing_params({'lr_scheduler': 'cosine'}, {'lr_scheduler': 'linear'})
        for config in [{'warmup_steps': True}, {'max_grad_norm': float('nan')},
                       {'weight_decay': -.1}, {'label_smoothing': 1.}, {'lr_scheduler': ''}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                comparison_params(config)


if __name__ == '__main__':
    unittest.main()
