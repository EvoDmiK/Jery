import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_laya_cv_experiment import aggregate_folds, FIXED_KEYS, validate_cv_plan
from run_laya_full_experiment import trial_settings, PARAMS


class CvExperimentTest(unittest.TestCase):
    def test_reference_child_all_non_lr_model_settings_are_immutable(self):
        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root / 'configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-20261021.json').read_text())
        config = json.loads((root / 'configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-model-20261021.json').read_text())
        validate_cv_plan(plan, config)
        config['head_learning_rate'] = .004
        config['models_dir'] = 'isolated-model-output'
        validate_cv_plan(plan, config)
        for key, value in [('warmup_steps', 75), ('label_smoothing', 0.),
                           ('eval_batch_size', 8), ('gradient_checkpointing', False)]:
            changed = {**config, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cv_plan(plan, changed)

    def test_cv_plan_rejects_searching_non_lr_conditions(self):
        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root / 'configs/training/laya-ten-tools-train600-lr-only-cv5-20261021.json').read_text())
        config = json.loads((root / 'configs/training/laya-ten-tools-train600-lr-only-cv5-model-20261021.json').read_text())
        validate_cv_plan(plan, config)
        for field, values in [('weight_decay', [0, .1]), ('warmup_steps', [10, 75]),
                              ('grad_accum', [1, 2]), ('lr_scheduler', ['linear', 'cosine'])]:
            changed = json.loads(json.dumps(plan))
            changed['hyperopt'][field] = values
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_cv_plan(changed, config)

    def test_fold_exports_drive_the_actual_model_override(self):
        plan = {'epochs': 32, 'seed': 3407, 'micro_batch': 4, 'grad_accum': 1,
                'export_dir': 'fold-1', 'child_params': sorted(PARAMS)}
        overrides, _ = trial_settings(plan, {}, {'head_learning_rate': .003, 'export_dir': 'fold-5'})
        self.assertEqual(overrides['export_dir'], 'fold-5')

    def test_aggregate_uses_validation_only_for_search_and_rejects_parameter_drift(self):
        config = {k: 1 for k in FIXED_KEYS}
        config['head_learning_rate'] = .003
        reports = [{'config': dict(config), 'selection': {'completed_epochs': 10, 'best_trained_score': score},
                    'validation': {'case_accuracy': score, 'decision_macro_f1': score},
                    'test': {'case_accuracy': .9, 'decision_macro_f1': .8}} for score in (.6, .8, .7, .8, .6)]
        result = aggregate_folds(reports)
        self.assertAlmostEqual(result['val_accuracy'], .7)
        self.assertAlmostEqual(result['cv_search_accuracy'], .7)
        self.assertGreater(result['val_accuracy_std'], 0)
        for report in reports:
            report['test']['case_accuracy'] = 0.
        self.assertEqual(aggregate_folds(reports)['cv_search_accuracy'], result['cv_search_accuracy'])
        reports[0]['config']['warmup_steps'] = 75
        with self.assertRaises(ValueError):
            aggregate_folds(reports)


if __name__ == '__main__':
    unittest.main()
