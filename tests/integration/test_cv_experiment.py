import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_laya_cv_experiment import aggregate_folds, FIXED_KEYS, validate_cv_plan, fold_trial_settings
from run_laya_full_experiment import trial_settings, PARAMS, hyperopt_candidate


class CvExperimentTest(unittest.TestCase):
    def test_fixed_lr_reference_first_then_nineteen_parameter_conditions(self):
        try:
            import hyperopt
        except ImportError:
            self.skipTest('Hyperopt is required for actual ask/tell verification')
        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root / 'configs/training/laya-ten-tools-fixedlr-params-cv5-20261022.json').read_text())
        config = json.loads((root / 'configs/training/laya-ten-tools-fixedlr-params-cv5-20261022-model.json').read_text())
        validate_cv_plan(plan, config)
        children = []
        seen = set()
        for index in range(20):
            candidate = hyperopt_candidate(plan, children)
            parameters = candidate['parameters']
            key = tuple(sorted(parameters.items()))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(parameters['head_learning_rate'], config['head_learning_rate'])
            if index == 0:
                self.assertEqual(parameters, plan['first_trial_parameters'])
            else:
                self.assertNotEqual(parameters, plan['first_trial_parameters'])
            overrides, tracked = trial_settings(plan, config, parameters)
            self.assertEqual(tracked['effective_batch_size'], 4 * parameters['grad_accum'])
            nested, model, fold_params = fold_trial_settings(plan, config, parameters, index + 1, Path('folds'), 'artifacts:parent')
            self.assertEqual(fold_params, tracked)
            from prepare_laya_cuda_experiment import validate_explicit_plan
            validate_explicit_plan(nested, model)
            for key in parameters:
                self.assertEqual(overrides[key], parameters[key])
                self.assertEqual(model[key], parameters[key])
            for fold in nested['trials']:
                actual, _ = trial_settings(nested, model, {k: v for k, v in fold.items() if k != 'trial_id'})
                self.assertEqual({**model, **actual}['grad_accum'], parameters['grad_accum'])
                self.assertEqual(actual['export_dir'], fold['export_dir'])
            changed_test = json.loads(json.dumps(children))
            for child in changed_test:
                child['final_test'] = 0.
            self.assertEqual(hyperopt_candidate(plan, changed_test), candidate)
            children.append({'status': 'finished', 'hyperopt': candidate,
                             'search_validation': .6 + index / 100, 'final_test': .99})
        changed = json.loads(json.dumps(plan))
        changed['first_trial_parameters']['weight_decay'] = 0.
        with self.assertRaises(ValueError):
            validate_cv_plan(changed, config)
        changed = json.loads(json.dumps(plan))
        changed['hyperopt']['head_lr'] = [.0001, .001]
        with self.assertRaises(ValueError):
            validate_cv_plan(changed, config)

    def test_duplicate_tpe_proposal_uses_unseen_condition_and_exhaustion_is_explicit(self):
        try:
            from hyperopt import tpe
        except ImportError:
            self.skipTest('Hyperopt required')
        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root / 'configs/training/laya-ten-tools-fixedlr-params-cv5-20261022.json').read_text())
        for key, value in plan['first_trial_parameters'].items():
            if key != 'head_learning_rate':
                plan['hyperopt'][key] = [value]
        plan['hyperopt']['weight_decay'] = [.1, .2]
        first = hyperopt_candidate(plan, [])
        children = [{'status': 'finished', 'hyperopt': first, 'search_validation': .75}]
        original_suggest = tpe.suggest
        def repeat_reference(*args, **kwargs):
            docs = original_suggest(*args, **kwargs)
            docs[0]['misc']['vals'] = json.loads(json.dumps(first['document']['misc']['vals']))
            return docs
        with mock.patch.object(tpe, 'suggest', side_effect=repeat_reference):
            next_trial = hyperopt_candidate(plan, children)
        self.assertEqual(next_trial['proposal'], 'unseen_grid')
        self.assertEqual(next_trial['parameters']['weight_decay'], .2)
        self.assertEqual(next_trial['document']['misc']['vals']['weight_decay'], [1])
        self.assertEqual(next_trial['document']['tid'], 1)
        children.append({'status': 'finished', 'hyperopt': next_trial, 'search_validation': .76})
        with self.assertRaisesRegex(ValueError, 'search space exhausted'):
            hyperopt_candidate(plan, children)

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
