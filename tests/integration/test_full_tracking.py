import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('full_tracking', ROOT / 'scripts/run_laya_full_experiment.py')
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class FullTrackingTest(unittest.TestCase):
    def test_proxy_outage_is_retryable_but_client_and_validation_errors_are_not(self):
        for code in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524):
            with self.subTest(code=code):
                self.assertTrue(tracking.transient_tracking_error(
                    urllib.error.HTTPError('https://tracking.invalid', code, 'outage', {}, None)))
        for code in (400, 401, 403, 404, 409, 501):
            with self.subTest(code=code):
                self.assertFalse(tracking.transient_tracking_error(
                    urllib.error.HTTPError('https://tracking.invalid', code, 'permanent', {}, None)))
        for error in (urllib.error.URLError('offline'), TimeoutError(), ConnectionError()):
            self.assertTrue(tracking.transient_tracking_error(error))
        self.assertFalse(tracking.transient_tracking_error(ValueError('invalid artifact')))

    def test_legacy_completion_accepts_backfilled_comparison_params_only(self):
        core = tracking.PARAMS | {'window_none_loss_weight'}
        def record(params, metrics=tracking.METRICS):
            return {'data': {'params': [{'key': k} for k in params],
                             'metrics': [{'key': k} for k in metrics]}}
        tracking.validate_logged_keys(record(core), core)
        tracking.validate_logged_keys(record(core | tracking.COMPARISON_PARAMS), core)
        tracking.validate_logged_keys(record(core | tracking.COMPARISON_PARAMS), core | tracking.COMPARISON_PARAMS)
        for params, metrics, required in [
            (core | {'private_detail'}, tracking.METRICS, core),
            (core - {'head_lr'}, tracking.METRICS, core),
            (core, tracking.METRICS - {'test_accuracy'}, core),
            (core, tracking.METRICS | {'private_metric'}, core),
            (core, tracking.METRICS, core | tracking.COMPARISON_PARAMS),
        ]:
            with self.subTest(params=params, metrics=metrics, required=required), self.assertRaises(ValueError):
                tracking.validate_logged_keys(record(params, metrics), required)

    def test_trial_overrides_drive_both_training_and_mlflow_parameters(self):
        plan = {'epochs': 32, 'seed': 3407, 'micro_batch': 4, 'grad_accum': 1,
                'export_dir': 'unchanged-split', 'child_params': sorted(tracking.PARAMS)}
        sampled = {'head_learning_rate': .0003, 'seed': 3411, 'micro_batch': 8, 'grad_accum': 2}
        overrides, params = tracking.trial_settings(plan, {}, sampled)
        self.assertEqual((overrides['seed'], params['seed']), (3411, 3411))
        self.assertEqual(params['effective_batch_size'], overrides['micro_batch'] * overrides['grad_accum'])
        self.assertEqual(params['effective_batch_size'], 16)
        self.assertEqual(overrides['export_dir'], 'unchanged-split')
        overrides, params = tracking.trial_settings(plan, {}, {'head_learning_rate': .0003})
        self.assertEqual((params['seed'], params['effective_batch_size']), (3407, 4))
        self.assertEqual(set(params), tracking.PARAMS)

    def test_ten_tool_f1_is_explicit_without_reinterpreting_legacy_metrics(self):
        result = {'decision_macro_f1': .25, 'tool_macro_f1': .8, 'macro_f1_scope': 'tools'}
        self.assertEqual(tracking.comparison_macro_f1(result), .8)
        self.assertEqual(tracking.comparison_macro_f1({'decision_macro_f1': .25}), .25)
        with self.assertRaises(KeyError):
            tracking.comparison_macro_f1({'decision_macro_f1': .25, 'macro_f1_scope': 'tools'})
        with self.assertRaises(ValueError):
            tracking.comparison_macro_f1({'decision_macro_f1': .25, 'macro_f1_scope': 'unknown'})

    @unittest.skipUnless(importlib.util.find_spec('hyperopt'), 'Requires the decision-training environment')
    def test_joint_search_tracks_weight_and_replays_only_trained_validation(self):
        plan = json.loads((ROOT / 'configs/training/laya-routing-v2-joint-search-20261019.json').read_text())
        self.assertEqual(tracking.child_param_keys(plan), tracking.PARAMS | {'window_none_loss_weight'})
        first = tracking.hyperopt_candidate(plan, [])
        self.assertIn(first['parameters']['window_none_loss_weight'], [.25, .5, 1.])
        self.assertTrue(1e-6 <= first['parameters']['head_learning_rate'] <= 1e-5)
        child = {'status': 'finished', 'hyperopt': first, 'search_validation': .08, 'final_validation': .1613}
        second = tracking.hyperopt_candidate(plan, [child])
        child['final_test'] = 1.
        self.assertEqual(second, tracking.hyperopt_candidate(plan, [child]))
        legacy = {**plan, 'training_policy_version': 'legacy'}
        with self.assertRaises(ValueError): tracking.child_param_keys(legacy)
        with self.assertRaises(ValueError):
            tracking.child_param_keys({**plan, 'child_params': [*plan['child_params'], 'private_detail']})

    def test_prepared_policy_and_configuration_cannot_be_swapped(self):
        config = {'backend': 'cuda', 'training_policy_version': tracking.POLICY_V2}
        plan = {'training_policy_version': tracking.POLICY_V2}
        tracking.validate_prepared_config(config, plan, dict(config))
        with self.assertRaises(ValueError):
            tracking.validate_prepared_config({'backend': 'mlx'}, plan, config)
        with self.assertRaises(ValueError):
            tracking.validate_prepared_config(config, {}, config)

    @unittest.skipUnless(importlib.util.find_spec('hyperopt'), 'Requires the decision-training environment')
    def test_optimizer_search_keeps_types_and_replays_into_training(self):
        plan = json.loads((ROOT / 'configs/training/laya-ten-tools-train600-joint20-20261021.json').read_text())
        config = json.loads((ROOT / 'configs/training/laya-ten-tools-train600-joint20-model-20261021.json').read_text())
        children = []
        for index in range(8):  # Includes TPE after its six startup trials.
            trial = tracking.hyperopt_candidate(plan, children)
            sampled = trial['parameters']
            self.assertIs(type(sampled['grad_accum']), int)
            self.assertIs(type(sampled['warmup_steps']), int)
            self.assertIn(sampled['lr_scheduler'], ['linear', 'cosine'])
            overrides, params = tracking.trial_settings(plan, config, sampled)
            for key in ('warmup_steps', 'grad_accum', 'max_grad_norm', 'lr_scheduler'):
                self.assertIn(sampled[key], plan['hyperopt'][key])
                self.assertEqual(overrides[key], sampled[key])
            self.assertIn(params['effective_batch_size'], [4, 8, 16])
            self.assertEqual(params['effective_batch_size'], 4 * sampled['grad_accum'])
            self.assertEqual(set(params), tracking.child_param_keys(plan))
            for key in tracking.COMPARISON_PARAMS:
                self.assertEqual(params[key], overrides.get(key, config[key]))
            self.assertEqual(json.loads(json.dumps(trial)), trial)
            children.append({'status': 'finished', 'search_validation': .6 + .01 * index,
                             'hyperopt': trial, 'final_test': 0.})
        expected = tracking.hyperopt_candidate(plan, children)
        for child in children:
            child['final_test'] = 1.
        self.assertEqual(tracking.hyperopt_candidate(plan, children), expected)

    def test_comparison_params_use_actual_override_then_config(self):
        plan = {'epochs': 32, 'seed': 3407, 'micro_batch': 4, 'grad_accum': 1,
                'export_dir': 'unchanged', 'child_params': sorted(tracking.PARAMS | tracking.COMPARISON_PARAMS)}
        config = {'warmup_steps': 10, 'max_grad_norm': 1., 'lr_scheduler': 'linear',
                  'weight_decay': .05, 'label_smoothing': 0.}
        _, params = tracking.trial_settings(plan, config, {'head_learning_rate': .003,
                                            'warmup_steps': 150, 'lr_scheduler': 'cosine'})
        self.assertEqual(params['warmup_steps'], 150)
        self.assertEqual(params['lr_scheduler'], 'cosine')
        self.assertEqual(params['weight_decay'], .05)
        self.assertEqual(params['label_smoothing'], 0.)

    def test_v2_search_uses_trained_score_below_serving_baseline(self):
        plan = {'training_policy_version': tracking.POLICY_V2}
        self.assertEqual(tracking.comparison_score({'final_validation': .1613, 'search_validation': .08}, plan), .08)
        with self.assertRaises(KeyError):
            tracking.comparison_score({'final_validation': .1613}, plan)
        report = {'selection': {'policy_version': tracking.POLICY_V2, 'completed_epochs': 10,
                               'selected_epoch': 0, 'best_score': .1613, 'best_trained_epoch': 7,
                               'best_trained_score': .14, 'search_score': .14},
                  'validation': {'epoch': 0, 'case_accuracy': .1613}, 'test': {'epoch': 0},
                  'before_validation': {'case_accuracy': .1613},
                  'best_trained': {'epoch': 7, 'validation_case_accuracy': .14}}
        self.assertEqual(tracking.trained_search_score(report, 10), .14)
        report['selection']['search_score'] = .1613
        with self.assertRaises(ValueError): tracking.trained_search_score(report, 10)
        report['selection']['search_score'] = .14
        report['selection']['search_score'] = float('nan')
        with self.assertRaises(ValueError): tracking.trained_search_score(report, 10)
        report['selection']['search_score'] = .14
        report['best_trained']['epoch'] = 0
        with self.assertRaises(ValueError): tracking.trained_search_score(report, 10)

    @unittest.skipUnless(importlib.util.find_spec('hyperopt'), 'Requires the decision-training environment')
    def test_v2_hyperopt_replay_is_not_floored_at_the_initial_model(self):
        from hyperopt import tpe
        plan = {'seed': 3407, 'training_policy_version': tracking.POLICY_V2,
                'hyperopt': {'head_lr': [1e-5, 1e-3], 'weight_decay': [0., .01],
                             'label_smoothing': [0., .05], 'startup_trials': 2}}
        trial = tracking.hyperopt_candidate(plan, [])
        children = [{'status': 'finished', 'final_validation': .1613, 'search_validation': .08, 'hyperopt': trial}]
        with mock.patch.object(tpe, 'suggest', wraps=tpe.suggest) as suggest:
            next_trial = tracking.hyperopt_candidate(plan, children)
            replay = suggest.call_args.args[2].trials
            self.assertAlmostEqual(replay[0]['result']['loss'], .92)
        self.assertEqual(next_trial['objective'], '1 - best_trained_validation_case_accuracy')

    def test_recovery_preserves_original_source_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('source.json', 'configuration.json'):
                path = Path(directory) / name
                tracking.write_json(path, {'original': True})
                before = path.read_bytes()
                tracking.write_json(path, {'original': False}, overwrite=False)
                self.assertEqual(path.read_bytes(), before)
            fresh = Path(directory) / 'fresh.json'
            tracking.write_json(fresh, {'new': True}, overwrite=False)
            self.assertEqual(json.loads(fresh.read_text()), {'new': True})

    @unittest.skipUnless(importlib.util.find_spec('hyperopt'), 'Requires the decision-training environment')
    def test_hyperopt_replay_uses_only_finished_validation(self):
        plan = {'seed': 3407, 'hyperopt': {'head_lr': [1e-5, 1e-3], 'weight_decay': [0., .01],
                                        'label_smoothing': [0., .05], 'startup_trials': 2}}
        children = []
        for index in range(3):
            trial = tracking.hyperopt_candidate(plan, children)
            self.assertEqual(trial['document']['tid'], index)
            self.assertTrue(1e-5 <= trial['parameters']['head_learning_rate'] <= 1e-3)
            children.append({'status': 'finished', 'final_validation': .2 + .01 * index, 'hyperopt': trial})
        first = tracking.hyperopt_candidate(plan, children)
        children[0]['final_test'] = 1.0
        children.append({'status': 'running', 'final_validation': 1.0})
        self.assertEqual(first, tracking.hyperopt_candidate(plan, children))

    def test_best_validation_epoch_and_early_stop_bounds(self):
        report = {'selection': {'completed_epochs': 5, 'selected_epoch': 0, 'best_score': .16},
                  'validation': {'epoch': 0, 'case_accuracy': .16}, 'test': {'epoch': 0}}
        self.assertEqual(tracking.final_epoch_bounds(report, 64, True), (0, 5))
        report['test']['epoch'] = 5
        with self.assertRaises(ValueError): tracking.final_epoch_bounds(report, 64, True)
        report['test']['epoch'] = 0
        report['validation']['case_accuracy'] = .15
        with self.assertRaises(ValueError): tracking.final_epoch_bounds(report, 64, True)

    def test_only_comparison_metrics_escape_json_detail(self):
        record = {'epoch': 7, 'train_loss': 0.25,
            'validation': {'case_accuracy': 0.4, 'decision_macro_f1': 0.3, 'coverage': 0.9,
                           'case': {'per_class': {'private_detail': 123}}, 'window': {'soft_ce': 5.0}},
            'environment': {'executable': '/example/python'}}
        saved = json.dumps(record, sort_keys=True)
        self.assertEqual(tracking.compact_epoch(record), {
            'train_loss': 0.25, 'val_accuracy': 0.4, 'val_macro_f1': 0.3, 'val_coverage': 0.9})
        self.assertEqual(json.dumps(record, sort_keys=True), saved)
        self.assertEqual(len(tracking.PARAMS), 4)
        self.assertEqual(len(tracking.METRICS), 7)
        record['train_loss'] = float('nan')
        with self.assertRaises(ValueError): tracking.compact_epoch(record)

    def test_partial_last_json_line_does_not_hide_corrupt_complete_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            path.write_text('{"epoch": 1}\n{"epoch":')
            self.assertEqual(tracking.json_lines(path), [{'epoch': 1}])
            path.write_text('{"epoch": 1}\ninvalid\n')
            with self.assertRaises(json.JSONDecodeError): tracking.json_lines(path)
            target = Path(folder) / 'state.json'
            tracking.write_json(target, {'status': 'running'})
            self.assertEqual(json.loads(target.read_text()), {'status': 'running'})


if __name__ == '__main__':
    unittest.main()
