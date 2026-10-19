import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('full_tracking', ROOT / 'scripts/run_laya_full_experiment.py')
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class FullTrackingTest(unittest.TestCase):
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
