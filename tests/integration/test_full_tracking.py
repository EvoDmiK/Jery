import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('full_tracking', ROOT / 'scripts/run_laya_full_experiment.py')
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


class FullTrackingTest(unittest.TestCase):
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
