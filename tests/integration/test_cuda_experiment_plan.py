import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('cuda_prepare', ROOT / 'scripts/prepare_laya_cuda_experiment.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CudaExperimentPlanTest(unittest.TestCase):
    def test_preparation_retries_transient_reads_uploads_only(self):
        operation = mock.Mock(side_effect=[HTTPError('url', 520, 'proxy', {}, None), TimeoutError(), 'proof'])
        with mock.patch.object(prepare.time, 'sleep') as sleep:
            self.assertEqual(prepare.tracking_retry(operation, 'payload'), 'proof')
            self.assertEqual(sleep.call_count, 2)
        for error in (HTTPError('url', 400, 'bad input', {}, None), ValueError('checksum mismatch')):
            operation = mock.Mock(side_effect=error)
            with self.assertRaises(type(error)), mock.patch.object(prepare.time, 'sleep') as sleep:
                prepare.tracking_retry(operation)
            self.assertEqual(operation.call_count, 1)
            sleep.assert_not_called()
        operation = mock.Mock(side_effect=TimeoutError())
        with mock.patch.object(prepare.time, 'sleep'), self.assertRaises(TimeoutError):
            prepare.tracking_retry(operation)
        self.assertEqual(operation.call_count, 8)

    def test_fixed_optimizer_choices_are_validated(self):
        for key, value in [('warmup_steps', -1), ('max_grad_norm', 0), ('lr_scheduler', 'typo'),
                           ('head_learning_rate', True), ('label_smoothing', False)]:
            with self.subTest(key=key):
                plan = copy.deepcopy(self.plan)
                plan['trials'][0][key] = value
                with self.assertRaises(ValueError): prepare.validate_explicit_plan(plan, self.config)

    def setUp(self):
        self.plan = json.loads((ROOT / 'configs/training/laya-routing-v2-controls-20261019.json').read_text())
        self.config = json.loads((ROOT / 'configs/model/laya-cuda-20261019.json').read_text())
        self.config.update(training_policy_version='laya-routing-v2', validation_callback=self.plan['validation_callback'])

    def test_only_negative_window_weight_varies(self):
        prepare.validate_explicit_plan(self.plan, self.config)
        trials = self.plan['trials']
        self.assertEqual([t['window_none_loss_weight'] for t in trials], [1., .5, .25])
        fixed = [{k: v for k, v in t.items() if k not in ('trial_id', 'window_none_loss_weight')} for t in trials]
        self.assertEqual(fixed, [fixed[0]] * 3)
        self.assertEqual(self.plan['epochs'], 10)
        self.assertEqual(self.plan['micro_batch'] * self.plan['grad_accum'], 4)

    def test_fixed_trial_batch_and_seed_overrides_are_strict(self):
        self.plan['trials'][0].update(seed=3408, micro_batch=8, grad_accum=2)
        prepare.validate_explicit_plan(self.plan, self.config)
        for key, value in [('seed', -1), ('seed', True), ('micro_batch', 0),
                           ('micro_batch', 4.5), ('grad_accum', False)]:
            with self.subTest(key=key, value=value):
                plan = copy.deepcopy(self.plan)
                plan['trials'][0][key] = value
                with self.assertRaises(ValueError): prepare.validate_explicit_plan(plan, self.config)

    def test_joint_search_validates_lower_lr_and_weight_choices(self):
        plan = json.loads((ROOT / 'configs/training/laya-routing-v2-joint-search-20261019.json').read_text())
        prepare.validate_explicit_plan(plan, self.config)
        for weights in ([], [0], [1.1], [float('nan')], [True]):
            bad = copy.deepcopy(plan)
            bad['hyperopt']['window_none_loss_weight'] = weights
            with self.assertRaises(ValueError): prepare.validate_explicit_plan(bad, self.config)
        bad = copy.deepcopy(plan)
        bad['hyperopt']['head_lr'] = [1e-5, 1e-6]
        with self.assertRaises(ValueError): prepare.validate_explicit_plan(bad, self.config)
        bad = copy.deepcopy(plan)
        bad['child_params'].remove('window_none_loss_weight')
        with self.assertRaises(ValueError): prepare.validate_explicit_plan(bad, self.config)

    def test_ten_tool_plan_and_model_task_must_match(self):
        plan = json.loads((ROOT / 'configs/training/laya-ten-tools-memorization-20261020.json').read_text())
        config = json.loads((ROOT / 'configs/model/laya-ten-tools-20261020.json').read_text())
        prepare.validate_explicit_plan(plan, config)
        plan.pop('evaluation_task')
        with self.assertRaises(ValueError): prepare.validate_explicit_plan(plan, config)

    def test_optimizer_search_choices_reject_bad_types_and_values(self):
        plan = json.loads((ROOT / 'configs/training/laya-ten-tools-train600-joint20-20261021.json').read_text())
        config = json.loads((ROOT / 'configs/training/laya-ten-tools-train600-joint20-model-20261021.json').read_text())
        prepare.validate_explicit_plan(plan, config)
        for key, choices in [('warmup_steps', []), ('warmup_steps', [True]),
                             ('warmup_steps', [-1]), ('warmup_steps', [1.5]),
                             ('grad_accum', [0]), ('grad_accum', [False]),
                             ('max_grad_norm', [float('nan')]), ('max_grad_norm', [0]),
                             ('max_grad_norm', [True]), ('lr_scheduler', ['typo']),
                             ('lr_scheduler', 'linear'), ('weight_decay', [-.1]),
                             ('weight_decay', [0., 0]), ('label_smoothing', [1.]),
                             ('label_smoothing', [float('nan')]), ('label_smoothing', [True])]:
            with self.subTest(key=key, choices=choices):
                bad = copy.deepcopy(plan)
                bad['hyperopt'][key] = choices
                with self.assertRaises(ValueError): prepare.validate_explicit_plan(bad, config)

    def test_rejects_mixed_search_or_invalid_weights_and_budget(self):
        for kind in ('mixed', 'weight', 'duplicate', 'short_budget', 'version'):
            with self.subTest(kind=kind):
                plan = copy.deepcopy(self.plan)
                if kind == 'mixed': plan['hyperopt'] = {'max_evals': 5}
                if kind == 'weight': plan['trials'][0]['window_none_loss_weight'] = 0
                if kind == 'duplicate': plan['trials'][1]['trial_id'] = plan['trials'][0]['trial_id']
                if kind == 'short_budget': plan['epochs'] = 9
                if kind == 'version': plan['training_policy_version'] = 'legacy'
                with self.assertRaises(ValueError): prepare.validate_explicit_plan(plan, self.config)


if __name__ == '__main__':
    unittest.main()
