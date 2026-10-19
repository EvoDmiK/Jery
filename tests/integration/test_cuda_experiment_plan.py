import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('cuda_prepare', ROOT / 'scripts/prepare_laya_cuda_experiment.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CudaExperimentPlanTest(unittest.TestCase):
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
