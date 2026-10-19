import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jery.model.laya_cuda import (RoutingPolicy, check_soft_targets, policy_settings, restore_head,
                                  resume_adds_steps, rewind_history, tensor_hash, window_none_weights)
from jery.model.laya_full import ValidationPolicy
from jery.model.laya_pilot import QUESTION


class CudaResumeHistoryTest(unittest.TestCase):
    def test_checkpoint_rewind_ignores_later_validation_and_preserves_stop(self):
        policy = ValidationPolicy(1, .004)
        policy.observe(0, .2)
        history = [{'epoch': 1, 'validation': {'case_accuracy': .1}},
                   {'epoch': 2, 'validation': {'case_accuracy': .9}}]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'steps.jsonl').write_text(''.join(json.dumps({'step': i}) + '\n' for i in range(1, 6)))
            rewind_history(out, history, 4, 1, policy)
            self.assertEqual([r['epoch'] for r in history], [1])
            self.assertEqual([json.loads(r)['step'] for r in (out / 'steps.jsonl').read_text().splitlines()], [1, 2, 3, 4])
            self.assertTrue(policy.stopped)
            self.assertEqual(policy.best_epoch, 0)


class RoutingPolicyV2Test(unittest.TestCase):
    def test_recovers_below_baseline_and_selects_trained_apart_from_serving(self):
        policy = RoutingPolicy(patience=2, min_delta=.004, min_epochs=3)
        policy.observe(0, .16)
        saved = [e for e, s in enumerate([.05, .04, .03, .02, .10, .12, .121], 1) if policy.observe(e, s)]
        # Bad epochs 2-3 are inside min_epochs; 4 and 5 would stop the legacy policy but 5 improves.
        self.assertFalse(policy.stopped)
        self.assertEqual(saved, [1, 5, 6, 7])  # Exact highest is saved even below min_delta.
        snap = policy.snapshot()
        self.assertEqual((snap['best_trained_epoch'], snap['best_trained_score'], snap['search_score']), (7, .121, .121))
        self.assertEqual((snap['best_epoch'], snap['best_score']), (0, .16))  # Baseline stays the serving fallback.
        self.assertEqual(snap['trained_bad_epochs'], 1)  # .121 is within min_delta of the .12 reference.
        policy.observe(8, .2)
        self.assertEqual((policy.best_epoch, policy.snapshot()['best_score']), (8, .2))

    def test_patience_counts_trained_epochs_only_after_min_epochs(self):
        policy = RoutingPolicy(patience=2, min_delta=.004, min_epochs=10)
        policy.observe(0, .9)  # A strong baseline must not make every trained epoch "bad".
        for epoch in range(1, 11):
            policy.observe(epoch, .1)
        self.assertEqual((policy.trained_bad_epochs, policy.stopped), (0, False))
        policy.observe(11, .1)
        policy.observe(12, .1)
        self.assertTrue(policy.stopped)
        self.assertEqual((policy.best_trained_epoch, policy.best_epoch), (1, 0))  # Ties keep the earliest epoch.

    def test_replay_is_exact_and_stopped_or_final_checkpoint_adds_no_steps(self):
        scores = [.05, .3, .2, .29, .1, .1]
        live = RoutingPolicy(1, .004, 2)
        live.observe(0, .16)
        history = []
        for epoch, score in enumerate(scores, 1):
            live.observe(epoch, score)
            history.append({'epoch': epoch, 'validation': {'case_accuracy': score}})
        replay = RoutingPolicy(1, .004, 2)
        replay.observe(0, .16)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / 'steps.jsonl').write_text(''.join(json.dumps({'step': i}) + '\n' for i in range(1, 7)))
            rewind_history(out, history, 3, 3, replay)
        self.assertEqual(len(history), 3)
        self.assertEqual((replay.best_trained_epoch, replay.trained_bad_epochs, replay.stopped), (2, 1, True))
        self.assertFalse(resume_adds_steps(replay, 3, 10))
        full = RoutingPolicy(5, .004, 10)
        full.observe(0, .1)
        full.observe(1, .2)
        self.assertFalse(resume_adds_steps(full, 10, 10))
        self.assertTrue(resume_adds_steps(full, 1, 10))

    def test_settings_and_weights(self):
        base = {'validation_callback': {'patience': 5, 'min_delta': .004}}
        self.assertIsNone(policy_settings(base))
        for bad in ({'window_none_loss_weight': .5}, {'training_policy_version': 'v3'},
                    {'training_policy_version': 'laya-routing-v2', 'window_none_loss_weight': 0},
                    {'training_policy_version': 'laya-routing-v2', 'window_none_loss_weight': 1.5}):
            with self.assertRaises(ValueError):
                policy_settings({**base, **bad})
        self.assertEqual(policy_settings({**base, 'training_policy_version': 'laya-routing-v2'}),
                         {'window_none_loss_weight': 1.0, 'min_epochs': 0})
        rows = [{'gold': {QUESTION: {'label': label}}} for label in ('a', 'window_none', 'window_none', 'no_tool')]
        weights, summary = window_none_weights(rows, [0, 1, 1, 2, 3], .25)
        self.assertEqual(weights, [1.0, .25, .25, .25, 1.0])
        self.assertEqual(summary['original_counts'], {'tool': 1, 'window_none': 2, 'no_tool': 1})
        self.assertEqual(summary['sampled_counts'], {'tool': 1, 'window_none': 3, 'no_tool': 1})
        self.assertAlmostEqual(summary['train_mean_weight'], 2.75 / 5)
        check_soft_targets([{'target': [.5, .5, 0], 'markers': [1, 2, 3]}])
        with self.assertRaises(ValueError):
            check_soft_targets([{'target': [.5, .4, 0], 'markers': [1, 2, 3]}])


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('unsloth'),
                     'Requires the decision-training environment')
class WeightedLossTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(0)
        self.logits = torch.randn(3, 4)
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        self.target = torch.tensor([[.5, .5, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])

    def test_unit_weights_equal_native_loss_with_and_without_smoothing(self):
        import torch
        from unsloth.models.decision import _decision_loss
        from jery.model.laya_cuda import weighted_decision_loss
        for smoothing in (0.0, .1):
            native = _decision_loss(self.logits, self.target, self.mask, label_smoothing=smoothing)
            ours = weighted_decision_loss(self.logits, self.target, self.mask, torch.ones(3), 1.0, smoothing)
            self.assertTrue(torch.allclose(native, ours, atol=1e-6))
        # Soft target: 0.5/0.5 is cross entropy against both options, not either alone.
        log_p = torch.log_softmax(self.logits[0, :3], -1)
        first = weighted_decision_loss(self.logits[:1], self.target[:1], self.mask[:1], torch.ones(1), 1.0)
        self.assertAlmostEqual(float(first), float(-(log_p[0] + log_p[1]) / 2), places=5)

    def test_microbatch_of_one_keeps_weight_and_inputs_stay_blind_and_encoder_frozen(self):
        import torch
        from unsloth.models.decision import DecisionDataCollator
        from jery.model.laya_cuda import LossWeightCollator, weighted_loss_step

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Embedding(10, 4).requires_grad_(False)
                self.head = torch.nn.Linear(4, 1)

            def forward(self, **inputs):
                self.seen = set(inputs)
                hidden = self.encoder(inputs['input_ids']).gather(1, inputs['marker_pos'][..., None].expand(-1, -1, 4))
                return self.head(hidden).squeeze(-1), None

        item = {'input_ids': [1, 2, 3, 4], 'markers': [1, 2, 3], 'qtype': 0, 'target': [0, 1, 0], 'keys': ['x']}
        collate = LossWeightCollator(DecisionDataCollator(0))
        grads = {}
        for weight in (1.0, .25):
            torch.manual_seed(1)
            model = Fake()
            loss, _ = weighted_loss_step(model, collate([{**item, 'loss_weight': weight}]), .5, 0.0)
            loss.backward()
            self.assertEqual(model.seen, {'input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'})
            self.assertIsNone(model.encoder.weight.grad)
            grads[weight] = model.head.weight.grad.clone()
        self.assertTrue(torch.allclose(grads[.25] * 4, grads[1.0]))
        self.assertFalse(torch.allclose(grads[.25], grads[1.0]))


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('safetensors'),
                     'Requires the decision-training environment')
class CudaHeadContractTest(unittest.TestCase):
    def test_head_restore_preserves_encoder_and_rejects_incomplete_or_extra_keys(self):
        import torch
        from safetensors.torch import save_file

        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(2, 2)
        model.head = torch.nn.Linear(2, 1)
        model.encoder.requires_grad_(False)
        encoder_before = tensor_hash(model.encoder.named_parameters())
        head = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'head.safetensors'
            save_file(head, str(path))
            with torch.no_grad():
                model.head.weight.add_(1)
            restore_head(model, path)
            self.assertTrue(torch.equal(model.head.weight, head['head.weight']))
            self.assertEqual(tensor_hash(model.encoder.named_parameters()), encoder_before)
            for invalid in ({'head.weight': head['head.weight']},
                            {**head, 'encoder.weight': model.encoder.weight.detach().clone()}):
                save_file(invalid, str(path))
                with self.assertRaises(ValueError):
                    restore_head(model, path)
                self.assertEqual(tensor_hash(model.encoder.named_parameters()), encoder_before)


if __name__ == '__main__':
    unittest.main()
