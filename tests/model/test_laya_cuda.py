import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jery.model.laya_cuda import (RoutingPolicy, check_soft_targets, check_ten_tool_rows, policy_settings,
                                  restore_head, resume_adds_steps, rewind_history, task_settings, tensor_hash,
                                  train_fit, window_none_weights)
from jery.model.laya_full import (ValidationPolicy, balanced_indices, case_gold, evaluate_split, group_cases,
                                  tool_macro_f1)
from jery.model.laya_pilot import QUESTION, TEN_TOOL_SCOPE, load_export, sha256

CONFIG = Path(__file__).resolve().parents[2] / 'configs/model'
TOOLS = [f'mcp__s__t{i}' for i in range(10)]


def ten_tool_row(source_id, gold_tool, order=TOOLS):
    """A decision-export-v2 fixed_ten_tools_select_only row: one window, ten tool options, no null options."""
    aliases = {f'T{i}': t for i, t in enumerate(order)}
    label = next(a for a, t in aliases.items() if t == gold_tool)
    return {'state': {'request': 'r', 'context': [], 'candidates': [{'alias': a, 'server': 's', 'name': t, 'summary': t}
                                                                     for a, t in aliases.items()]},
            'questions': {QUESTION: {'type': 'choice', 'instructions': 'i', 'criteria': {a: a for a in aliases}}},
            'gold': {QUESTION: {'label': label, 'probabilities': {label: 1.0}}},
            'metadata': {'source_id': source_id, 'window_index': 0, 'window_count': 1, 'export_id': f'{source_id}:0',
                         'window_tools': list(order), 'alias_to_tool': aliases, 'original_candidate_tools': list(order),
                         'original_decision': 'select_tools', 'acceptable_choices': [{'tool': gold_tool}]}}


class FakeScorer:
    """Picks a fixed tool per source; must never be asked for a second-round window."""
    def __init__(self, picks):
        self.picks = picks

    def top(self, items):
        choices = [next(a for a, t in item['alias_to_tool'].items() if t == self.picks[item['source_id']]) for item in items]
        return choices, [{c: 1.0} for c in choices]

    def __call__(self, windows):
        raise AssertionError('single-window cases must not build a final round')


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


class TenToolTaskTest(unittest.TestCase):
    def setUp(self):
        shuffled = TOOLS[3:] + TOOLS[:3]
        self.rows = {'train': [ten_tool_row(f'tr{i}', TOOLS[i % 10], shuffled if i % 2 else TOOLS) for i in range(20)],
                     'validation': [ten_tool_row('v0', TOOLS[0])], 'test': [ten_tool_row('te0', TOOLS[1])]}
        self.export = {'export_version': 'decision-export-v2', 'export_scope': TEN_TOOL_SCOPE,
                       'evaluation_task': 'ten_tool_choice'}

    def test_ten_options_accepted_on_single_window_path_without_label_leakage(self):
        check_ten_tool_rows(self.rows, self.export)
        rows = self.rows['train']
        with self.assertRaises(ValueError):
            group_cases(rows)  # Legacy windows stay at most 8 tools.
        cases = group_cases(rows, 10)
        flat = json.dumps(cases)
        for leaked in ('gold', 'original_decision', 'acceptable_choices', 'probabilities'):
            self.assertNotIn(leaked, flat)
        items = [{'alias_to_tool': r['metadata']['alias_to_tool'], 'source_id': r['metadata']['source_id']} for r in rows]
        picks = {r['metadata']['source_id']: TOOLS[0] for r in rows}
        metrics, predictions, _ = evaluate_split(FakeScorer(picks), rows, items, cases, case_gold(rows), 'ten_tool_choice')
        self.assertEqual({p['path'] for p in predictions}, {'single'})
        self.assertEqual(metrics['case_accuracy'], .1)
        self.assertEqual(metrics['decision_macro_f1'], 1.0)  # Decision-level F1 is trivially perfect here...
        self.assertEqual(metrics['macro_f1_scope'], 'tools')  # ...so the tool scope is explicit.
        self.assertEqual(metrics['tool_macro_f1'], round((2 * .1 / 1.1) / 10, 4))
        legacy, _, _ = evaluate_split(FakeScorer(picks), rows, items, cases, case_gold(rows))
        self.assertNotIn('tool_macro_f1', legacy)
        self.assertNotIn('macro_f1_scope', legacy)
        self.assertNotIn('per_tool', legacy['case'])

    def test_invalid_candidates_and_contracts_are_rejected(self):
        alternatives = json.loads(json.dumps(self.rows))
        alternatives['test'][0]['metadata']['acceptable_choices'].append({'tool': TOOLS[1], 'action': 'other'})
        check_ten_tool_rows(alternatives, self.export)  # Action alternatives of the same tool are one tool.
        def broken(change):
            row = json.loads(json.dumps(self.rows['validation'][0]))
            change(row)
            return {**self.rows, 'validation': [row]}
        q = lambda r: r['questions'][QUESTION]
        cases = [lambda r: q(r)['criteria'].pop('T9'),
                 lambda r: q(r)['criteria'].update(window_none='w'),
                 lambda r: r['metadata'].update(original_decision='unsupported'),
                 lambda r: r['metadata']['acceptable_choices'].append({'tool': TOOLS[1]}),
                 lambda r: r['gold'][QUESTION].update(label='T1'),
                 lambda r: r['gold'][QUESTION]['probabilities'].update(T1=.5),
                 lambda r: r['metadata'].update(window_count=2)]
        for change in cases:
            with self.assertRaises(ValueError):
                check_ten_tool_rows(broken(change), self.export)
        with self.assertRaises(ValueError):
            check_ten_tool_rows({**self.rows, 'test': self.rows['test'] * 2}, self.export)  # Two windows per source.
        with self.assertRaises(ValueError):
            check_ten_tool_rows(self.rows, {'export_scope': 'exhaustive_original_candidates'})
        with self.assertRaises(ValueError):
            check_ten_tool_rows(self.rows, {**self.export, 'evaluation_task': None})

    def test_load_export_accepts_only_v1_or_the_explicit_v2_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows['train']))
            split = {'train': {'sha256': sha256(folder / 'train.jsonl'), 'rows': 20}}
            for manifest, ok in (({'export_version': 'decision-export-v1'}, True),
                                 ({'export_version': 'decision-export-v2', 'export_scope': TEN_TOOL_SCOPE}, True),
                                 ({'export_version': 'decision-export-v2', 'export_scope': 'other'}, False),
                                 ({'export_version': 'decision-export-v2'}, False)):
                (folder / 'manifest.json').write_text(json.dumps({**manifest, 'splits': split}))
                if not ok:
                    with self.assertRaises(ValueError):
                        load_export(folder, ('train',))
                    continue
                rows, info = load_export(folder, ('train',))
                self.assertEqual(len(rows['train']), 20)
                self.assertEqual('export_scope' in info, manifest['export_version'] == 'decision-export-v2')

    def test_zero_null_buckets_only_when_truly_empty_and_every_row_retained(self):
        ratios = {'tool': 1, 'window_none': 0, 'no_tool': 0, 'clarify': 0}
        indices, summary = balanced_indices(self.rows['train'], ratios, 3407)
        self.assertEqual(sorted(indices), list(range(20)))
        self.assertEqual(summary['sampled_counts'], {'tool': 20, 'window_none': 0, 'no_tool': 0, 'clarify': 0})
        with_null = self.rows['train'] + [{'gold': {QUESTION: {'label': 'window_none'}}}]
        with self.assertRaises(ValueError):
            balanced_indices(with_null, ratios, 3407)  # Would silently discard a source window.
        for bad in ({**ratios, 'tool': 0}, {**ratios, 'no_tool': -1}, {**ratios, 'clarify': True}):
            with self.assertRaises(ValueError):
                balanced_indices(self.rows['train'], bad, 3407)
        with self.assertRaises(ValueError):  # Legacy positive ratios still demand every bucket.
            balanced_indices(self.rows['train'], {'tool': 4, 'window_none': 4, 'no_tool': 1, 'clarify': 1}, 3407)

    def test_tool_macro_f1_counts_wrong_picks_against_both_tools(self):
        gold = {s: {'acceptable_choices': [{'tool': t}]} for s, t in (('a', 'x'), ('b', 'x'), ('c', 'y'))}
        f1, per_tool = tool_macro_f1(gold, [{'source_id': 'a', 'tool': 'x'}, {'source_id': 'b', 'tool': 'y'},
                                            {'source_id': 'c', 'tool': 'y'}])
        self.assertEqual(per_tool['x'], {'precision': 1.0, 'recall': .5, 'f1': .6667, 'support': 2})
        self.assertEqual(per_tool['y'], {'precision': .5, 'recall': 1.0, 'f1': .6667, 'support': 1})
        self.assertEqual(f1, .6667)

    def test_train_fit_reports_last_and_best_without_selecting(self):
        history = [{'epoch': e, 'validation': {'case_accuracy': .1}, **({'train_evaluation': {'case_accuracy': a}} if a else {})}
                   for e, a in ((1, .3), (2, .95), (3, .95), (4, .9))]
        self.assertEqual(train_fit(history), {'last_trained_epoch': 4, 'last_trained_accuracy': .9, 'best_accuracy': .95,
                                              'best_epoch': 2, 'used_for_selection': False})
        self.assertIsNone(train_fit([{'epoch': 1, 'validation': {'case_accuracy': .1}}]))

    def test_task_config_and_legacy_compatibility(self):
        config = json.loads((CONFIG / 'laya-ten-tools-20261020.json').read_text())
        legacy = json.loads((CONFIG / 'laya-cuda-routing-v2-20261019.json').read_text())
        self.assertEqual(task_settings(config), 'ten_tool_choice')
        self.assertEqual(policy_settings(config), {'window_none_loss_weight': 1.0, 'min_epochs': 64})
        self.assertIsNone(task_settings(legacy))
        for bad in ({'max_window_tools': 10}, {'evaluation_task': 'other', 'max_window_tools': 10},
                    {'evaluation_task': 'ten_tool_choice', 'max_window_tools': 8}):
            with self.assertRaises(ValueError):
                task_settings(bad)
        policy = RoutingPolicy(config['validation_callback']['patience'], config['validation_callback']['min_delta'], 64)
        policy.observe(0, .1)
        for epoch in range(1, 65):
            policy.observe(epoch, .1)
        self.assertFalse(policy.stopped)  # Patience never fires inside min_epochs=64...
        self.assertFalse(resume_adds_steps(policy, 64, config['epochs']))  # ...and a completed run adds no epochs.
        self.assertTrue(resume_adds_steps(policy, 63, config['epochs']))


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
