import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jery.model.laya_cuda import restore_head, rewind_history, tensor_hash
from jery.model.laya_full import ValidationPolicy


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
