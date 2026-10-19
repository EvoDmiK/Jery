"""Short CUDA engineering probe; does not resume or select a Hyperopt trial.

PYTHONPATH=src python scripts/probe_laya_cuda.py --output outputs/laya-cuda-probe
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from jery.model.laya_full import balanced_indices, write_json
from jery.model.laya_pilot import QUESTION, load_export, sequence_budget, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/model/laya-balanced-20261018.json'))
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--memory-fraction', type=float, default=0.30)
    args = parser.parse_args()
    if args.steps < 8 or min(args.micro_batch, args.accumulation) < 1 or not 0 < args.memory_fraction <= 0.35:
        parser.error('steps >= 8, positive batches and 0 < memory-fraction <= 0.35 required')
    args.output.mkdir(parents=True, exist_ok=False)
    # Unsloth must patch transformers before its import.
    import unsloth
    import torch
    from transformers import TrainingArguments, TrainerCallback
    from unsloth import FastDecisionModel, DecisionTrainer
    from unsloth.models import decision
    from safetensors.torch import save_file, load_file

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    config = json.loads(args.config.read_text())
    model_dir = Path(config['model_dir'])
    expected = '9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204'
    if sha256(model_dir / 'model.safetensors') != expected:
        raise ValueError('Base model differs from the measured M4 model')
    rows, hashes = load_export(Path(config['export_dir']), splits=('train',))
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(model_dir), max_seq_length=config['max_seq_length'], dtype=torch.float16,
        local_files_only=True, random_state=config['seed'])
    FastDecisionModel.freeze_backbone(model)
    # MLX does not serve the upstream action head; train exactly the same routing components.
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(('head.', 'scorer.', 'type_emb.')))
    items, dataset_report = FastDecisionModel.build_dataset(rows['train'], tokenizer, model)
    if len(items) != len(rows['train']) or dataset_report['skipped']:
        raise ValueError('Training rows were skipped')
    common = decision._laya().common
    budgets = [sequence_budget(tokenizer, common, row['state'], decision._internal(row['questions'][QUESTION]),
                               config['max_seq_length'], model.decision_config['head_max_len'])
               for row in rows['train']]
    if any(b['options_cut'] or b['state_cut'] or b['instruction_cut'] or b['option_text_collisions'] for b in budgets):
        raise ValueError('Input budget loses information')
    indices, balance = balanced_indices(rows['train'], config['train_bucket_ratios'], config['seed'])
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(n.startswith('encoder.') for n in trainable):
        raise AssertionError('Expected head-only training')
    if sum(p.numel() for p in trainable.values()) != 14770945:
        raise AssertionError('Routing head differs from the M4 parameter set')
    before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    frozen = next(p for p in model.encoder.parameters())
    frozen_before = frozen.detach().cpu().clone()
    gpu_before = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                                          '--format=csv,noheader'], text=True).strip()
    # Inference first also warms ModernBERT's rotary/layer-norm path before compilation.
    probe_items = items[:8]
    logits_before, _ = decision._decision_logits(model, tokenizer, probe_items)
    step_times = []

    class Timer(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            for key in ('loss', 'grad_norm'):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise AssertionError(f'Nonfinite {key} at step {state.global_step}')

        def on_step_begin(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_step_end(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - self.started)
            write_json(args.output / 'progress.json', {'steps': state.global_step, 'step_seconds': step_times[-1]})

    training_args = TrainingArguments(
        output_dir=str(args.output / 'trainer'), max_steps=args.steps,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=args.accumulation,
        learning_rate=config['head_learning_rate'], weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'], lr_scheduler_type='linear',
        max_grad_norm=config['max_grad_norm'], bf16=True, seed=config['seed'],
        save_strategy='no', logging_steps=1, report_to=[], dataloader_num_workers=0,
        optim='adamw_torch', disable_tqdm=True)
    trainer = DecisionTrainer(model=model, tokenizer=tokenizer, args=training_args,
                              head_learning_rate=config['head_learning_rate'],
                              label_smoothing=config.get('label_smoothing', 0.0),
                              train_dataset=[items[i] for i in indices], callbacks=[Timer()])
    started = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    if not math.isfinite(result.training_loss):
        raise AssertionError('Nonfinite loss')
    if not any(not torch.equal(before[n], p.detach().cpu()) for n, p in trainable.items()):
        raise AssertionError('Head did not update')
    if not torch.equal(frozen_before, frozen.detach().cpu()):
        raise AssertionError('Frozen encoder changed')
    head = {n: p.detach().cpu().contiguous() for n, p in trainable.items()}
    save_file(head, str(args.output / 'head.safetensors'))
    restored = load_file(str(args.output / 'head.safetensors'))
    if any(not torch.equal(head[n], restored[n]) for n in head):
        raise AssertionError('Head save/reload differs')
    logits_after, _ = decision._decision_logits(model, tokenizer, probe_items)
    if any(not torch.isfinite(x).all() for x in (*logits_before, *logits_after)):
        raise AssertionError('Nonfinite inference logits')
    missing, unexpected = model.load_state_dict(restored, strict=False)
    if unexpected or any(n in trainable for n in missing):
        raise AssertionError('Head cannot be loaded into the model')
    logits_restored, _ = decision._decision_logits(model, tokenizer, probe_items)
    if any(not torch.equal(a, b) for a, b in zip(logits_after, logits_restored)):
        raise AssertionError('Predictions differ after head reload')
    steady = step_times[5:]
    report = {
        'status': 'succeeded', 'scope': 'short engineering probe on synthetic unreviewed data; not full training or accuracy evaluation',
        'backend': 'Unsloth DecisionTrainer / PyTorch CUDA', 'gpu_before': gpu_before,
        'precision': 'FP16 frozen encoder weights, FP32 head weights, BF16 autocast',
        'gpu_after': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                                               '--format=csv,noheader'], text=True).strip(),
        'packages': {p: importlib.metadata.version(p) for p in ('torch', 'unsloth', 'unsloth_zoo', 'transformers', 'laya')},
        'configuration': config, 'probe_source_sha256': sha256(Path(__file__)),
        'base_model_sha256': expected, 'export': hashes, 'dataset': dataset_report,
        'balance': {k: v for k, v in balance.items() if k != 'indices'},
        'max_train_tokens': max(b['sequence_tokens'] for b in budgets),
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'steps': len(step_times), 'micro_batch': args.micro_batch, 'accumulation': args.accumulation,
        'effective_batch': args.micro_batch * args.accumulation, 'training_loss': result.training_loss,
        'train_seconds_including_compile': train_seconds, 'step_seconds': step_times,
        'steady_median_step_seconds': statistics.median(steady),
        'steady_examples_per_second': args.micro_batch * args.accumulation / statistics.median(steady),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'memory_fraction': args.memory_fraction,
        'head_updated': True, 'frozen_encoder_sample_unchanged': True, 'head_save_reload_equal': True,
        'head_reload_logits_equal': True,
    }
    write_json(args.output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
