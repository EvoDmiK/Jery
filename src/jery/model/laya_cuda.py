"""CUDA adapter for the full-data Laya training contract; invoke through laya_full with backend=cuda."""
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from jery.model.laya_full import (ROOT, SPLITS, QUESTION, JsonLines, Scorer, ValidationPolicy,
                                  balanced_indices, case_gold, evaluate_split, group_cases,
                                  repair_json_lines, write_json)
from jery.model.laya_pilot import MemoryGuard, load_export, sha256


class CudaMemoryGuard(MemoryGuard):
    def used(self):
        import psutil
        return psutil.Process().memory_info().rss


class CudaScorer(Scorer):
    def __init__(self, model, tokenizer, batch_size):
        from unsloth.models import decision
        self.model, self.tokenizer, self.batch_size = model, tokenizer, batch_size
        self.common, self.pad = decision._laya().common, tokenizer.pad_token_id
        self.max_len = int(model.decision_config['max_len'])
        self.head_max_len = int(model.decision_config['head_max_len'])

    def logits(self, items):
        from unsloth.models.decision import _logits
        return _logits(self.model, items, self.pad, batch_size=self.batch_size)


def tensor_hash(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def restore_head(model, path):
    from safetensors.torch import load_file
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    state = load_file(str(path))
    if set(state) != expected:
        raise ValueError('Saved head keys differ from trainable routing parameters')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or expected.intersection(missing):
        raise ValueError('Incomplete head restore')


def load_cuda_model(config):
    import unsloth
    import torch
    from unsloth import FastDecisionModel
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(config['model_dir']), local_files_only=True, max_seq_length=config['max_seq_length'],
        dtype=torch.float16, random_state=config['seed'],
        use_gradient_checkpointing=config['gradient_checkpointing'])
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(('head.', 'scorer.', 'type_emb.')))
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 14770945:
        raise ValueError('Routing parameter set differs from the measured model')
    return model, tokenizer


def rewind_history(out, history, step, epoch, policy):
    """Keep only durable checkpoint history and replay validation selection before deciding to train."""
    history[:] = [h for h in history if h['epoch'] <= epoch]
    (out / 'metrics.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in history))
    kept = [r for r in (out / 'steps.jsonl').read_text().splitlines() if json.loads(r)['step'] <= step]
    (out / 'steps.jsonl').write_text(''.join(r + '\n' for r in kept))
    for record in history:
        policy.observe(record['epoch'], record['validation']['case_accuracy'])


def run(config, run_id, out, log, probe=False, resume=False):
    import unsloth  # Must patch before transformers.
    import torch
    from transformers import TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    from unsloth import FastDecisionModel, DecisionTrainer
    from safetensors.torch import save_file

    if probe:
        raise ValueError('Use scripts/probe_laya_cuda.py for short CUDA probes')
    fraction = config['cuda_memory_fraction']
    if not torch.cuda.is_available() or not 0 < fraction <= 0.35:
        raise ValueError('CUDA and memory fraction in (0, 0.35] required')
    torch.cuda.set_per_process_memory_fraction(fraction)
    started = time.perf_counter()
    weights = ROOT / config['models_dir'] / run_id
    weights.mkdir(parents=True, exist_ok=resume)
    model_dir = Path(config['model_dir'])
    model_sha = sha256(model_dir / 'model.safetensors')
    if model_sha != config['base_model_sha256']:
        raise ValueError('Base model checksum differs')
    rows, export = load_export(Path(config['export_dir']), SPLITS)
    cases = {s: group_cases(rows[s]) for s in SPLITS}
    gold = {s: case_gold(rows[s]) for s in SPLITS}
    report = {'run_id': run_id, 'config': config, 'backend': 'cuda', 'export': export,
              'provenance': {'model': {'revision': config['model_revision'], 'model_safetensors_sha256': model_sha},
                             'packages': {p: importlib.metadata.version(p) for p in
                                          ('torch', 'unsloth', 'unsloth_zoo', 'transformers', 'laya')},
                             'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                             'source_sha256': sha256(Path(__file__)),
                             'gpu_before': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                                                                   '--format=csv,noheader'], text=True).strip()},
              'data': {s: {'windows': len(rows[s]), 'cases': len(cases[s])} for s in SPLITS}}

    model, tokenizer = load_cuda_model(config)
    encoder_sha = tensor_hash(model.encoder.named_parameters())
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    scorer = CudaScorer(model, tokenizer, config['eval_batch_size'])
    report['parameters'] = {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                            'encoder_sha256': encoder_sha,
                            'precision': 'FP16 frozen encoder, FP32 routing head, BF16 autocast'}
    items, budgets = {}, {}
    for split in SPLITS:
        encoded = [scorer.encode(row) for row in rows[split]]
        # Features come from the blind encoder; gold is added only as loss targets by the native builder.
        items[split], built = FastDecisionModel.build_dataset(rows[split], tokenizer, model)
        if built['skipped'] or len(items[split]) != len(rows[split]):
            raise ValueError(f'{split}: skipped training/evaluation rows')
        for item, (blind_item, _) in zip(items[split], encoded):
            if any(item[k] != blind_item[k] for k in ('input_ids', 'markers', 'qtype')):
                raise ValueError('Native dataset features differ from blind encoding')
            item['keys'] = blind_item['keys']
        budgets[split] = {'windows': len(encoded), 'max_tokens': max(b['sequence_tokens'] for _, b in encoded), 'cuts': 0}
    write_json(out / 'token_budget.json', budgets)
    indices, balance = balanced_indices(rows['train'], config['train_bucket_ratios'], config['seed'])
    train_items = [items['train'][i] for i in indices]
    write_json(out / 'balance.json', balance)
    report['balance'] = {k: v for k, v in balance.items() if k != 'indices'}
    policy_config = config['validation_callback']
    policy = ValidationPolicy(policy_config['patience'], policy_config['min_delta'])
    if resume:
        for name in ('metrics.jsonl', 'steps.jsonl'):
            repair_json_lines(out / name)
    history = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()] if resume else []
    metrics_log, steps_log = JsonLines(out / 'metrics.jsonl'), JsonLines(out / 'steps.jsonl')

    def save_head(destination):
        destination.mkdir(parents=True, exist_ok=True)
        save_file({n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad},
                  str(destination / 'trainable.safetensors'))

    def validate():
        return evaluate_split(scorer, rows['validation'], items['validation'], cases['validation'], gold['validation'])

    if resume:
        report['before_validation'] = json.loads((out / 'before_validation.json').read_text())
    else:
        report['before_validation'], _, _ = validate()
        write_json(out / 'before_validation.json', report['before_validation'])
    policy.observe(0, report['before_validation']['case_accuracy'])
    if not resume:
        save_head(weights / 'best-epoch-0')
    log(f"baseline validation: {report['before_validation']['case_accuracy']:.4f}; full export token budgets: {budgets}")
    step_times, epoch_losses = [], []
    train_started = time.perf_counter()

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            control.should_training_stop = policy.stopped

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - self.step_started)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or 'loss' not in logs:
                return
            if any(not math.isfinite(float(logs[k])) for k in ('loss', 'grad_norm') if k in logs):
                raise FloatingPointError('Nonfinite loss or gradient')
            epoch_losses.append(float(logs['loss']))
            steps_log.write({'step': state.global_step, 'epoch': state.epoch, 'loss': float(logs['loss']),
                             'grad_norm': float(logs.get('grad_norm', 0)), 'seconds': time.perf_counter() - train_started,
                             'step_seconds': step_times[-1], 'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9})

        def on_epoch_end(self, args, state, control, **kwargs):
            # Capture training time before callback evaluation and checkpoint I/O.
            train_seconds = sum(step_times)
            validation, predictions, windows = validate()
            epoch = round(state.epoch)
            if policy.observe(epoch, validation['case_accuracy']):
                save_head(weights / f'best-epoch-{epoch}')
            record = {'epoch': epoch, 'steps': state.global_step,
                      'train_loss': statistics.fmean(epoch_losses) if epoch_losses else None,
                      'train_seconds': train_seconds, 'validation': validation, 'callback': policy.snapshot(),
                      'time': datetime.now(timezone.utc).isoformat()}
            metrics_log.write(record)
            history.append(record)
            epoch_losses.clear()
            write_json(out / 'callback.json', policy.snapshot())
            control.should_training_stop = policy.stopped
            log(f"epoch {epoch}: train {train_seconds:.1f}s, val {validation['seconds']['total']:.1f}s, accuracy {validation['case_accuracy']:.4f}")

    args = TrainingArguments(
        output_dir=str(weights / 'trainer'), num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['micro_batch'], gradient_accumulation_steps=config['grad_accum'],
        learning_rate=config['head_learning_rate'], warmup_steps=config['warmup_steps'],
        lr_scheduler_type=config['lr_scheduler'], weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'], bf16=True, seed=config['seed'], optim='adamw_torch',
        save_strategy='epoch', save_total_limit=2, logging_steps=1, report_to=[], disable_tqdm=True,
        dataloader_num_workers=0)
    trainer = DecisionTrainer(model=model, tokenizer=tokenizer, args=args, train_dataset=train_items,
                              head_learning_rate=config['head_learning_rate'],
                              label_smoothing=config.get('label_smoothing', 0), callbacks=[Progress()])
    checkpoint = get_last_checkpoint(args.output_dir) if resume else None
    if resume and checkpoint is None:
        raise ValueError('No native CUDA checkpoint to resume; MLX checkpoints are incompatible')
    checkpoint_epoch = 0
    if resume:
        saved_state = json.loads((Path(checkpoint) / 'trainer_state.json').read_text())
        checkpoint_epoch = int(saved_state['epoch'])
        rewind_history(out, history, saved_state['global_step'], checkpoint_epoch, policy)
        if checkpoint_epoch and not any(h['epoch'] == checkpoint_epoch for h in history):
            raise ValueError('Checkpoint is missing its durable validation record')
    if resume and (policy.stopped or checkpoint_epoch >= config['epochs']):
        # A post-training upload/evaluation failure must not add an optimizer step after an eligible stop.
        trainer._load_from_checkpoint(checkpoint)
        output_metrics = {'resume_without_new_steps': True}
    else:
        output_metrics = trainer.train(resume_from_checkpoint=checkpoint).metrics
    if not history or not (policy.stopped or history[-1]['epoch'] == config['epochs']):
        raise RuntimeError('Training ended before the declared epochs or stopping policy')
    report['train'] = {**output_metrics, 'wall_seconds': time.perf_counter() - train_started,
                       'optimizer_seconds': sum(step_times), 'step_seconds': step_times,
                       'steps_per_epoch': math.ceil(math.ceil(len(train_items) / config['micro_batch']) / config['grad_accum']),
                       'train_windows': len(train_items), 'effective_batch_size': config['micro_batch'] * config['grad_accum']}
    deltas = {n: float((p.detach().cpu() - before[n]).abs().max()) for n, p in model.named_parameters() if n in before}
    report['updates'] = {'changed': sum(v > 0 for v in deltas.values()), 'max_abs_delta': max(deltas.values()),
                         'encoder_unchanged': tensor_hash(model.encoder.named_parameters()) == encoder_sha}
    if not report['updates']['changed'] or not report['updates']['encoder_unchanged']:
        raise RuntimeError('Head/frozen encoder update checks failed')
    completed, selected = history[-1]['epoch'], policy.best_epoch
    save_head(weights / 'last-trained')
    last_head = weights / 'last-trained' / 'trainable.safetensors'
    report['last_trained'] = {'artifact': str(last_head.relative_to(ROOT)), 'artifact_sha256': sha256(last_head),
                              'epoch': completed, 'validation_case_accuracy': history[-1]['validation']['case_accuracy'],
                              'selected_for_serving': selected == completed,
                              'format': 'unsloth-torch-laya-routing-v1', 'base_model_safetensors_sha256': model_sha,
                              'loader': 'load_cuda_model(config), then restore_head(model, artifact)'}
    write_json(last_head.parent / 'head.json', report['last_trained'])
    report['selection'] = {**policy.snapshot(), 'completed_epochs': completed, 'selected_epoch': selected,
                           'max_epochs': config['epochs'], 'reason': 'early_stopping' if policy.stopped else 'max_epochs'}
    restore_head(model, weights / f'best-epoch-{selected}' / 'trainable.safetensors')
    validation, predictions, windows = validate()
    final = weights / 'final'
    save_head(final)
    artifact = final / 'trainable.safetensors'
    write_json(final / 'head.json', {'format': 'unsloth-torch-laya-routing-v1', 'base_revision': config['model_revision'],
                                    'base_model_safetensors_sha256': model_sha, 'trainable_sha256': sha256(artifact),
                                    'epochs': selected, 'completed_epochs': completed,
                                    'loader': 'jery.model.laya_cuda.load_cuda_model(config), then restore_head(model, artifact)'})
    # New instance verifies that the persisted artifact can actually be served.
    del trainer, model, scorer
    gc.collect()
    torch.cuda.empty_cache()
    model, tokenizer = load_cuda_model(config)
    restore_head(model, artifact)
    scorer = CudaScorer(model, tokenizer, config['eval_batch_size'])
    restored, restored_predictions, restored_windows = validate()
    diff = max(abs(a['probabilities'][k] - b['probabilities'][k])
               for a, b in zip(windows, restored_windows) for k in a['probabilities'])
    same = predictions == restored_predictions
    report['restore'] = {'artifact': str(artifact.relative_to(ROOT)), 'artifact_sha256': sha256(artifact),
                         'artifact_bytes': artifact.stat().st_size, 'format': 'unsloth-torch-laya-routing-v1',
                         'same_validation_case_predictions': same, 'validation_max_abs_probability_diff': diff,
                         'encoder_unchanged': tensor_hash(model.encoder.named_parameters()) == encoder_sha}
    if not same or diff > 1e-6 or not report['restore']['encoder_unchanged']:
        raise RuntimeError('Saved head differs after reload')
    report['validation'] = {'epoch': selected, **restored}
    if (out / 'test.json').exists():
        saved = json.loads((out / 'test.json').read_text())
        if saved['artifact_sha256'] != sha256(artifact):
            raise RuntimeError('Existing test was scored with another artifact')
        test, test_predictions, test_windows = saved['metrics'], saved['cases'], saved['windows']
    else:
        test, test_predictions, test_windows = evaluate_split(scorer, rows['test'], items['test'], cases['test'], gold['test'])
        write_json(out / 'test.json', {'artifact_sha256': sha256(artifact), 'metrics': test,
                                       'cases': test_predictions, 'windows': test_windows})
    report['test'] = {'epoch': selected, **test}
    report['timing'] = {'end_to_end_seconds': time.perf_counter() - started,
                        'training_call_seconds_including_validation_and_checkpoints': report['train']['wall_seconds'],
                        'optimizer_seconds': sum(step_times),
                        'steady_median_step_seconds': statistics.median(step_times[5:]) if len(step_times) > 5 else None}
    report['memory'] = {'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
                        'cuda_peak_reserved_gib': torch.cuda.max_memory_reserved() / 2 ** 30}
    return report, {'validation_cases': restored_predictions, 'validation_windows': restored_windows,
                    'test_cases': test_predictions, 'test_windows': test_windows}
