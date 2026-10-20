"""Read an existing native CUDA checkpoint and continue one optimizer step in a separate output directory."""
import argparse
import json
from pathlib import Path

from jery.model.laya_cuda import (LossWeightCollator, check_soft_targets, load_cuda_model, policy_settings,
                                  weighted_trainer, window_none_weights)
from jery.model.laya_full import balanced_indices, write_json
from jery.model.laya_pilot import load_export, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='Completed run config.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import unsloth
    import torch
    from transformers import TrainingArguments, TrainerCallback
    from unsloth import FastDecisionModel, DecisionTrainer
    from unsloth.models.decision import DecisionDataCollator, _logits
    config = json.loads(args.config.read_text())
    torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
    if sha256(Path(config['model_dir']) / 'model.safetensors') != config['base_model_sha256']:
        raise ValueError('Base model differs')
    original_state = json.loads((args.checkpoint / 'trainer_state.json').read_text())
    step = original_state['global_step']
    # Checkpoint files are read-only: record all hashes before and after native loading.
    original_hashes = {p.name: sha256(p) for p in args.checkpoint.iterdir() if p.is_file()}
    rows, _ = load_export(Path(config['export_dir']), ('train',))
    model, tokenizer = load_cuda_model(config)
    items, built = FastDecisionModel.build_dataset(rows['train'], tokenizer, model)
    if built['skipped'] or len(items) != len(rows['train']):
        raise ValueError('Skipped training windows')
    indices, _ = balanced_indices(rows['train'], config['train_bucket_ratios'], config['seed'])
    train_items = [items[i] for i in indices]
    trainer_class, trainer_extra = DecisionTrainer, {}
    settings = policy_settings(config)
    weighting = None
    if settings:
        check_soft_targets(train_items)
        weights, weighting = window_none_weights(rows['train'], indices, settings['window_none_loss_weight'])
        train_items = [{**item, 'loss_weight': w} for item, w in zip(train_items, weights)]
        trainer_class = weighted_trainer(DecisionTrainer, weighting['train_mean_weight'])
        trainer_extra = {'data_collator': LossWeightCollator(DecisionDataCollator(tokenizer.pad_token_id))}
    _logits(model, items[:4], tokenizer.pad_token_id, batch_size=4)
    observed = {}

    class Witness(TrainerCallback):
        def on_train_begin(self, training_args, state, control, optimizer=None, **kwargs):
            observed['start_step'] = state.global_step
            observed['optimizer_steps_before'] = sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})

    training_args = TrainingArguments(
        output_dir=str(args.output / 'trainer'), max_steps=step + 1, bf16=True,
        per_device_train_batch_size=config['micro_batch'], gradient_accumulation_steps=config['grad_accum'],
        learning_rate=config['head_learning_rate'], warmup_steps=config['warmup_steps'],
        lr_scheduler_type=config['lr_scheduler'], weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'], seed=config['seed'], optim='adamw_torch',
        save_strategy='no', report_to=[], disable_tqdm=True, logging_steps=1)
    trainer = trainer_class(model=model, tokenizer=tokenizer, args=training_args,
                            train_dataset=train_items, head_learning_rate=config['head_learning_rate'],
                            label_smoothing=config.get('label_smoothing', 0), callbacks=[Witness()], **trainer_extra)
    trainer.train(resume_from_checkpoint=str(args.checkpoint))
    observed['end_step'] = trainer.state.global_step
    observed['optimizer_steps_after'] = sorted({int(s['step']) for s in trainer.optimizer.state.values() if 'step' in s})
    observed['source_checkpoint_unchanged'] = original_hashes == {p.name: sha256(p) for p in args.checkpoint.iterdir() if p.is_file()}
    if (observed['start_step'] != step or observed['end_step'] != step + 1
            or observed['optimizer_steps_before'] != [step] or observed['optimizer_steps_after'] != [step + 1]
            or not observed['source_checkpoint_unchanged']):
        raise AssertionError(observed)
    observed.update(status='succeeded', scope='checkpoint restore compatibility; one extra step at saved LR, no accuracy claim',
                    source_checkpoint=str(args.checkpoint), checkpoint_sha256=original_hashes,
                    training_policy_version=config.get('training_policy_version'), loss_weighting=weighting)
    write_json(args.output / 'report.json', observed)
    print(json.dumps(observed))


if __name__ == '__main__':
    main()
