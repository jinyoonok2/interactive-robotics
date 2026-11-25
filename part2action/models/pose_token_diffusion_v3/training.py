"""Fresh v3 policy training with frozen DINO/T5 and a separate progress query stream."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
import yaml
from diagnostics.storage import result_dir, configure_cache_environment
from diagnostics.run_health import atomic_json

from models.pose_token_diffusion_v2.completion import BalancedCompletionSampler
from .checkpoint import checkpoint_state, save_checkpoint
from .config import load_config
from .dataset import build_dataset, collate, observation_from_batch, ProgressDataset
from .losses import compute_losses
from .metrics import component_rows, write_report
from .policy import build_policy
from .training_state import load_training_state, save_training_state, archive_incomplete_epochs

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def write(path, value):
    atomic_json(path, value)


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_step(model, batch, optimizer, weights=None, *, progress_batch=None, amp=False,
               grad_clip=1., collect_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(batch['rgb'].device.type, dtype=torch.bfloat16, enabled=amp):
        output = model(observation_from_batch(batch), target_actions=batch['action_chunk'])
        loss, metrics = compute_losses(output, batch, model, weights)
        if progress_batch is not None:
            logits = model.progress(observation_from_batch(progress_batch))
            progress = F.binary_cross_entropy_with_logits(logits.float(), progress_batch['progress_target'].float())
            loss = loss + float((weights or {}).get('progress_weight', .5))*progress
            metrics.update(progress=progress.detach(), total=loss.detach())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite v3 training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite v3 gradients')
    gradients = {name: float(p.grad.float().norm()) if p.grad is not None else 0.
                 for name, p in model.named_parameters() if p.requires_grad} if collect_gradients else {}
    optimizer.step()
    return {k: float(v) for k, v in metrics.items()}, gradients


def validate(model, loader, progress_loader, weights, directory, *, epoch, device,
             amp=False, max_batches=None, seed=1042):
    previous = model.training
    model.eval()
    rows, progress_rows, totals, count = [], [], {}, 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            for i, raw in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = move_batch(raw, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    output = model(observation_from_batch(batch), target_actions=batch['action_chunk'])
                    loss, metrics = compute_losses(output, batch, model, weights)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite v3 validation')
                current = component_rows(output, batch)
                rows.extend(current)
                count += len(current)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + float(value)*len(current)
            for i, raw in enumerate(progress_loader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = move_batch(raw, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    probabilities = model.progress(observation_from_batch(batch)).float().sigmoid().cpu().tolist()
                progress_rows.extend({**meta, 'probability': p} for meta, p in zip(batch['meta'], probabilities))
        if not rows or not progress_rows:
            raise ValueError('Validation must cover action and progress observations')
        return write_report(directory, rows, epoch=epoch, losses={k: v/count for k, v in totals.items()},
                            progress_rows=progress_rows)
    finally:
        model.train(previous)


def source_paths():
    paths = list(PACKAGE.rglob('*.py')) + list((PACKAGE/'configs').glob('*.yaml'))
    for package in ('pose_token_diffusion', 'pose_token_diffusion_v2'):
        paths += list(PACKAGE.with_name(package).rglob('*.py'))
    paths += [ROOT/'part2action'/name for name in ('models/backbone.py', 'models/heads.py',
        'models/temporal.py', 'models/hierarchical.py', 'models/diffusion_transformer.py',
        'models/part_grounded_diffusion/verify_sources.py', 'models/part_grounded_diffusion/preserved_sources.json',
        'data/partinstruct_loader.py', 'data/targets.py', 'data/geometry.py',
        'scripts/train_pose_token_diffusion_v3.py', 'scripts/run_pose_token_diffusion_v3.py',
        'slurm/pose_token_diffusion_v3.sbatch', 'slurm/project_storage_env.sh',
        'diagnostics/storage.py', 'diagnostics/run_health.py')]
    return sorted(set(paths))


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(checkpoint_state(model).items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(*, health=None):
    health = health or SimpleNamespace(context={}, event=lambda *args, **kwargs: None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--resume', type=Path, help='Same-run epoch-boundary training snapshot, including optimizer and RNG')
    args = parser.parse_args()
    configure_cache_environment()
    args.out_dir = result_dir(args.out_dir)
    if args.resume and (args.smoke or args.resume.resolve().parent != args.out_dir):
        raise ValueError('Resume needs a training-state file inside this same non-smoke output directory')
    health.event('checking_recipe', output=str(args.out_dir), resume=str(args.resume) if args.resume else None)
    recipe = yaml.safe_load(args.config.read_text())
    if set(recipe) != {'model_config', 'data', 'goal_labels', 'progress', 'train', 'losses'}:
        raise ValueError('Expected an independent v3 recipe')
    options = recipe['train']
    if options.get('initialization') != 'random_policy_frozen_backbones' or 'init_checkpoint' in options:
        raise ValueError('Initial v3 experiments require fresh policy weights and frozen backbones')
    if type(options.get('epochs')) is not int or options['epochs'] < 1:
        raise ValueError('Epoch count must be a positive integer')
    limit = options.get('validation_max_batches')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Validation batch limit must be positive or null')
    timeout = options.get('data_timeout_seconds', 180)
    if type(timeout) not in (int, float) or not 0 < timeout < float('inf'):
        raise ValueError('Data timeout must be finite and positive')
    data = recipe['data']
    manifest = data.get('demo_manifest')
    if not manifest:
        raise ValueError('Prepare an explicit whole-demonstration pilot manifest first')
    model_path = (args.config.resolve().parent/recipe['model_config']).resolve()
    config = load_config(model_path)
    if config.pose_source != 'predicted':
        raise ValueError('Initial v3 pilot trains and deploys predicted geometric goals')
    if (data['action_chunk'], data['n_obs_steps'], data['max_skill_slots']) != (config.action_horizon, config.n_obs_steps, config.max_skill_slots):
        raise ValueError('V3 model/data contracts differ')
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation for pretrained-backbone training')
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError('Choose a fresh v3 output directory')
    if args.resume and not args.out_dir.is_dir():
        raise FileNotFoundError('Cannot resume a missing training directory')
    recipe_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    paths = source_paths() + [args.config.resolve(), model_path, Path(manifest).resolve()]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt = None
    if args.smoke_receipt:
        receipt = json.loads(args.smoke_receipt.read_text())
        if receipt.get('status') != 'passed' or receipt.get('recipe_sha256') != recipe_hash or receipt.get('source_sha256') != source_hashes:
            raise ValueError('Smoke receipt must match this exact recipe, manifest, and source')
    elif not args.smoke:
        raise ValueError('Training requires --smoke-receipt from the same recipe and sources')
    seed = int(options['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    health.event('indexing_datasets')
    train = build_dataset({**data, 'split': 'train'}, recipe['goal_labels'])
    validation = build_dataset({**data, 'split': 'validation'}, recipe['goal_labels'])
    if not train.split_info['demonstrations'] or not validation.split_info['demonstrations'] or not set(map(tuple, train.split_info['demonstrations'])).isdisjoint(map(tuple, validation.split_info['demonstrations'])):
        raise ValueError('Train/validation must contain disjoint complete demonstrations')
    progress_train = ProgressDataset(manifest, split='train', n_obs_steps=config.n_obs_steps, **recipe['progress'])
    progress_validation = ProgressDataset(manifest, split='validation', n_obs_steps=config.n_obs_steps, **recipe['progress'])
    if args.smoke:
        indices, seen = [], set()
        for i, spec in enumerate(train._samples):
            key = spec.file_path, spec.demo_key
            if key in seen:
                continue
            seen.add(key)
            phases = train.targets_for(spec)
            for phase in phases:
                candidate = i + phase['start'] // int(data.get('sample_stride', 1))
                if candidate < len(train) and train._samples[candidate].demo_key == spec.demo_key:
                    indices.append(candidate)
            if len(indices) >= data['batch_size']:
                break
        train_data = Subset(train, indices[:data['batch_size']])
    else:
        train_data = train
    train.close()
    validation.close()
    workers = 0 if args.smoke else data['num_workers']
    # Keep shuffle RNG separate from worker-start RNG. Creating new workers on
    # resume must not change the next epoch's sample order. Dataset reads and
    # targets are deterministic; no stochastic worker augmentations are used.
    generators = {name: torch.Generator().manual_seed(seed+offset) for name, offset in (
        ('shuffle', 0), ('workers', 4000), ('validation', 1000),
        ('progress', 2001), ('progress_validation', 3000))}
    loader = DataLoader(train_data, batch_size=data['batch_size'],
        sampler=RandomSampler(train_data, generator=generators['shuffle']), collate_fn=collate,
        num_workers=workers, generator=generators['workers'], pin_memory=data['pin_memory'],
        **({'multiprocessing_context': 'spawn', 'persistent_workers': True,
            'timeout': timeout} if workers else {}))
    val_loader = DataLoader(validation, batch_size=data['batch_size'], collate_fn=collate,
        generator=generators['validation'])
    progress_sampler = BalancedCompletionSampler(progress_train, len(loader), seed+2000)
    progress_loader = DataLoader(progress_train, batch_size=4,
        sampler=progress_sampler, collate_fn=collate, generator=generators['progress'])
    progress_val_loader = DataLoader(progress_validation, batch_size=data['batch_size'], collate_fn=collate,
        generator=generators['progress_validation'])
    if not args.resume:
        args.out_dir.mkdir(parents=True)
        write(args.out_dir/'config_resolved.json', {'model': config.to_dict(), **recipe, 'smoke': args.smoke})
        write(args.out_dir/'data_split.json', {'train': train.split_info, 'validation': validation.split_info, 'overlap': 0})
        write(args.out_dir/'progress_labels.json', {'train': progress_train.audit, 'validation': progress_validation.audit})
        snapshot = args.out_dir/'source_snapshot'
        snapshot.mkdir()
        for i, path in enumerate(sorted(set(paths))):
            shutil.copy2(path, snapshot/f'{i:03d}_{path.name}')
        write(args.out_dir/'source_hashes.json', source_hashes)
    health.event('loading_frozen_backbones', train_frames=len(train), validation_frames=len(validation))
    print(json.dumps({'stage': 'loading_frozen_backbones', 'train_frames': len(train),
        'validation_frames': len(validation), 'gpu': torch.cuda.get_device_name()}), flush=True)
    model = build_policy(config)
    initial = state_digest(model)
    if receipt and receipt['initial_trainable_state_sha256'] != initial:
        raise ValueError('Smoke and training initializations differ')
    initialization = {'checkpoint_loaded': False,
        'mode': 'random_policy_frozen_backbones', 'initial_trainable_state_sha256': initial,
        'visual_backbone': config.visual_backbone, 'language_backbone': config.language_backbone, 'seed': seed}
    if not args.resume:
        write(args.out_dir/'initialization.json', initialization)
    model.cuda()
    device = torch.device('cuda')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=options['lr'], weight_decay=options['weight_decay'])
    history = []
    identity = {'recipe_sha256': recipe_hash, 'source_sha256': source_hashes}
    start_epoch, cumulative = 1, 0
    if args.resume:
        state = load_training_state(model, optimizer, args.resume, generators=generators,
            progress_sampler=progress_sampler, identity=identity)
        if state['initialization'] != initialization or json.loads((args.out_dir/'initialization.json').read_text()) != initialization:
            raise ValueError('Resumed run initialization differs from its original smoke receipt')
        history = state['history']
        completed = history[-1]['epoch']
        start_epoch, cumulative = completed+1, history[-1]['optimizer_steps']
        archive_incomplete_epochs(args.out_dir, completed)
        write(args.out_dir/'history.json', history)
        events_path = args.out_dir/'resume_events.json'
        events = json.loads(events_path.read_text()) if events_path.exists() else []
        events.append({'checkpoint': str(args.resume.resolve()), 'completed_epoch': completed,
                       'optimizer_steps': cumulative, 'health_directory': str(getattr(health, 'directory', ''))})
        write(events_path, events)
        health.event('resumed_completed_epoch', epoch=completed, optimizer_steps=cumulative)
    def validate_epoch(epoch):
        return validate(model, val_loader, progress_val_loader, recipe['losses'],
            args.out_dir/'validation'/f'epoch_{epoch:03d}', epoch=epoch, device=device,
            amp=options['amp'], max_batches=1 if args.smoke else limit, seed=seed+1000)
    if not args.smoke and not args.resume:
        health.event('validating', epoch=0)
        report = validate_epoch(0)
        history.append({'epoch': 0, 'validation': report['overall']})
        write(args.out_dir/'history.json', history)
    for epoch in range(start_epoch, (1 if args.smoke else options['epochs'])+1):
        health.event('starting_epoch', epoch=epoch, step=0, optimizer_steps=cumulative)
        progress_iterator = iter(progress_loader)
        totals, seen = {}, 0
        health.context.update(stage='loading_next_batch', next_step=1)
        for step, raw in enumerate(loader, 1):
            health.context.update(stage='training_batch', epoch=epoch, step=step,
                                  action_samples=raw['meta'], optimizer_steps=cumulative+step-1)
            progress_raw = next(progress_iterator)
            health.context['progress_samples'] = progress_raw['meta']
            metrics, gradients = train_step(model, move_batch(raw, device), optimizer, recipe['losses'],
                progress_batch=move_batch(progress_raw, device), amp=options['amp'],
                grad_clip=options['grad_clip'], collect_gradients=args.smoke)
            seen += len(raw['meta'])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value*len(raw['meta'])
            if step == 1 or step % options.get('log_every_steps', 50) == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'steps_per_epoch': len(loader), **metrics}), flush=True)
                health.event('training', epoch=epoch, step=step, optimizer_steps=cumulative+step,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved())
            health.context.update(stage='loading_next_batch', next_step=step+1)
            if args.smoke:
                break
        cumulative += step
        health.event('validating', epoch=epoch, optimizer_steps=cumulative)
        report = validate_epoch(epoch)
        row = {'epoch': epoch, 'optimizer_steps': cumulative,
               'training_losses': {k: v/seen for k, v in totals.items()}, 'validation': report['overall']}
        history.append(row)
        save_checkpoint(model, args.out_dir/f'epoch_{epoch:03d}.pt', training=row)
        write(args.out_dir/'history.json', history)
        if not args.smoke:
            health.event('saving_training_state', epoch=epoch, optimizer_steps=cumulative)
            save_training_state(model, optimizer, args.out_dir/'training_latest.pt',
                history=history, generators=generators, progress_sampler=progress_sampler,
                identity=identity, initialization=initialization)
        health.event('epoch_completed', epoch=epoch, optimizer_steps=cumulative)
        print(json.dumps({'stage': 'epoch_completed', **row}), flush=True)
        if args.smoke:
            for prefix in ('blocks.goal_head.position.', 'blocks.goal_head.rotation.', 'blocks.goal_head.opening.',
                           'blocks.grounding.', 'blocks.progress_monitor.', 'blocks.action_decoder.'):
                if not any(v > 0 for k, v in gradients.items() if k.startswith(prefix)):
                    raise ValueError('Missing smoke gradient: ' + prefix)
            raw = next(iter(val_loader))
            batch = move_batch(raw, device)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=options['amp']):
                sampled = model(observation_from_batch(batch)).actions['action_chunk']
            if not torch.isfinite(sampled).all():
                raise ValueError('Nonfinite sampled actions')
            probe = args.out_dir/'recovery_probe.pt'
            save_training_state(model, optimizer, probe, history=history, generators=generators,
                progress_sampler=progress_sampler, identity=identity, initialization=initialization)
            saved_digest = state_digest(model)
            expected_rng = torch.rand(16, device=device)
            load_training_state(model, optimizer, probe, generators=generators,
                progress_sampler=progress_sampler, identity=identity)
            if state_digest(model) != saved_digest or not torch.equal(expected_rng, torch.rand(16, device=device)):
                raise ValueError('GPU training-state recovery roundtrip failed')
            probe.unlink()
            write(args.out_dir/'smoke.json', {'status': 'passed', 'recipe_sha256': recipe_hash,
                'source_sha256': source_hashes, 'initial_trainable_state_sha256': initial,
                'optimizer_rng_recovery_roundtrip': 'passed',
                'losses': metrics, 'gradient_norms': gradients, 'sampled_action_shape': list(sampled.shape),
                'scope': 'One real pretrained-backbone GPU update, held-out diagnostics, action sampling and optimizer/CUDA-RNG recovery roundtrip; not policy performance.'})
    write(args.out_dir/'completion.json', {'status': 'completed', 'smoke': args.smoke, 'optimizer_steps': cumulative})
    for dataset in (train, validation, progress_train, progress_validation):
        dataset.close()


if __name__ == '__main__':
    main()
