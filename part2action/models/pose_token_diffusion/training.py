"""Train/smoke the independent model family. Existing trainers are not modified."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data.partinstruct_loader import collate_part2action
from models.part_grounded_diffusion.verify_sources import verify_legacy_sources
from .config import load_config
from .policy import build_policy
from .checkpoint import initialize_from_legacy, load_checkpoint, save_checkpoint
from .dataset import build_dataset, observation_from_batch, provided_pose_from_batch
from .losses import compute_losses

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def train_step(model, batch, optimizer, weights, *, amp=False, grad_clip=1., collect_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    observation = observation_from_batch(batch)
    goal = provided_pose_from_batch(batch) if model.config.pose_source == 'provided' else None
    with torch.autocast(device_type=observation.rgb.device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(observation, provided_pose=goal, target_actions=batch['action_chunk'])
        loss, parts = compute_losses(out, batch, model, weights)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite pose-token training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
    if not torch.isfinite(norm):
        raise ValueError('Nonfinite pose-token training gradients')
    gradients = {name: float(parameter.grad.float().norm()) if parameter.grad is not None else 0.
                 for name, parameter in model.named_parameters() if parameter.requires_grad} if collect_gradients else {}
    optimizer.step()
    return {key: float(value) for key, value in parts.items()}, gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PACKAGE/'configs/train.yaml')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--pose-source', choices=['provided', 'predicted', 'disabled'])
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--smoke', action='store_true', help='One real-data GPU update and full diffusion sample')
    parser.add_argument('--expected-hashes', type=Path)
    args = parser.parse_args()
    if args.expected_hashes:
        expected = json.loads(args.expected_hashes.read_text())['source_sha256']
        for name, digest in expected.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Submitted source changed: '+name)
    verify_legacy_sources()
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation for actual DINO/T5 training or smoke')
    recipe = yaml.safe_load(args.config.read_text())
    if set(recipe) != {'model_config', 'data', 'train', 'losses'}:
        raise ValueError('Expected the independent model/data/train/loss recipe')
    config = load_config(args.config.resolve().parent/recipe['model_config'])
    if args.pose_source:
        config = replace(config, pose_source=args.pose_source)
    if recipe['data']['action_chunk'] != config.action_horizon or recipe['data']['n_obs_steps'] != config.n_obs_steps:
        raise ValueError('Model and data horizons/history differ')
    seed = int(recipe['train']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.out_dir.exists():
        raise FileExistsError('Choose a fresh pose-token run directory')
    args.out_dir.mkdir(parents=True)
    paths = list(PACKAGE.rglob('*.py'))+list((PACKAGE/'configs').glob('*.yaml'))
    paths += [REPO/'part2action/models/backbone.py', REPO/'part2action/models/heads.py',
              REPO/'part2action/models/hierarchical.py', REPO/'part2action/models/temporal.py',
              REPO/'part2action/data/geometry.py', REPO/'part2action/data/partinstruct_loader.py',
              REPO/'part2action/data/oracle_grasp_pose_v1.py', REPO/'part2action/data/targets.py',
              REPO/'part2action/models/part_grounded_diffusion/verify_sources.py',
              REPO/'part2action/models/part_grounded_diffusion/preserved_sources.json',
              REPO/'part2action/scripts/train_pose_token_diffusion.py',
              REPO/'part2action/slurm/pose_token_diffusion.sbatch']
    snapshot = args.out_dir/'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for i, path in enumerate(sorted(set(paths))):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, snapshot/f'{i:03d}_{path.name}')
    dump(args.out_dir/'source_hashes.json', hashes)
    data = dict(recipe['data'])
    if args.smoke:
        data.update(max_demos_per_file=1, num_workers=0, batch_size=2)
    dump(args.out_dir/'config_resolved.json', {'model': config.to_dict(), **recipe, 'data': data, 'smoke': args.smoke})
    dump(args.out_dir/'data_files.json', [{'path': str(Path(name).resolve()), 'bytes': Path(name).stat().st_size,
         'mtime_ns': Path(name).stat().st_mtime_ns} for name in data['train_hdf5']])
    dataset = build_dataset(data)
    available_samples, smoke_indices = len(dataset), None
    if args.smoke:
        smoke_indices = []
        for i, spec in enumerate(dataset._samples):
            target = dataset.target_for(spec)
            if target['valid'] and spec.t < target['closure_action_index']:
                smoke_indices.append(i)
                if len(smoke_indices) == data['batch_size']:
                    break
        if len(smoke_indices) != data['batch_size']:
            raise ValueError('Smoke needs a full batch of real preclosure target-pose samples')
        dataset = Subset(dataset, smoke_indices)
    dump(args.out_dir/'dataset_summary.json', {'available_samples': available_samples,
         'used_samples': len(dataset), 'smoke_sample_indices': smoke_indices})
    loader = DataLoader(dataset, batch_size=data['batch_size'], shuffle=True,
        num_workers=data['num_workers'], pin_memory=data['pin_memory'], collate_fn=collate_part2action,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    if not len(loader):
        raise ValueError('Training dataset does not contain a complete batch')
    model = build_policy(config)
    checkpoint = args.init_checkpoint or Path(recipe['train']['init_checkpoint'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('architecture') == config.architecture:
        receipt = load_checkpoint(model, checkpoint, allow_pose_source_change=True)
        receipt['source_pose_source'] = payload['config']['pose_source']
        receipt['source'] = str(checkpoint.resolve())
        receipt['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    else:
        receipt = initialize_from_legacy(model, checkpoint)
    del payload
    from .checkpoint import checkpoint_state
    initial_digest = hashlib.sha256()
    for name, value in sorted(checkpoint_state(model).items()):
        initial_digest.update(name.encode())
        initial_digest.update(value.contiguous().numpy().tobytes())
    receipt['initial_trainable_state_sha256'] = initial_digest.hexdigest()
    dump(args.out_dir/'initialization.json', receipt)
    model.to('cuda')
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    dump(args.out_dir/'parameter_counts.json', {'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in parameters.values())})
    optimizer = torch.optim.AdamW(parameters.values(), lr=recipe['train']['lr'],
                                  weight_decay=recipe['train']['weight_decay'])
    history = []
    epochs = 1 if args.smoke else int(recipe['train']['epochs'])
    for epoch in range(epochs):
        running, order, count = {}, hashlib.sha256(), 0
        for raw in loader:
            order.update(repr(raw['meta']).encode())
            batch = {key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            parts, gradients = train_step(model, batch, optimizer, recipe['losses'],
                amp=recipe['train']['amp'], grad_clip=recipe['train']['grad_clip'], collect_gradients=args.smoke)
            count += 1
            for name, value in parts.items():
                running[name] = running.get(name, 0.)+value
            if args.smoke:
                model.eval()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=recipe['train']['amp']):
                    sampled = model(observation_from_batch(batch), provided_pose=provided_pose_from_batch(batch)
                                    if config.pose_source == 'provided' else None).actions['action_chunk']
                if sampled.shape != batch['action_chunk'].shape or not torch.isfinite(sampled).all():
                    raise ValueError('Smoke produced invalid action samples')
                required = ['blocks.encoder.scene_projection.0.weight',
                            'blocks.action_decoder.denoiser.layers.layers.0.multihead_attn.in_proj_weight']
                if config.pose_source == 'provided':
                    required += ['blocks.pose_tokenizer.translation.0.weight', 'blocks.pose_tokenizer.orientation.0.weight']
                if config.pose_source == 'predicted':
                    required += ['blocks.pose_head.position.weight', 'blocks.pose_head.rotation.weight']
                if any(gradients[key] <= 0 for key in required):
                    raise ValueError('Smoke missing scene/cross-attention gradients')
                dump(args.out_dir/'smoke.json', {'status': 'passed', 'losses': parts,
                    'gradient_norms': gradients, 'action_shape': list(sampled.shape),
                    'valid_pose_labels': int(batch['pose_supervision_valid'].sum()),
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
                    'note': 'One optimization step only; not a trained policy or performance evaluation.'})
                break
            if count % 50 == 0:
                print(json.dumps({'epoch': epoch+1, 'step': count, 'loss': parts['total']}), flush=True)
        history.append({'epoch': epoch+1, 'optimizer_steps': count, 'batch_order_sha256': order.hexdigest(),
                        **{name: value/count for name, value in running.items()}})
        save_checkpoint(model, args.out_dir/f'epoch_{epoch+1:03d}.pt', training=history[-1])
        dump(args.out_dir/'history.json', history)
    dump(args.out_dir/'completion.json', {'status': 'completed', 'smoke': args.smoke,
        'epochs': len(history), 'optimizer_steps': sum(row['optimizer_steps'] for row in history)})
    print(json.dumps({'status': 'completed', 'out_dir': str(args.out_dir), 'smoke': args.smoke}), flush=True)


if __name__ == '__main__':
    main()
