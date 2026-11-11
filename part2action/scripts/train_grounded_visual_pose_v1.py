"""Train the visual grounding/pose extension without exposing labels to forward.

Uses the existing action/hierarchy losses and dataloader, plus supervised pose,
applicability and soft-region losses. This does not modify or launch GT V1 jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F

from _common import ROOT, load_yaml, resolve_paths, set_seed, save_json
from data.grounded_grasp_pose_v1 import GroundedGraspPoseDatasetV1
from data.partinstruct_loader import PartInstructDataset
from models.grounded_visual_pose_policy_v1 import GroundedVisualPosePolicyV1
from models.part2action_model import Part2ActionModel
from models.pose_conditioning_v1 import quaternion_matrix_xyzw
from train_visual_pose_v1 import (
    build_dataloader, compute_losses as baseline_losses, curriculum_state,
    freeze_prefixes, optimizer_groups, trainable_state_dict)


def build_dataset(data):
    parameters = inspect.signature(PartInstructDataset.__init__).parameters
    kwargs = {k: v for k, v in data.items() if k in parameters and k != 'self'}
    kwargs['hdf5_paths'] = data['train_hdf5']
    return GroundedGraspPoseDatasetV1(**kwargs)


def build_model(cfg):
    parameters = inspect.signature(Part2ActionModel.__init__).parameters
    kwargs = {k: v for k, v in cfg['model'].items() if k in parameters and k != 'self'}
    unknown = set(cfg['model']) - set(kwargs)
    if unknown:
        raise ValueError(f'Unknown model options: {sorted(unknown)}')
    kwargs.update(heads=cfg['heads'], action_chunk=cfg['data']['action_chunk'],
                  n_obs_steps=cfg['data']['n_obs_steps'])
    return GroundedVisualPosePolicyV1(**kwargs, **cfg['predicted_pose'])


def forward_kwargs(batch, hierarchy, probability=1., include_targets=True):
    # Explicit allowlist: future pose/contact/part labels never cross this API.
    kwargs = {key: batch[key] for key in ('agentview_pcd', 'tcp_pose', 'gripper_state')}
    if hierarchy:
        kwargs.update(skill_valid_mask=batch['skill_valid_mask'],
                      oracle_skill_instructions=batch['instruction'],
                      scheduled_sampling_prob=probability)
        if include_targets:
            kwargs['skill_plan'] = batch['skill_plan']
    return kwargs


def compute_losses(out, batch, weights, active=frozenset({'action'})):
    total, parts = baseline_losses(out, batch, weights, active)
    valid = batch['grasp_pose_valid'].bool()
    identity = batch['grasp_pose_target'].new_tensor([0., 0., 0., 1., 0., 0., 0.])
    target = torch.where(valid[:, None], batch['grasp_pose_target'], identity)
    rotation = quaternion_matrix_xyzw(target[:, :4])
    position_error = out['predicted_pose_position_world'] - target[:, 4:7]
    translation = F.smooth_l1_loss(position_error / .1, torch.zeros_like(position_error), reduction='none').mean(-1)
    orientation = (out['predicted_pose_rotation_world'] - rotation).square().sum((-1, -2)) / 6
    denominator = valid.sum().clamp(min=1)
    for name, per_sample in [('pose_position', translation), ('pose_rotation', orientation)]:
        loss = (per_sample * valid).sum() / denominator
        total = total + weights.get(name + '_weight', 1.) * loss
        parts[name] = loss.detach()
    applicability = F.binary_cross_entropy_with_logits(
        out['pose_applicability_logits'].float(), valid.float())
    total = total + weights.get('pose_applicability_weight', .1) * applicability
    parts['pose_applicability'] = applicability.detach()
    parts['pose_valid_fraction'] = valid.float().mean().detach()
    parts['pose_position_mm'] = ((position_error.norm(dim=-1) * valid).sum() / denominator * 1000).detach()
    region_weight = weights.get('grounding_region_weight', 0.)
    if region_weight:
        target_region = batch['point_part_labels'].float()
        counts = target_region.sum(-1)
        region_valid = batch['point_part_valid'].bool() & (counts > 0)
        distribution = target_region / counts.clamp(min=1)[:, None]
        per_sample = -(distribution * out['contact_point_logits'].float().log_softmax(-1)).sum(-1)
        region_loss = (per_sample * region_valid).sum() / region_valid.sum().clamp(min=1)
        total = total + region_weight * region_loss
        parts['grounding_region'] = region_loss.detach()
    parts['total'] = total.detach()
    return total, parts


def load_initialization(model, checkpoint):
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = source['model']
    current = model.state_dict()
    dropped = ('heatmap_head.', 'approach_head.', 'contact_head.',
               'contact_xyz_head.', 'contact_attention.token.')
    unexpected = [k for k in state if k not in current and not k.startswith(dropped)]
    if unexpected:
        raise ValueError(f'Unexpected checkpoint parameters: {unexpected}')
    shared = {k: v for k, v in state.items() if k in current}
    missing, extra = model.load_state_dict(shared, strict=False)
    allowed = ('visual.model.', 'text.model.', 'pose_head.', 'pose_conditioner.')
    incompatible = [k for k in missing if not k.startswith(allowed)]
    if incompatible or extra:
        raise ValueError(f'Incomplete initialization: {incompatible}, {extra}')
    loaded = model.state_dict()
    for key, value in shared.items():
        if not torch.equal(loaded[key].cpu(), value.cpu()):
            raise ValueError(f'Initialization differs at {key}')
    return dict(source=str(checkpoint), shared_tensors=len(shared), new_tensors=missing,
                omitted_unused_tensors=[k for k in state if k not in current])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--override-out')
    parser.add_argument('--expected-hashes', type=Path, help='Frozen submission plan with source_sha256')
    parser.add_argument('--dry-run', action='store_true', help='One real-data optimization step and diffusion sample')
    args = parser.parse_args()
    if args.expected_hashes:
        expected = json.loads(args.expected_hashes.read_text())
        for path, digest in expected['source_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Submitted source/input changed: ' + path)
    cfg = resolve_paths(load_yaml(args.config), ROOT)
    if args.override_out:
        cfg['output_dir'] = args.override_out
    out_dir = Path(cfg['output_dir'])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or cfg['train']['device'] != 'cuda':
        raise RuntimeError('Use a Slurm GPU allocation for the full visual model')
    set_seed(cfg['train']['seed'])
    source_files = [Path(__file__), Path(args.config),
                    ROOT / 'models/grounded_visual_pose_policy_v1.py',
                    ROOT / 'models/grounded_pose_head_v1.py', ROOT / 'data/grounded_grasp_pose_v1.py',
                    ROOT / 'data/oracle_grasp_pose_v1.py', ROOT / 'data/partinstruct_loader.py',
                    ROOT / 'scripts/train_visual_pose_v1.py', ROOT / 'models/part2action_model.py',
                    ROOT / 'models/heads.py', ROOT / 'models/pose_conditioning_v1.py',
                    ROOT / 'models/visual_pose_policy_v1.py', ROOT / 'models/hierarchical.py',
                    ROOT / 'models/temporal.py', ROOT / 'models/backbone.py',
                    ROOT / 'models/diffusion_transformer.py', ROOT / 'data/targets.py',
                    ROOT / 'data/geometry.py', ROOT / 'scripts/_common.py']
    snapshot = out_dir / 'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for index, path in enumerate(source_files):
        hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path, snapshot / f'{index:02d}_{path.name}')
    save_json(out_dir / 'source_hashes.json', hashes)
    save_json(out_dir / 'config_resolved.json', cfg)
    dataset = build_dataset(cfg['data'])
    loader = build_dataloader(dataset, cfg['data'])
    model = build_model(cfg)
    initialization = load_initialization(model, cfg['train']['init_checkpoint'])
    initialization['checkpoint_sha256'] = hashlib.sha256(Path(cfg['train']['init_checkpoint']).read_bytes()).hexdigest()
    save_json(out_dir / 'initialization_verification.json', initialization)
    freeze_prefixes(model, cfg['train']['freeze_prefixes'])
    model.to('cuda').train()
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, cfg['train']['lr'], cfg['train'].get('lr_scales', {})),
        weight_decay=cfg['train']['weight_decay'])
    history = []
    for epoch in range(1 if args.dry_run else cfg['train']['epochs']):
        _, probability, weights = curriculum_state(epoch, cfg)
        running, count = {}, 0
        order = hashlib.sha256()
        for batch in loader:
            order.update(repr(batch['meta']).encode())
            batch = {k: v.to('cuda', non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['train']['amp']):
                out = model(batch['rgb'], batch['task_instruction'], target_action=batch['action_chunk'],
                            **forward_kwargs(batch, model.use_hierarchy, probability))
                loss, parts = compute_losses(out, batch, weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if args.dry_run:
                names = ('pose_head.applicability.weight', 'pose_head.translation.weight',
                         'pose_head.rotation.weight', 'pose_conditioner.projection.2.weight',
                         'contact_attention.point_proj.weight', 'fusion.visual_proj.weight')
                parameters = dict(model.named_parameters())
                gradients = {name:float(parameters[name].grad.float().norm())
                             if parameters[name].grad is not None else 0. for name in names}
                if any(gradients[name] <= 0 for name in (
                    'pose_head.applicability.weight', 'pose_conditioner.projection.2.weight',
                    'fusion.visual_proj.weight')):
                    raise ValueError(f'Missing expected new/shared gradients: {gradients}')
            norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), cfg['train']['grad_clip'])
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite training gradients')
            optimizer.step()
            count += 1
            for key, value in parts.items():
                running[key] = running.get(key, 0.) + float(value)
            if args.dry_run:
                model.eval()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg['train']['amp']):
                    sampled = model(batch['rgb'], batch['task_instruction'],
                                    **{k:batch[k] for k in ('agentview_pcd','tcp_pose','gripper_state')})['action_chunk']
                if sampled.shape != batch['action_chunk'].shape or not torch.isfinite(sampled).all():
                    raise ValueError('Invalid sampled action chunk')
                save_json(out_dir / 'preflight_checks.json',dict(
                    status='passed', loss=float(loss), gradient_norms=gradients,
                    pose_valid_samples=int(batch['grasp_pose_valid'].sum()),
                    rgb_shape=list(batch['rgb'].shape),scene_shape=list(batch['agentview_pcd'].shape),
                    sampled_action_shape=list(sampled.shape),
                    sampled_using='observation-only predicted hierarchy, grounding and pose',
                    frozen_foundations=all(not p.requires_grad for name,p in model.named_parameters()
                                           if name.startswith(('visual.','text.')))))
                break
            if count % cfg['train']['log_every'] == 0:
                print(json.dumps(dict(epoch=epoch, step=count, loss=float(loss))), flush=True)
        history.append(dict(epoch=epoch, optimizer_steps=count, batch_order_sha256=order.hexdigest(),
                            **{k: v / max(1, count) for k, v in running.items()}))
        torch.save(dict(model=trainable_state_dict(model), cfg=cfg, epoch=epoch,
                        history=history, trainable_only=True,
                        architecture='grounded_visual_pose_v1'), out_dir / 'last.pt')
        save_json(out_dir / 'history.json', history)
    save_json(out_dir / 'completion.json', dict(status='completed', dry_run=args.dry_run,
                                               epochs=len(history)))


if __name__ == '__main__':
    main()
