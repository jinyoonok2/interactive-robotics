"""New-family checkpoint I/O with identity checks and no overwrite behavior."""
from __future__ import annotations

from pathlib import Path
import hashlib

import torch

from . import ARCHITECTURE_ID
from .policy import PoseTokenDiffusionPolicy


def _require_new_policy(model):
    if not isinstance(model, PoseTokenDiffusionPolicy):
        raise TypeError('These checkpoint helpers only accept PoseTokenDiffusionPolicy')


def checkpoint_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not key.startswith('blocks.encoder.backbones.')}


def save_checkpoint(model: PoseTokenDiffusionPolicy, path: str | Path, *, training=None):
    _require_new_policy(model)
    payload = {'architecture': ARCHITECTURE_ID, 'checkpoint_schema': 2,
               'config': model.config.to_dict(), 'model': checkpoint_state(model),
               'frozen_backbones_external': True, 'training': training}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects both old checkpoints and earlier new runs.
    with path.open('xb') as handle:
        torch.save(payload, handle)


def load_checkpoint(model: PoseTokenDiffusionPolicy, path: str | Path, *, allow_pose_source_change=False):
    _require_new_policy(model)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('architecture') != ARCHITECTURE_ID or payload.get('checkpoint_schema') != 2:
        raise ValueError('Checkpoint is not an implemented pose_token_diffusion_v1 checkpoint; use the explicit initializer for legacy weights')
    previous = dict(payload.get('config') or {})
    current = model.config.to_dict()
    if allow_pose_source_change:
        previous.pop('pose_source', None)
        current.pop('pose_source', None)
    if previous != current:
        raise ValueError('Checkpoint configuration differs from the requested new model')
    incoming, expected = payload.get('model'), checkpoint_state(model)
    if not isinstance(incoming, dict) or incoming.keys() != expected.keys():
        raise ValueError('Checkpoint parameter keys do not match the new model blocks')
    # load_state_dict can modify matching tensors before reporting an error.
    # Check every shape/type first so incompatible files are rejected unchanged.
    for key, value in incoming.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError('Checkpoint parameter shape/dtype differs: '+key)
    missing, unexpected = model.load_state_dict(incoming, strict=False)
    if unexpected or any(not key.startswith('blocks.encoder.backbones.') for key in missing):
        raise ValueError('Unexpected checkpoint load result')
    return {'architecture': ARCHITECTURE_ID, 'loaded_tensors': len(incoming)}


def initialize_from_legacy(model, path):
    """Explicit, shape-checked warm start; original files/weights are read only.

    Retain compatible temporal/PointNet/part/skill modules and action self-attention
    and feedforward weights. Cross-attention, token fusion and pose blocks are new.
    This is an initialization transfer, not resuming the original architecture.
    """
    _require_new_policy(model)
    path = Path(path)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    source = payload.get('model', {})
    if not any(key.startswith('skill_planner.') for key in source) or 'action_head.net.input.weight' not in source:
        raise ValueError('Expected a hierarchical visual Transformer diffusion initialization')
    target = model.state_dict()
    for name in ('action_scale', 'action_offset', 'betas', 'alphas', 'alpha_bars'):
        old_key, new_key = 'action_head.'+name, 'blocks.action_decoder.'+name
        if old_key not in source or source[old_key].shape != target[new_key].shape or not torch.allclose(source[old_key].float(), target[new_key].float(), atol=1e-7, rtol=1e-5):
            raise ValueError('Legacy action normalization/diffusion schedule differs: '+name)
    transferred, mapping, skipped = {}, {}, []
    for key, value in source.items():
        destination = None
        if key.startswith(('temporal.', 'pcd_encoder.', 'part_gate.', 'skill_planner.', 'phase_selector.', 'termination_head.')):
            destination = 'blocks.encoder.'+key
        elif key.startswith('action_head.net.encoder.layers.'):
            suffix = key[len('action_head.net.encoder.layers.'):]
            suffix = suffix.replace('.norm2.', '.norm3.')
            destination = 'blocks.action_decoder.denoiser.layers.layers.'+suffix
        elif key.startswith('action_head.net.'):
            destination = 'blocks.action_decoder.denoiser.'+key[len('action_head.net.'):]
        elif key.startswith('action_head.time.'):
            destination = 'blocks.action_decoder.time.'+key[len('action_head.time.'):]
        elif key in {'action_head.action_scale', 'action_head.action_offset', 'action_head.betas', 'action_head.alphas', 'action_head.alpha_bars'}:
            destination = 'blocks.action_decoder.'+key[len('action_head.'):]
        if destination in target and value.shape == target[destination].shape and value.dtype == target[destination].dtype:
            transferred[destination], mapping[key] = value, destination
        elif destination is not None:
            skipped.append(key)
    for prefix in ('blocks.encoder.pcd_encoder.', 'blocks.encoder.skill_planner.', 'blocks.action_decoder.denoiser.input.'):
        if not any(key.startswith(prefix) for key in transferred):
            raise ValueError('Initialization is incompatible with required block: '+prefix)
    model.load_state_dict(transferred, strict=False)
    for key, value in transferred.items():
        if not torch.equal(model.state_dict()[key].cpu(), value.cpu()):
            raise ValueError('Initialization transfer differs at '+key)
    return {'source': str(path.resolve()), 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'loaded_tensors': len(transferred), 'mapping': mapping, 'incompatible_omitted': skipped,
            'new_tensors': [key for key in checkpoint_state(model) if key not in transferred],
            'note': 'Only compatible blocks transferred; new pose/feature fusion/cross-attention weights are newly initialized.'}
