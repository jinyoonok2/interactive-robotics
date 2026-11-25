"""Strict v3 checkpoints; no loading or conversion of an earlier policy."""
from pathlib import Path
import torch

from . import ARCHITECTURE_ID
from .policy import PoseTokenDiffusionV3Policy


def checkpoint_state(model):
    if not isinstance(model, PoseTokenDiffusionV3Policy):
        raise TypeError('Expected a v3 policy')
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith('blocks.encoder.backbones.')}


def save_checkpoint(model, path, *, training=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save({'architecture': ARCHITECTURE_ID, 'checkpoint_schema': 1,
            'config': model.config.to_dict(), 'model': checkpoint_state(model),
            'frozen_backbones_external': True, 'training': training}, handle)


def load_checkpoint(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('architecture') != ARCHITECTURE_ID or payload.get('checkpoint_schema') != 1:
        raise ValueError('Expected v3 checkpoint; earlier architectures are incompatible')
    if payload.get('config') != model.config.to_dict():
        raise ValueError('V3 checkpoint configuration mismatch')
    expected, incoming = checkpoint_state(model), payload.get('model')
    if not isinstance(incoming, dict) or expected.keys() != incoming.keys():
        raise ValueError('V3 checkpoint key mismatch')
    for key, value in incoming.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError('V3 checkpoint tensor mismatch: ' + key)
    missing, unexpected = model.load_state_dict(incoming, strict=False)
    if unexpected or any(not k.startswith('blocks.encoder.backbones.') for k in missing):
        raise ValueError('Unexpected v3 load result')
    return payload.get('training')
