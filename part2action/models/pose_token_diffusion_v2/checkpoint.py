"""V2-only checkpoints; existing v1 files are rejected, never converted silently."""
from pathlib import Path
import torch

from . import ARCHITECTURE_ID
from .policy import PoseTokenDiffusionV2Policy


def checkpoint_state(model):
    if not isinstance(model,PoseTokenDiffusionV2Policy):
        raise TypeError('V2 checkpoint helpers require a v2 policy')
    return {key:value.detach().cpu() for key,value in model.state_dict().items()
            if not key.startswith('blocks.encoder.backbones.')}


def save_checkpoint(model,path,*,training=None):
    payload = {'architecture':ARCHITECTURE_ID,'checkpoint_schema':1,'config':model.config.to_dict(),
        'model':checkpoint_state(model),'frozen_backbones_external':True,'training':training}
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as handle:
        torch.save(payload,handle)


def load_checkpoint(model,path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('architecture') != ARCHITECTURE_ID or payload.get('checkpoint_schema') != 1:
        raise ValueError('Expected a v2 checkpoint; v1/legacy policies need a separate explicit experiment')
    if payload.get('config') != model.config.to_dict():
        raise ValueError('V2 checkpoint configuration mismatch')
    expected,incoming = checkpoint_state(model),payload.get('model')
    if not isinstance(incoming,dict) or incoming.keys() != expected.keys():
        raise ValueError('V2 checkpoint parameter-key mismatch')
    for key,value in incoming.items():
        if not isinstance(value,torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError('V2 checkpoint tensor mismatch: '+key)
    missing,unexpected = model.load_state_dict(incoming,strict=False)
    if unexpected or any(not key.startswith('blocks.encoder.backbones.') for key in missing):
        raise ValueError('Unexpected v2 checkpoint load result')
    return payload.get('training')
