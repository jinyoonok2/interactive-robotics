"""Read-only reuse of frozen encoder classes, with fresh module instances.

Loading is explicit and lazy. Importing the new package does not load DINO/T5,
touch checkpoints, patch legacy classes, or download pretrained weights.
"""
from __future__ import annotations

import os

import torch
from torch import nn

from .config import PoseTokenDiffusionConfig


class FrozenBackbones(nn.Module):
    def __init__(self, visual: nn.Module, language: nn.Module):
        super().__init__()
        self.visual, self.language = visual, language
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Parent .train() must not enable dropout in the frozen encoders.
        super().train(False)
        return self

    def _apply(self, function):
        super()._apply(function)
        # FrozenT5 places its input tokens on device_str, so retain that explicit
        # placement when the surrounding policy moves to a GPU.
        if hasattr(self.language, 'device_str') and hasattr(self.language, 'model'):
            self.language.model.to(self.language.device_str)
        return self

    @torch.no_grad()
    def encode_visual(self, rgb):
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB (B,T,3,H,W) or (B,3,H,W)')
        b, t, c, h, w = rgb.shape
        patches = self.visual(rgb.reshape(b*t, c, h, w))
        return patches.reshape(b, t, patches.shape[1], patches.shape[2])

    @torch.no_grad()
    def encode_language(self, instructions):
        return self.language(instructions)

    @torch.no_grad()
    def forward(self, rgb, instructions):
        patches = self.encode_visual(rgb)
        words, mask = self.encode_language(instructions)
        return {'visual_history': patches,
                'language_tokens': words, 'language_mask': mask}


def load_frozen_backbones(config: PoseTokenDiffusionConfig, *, img_size=252, text_device='cpu',
                          visual_hub_dir=None, visual_checkpoint=None):
    """Explicitly load existing local pretrained encoders into new instances.

Use a GPU allocation for full-model experiments. Tests inject tiny encoders
instead. RGB/3D alignment, temporal processing, and feature projections belong
to the observation-encoder block, not this frozen-backbone adapter.
"""
    override = os.environ.get('FLAN_T5_MODEL')
    if override and override != config.language_backbone:
        raise ValueError('FLAN_T5_MODEL conflicts with the new model configuration')
    from models.part_grounded_diffusion.verify_sources import verify_legacy_sources
    verify_legacy_sources(only=['part2action/models/backbone.py'])
    from models.backbone import FrozenDINOv2, FrozenT5
    visual = FrozenDINOv2(model_name=config.visual_backbone, img_size=img_size,
                          local_hub_dir=visual_hub_dir, local_ckpt=visual_checkpoint)
    language = FrozenT5(model_name=config.language_backbone, device=text_device)
    return FrozenBackbones(visual, language)
