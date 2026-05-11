"""Top-level model that ties backbone, fusion, and heads together.

The same backbone + fusion is shared by all tracks. The presence of a head is
controlled at construction time via the `heads` set, while action and temporal
variants are selected via config.
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

import torch
import torch.nn as nn

from .backbone import FrozenDINOv2, FrozenT5
from .heads import (
    ActionChunkHead,
    ApproachHead,
    ContactHead2D,
    CrossAttentionFusion,
    DiffusionActionHead,
    HeatmapHead,
)
from .temporal import build_temporal_encoder


_VALID_HEADS = {"heatmap", "contact", "approach", "action"}


class Part2ActionModel(nn.Module):
    def __init__(
        self,
        heads: Iterable[str] = ("heatmap",),
        img_size: int = 252,
        out_size: int = 96,
        action_chunk: int = 8,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_fusion_layers: int = 2,
        text_device: str = "cpu",
        action_head_type: str = "mlp",
        diffusion_steps: int = 50,
        temporal_encoder_type: str = "none",
        n_obs_steps: int = 1,
        temporal_layers: int = 1,
        temporal_heads: int = 4,
    ) -> None:
        super().__init__()
        heads_set: Set[str] = set(heads)
        unknown = heads_set - _VALID_HEADS
        if unknown:
            raise ValueError(f"Unknown heads: {unknown}; valid: {_VALID_HEADS}")
        if not heads_set:
            raise ValueError("At least one head is required")
        self.active_heads = heads_set
        self.action_head_type = action_head_type.lower()
        if self.action_head_type not in {"mlp", "diffusion"}:
            raise ValueError("action_head_type must be 'mlp' or 'diffusion'")
        self.n_obs_steps = max(1, int(n_obs_steps))

        self.visual = FrozenDINOv2(img_size=img_size)
        self.text = FrozenT5(device=text_device)
        self.temporal = build_temporal_encoder(
            encoder_type=temporal_encoder_type,
            embed_dim=self.visual.embed_dim,
            n_obs_steps=self.n_obs_steps,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
        )

        self.fusion = CrossAttentionFusion(
            visual_dim=self.visual.embed_dim,
            text_dim=self.text.embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_fusion_layers,
        )

        if "heatmap" in heads_set:
            self.heatmap_head = HeatmapHead(in_dim=hidden_dim, grid=self.visual.grid, out_size=out_size)
        if "contact" in heads_set:
            self.contact_head = ContactHead2D(in_dim=hidden_dim)
        if "approach" in heads_set:
            self.approach_head = ApproachHead(in_dim=hidden_dim)
        if "action" in heads_set:
            if self.action_head_type == "diffusion":
                self.action_head = DiffusionActionHead(
                    in_dim=hidden_dim,
                    chunk=action_chunk,
                    action_dim=action_dim,
                    num_steps=diffusion_steps,
                )
            else:
                self.action_head = ActionChunkHead(in_dim=hidden_dim, chunk=action_chunk, action_dim=action_dim)

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def forward(self, rgb: torch.Tensor, instructions, target_action: Optional[torch.Tensor] = None):
        text_tok, text_mask = self.text(instructions)
        text_tok = text_tok.to(rgb.device)
        text_mask = text_mask.to(rgb.device)

        with torch.no_grad():
            if rgb.ndim == 5:
                bsz, steps, channels, height, width = rgb.shape
                rgb_flat = rgb.reshape(bsz * steps, channels, height, width)
                visual_flat = self.visual(rgb_flat)
                visual_tok = visual_flat.reshape(bsz, steps, visual_flat.shape[1], visual_flat.shape[2])
            elif rgb.ndim == 4:
                visual_tok = self.visual(rgb)
            else:
                raise RuntimeError(f"Expected RGB shape (B,C,H,W) or (B,T,C,H,W), got {tuple(rgb.shape)}")
        visual_tok = self.temporal(visual_tok)
        fused = self.fusion(visual_tok, text_tok, text_attn_mask=text_mask)

        out: dict = {"fused": fused}
        if "heatmap" in self.active_heads:
            out["heatmap_logits"] = self.heatmap_head(fused).squeeze(1)
        if "contact" in self.active_heads:
            out["contact_xy"] = self.contact_head(fused)
        if "approach" in self.active_heads:
            out["approach_dir"] = self.approach_head(fused)
        if "action" in self.active_heads:
            if self.action_head_type == "diffusion":
                out.update(self.action_head(fused, target_action=target_action))
            else:
                out["action_chunk"] = self.action_head(fused)
        return out
