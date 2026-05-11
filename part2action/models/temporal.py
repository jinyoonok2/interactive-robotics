"""Temporal encoders for DINOv2 patch-token histories.

The rest of part2action expects visual tokens shaped (B, N_patches, D). A
temporal encoder consumes multiple frame-token grids (B, T, N_patches, D) and
compresses time back to the same shape, keeping the downstream fusion + heads
unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IdentityTemporalEncoder(nn.Module):
    """No temporal modeling; returns the last frame's patch tokens."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            return tokens
        if tokens.ndim != 4:
            raise RuntimeError(f"Expected (B,N,D) or (B,T,N,D), got {tuple(tokens.shape)}")
        return tokens[:, -1]


class TemporalPatchTransformer(nn.Module):
    """Small transformer over time for each patch location.

    Input:  (B, T, N_patches, D)
    Output: (B, N_patches, D)

    Each spatial patch keeps its identity while the transformer summarizes its
    short motion history. This preserves the patch grid needed by heatmap and
    contact heads.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_steps: int = 8,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.pos = nn.Parameter(torch.zeros(1, self.max_steps, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            return tokens
        if tokens.ndim != 4:
            raise RuntimeError(f"Expected (B,N,D) or (B,T,N,D), got {tuple(tokens.shape)}")

        bsz, steps, n_patches, dim = tokens.shape
        if steps > self.max_steps:
            raise ValueError(f"n_obs_steps={steps} exceeds max_steps={self.max_steps}")

        x = tokens.permute(0, 2, 1, 3).reshape(bsz * n_patches, steps, dim)
        x = x + self.pos[:, :steps].to(dtype=x.dtype)
        x = self.encoder(x)
        x = self.norm(x[:, -1])
        return x.reshape(bsz, n_patches, dim)


def build_temporal_encoder(
    encoder_type: str,
    embed_dim: int,
    n_obs_steps: int,
    num_layers: int = 1,
    num_heads: int = 4,
) -> nn.Module:
    kind = encoder_type.lower()
    if kind in {"none", "identity"} or n_obs_steps <= 1:
        return IdentityTemporalEncoder()
    if kind in {"transformer", "patch_transformer"}:
        return TemporalPatchTransformer(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_steps=max(1, int(n_obs_steps)),
        )
    raise ValueError("temporal_encoder_type must be 'none' or 'transformer'")
