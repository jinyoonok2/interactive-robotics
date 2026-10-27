"""Bidirectional action-token denoiser using the existing pooled policy context."""
from __future__ import annotations

import torch
from torch import nn


class ActionTokenDenoiser(nn.Module):
    """Attend across noisy action steps; observations are already fused upstream.

    This deliberately retains the MLP denoiser's pooled conditioning input.
    Separate visual/language/contact memory tokens are a different experiment.
    Diffusion predicts all steps together, so no autoregressive mask is used.
    """

    def __init__(self, action_dim: int, chunk: int, hidden: int,
                 layers: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if layers < 1 or heads < 1 or hidden % heads:
            raise ValueError("Transformer requires positive layers/heads and hidden divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("Transformer dropout must be in [0,1)")
        self.chunk = int(chunk)
        self.input = nn.Linear(action_dim, hidden)
        self.position = nn.Parameter(torch.empty(1, chunk, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        # PyTorch clones initial layer weights; initialize layers independently.
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, action_dim))

    def forward(self, noisy_action, condition, diffusion_time):
        if noisy_action.ndim != 3 or noisy_action.shape[1] != self.chunk:
            raise ValueError("Expected noisy actions (B, configured horizon, action_dim)")
        tokens = self.input(noisy_action)
        tokens = (tokens + self.position.to(tokens.dtype)
                  + condition[:, None, :] + diffusion_time[:, None, :])
        return self.output(self.encoder(tokens))
