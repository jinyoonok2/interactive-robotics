"""Optional pose-aware phase progress; the compact recipe retains legacy progress."""
import torch
from torch import nn


class PoseAwareProgress(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.translation_scale_m
        self.network = nn.Sequential(nn.Linear(config.hidden_dim*3+9, config.hidden_dim),
            nn.GELU(), nn.Linear(config.hidden_dim, 1))

    def forward(self, encoded, memory):
        error = memory.relative_error.clone()
        error[:, :3] = error[:, :3]/self.scale
        inputs = torch.cat([encoded.visual_tokens.mean(1), encoded.language_tokens[:, 0],
                            encoded.robot_tokens[:, 0], error], -1).to(encoded.visual_tokens.dtype)
        return self.network(inputs).squeeze(-1)
