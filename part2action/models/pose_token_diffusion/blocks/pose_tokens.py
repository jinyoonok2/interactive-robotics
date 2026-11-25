"""Separate translation/orientation memory tokens; no addition to scene features."""
import torch
from torch import nn
from ..contracts import PoseMemory
from .geometry import relative_pose


class RelativePoseTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale, self.width = config.translation_scale_m, config.hidden_dim
        self.translation = nn.Sequential(nn.Linear(4, self.width), nn.GELU(), nn.Linear(self.width, self.width))
        self.orientation = nn.Sequential(nn.Linear(7, self.width), nn.GELU(), nn.Linear(self.width, self.width))

    def forward(self, encoded, target):
        b = encoded.tcp_pose.shape[0]
        if target is None:
            return PoseMemory(encoded.visual_tokens.new_zeros(b, 2, self.width),
                torch.zeros(b, 2, device=encoded.tcp_pose.device, dtype=torch.bool),
                encoded.tcp_pose.new_zeros(b, 9))
        error = relative_pose(encoded.tcp_pose, target)
        confidence = target.valid.float()
        if target.applicability_logits is not None:
            confidence = confidence*target.applicability_logits.float().sigmoid()
        dtype = encoded.visual_tokens.dtype
        translation = self.translation(torch.cat([error[:, :3]/self.scale, confidence[:, None]], -1).to(dtype))
        orientation = self.orientation(torch.cat([error[:, 3:], confidence[:, None]], -1).to(dtype))
        tokens = torch.stack([translation, orientation], 1)*confidence[:, None, None]
        return PoseMemory(tokens, target.valid[:, None].expand(-1, 2), error)
