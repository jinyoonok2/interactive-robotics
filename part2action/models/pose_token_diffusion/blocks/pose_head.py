"""Predict a task-specific target TCP pose from local appearance and geometry."""
import torch
from torch import nn
from ..contracts import TargetPose
from .geometry import quaternion_matrix, rotation_matrix


class TargetPoseHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.translation_scale_m
        width = config.hidden_dim
        self.features = nn.Sequential(nn.Linear(width*4+12, width), nn.GELU(),
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.position = nn.Linear(width, 3)
        self.rotation = nn.Linear(width, 6)
        self.applicability = nn.Linear(width, 1)
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)
        nn.init.normal_(self.rotation.weight, std=.001)
        with torch.no_grad():
            self.rotation.bias.copy_(torch.tensor([1., 0., 0., 1., 0., 0.]))

    def forward(self, encoded, region):
        with torch.autocast(device_type=encoded.tcp_pose.device.type, enabled=False):
            rotation = quaternion_matrix(encoded.tcp_pose[:, :4])
            local_center = (rotation.transpose(-1, -2) @ (region.center_world-encoded.tcp_pose[:, 4:7].float())[..., None]).squeeze(-1)
            covariance = region.covariance_world
            if covariance is None:
                covariance = torch.zeros(rotation.shape, device=rotation.device)
            local_covariance = rotation.transpose(-1, -2) @ covariance.float() @ rotation
        features = torch.cat([region.tokens[:, 0], encoded.visual_tokens.mean(1),
            encoded.language_tokens[:, 0], encoded.robot_tokens[:, 0], local_center/self.scale,
            local_covariance.flatten(1)/self.scale**2], -1).to(encoded.visual_tokens.dtype)
        features = self.features(features)
        local_position = local_center+self.scale*self.position(features).float()
        relative_rotation = rotation_matrix(self.rotation(features))
        with torch.autocast(device_type=rotation.device.type, enabled=False):
            world_position = encoded.tcp_pose[:, 4:7].float()+(rotation @ local_position[..., None]).squeeze(-1)
            world_rotation = rotation @ relative_rotation
        return TargetPose(world_position, world_rotation,
            torch.ones(world_position.shape[0], device=world_position.device, dtype=torch.bool),
            self.applicability(features).squeeze(-1))
