"""Target TCP-pose conditioning for the existing visual diffusion policy.

This module changes features before the original action head. It never emits,
copies, interpolates, or executes robot commands. GT poses are explicit oracle
inputs in this first experiment; a future estimator can supply the same API.
"""
from __future__ import annotations

import torch
from torch import nn


def quaternion_matrix_xyzw(quaternion):
    q = quaternion.float()
    if q.shape[-1] != 4 or not torch.isfinite(q).all():
        raise ValueError("Expected finite xyzw quaternions")
    norm = q.norm(dim=-1, keepdim=True)
    if (norm < 1e-6).any():
        raise ValueError("Zero quaternion is not an orientation")
    x, y, z, w = (q / norm).unbind(-1)
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def relative_pose_features(tcp_pose, goal_pose, valid, scale_m=.1):
    """10D: TCP-frame goal displacement, rotation6D-minus-identity, valid bit."""
    if scale_m <= 0 or tcp_pose.ndim != 2 or tcp_pose.shape[-1] != 7:
        raise ValueError("Expected TCP (B,7), xyzw then world XYZ, and positive scale")
    if goal_pose.shape != tcp_pose.shape:
        raise ValueError("Goal and current TCP poses must have the same (B,7) shape")
    valid = valid.reshape(-1).bool()
    if valid.shape[0] != tcp_pose.shape[0]:
        raise ValueError("One validity flag is required per sample")
    identity = tcp_pose.new_tensor([0., 0., 0., 1., 0., 0., 0.])
    current = torch.where(valid[:, None], tcp_pose.detach(), identity).float()
    goal = torch.where(valid[:, None], goal_pose.detach(), identity).float()
    # Keep geometric transforms in float32 under mixed-precision training.
    with torch.autocast(device_type=current.device.type, enabled=False):
        rotation = quaternion_matrix_xyzw(current[:, :4])
        target_rotation = quaternion_matrix_xyzw(goal[:, :4])
        displacement = (rotation.transpose(-1, -2) @
                        (goal[:, 4:] - current[:, 4:])[..., None]).squeeze(-1) / scale_m
        relative = rotation.transpose(-1, -2) @ target_rotation
    six = relative[:, :, :2].reshape(-1, 6)
    identity_six = six.new_tensor([1., 0., 0., 1., 0., 0.])
    encoded = torch.cat([displacement, six - identity_six, valid.float()[:, None]], -1)
    if not torch.isfinite(encoded).all():
        raise ValueError("Nonfinite supplied valid target pose")
    return torch.where(valid[:, None], encoded, torch.zeros_like(encoded))


class TargetPoseConditionerV1(nn.Module):
    def __init__(self, hidden_dim, scale_m=.1):
        super().__init__()
        self.scale_m = float(scale_m)
        self.projection = nn.Sequential(nn.Linear(10, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, hidden_dim))
        # Start exactly at the old visual policy in BOTH experiment arms.
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, features, tcp_pose, goal_pose=None, valid=None, enabled=True):
        batch = features.shape[0]
        if not enabled:
            geometry = features.new_zeros(batch, 10)
        else:
            if goal_pose is None or valid is None or tcp_pose is None:
                raise ValueError("Oracle pose conditioning requires explicit target, validity and TCP")
            geometry = relative_pose_features(tcp_pose, goal_pose, valid, self.scale_m)
            geometry = geometry.to(features.device, features.dtype)
        signal = self.projection(geometry) - self.projection(torch.zeros_like(geometry))
        if features.ndim == 3:
            signal = signal[:, None]
        return features + signal, geometry
