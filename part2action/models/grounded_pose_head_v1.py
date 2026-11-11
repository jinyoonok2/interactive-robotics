"""Shared scene grounding -> local RGB/geometry -> target TCP pose.

Inputs are observation features, measured robot state and inferred skill only.
Part masks, contact proxies and grasp poses are loss targets, never inputs.
This is a single-pose predictor, not a servo or an action decoder.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .pose_conditioning_v1 import quaternion_matrix_xyzw


def rotation_from_six(six):
    """Row-interleaved first two columns, matching pose_conditioning_v1."""
    columns = six.float().reshape(-1, 3, 2)
    raw_first = columns[..., 0]
    fallback_first = torch.zeros_like(raw_first)
    fallback_first[:, 0] = 1
    first = F.normalize(torch.where(raw_first.norm(dim=-1, keepdim=True) > 1e-6,
                                    raw_first, fallback_first), dim=-1)
    second = columns[..., 1] - (first * columns[..., 1]).sum(-1, keepdim=True) * first
    axis = F.one_hot(first.abs().argmin(-1), num_classes=3).to(first.dtype)
    fallback_second = axis - (axis * first).sum(-1, keepdim=True) * first
    second = F.normalize(torch.where(second.norm(dim=-1, keepdim=True) > 1e-6,
                                     second, fallback_second), dim=-1)
    return torch.stack([first, second, torch.cross(first, second, dim=-1)], -1)


class PartRegionGroundingV1(nn.Module):
    """Reuse the baseline contact scorer as soft, skill-conditioned grounding.

The old unused contact-token MLP is omitted. A region-weighted position remains
available for auxiliary contact supervision and as a pose prediction origin.
It is never sent separately to the action decoder.
"""
    def __init__(self, point_dim, visual_dim, skill_dim, hidden_dim):
        super().__init__()
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.skill_proj = nn.Linear(skill_dim, hidden_dim)
        self.geometry_proj = nn.Linear(6, hidden_dim)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, point_tokens, world_xyz, visual_tokens, skill, tcp_pose):
        if not torch.isfinite(world_xyz).all() or not torch.isfinite(point_tokens).all():
            raise ValueError("Grounding requires finite observed scene points/features")
        relative = world_xyz - tcp_pose[:, None, 4:7]
        context = self.visual_proj(visual_tokens.mean(1)) + self.skill_proj(skill)
        logits = self.score(self.point_proj(point_tokens) + self.geometry_proj(
            torch.cat([world_xyz, relative], -1)) + context[:, None]).squeeze(-1)
        weights = logits.float().softmax(-1)
        center = (weights[..., None] * world_xyz.float()).sum(1)
        local = (weights[..., None] * point_tokens).sum(1)
        with torch.autocast(device_type=world_xyz.device.type, enabled=False):
            centered = world_xyz.float() - center[:, None]
            covariance = torch.einsum('bn,bni,bnj->bij', weights, centered, centered)
        return dict(logits=logits, weights=weights, center=center,
                    local_geometry=local, covariance=covariance)


class GroundedTargetPoseHeadV1(nn.Module):
    def __init__(self, point_dim, visual_dim, skill_dim, hidden_dim, scale_m=.1):
        super().__init__()
        self.scale_m = float(scale_m)
        self.query = nn.Linear(point_dim + skill_dim, visual_dim)
        self.local_visual = nn.MultiheadAttention(visual_dim, 4, batch_first=True)
        # Local geometry, local visual context, skill, relative region center,
        # region covariance, measured TCP quaternion/XYZ, and finger opening.
        self.features = nn.Sequential(
            nn.Linear(point_dim + visual_dim + skill_dim + 3 + 9 + 8, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.translation = nn.Linear(hidden_dim, 3)
        self.rotation = nn.Linear(hidden_dim, 6)
        self.applicability = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.translation.weight)
        nn.init.zeros_(self.translation.bias)
        # Nondegenerate initial rotation, with small weights retaining gradients.
        nn.init.normal_(self.rotation.weight, std=.001)
        with torch.no_grad():
            self.rotation.bias.copy_(torch.tensor([1., 0., 0., 1., 0., 0.]))

    def forward(self, region, visual_tokens, skill, tcp_pose, gripper_state):
        local = region['local_geometry'].to(visual_tokens.dtype)
        query = self.query(torch.cat([local, skill], -1))[:, None]
        visual, _ = self.local_visual(query, visual_tokens, visual_tokens, need_weights=False)
        current_rotation = quaternion_matrix_xyzw(tcp_pose[:, :4])
        delta = region['center'] - tcp_pose[:, 4:7].float()
        with torch.autocast(device_type=tcp_pose.device.type, enabled=False):
            local_center = (current_rotation.transpose(-1, -2) @ delta[..., None]).squeeze(-1)
            local_covariance = current_rotation.transpose(-1, -2) @ region['covariance'] @ current_rotation
        robot = torch.cat([tcp_pose, gripper_state.reshape(-1, 1)], -1)
        h = self.features(torch.cat([
            local, visual[:, 0], skill, local_center / self.scale_m,
            local_covariance.flatten(1) / self.scale_m**2, robot], -1).to(local.dtype))
        local_position = local_center + self.scale_m * self.translation(h).float()
        relative_rotation = rotation_from_six(self.rotation(h))
        with torch.autocast(device_type=tcp_pose.device.type, enabled=False):
            world_position = tcp_pose[:, 4:7].float() + (
                current_rotation @ local_position[..., None]).squeeze(-1)
            world_rotation = current_rotation @ relative_rotation
        identity_six = local_position.new_tensor([1., 0., 0., 1., 0., 0.])
        logits = self.applicability(h).squeeze(-1)
        geometry = torch.cat([local_position / self.scale_m,
                              relative_rotation[:, :, :2].reshape(-1, 6) - identity_six,
                              torch.ones_like(logits[:, None])], -1)
        # Predicted support, never a GT phase/validity gate. All samples retain
        # their direct visual/action path even when grasp conditioning is weak.
        geometry = geometry * logits.sigmoid()[:, None]
        return dict(predicted_pose_position_world=world_position,
                    predicted_pose_rotation_world=world_rotation,
                    pose_applicability_logits=logits, pose_features=geometry,
                    local_visual_features=visual[:, 0])


class PredictedPoseConditionerV1(nn.Module):
    """Same 10 -> hidden -> hidden interface as the GT module, with gradients."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(10, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, features, geometry, enabled=True):
        geometry = geometry.to(features.dtype)
        if not enabled:
            geometry = torch.zeros_like(geometry)
        signal = self.projection(geometry) - self.projection(torch.zeros_like(geometry))
        return features + signal[:, None], geometry
