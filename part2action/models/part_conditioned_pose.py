"""Matched pose heads with optional oracle point and a point-independent origin.

Both variants use the same 52D MLP and part-centroid-relative output decoder.
The first 49 features, their neighborhood selection and the output origin never
read an interaction point or future pose. Only the final three channels differ.
This is a GT-geometry diagnostic, not a deployed visual policy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from .interaction_pose_head import rotation6d, rotation_matrix6d

FEATURE_DIM = 52
VARIANTS = ('without_point', 'with_oracle_point')
GEOMETRY_PRIVILEGES = ('target_part_cloud_world_m', 'object_cloud_world_m')


def _cloud(value):
    cloud = np.asarray(value, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 16 or not np.isfinite(cloud).all():
        raise ValueError('Expected at least 16 finite XYZ geometry points')
    return cloud


def part_pose_features(tcp, opening, part_cloud, object_cloud, interaction_point=None):
    part, obj = _cloud(part_cloud), _cloud(object_cloud)
    tcp = np.asarray(tcp, dtype=np.float64)
    if tcp.shape != (7,) or not np.isfinite(tcp).all():
        raise ValueError('Invalid measured TCP pose')
    origin = part.mean(0)
    centered = part - origin
    # Ties are resolved with current coordinates, not source point order or GT.
    order = np.lexsort((part[:, 2], part[:, 1], part[:, 0], np.sum(centered ** 2, axis=1)))
    local = centered[order[:min(64, len(part))]]
    triangle = np.triu_indices(3)
    point_features = np.zeros(3)
    if interaction_point is not None:
        point = np.asarray(interaction_point)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid supplied point')
        point_features = (point - tcp[4:]) / .1
    features = np.r_[tcp[4:] / .5, rotation6d(Rotation.from_quat(tcp[:4]).as_matrix()),
                     float(np.asarray(opening).item()) / .04,
                     (obj.mean(0) - tcp[4:]) / .1, np.ptp(obj, axis=0) / .1,
                     (origin - tcp[4:]) / .1, np.ptp(part, axis=0) / .1,
                     centered.std(0) / .05, np.cov(centered.T)[triangle] / .0025,
                     local.mean(0) / .05, local.std(0) / .05,
                     local.min(0) / .05, local.max(0) / .05,
                     np.cov(local.T)[triangle] / .0025, point_features]
    if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
        raise ValueError('Invalid point-independent feature vector')
    return features.astype(np.float32), origin.astype(np.float32)


def sample_features(sample, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown pose variant: ' + variant)
    permissions = GEOMETRY_PRIVILEGES + (('reference_interaction_point_world_m',) if variant == 'with_oracle_point' else ())
    inputs = sample.model_inputs(allow_privileged=permissions)
    return part_pose_features(inputs['tcp_pose_xyzw_xyz'], inputs['gripper_opening_m'],
                              inputs['target_part_cloud_world_m'], inputs['object_cloud_world_m'],
                              inputs.get('reference_interaction_point_world_m'))


class PartConditionedPoseHead(nn.Module):
    def __init__(self, hidden=64, position_scale_m=.1):
        super().__init__()
        self.hidden, self.position_scale_m = int(hidden), float(position_scale_m)
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9))
        with torch.no_grad():
            self.net[-1].bias[:3].zero_()
            self.net[-1].bias[3:].copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, features, origin):
        raw = self.net(features)
        # Same linear decoder for BOTH variants; no point-based output shortcut.
        return {'position': origin + self.position_scale_m * raw[:, :3],
                'rotation6d': raw[:, 3:], 'rotation': rotation_matrix6d(raw[:, 3:])}

    @torch.no_grad()
    def predict(self, features, origin):
        value = self(torch.as_tensor(features[None], dtype=torch.float32),
                     torch.as_tensor(origin[None], dtype=torch.float32))
        matrix = value['rotation'][0].cpu().numpy()
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3) or np.linalg.det(matrix) < .99:
            raise ValueError('Degenerate predicted rotation')
        return np.r_[Rotation.from_matrix(matrix).as_quat(), value['position'][0].cpu().numpy()]
