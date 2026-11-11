"""Predict a TCP offset/orientation around an oracle target-part surface point.

This diagnostic uses annotated part geometry. Its surface proxy is not a measured
finger contact and does not validate deployed localization. Opening remains the
responsibility of the separately frozen low-level actor.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

PARTS = ('front', 'back', 'left', 'right', 'top', 'bottom', 'handle', 'body', 'other')
FEATURE_DIM = 52


def rotation6d(matrix):
    value = np.asarray(matrix)
    return np.concatenate([value[..., :, 0], value[..., :, 1]], axis=-1)


def rotation_matrix6d(value):
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    raw_second = value[..., 3:6]
    second = F.normalize(raw_second - (first * raw_second).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-6)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def pose_features(point, part_cloud, object_cloud, tcp, opening, part_name):
    point = np.asarray(point, dtype=float)
    part = np.asarray(part_cloud, dtype=float)[:, :3]
    obj = np.asarray(object_cloud, dtype=float)[:, :3]
    tcp = np.asarray(tcp, dtype=float)
    part = part[np.isfinite(part).all(axis=1)]
    obj = obj[np.isfinite(obj).all(axis=1)]
    if len(part) < 16 or len(obj) < 16 or not np.isfinite(point).all():
        raise ValueError('Insufficient finite target-part/object geometry')
    nearest = np.argsort(np.linalg.norm(part - point, axis=1))[:min(64, len(part))]
    local = part[nearest] - point
    covariance = np.cov(local.T)
    semantic = np.zeros(len(PARTS))
    semantic[PARTS.index(part_name) if part_name in PARTS else -1] = 1
    features = np.r_[point / .5, (point - tcp[4:]) / .1,
        rotation6d(Rotation.from_quat(tcp[:4]).as_matrix()), opening / .04,
        (obj.mean(0) - point) / .1, (part.mean(0) - point) / .1,
        local.mean(0) / .05, local.std(0) / .05, local.min(0) / .05, local.max(0) / .05,
        covariance[np.triu_indices(3)] / .0025,
        np.ptp(obj, axis=0) / .1, np.ptp(part, axis=0) / .1, semantic]
    if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
        raise ValueError('Invalid pose-head inputs')
    return features.astype(np.float32)


class InteractionPoseHead(nn.Module):
    def __init__(self, hidden=64, max_offset_m=.1):
        super().__init__()
        self.hidden = hidden
        self.max_offset_m = max_offset_m
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9))
        with torch.no_grad():
            self.net[-1].bias[:3].zero_()
            self.net[-1].bias[3:].copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, features, point):
        raw = self.net(features)
        offset = self.max_offset_m * torch.tanh(raw[:, :3])
        matrix = rotation_matrix6d(raw[:, 3:])
        return {'position': point + offset, 'offset': offset, 'rotation': matrix,
                'rotation6d': raw[:, 3:]}

    @torch.no_grad()
    def predict(self, features, point):
        value = self(torch.as_tensor(features[None], dtype=torch.float32),
                     torch.as_tensor(np.asarray(point)[None], dtype=torch.float32))
        matrix = value['rotation'][0].cpu().numpy()
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3) or np.linalg.det(matrix) < .99:
            raise ValueError('Degenerate predicted orientation')
        return np.r_[Rotation.from_matrix(matrix).as_quat(), value['position'][0].cpu().numpy()]
