"""Small learned goal-pose actor for an isolated low-level control experiment.

The goal affects neural INPUTS only. World commands are decoded from predicted
TCP-relative command offsets, never by adding/copying the supplied goal or by
executing an analytic interpolation. This is not the full visual Part2Action actor.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn


def actor_features(tcp, fingers, goal, indices, close, end):
    tcp = np.atleast_2d(np.asarray(tcp, dtype=float))
    goal = np.broadcast_to(np.asarray(goal, dtype=float), tcp.shape)
    count = len(tcp)
    fingers = np.broadcast_to(np.asarray(fingers).reshape(-1), (count,))
    indices = np.broadcast_to(np.asarray(indices).reshape(-1), (count,))
    current_rotation = Rotation.from_quat(tcp[:, :4])
    rotation_features = current_rotation.as_matrix()[:, :, :2].reshape(count, 6)
    relative_xyz = (goal[:, 4:] - tcp[:, 4:]) / .1
    relative_rotation = (current_rotation.inv() * Rotation.from_quat(goal[:, :4])).as_rotvec() / np.pi
    # Privileged source-event timing is fixed across all conditions in this
    # diagnostic. It is not a learned task/phase/closure-time predictor.
    clock = np.stack([(indices - 1) / max(close - 1, 1),
                      (end - indices) / max(end - 1, 1)], axis=-1)
    features = np.concatenate([tcp[:, 4:] / .5, rotation_features,
                               fingers[:, None] / .04, clock,
                               relative_xyz, relative_rotation], axis=-1)
    if features.shape != (count, 18) or not np.isfinite(features).all():
        raise ValueError('Invalid actor features')
    return features.astype(np.float32)


def action_targets(tcp, actions):
    tcp, actions = np.asarray(tcp), np.asarray(actions)
    rotation = (Rotation.from_quat(tcp[:, :4]).inv()
                * Rotation.from_euler('xyz', actions[:, 3:6])).as_rotvec() / np.pi
    targets = np.concatenate([(actions[:, :3] - tcp[:, 4:]) / .1,
                              rotation, (actions[:, 6:7] > .02).astype(float)], axis=-1)
    return targets.astype(np.float32)


def decode_action(prediction, current_tcp):
    """Decode network outputs using only current state; no goal parameter."""
    prediction = np.asarray(prediction, dtype=float)
    current = np.asarray(current_tcp, dtype=float)
    if prediction.shape != (7,) or not np.isfinite(prediction).all():
        raise ValueError('The neural actor must output seven finite values')
    xyz = current[4:] + .1 * prediction[:3]
    rotation = Rotation.from_quat(current[:4]) * Rotation.from_rotvec(np.pi * prediction[3:6])
    opening = .04 if prediction[6] >= 0 else 0.
    return np.r_[xyz, rotation.as_euler('xyz'), opening]


class PoseGuidedActor(nn.Module):
    def __init__(self, pose_enabled=True, hidden=128):
        super().__init__()
        self.pose_enabled = bool(pose_enabled)
        self.hidden = int(hidden)
        self.net = nn.Sequential(nn.Linear(18, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 7))

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 18:
            raise ValueError('Expected (B,18) state/clock/pose features')
        if not self.pose_enabled:
            features = torch.cat([features[:, :12], torch.zeros_like(features[:, 12:])], dim=-1)
        return self.net(features)

    @torch.no_grad()
    def predict(self, tcp, fingers, goal, index, close, end):
        features = actor_features(tcp, fingers, goal, index, close, end)
        device = next(self.parameters()).device
        output = self(torch.as_tensor(features, device=device))[0].cpu().numpy()
        return decode_action(output, tcp)
