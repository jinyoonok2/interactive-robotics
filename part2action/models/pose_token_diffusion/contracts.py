"""Data boundaries between replaceable pose-first blocks.

All 3D outputs use world coordinates in meters. TCP input is quaternion xyzw
followed by world XYZ. TargetPose is a desired gripper/TCP pose, not object pose.
No part masks, future pose labels, or reference time are observation fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class Observation:
    rgb: torch.Tensor
    scene_points_camera: torch.Tensor
    instructions: Sequence[str] | None
    tcp_pose: torch.Tensor
    gripper_opening: torch.Tensor
    plan_slots: torch.Tensor | None = None
    slot_valid_logits: torch.Tensor | None = None
    current_phase: torch.Tensor | None = None
    force_current_phase: bool = False

    def validate(self):
        if self.rgb.ndim not in (4, 5) or self.rgb.shape[-3] != 3:
            raise ValueError('RGB must be (B,3,H,W) or (B,T,3,H,W)')
        b = self.rgb.shape[0]
        if self.scene_points_camera.ndim != 3 or self.scene_points_camera.shape[0] != b or self.scene_points_camera.shape[-1] != 3:
            raise ValueError('Observed camera-frame scene points must be (B,N,3)')
        if self.tcp_pose.shape != (b, 7) or self.gripper_opening.shape != (b, 1):
            raise ValueError('Measured TCP/opening must be (B,7)/(B,1)')
        if self.instructions is None and self.plan_slots is None:
            raise ValueError('Instructions are required unless reusing a predicted plan')
        if self.instructions is not None and (isinstance(self.instructions, str) or len(self.instructions) != b or not all(isinstance(text, str) for text in self.instructions)):
            raise ValueError('One instruction string is required per batch element')
        if (self.plan_slots is None) != (self.slot_valid_logits is None):
            raise ValueError('Cached plan slots and validity logits must be provided together')
        if any(value.device != self.rgb.device for value in (self.scene_points_camera, self.tcp_pose, self.gripper_opening)):
            raise ValueError('All observation tensors must share a device')


@dataclass(frozen=True)
class EncodedObservation:
    visual_tokens: torch.Tensor
    scene_tokens: torch.Tensor
    scene_xyz_world: torch.Tensor
    language_tokens: torch.Tensor
    robot_tokens: torch.Tensor
    tcp_pose: torch.Tensor
    scene_valid_mask: torch.Tensor | None = None
    language_valid_mask: torch.Tensor | None = None
    hierarchy: dict | None = None

    def validate(self, batch_size, hidden_dim):
        for name in ('visual_tokens', 'scene_tokens', 'language_tokens', 'robot_tokens'):
            value = getattr(self, name)
            if value.ndim != 3 or value.shape[0] != batch_size or value.shape[1] == 0 or value.shape[2] != hidden_dim:
                raise ValueError(name+' must be nonempty (B,N,hidden_dim)')
        if self.scene_xyz_world.shape != (*self.scene_tokens.shape[:2], 3):
            raise ValueError('Scene positions must correspond one-to-one to scene tokens')
        if self.tcp_pose.shape != (batch_size, 7):
            raise ValueError('Encoded observation must retain measured TCP pose')


@dataclass(frozen=True)
class GroundedRegion:
    tokens: torch.Tensor
    center_world: torch.Tensor
    logits: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    covariance_world: torch.Tensor | None = None

    def validate(self, batch_size, hidden_dim):
        if self.tokens.ndim != 3 or self.tokens.shape[0] != batch_size or self.tokens.shape[1] == 0 or self.tokens.shape[2] != hidden_dim:
            raise ValueError('Grounded tokens must be nonempty (B,N,hidden_dim)')
        if self.center_world.shape != (batch_size, 3):
            raise ValueError('Region center must be world XYZ (B,3)')


@dataclass(frozen=True)
class TargetPose:
    position_world: torch.Tensor
    rotation_world: torch.Tensor
    valid: torch.Tensor
    applicability_logits: torch.Tensor | None = None

    def validate(self, batch_size):
        if self.position_world.shape != (batch_size, 3) or self.rotation_world.shape != (batch_size, 3, 3):
            raise ValueError('Target TCP position/rotation must be (B,3)/(B,3,3)')
        if self.valid.shape != (batch_size,) or self.valid.dtype != torch.bool:
            raise ValueError('Target validity must be boolean (B,)')
        if not torch.isfinite(self.position_world[self.valid]).all() or not torch.isfinite(self.rotation_world[self.valid]).all():
            raise ValueError('Valid targets must have finite position and rotation')


@dataclass(frozen=True)
class PoseMemory:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    relative_error: torch.Tensor

    def validate(self, batch_size, hidden_dim):
        if self.tokens.ndim != 3 or self.tokens.shape[0] != batch_size or self.tokens.shape[1] == 0 or self.tokens.shape[2] != hidden_dim:
            raise ValueError('Pose memory must be nonempty (B,N,hidden_dim)')
        if self.valid_mask.shape != self.tokens.shape[:2] or self.valid_mask.dtype != torch.bool:
            raise ValueError('Pose memory needs a boolean (B,N) validity mask')
        if self.relative_error.shape != (batch_size, 9):
            raise ValueError('Relative pose error is TCP-frame translation (3) + relative rotation columns (6)')
