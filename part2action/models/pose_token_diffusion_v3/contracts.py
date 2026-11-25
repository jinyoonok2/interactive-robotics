"""V3 observations explicitly contain the current semantic spec and past anchor.

Annotation skill specs are permitted training inputs. Future geometric targets,
part masks, phase indices, and completion labels are never observation fields.
"""
from dataclasses import dataclass
import torch

from models.pose_token_diffusion.contracts import EncodedObservation, GroundedRegion, PoseMemory
from .skills import SkillSpec


@dataclass(frozen=True)
class SkillObservation:
    rgb: torch.Tensor
    scene_points_camera: torch.Tensor
    tcp_pose: torch.Tensor
    gripper_opening: torch.Tensor
    skills: tuple[SkillSpec, ...]
    skill_start_tcp: torch.Tensor

    def validate(self):
        if self.rgb.ndim not in (4, 5) or self.rgb.shape[-3] != 3 or self.rgb.shape[-1] != self.rgb.shape[-2]:
            raise ValueError('Expected square RGB frames in BCHW or BTCHW form')
        batch = self.rgb.shape[0]
        if self.scene_points_camera.ndim != 3 or self.scene_points_camera.shape[0] != batch or self.scene_points_camera.shape[-1] != 3 or not self.scene_points_camera.shape[1]:
            raise ValueError('Observed camera-frame scene must be nonempty (B,N,3)')
        if self.tcp_pose.shape != (batch, 7) or self.skill_start_tcp.shape != (batch, 7) or self.gripper_opening.shape != (batch, 1):
            raise ValueError('Invalid measured TCP, skill-start anchor, or opening')
        if len(self.skills) != batch or not all(isinstance(s, SkillSpec) for s in self.skills):
            raise ValueError('One validated current SkillSpec is required per observation')
        for tensor in (self.rgb, self.scene_points_camera, self.tcp_pose, self.skill_start_tcp, self.gripper_opening):
            if tensor.device != self.rgb.device or not torch.isfinite(tensor).all():
                raise ValueError('Observation tensors must be finite and share one device')
        if (self.tcp_pose[:, :4].norm(dim=-1) < 1e-6).any() or (self.skill_start_tcp[:, :4].norm(dim=-1) < 1e-6).any():
            raise ValueError('Measured quaternion cannot be zero')
