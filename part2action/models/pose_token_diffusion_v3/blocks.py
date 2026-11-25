"""Current-skill goal prediction and an independent observation-conditioned monitor."""
import torch
from torch import nn

from models.pose_token_diffusion.blocks.geometry import quaternion_matrix, rotation_matrix
from models.pose_token_diffusion_v2.contracts import SkillGoalBank
from models.pose_token_diffusion_v2.targets import SKILLS as GOAL_SKILLS


class CurrentSkillGoalHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.scale = config.translation_scale_m
        self.features = nn.Sequential(nn.Linear(width * 4 + 12, width), nn.GELU(), nn.LayerNorm(width),
                                      nn.Linear(width, width), nn.GELU())
        self.position = nn.Linear(width, 3)
        self.rotation = nn.Linear(width, 6)
        self.opening = nn.Linear(width, 1)
        self.applicability = nn.Linear(width, 3)
        nn.init.zeros_(self.position.weight)
        nn.init.zeros_(self.position.bias)
        nn.init.normal_(self.rotation.weight, std=.001)
        with torch.no_grad():
            self.rotation.bias.copy_(torch.tensor([1., 0., 0., 1., 0., 0.]))

    def forward(self, encoded, region, skills, skill_start_tcp):
        with torch.autocast(device_type=encoded.tcp_pose.device.type, enabled=False):
            current = quaternion_matrix(encoded.tcp_pose[:, :4])
            local = (current.transpose(-1, -2) @ (region.center_world - encoded.tcp_pose[:, 4:7])[..., None]).squeeze(-1)
            covariance = current.transpose(-1, -2) @ region.covariance_world.float() @ current
        features = self.features(torch.cat([region.tokens[:, 0], encoded.visual_tokens.mean(1),
            encoded.language_tokens[:, 0], encoded.robot_tokens[:, 0],
            local / self.scale, covariance.flatten(1) / self.scale ** 2], -1).to(encoded.visual_tokens.dtype))
        offset = self.scale * self.position(features).float()
        relative_rotation = rotation_matrix(self.rotation(features))
        with torch.autocast(device_type=current.device.type, enabled=False):
            moving = torch.tensor([s.kind == 'move_gripper' for s in skills], device=current.device)
            # A move predicts displacement from the latched skill start, never
            # from an ever-changing current height. Contact goals track their
            # visually grounded part as observations change.
            basis = torch.where(moving[:, None, None], quaternion_matrix(skill_start_tcp[:, :4]), current)
            base = torch.where(moving[:, None], skill_start_tcp[:, 4:7], region.center_world)
            position = base + (basis @ offset[..., None]).squeeze(-1)
            rotation = current @ relative_rotation
        # Skill identity is an explicit planner input, not a second hidden skill classifier.
        ids = torch.tensor([GOAL_SKILLS.index(s.kind) for s in skills], device=position.device)
        logits = 20 * torch.nn.functional.one_hot(ids, len(GOAL_SKILLS)).float()
        return SkillGoalBank(position[:, None], rotation[:, None],
            .04 * self.opening(features).float().sigmoid(),
            torch.ones(len(skills), 1, 3, device=position.device, dtype=torch.bool),
            logits[:, None], self.applicability(features)[:, None])


class SkillProgressMonitor(nn.Module):
    """Predict annotated transition readiness, not verified physical success."""
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.net = nn.Sequential(nn.Linear(width * 4, width), nn.GELU(),
                                 nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1))

    def forward(self, encoded, region):
        features = torch.cat([encoded.visual_tokens.mean(1), region.tokens[:, 0],
                              encoded.language_tokens[:, 0], encoded.robot_tokens[:, 0]], -1)
        return self.net(features).squeeze(-1)
