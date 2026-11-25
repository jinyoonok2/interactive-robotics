"""Visual/scene encoding plus explicit current-skill conditioning, without plan slots."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from data.geometry import camera_to_world_torch
from models.heads import PointCloudEncoder, SkillConditionedPartGate
from models.temporal import build_temporal_encoder
from models.pose_token_diffusion.contracts import EncodedObservation
from models.pose_token_diffusion.blocks.geometry import sample_image_features
from .skills import SKILLS, INTERACTIONS, DIRECTIONS


class SkillObservationEncoder(nn.Module):
    def __init__(self, config, backbones):
        super().__init__()
        self.config, self.backbones = config, backbones
        width, visual_dim = config.hidden_dim, backbones.visual.embed_dim
        self.temporal = build_temporal_encoder('transformer', visual_dim, config.n_obs_steps,
                                               config.temporal_layers, config.attention_heads)
        self.pcd_encoder = PointCloudEncoder(out_dim=visual_dim)
        self.word_projection = nn.Linear(backbones.language.embed_dim, width)
        self.skill_embedding = nn.Embedding(len(SKILLS), width)
        self.interaction_embedding = nn.Embedding(len(INTERACTIONS), width)
        self.direction_embedding = nn.Embedding(len(DIRECTIONS), width)
        self.distance_projection = nn.Linear(2, width)
        self.skill_norm = nn.LayerNorm(width)
        self.part_gate = SkillConditionedPartGate(visual_dim, width) if config.use_part_gate else None
        self.visual_projection = nn.Linear(visual_dim + 2, width)
        self.scene_projection = nn.Sequential(nn.Linear(visual_dim * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.robot_projection = nn.Sequential(nn.Linear(15, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, observation):
        words, mask = self.backbones.encode_language([s.language for s in observation.skills])
        device = observation.rgb.device
        words, mask = self.word_projection(words.to(device)), mask.to(device).bool()
        pooled = (words * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        ids = lambda names, key: torch.tensor([names.index(getattr(s, key)) for s in observation.skills], device=device)
        distance = torch.tensor([[s.distance_m or 0., float(s.distance_m is not None)] for s in observation.skills], device=device)
        skill = self.skill_norm(pooled + self.skill_embedding(ids(SKILLS, 'kind')) +
            self.interaction_embedding(ids(INTERACTIONS, 'interaction')) +
            self.direction_embedding(ids(DIRECTIONS, 'direction')) + self.distance_projection(distance.to(words.dtype)))
        raw = self.temporal(self.backbones.encode_visual(observation.rgb))
        gate_logits = None
        if self.part_gate is not None:
            raw, gate_logits = self.part_gate(raw, skill.to(raw.dtype))
        batch, patches, channels = raw.shape
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError('DINO patches must form a square grid')
        height, width = observation.rgb.shape[-2:]
        aligned, valid = sample_image_features(raw.transpose(1, 2).reshape(batch, channels, side, side),
            observation.scene_points_camera, height, width, self.config.camera_fov_degrees)
        if not valid.any(-1).all():
            raise ValueError('Each observation requires visible scene points')
        with torch.autocast(device_type=device.type, enabled=False):
            world = camera_to_world_torch(observation.scene_points_camera.float())
        points, _ = self.pcd_encoder(agentview_pcd=observation.scene_points_camera,
                                      tcp_pose=observation.tcp_pose, return_tokens=True)
        scene = self.scene_projection(torch.cat([points, aligned.to(points.dtype)], -1))
        grid = min(self.config.visual_token_grid, side)
        reduced = F.adaptive_avg_pool2d(raw.transpose(1, 2).reshape(batch, channels, side, side), (grid, grid))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, grid, device=device), torch.linspace(-1, 1, grid, device=device), indexing='ij')
        xy = torch.stack([xx, yy], -1).reshape(1, -1, 2).expand(batch, -1, -1).to(raw.dtype)
        visual = self.visual_projection(torch.cat([reduced.flatten(2).transpose(1, 2), xy], -1))
        robot = self.robot_projection(torch.cat([observation.tcp_pose, observation.gripper_opening,
                                                 observation.skill_start_tcp], -1).to(raw.dtype))[:, None]
        context = torch.cat([skill[:, None], words], 1).to(visual.dtype)
        context_valid = torch.cat([torch.ones(batch, 1, device=device, dtype=torch.bool), mask], 1)
        hierarchy = {'selected_skill': skill, 'gate_logits': gate_logits}
        return EncodedObservation(visual, scene, world, context, robot, observation.tcp_pose,
                                  valid, context_valid, hierarchy)
