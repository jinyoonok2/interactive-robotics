"""Shared RGB/3D encoder with measured-state inputs and a learned skill plan."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from data.geometry import camera_to_world_torch
from models.heads import PointCloudEncoder, PartMaskGate
from models.hierarchical import SkillPlanDecoder, PhaseSelector, SkillTerminationHead
from models.temporal import build_temporal_encoder
from ..contracts import EncodedObservation
from .geometry import sample_image_features


class SharedObservationEncoder(nn.Module):
    def __init__(self, config, backbones):
        super().__init__()
        self.config, self.backbones = config, backbones
        visual_dim, text_dim = backbones.visual.embed_dim, backbones.language.embed_dim
        self.temporal = build_temporal_encoder('transformer', visual_dim, config.n_obs_steps,
            config.temporal_layers, config.attention_heads)
        self.pcd_encoder = PointCloudEncoder(out_dim=visual_dim)
        self.part_gate = PartMaskGate(visual_dim) if config.use_part_gate else None
        self.skill_planner = SkillPlanDecoder(text_dim, config.hidden_dim, config.max_skill_slots,
                                              config.hierarchy_layers, config.attention_heads)
        self.phase_selector = PhaseSelector(visual_dim, text_dim, config.hidden_dim)
        self.termination_head = SkillTerminationHead(visual_dim, text_dim, config.hidden_dim)
        self.visual_projection = nn.Linear(visual_dim+2, config.hidden_dim)
        self.scene_projection = nn.Sequential(nn.Linear(visual_dim*2, config.hidden_dim),
                                              nn.GELU(), nn.LayerNorm(config.hidden_dim))
        self.language_projection = nn.Linear(text_dim, config.hidden_dim)
        self.robot_projection = nn.Sequential(nn.Linear(8, config.hidden_dim), nn.GELU(),
                                              nn.Linear(config.hidden_dim, config.hidden_dim))

    def forward(self, observation):
        raw = self.temporal(self.backbones.encode_visual(observation.rgb))
        gate_logits = None
        if self.part_gate is not None:
            raw, gate_logits = self.part_gate(raw)
        b, patches, channels = raw.shape
        side = math.isqrt(patches)
        if side*side != patches:
            raise ValueError('DINO patches must form a square feature map')
        height, width = observation.rgb.shape[-2:]
        if height != width:
            raise ValueError('This calibrated PartGym adapter currently requires square camera frames')
        camera = observation.scene_points_camera.float()
        if not torch.isfinite(camera).all():
            raise ValueError('Observed scene cloud contains nonfinite coordinates')
        aligned, valid = sample_image_features(raw.transpose(1, 2).reshape(b, channels, side, side),
            camera, height, width, self.config.camera_fov_degrees)
        if not valid.any(-1).all():
            raise ValueError('Each observation needs visible scene points in the calibrated camera')
        with torch.autocast(device_type=camera.device.type, enabled=False):
            world = camera_to_world_torch(camera)
        points, _ = self.pcd_encoder(agentview_pcd=camera, tcp_pose=observation.tcp_pose, return_tokens=True)
        scene = points.masked_fill(~valid[..., None], torch.finfo(points.dtype).min).max(1).values
        if observation.plan_slots is None:
            words, mask = self.backbones.encode_language(observation.instructions)
            slots, validity = self.skill_planner(words.to(raw.device), mask.to(raw.device))
        else:
            slots, validity = observation.plan_slots, observation.slot_valid_logits
            if slots.shape[:2] != (b, self.config.max_skill_slots) or validity.shape != slots.shape[:2]:
                raise ValueError('Cached plan shape does not match the skill-planner configuration')
        counts = (validity.sigmoid() >= .5).sum(-1).clamp(min=1)
        indices = torch.arange(slots.shape[1], device=slots.device)[None]
        slot_valid = indices < counts[:, None]
        current = observation.current_phase
        if current is not None:
            current = current.to(slots.device).clamp(min=0)
            current = torch.minimum(current, counts-1)
        # Supervised logits retain all slots so an initially wrong predicted plan
        # length cannot mask the true training phase out of its own loss.
        phase_logits = self.phase_selector(raw, slots, scene, None, current, True)
        selection_logits = phase_logits.masked_fill(~slot_valid, torch.finfo(phase_logits.dtype).min)
        phase = current if observation.force_current_phase and current is not None else selection_logits.argmax(-1)
        skill = slots.gather(1, phase[:, None, None].expand(-1, 1, slots.shape[-1])).squeeze(1)
        hierarchy = dict(plan_slots=slots, slot_valid_logits=validity, phase_logits=phase_logits,
            predicted_phase=phase, selected_skill=skill,
            termination_logits=self.termination_head(raw, skill, scene))
        if gate_logits is not None:
            hierarchy['gate_logits'] = gate_logits
        grid_size = min(self.config.visual_token_grid, side)
        reduced = F.adaptive_avg_pool2d(raw.transpose(1, 2).reshape(b, channels, side, side), (grid_size, grid_size))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, grid_size, device=raw.device),
                               torch.linspace(-1, 1, grid_size, device=raw.device), indexing='ij')
        xy = torch.stack([xx, yy], -1).reshape(1, -1, 2).expand(b, -1, -1).to(raw.dtype)
        visual = self.visual_projection(torch.cat([reduced.flatten(2).transpose(1, 2), xy], -1))
        fused_points = self.scene_projection(torch.cat([points, aligned.to(points.dtype)], -1))
        language = self.language_projection(torch.cat([skill[:, None], slots], 1))
        language_valid = torch.cat([torch.ones(b, 1, device=raw.device, dtype=torch.bool), slot_valid], 1)
        robot = self.robot_projection(torch.cat([observation.tcp_pose, observation.gripper_opening], -1).to(raw.dtype))[:, None]
        return EncodedObservation(visual, fused_points, world, language, robot,
            observation.tcp_pose, valid, language_valid, hierarchy)

    @torch.no_grad()
    def encode_skill_targets(self, skill_plans):
        device = next(self.robot_projection.parameters()).device
        output = torch.zeros(len(skill_plans), self.config.max_skill_slots,
                             self.backbones.language.embed_dim, device=device)
        flat = [text for plan in skill_plans for text in plan]
        if flat:
            words, mask = self.backbones.encode_language(flat)
            pooled = F.normalize((words*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp(min=1), dim=-1).to(device)
            offset = 0
            for i, plan in enumerate(skill_plans):
                output[i, :len(plan)] = pooled[offset:offset+len(plan)]
                offset += len(plan)
        return output
