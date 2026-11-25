"""V2 encoder: query the same completion MLP for every learned skill slot.

Parameters/initialization match the shared encoder. The extra logits allow
completion supervision to name its query slot without forcing the actor's phase.
The v1 implementation is kept unchanged.
"""
import math
import torch
from torch.nn import functional as F

from data.geometry import camera_to_world_torch
from models.pose_token_diffusion.blocks.encoder import SharedObservationEncoder
from models.pose_token_diffusion.blocks.geometry import sample_image_features
from models.pose_token_diffusion.contracts import EncodedObservation


class CompletionObservationEncoder(SharedObservationEncoder):
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
            raise ValueError('This calibrated PartGym adapter requires square camera frames')
        camera = observation.scene_points_camera.float()
        if not torch.isfinite(camera).all():
            raise ValueError('Observed scene cloud contains nonfinite coordinates')
        aligned, valid = sample_image_features(raw.transpose(1, 2).reshape(b, channels, side, side),
            camera, height, width, self.config.camera_fov_degrees)
        if not valid.any(-1).all():
            raise ValueError('Each observation needs visible scene points')
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
                raise ValueError('Cached plan shape differs from configuration')
        counts = (validity.sigmoid() >= .5).sum(-1).clamp(min=1)
        slot_valid = torch.arange(slots.shape[1], device=slots.device)[None] < counts[:, None]
        current = observation.current_phase
        if current is not None:
            current = torch.minimum(current.to(slots.device).clamp(min=0), counts-1)
        phase_logits = self.phase_selector(raw, slots, scene, None, current, True)
        selection_logits = phase_logits.masked_fill(~slot_valid, torch.finfo(phase_logits.dtype).min)
        phase = current if observation.force_current_phase and current is not None else selection_logits.argmax(-1)
        skill = slots.gather(1, phase[:, None, None].expand(-1, 1, slots.shape[-1])).squeeze(1)
        completion = torch.stack([self.termination_head(raw, slots[:, i], scene)
                                  for i in range(slots.shape[1])], dim=1)
        hierarchy = dict(plan_slots=slots, slot_valid_logits=validity, phase_logits=phase_logits,
            predicted_phase=phase, selected_skill=skill,
            termination_logits=completion.gather(1, phase[:, None]).squeeze(1),
            completion_logits_by_phase=completion)
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
