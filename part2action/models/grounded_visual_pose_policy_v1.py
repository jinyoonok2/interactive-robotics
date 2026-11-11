"""Visual diffusion policy with one shared grounding-to-pose branch.

Versioned separately from the queued supplied-GT-pose experiment. Shared
encoders/hierarchy/fusion and the diffusion action decoder retain their names.
No GT pose, part cloud, contact target or event clock is accepted by forward.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from data.geometry import camera_to_world_torch
from .part2action_model import Part2ActionModel
from .grounded_pose_head_v1 import (
    PartRegionGroundingV1, GroundedTargetPoseHeadV1, PredictedPoseConditionerV1)


class GroundedVisualPosePolicyV1(Part2ActionModel):
    def __init__(self, *args, pose_conditioning_enabled=True, pose_scale_m=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.action_head_type != 'diffusion' or self.split_action_heads or
            not self.use_pcd or not self.use_gripper_state or
            self.contact_conditioning_enabled or self.contact_conditioning_stage != 'late' or
            self.use_contact_relative_actions or self.use_contact_residual or
            self.use_near_contact_actions or self.use_action_deltas or
            self.use_skill_gate or self.use_point_part_head):
            raise ValueError('Grounded pose V1 requires the original world-action visual diffusion recipe; direct contact conditioning stays off')
        self.pose_conditioning_enabled = bool(pose_conditioning_enabled)
        hidden = kwargs.get('hidden_dim', 256)
        with torch.random.fork_rng(devices=[]):
            self.contact_attention = PartRegionGroundingV1(
                self.visual.embed_dim, self.visual.embed_dim, self.text.embed_dim, hidden)
            self.pose_head = GroundedTargetPoseHeadV1(
                self.visual.embed_dim, self.visual.embed_dim, self.text.embed_dim, hidden, pose_scale_m)
            self.pose_conditioner = PredictedPoseConditionerV1(hidden)
        # These heads have no active loss/action role in this recipe. The visual
        # part gate and shared grounding scorer remain trainable.
        self.active_heads = {'action'}
        for name in ('heatmap_head', 'approach_head', 'contact_head', 'contact_xyz_head'):
            if hasattr(self, name):
                delattr(self, name)

    def forward(self, rgb, instructions, target_action=None, agentview_pcd=None,
                tcp_pose=None, gripper_state=None, skill_plan=None,
                skill_valid_mask=None, phase_index=None, oracle_skill_instructions=None,
                scheduled_sampling_prob=1., current_phase=None, force_current_phase=False,
                hierarchy_mode='auto', plan_slots_override=None,
                slot_valid_logits_override=None, gate_mode='predicted'):
        if gate_mode != 'predicted':
            raise ValueError('Grounded pose deployment always uses predicted grounding')
        if agentview_pcd is None or tcp_pose is None or gripper_state is None:
            raise ValueError('RGB, observed scene PCD and measured TCP/opening are required')
        text_tok = text_mask = None
        reusing_plan = self.use_hierarchy and hierarchy_mode != 'direct' and plan_slots_override is not None
        if not reusing_plan:
            if instructions is None:
                raise ValueError('Instructions required unless reusing a predicted plan')
            text_tok, text_mask = self.text(instructions)
            text_tok, text_mask = text_tok.to(rgb.device), text_mask.to(rgb.device)
        raw_visual = self._encode_visual(rgb)
        visual = raw_visual
        gate_logits = None
        if self.use_part_gate:
            visual, gate_logits = self.part_gate(raw_visual)
        point_tokens, scene = self.pcd_encoder(
            agentview_pcd=agentview_pcd, tcp_pose=tcp_pose, return_tokens=True)
        hierarchy_out = {}
        if self.use_hierarchy and hierarchy_mode != 'direct':
            if plan_slots_override is None:
                slots, valid_logits = self.skill_planner(text_tok, text_mask)
            else:
                if slot_valid_logits_override is None:
                    raise ValueError('Slot validity required with plan override')
                slots, valid_logits = plan_slots_override, slot_valid_logits_override
            if skill_valid_mask is None:
                counts = (valid_logits.sigmoid() >= .5).sum(-1).clamp(min=1)
                selector_valid = torch.arange(valid_logits.shape[1], device=rgb.device)[None] < counts[:, None]
            else:
                selector_valid = skill_valid_mask.bool()
            phase_logits = self.phase_selector(visual, slots, scene, selector_valid,
                                               current_phase, self.phase_selector_use_3d)
            predicted_phase = phase_logits.argmax(-1)
            if force_current_phase and current_phase is not None:
                predicted_phase = current_phase.clamp(0, slots.shape[1]-1)
            skill = slots.gather(1, predicted_phase[:, None, None].expand(-1, 1, slots.shape[-1])).squeeze(1)
            if oracle_skill_instructions is not None:
                oracle = self.encode_pooled_skill_texts(list(oracle_skill_instructions), rgb.device)
                if hierarchy_mode == 'oracle' or scheduled_sampling_prob <= 0:
                    skill = oracle
                elif self.training and scheduled_sampling_prob < 1:
                    choose = torch.rand(rgb.shape[0], device=rgb.device) < scheduled_sampling_prob
                    skill = torch.where(choose[:, None], skill, oracle)
            hierarchy_out = dict(plan_slots=slots, slot_valid_logits=valid_logits,
                phase_logits=phase_logits, predicted_phase=predicted_phase, selected_skill=skill,
                termination_logits=self.termination_head(visual, skill, scene))
            if skill_plan is not None:
                hierarchy_out['skill_target_slots'] = self.encode_skill_plans(skill_plan, rgb.device)
        else:
            skill = F.normalize(self._masked_pool(text_tok, text_mask), dim=-1)

        with torch.autocast(device_type=agentview_pcd.device.type, enabled=False):
            world_xyz = camera_to_world_torch(agentview_pcd[..., :3].float())
        region = self.contact_attention(point_tokens, world_xyz, visual, skill, tcp_pose)
        pose = self.pose_head(region, visual, skill, tcp_pose, gripper_state)
        visual_scene = visual + scene[:, None].to(visual.dtype)
        if self.use_hierarchy and hierarchy_mode != 'direct':
            fused = self.fusion(visual_scene, skill[:, None],
                text_attn_mask=torch.ones(rgb.shape[0], 1, device=rgb.device, dtype=torch.long))
        else:
            fused = self.fusion(visual_scene, text_tok, text_mask)
        action_features = self._condition_on_gripper(fused, gripper_state)
        if self.use_unified_contact_conditioning:
            # Preserve the loaded baseline's learned zero-input constant. No
            # predicted contact coordinate enters this action-conditioning path.
            action_features, _ = self.contact_geometry_conditioner(
                action_features, region['center'], tcp_pose, enabled=False)
        action_features, geometry = self.pose_conditioner(
            action_features, pose['pose_features'], enabled=self.pose_conditioning_enabled)
        out = dict(fused=fused, gate_mode=gate_mode, **hierarchy_out, **pose,
                   contact_xyz=region['center'], contact_point_logits=region['logits'],
                   grounding_weights=region['weights'], pose_conditioning_features=geometry)
        if gate_logits is not None:
            out['gate_logits'] = gate_logits
        out.update(self.action_head(action_features, target_action=target_action))
        return out
