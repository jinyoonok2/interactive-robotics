"""Versioned visual baseline with an added supplied-TCP-pose feature module.

Inherited construction and shared parameter names are unchanged. The versioned
forward below copies the baseline forward with only the explicit pose arguments
and marked feature-conditioning insertion. Original source is preserved.
Baseline forward source SHA256: d381b88d487a683b60b0f752b2570c4ffe586ab81e1e18524ac21eb6c9018fa2
"""
from typing import Optional
import torch
import torch.nn.functional as F
from data.geometry import camera_to_world_torch
from .part2action_model import Part2ActionModel
from .pose_conditioning_v1 import TargetPoseConditionerV1


class VisualPosePolicyV1(Part2ActionModel):
    def __init__(self, *args, pose_conditioning_enabled=False, pose_scale_m=.1, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_head_type != "diffusion" or self.split_action_heads:
            raise ValueError("Visual pose V1 retains the monolithic diffusion action head")
        self.pose_conditioning_enabled = bool(pose_conditioning_enabled)
        with torch.random.fork_rng(devices=[]):
            self.pose_conditioner = TargetPoseConditionerV1(
                kwargs.get("hidden_dim", 256), pose_scale_m)

    def forward(
        self, rgb: torch.Tensor, instructions, target_action: Optional[torch.Tensor] = None,
        agentview_pcd: Optional[torch.Tensor] = None, tcp_pose: Optional[torch.Tensor] = None,
        skill_plan: Optional[list[list[str]]] = None,
        skill_valid_mask: Optional[torch.Tensor] = None,
        phase_index: Optional[torch.Tensor] = None,
        oracle_skill_instructions: Optional[list[str]] = None,
        scheduled_sampling_prob: float = 1.0,
        current_phase: Optional[torch.Tensor] = None, force_current_phase: bool = False,
        hierarchy_mode: str = "auto", plan_slots_override: Optional[torch.Tensor] = None,
        slot_valid_logits_override: Optional[torch.Tensor] = None,
        gate_mode: str = "predicted",
        gt_part_mask: Optional[torch.Tensor] = None,
        oracle_gate_logit_scale: float = 8.0,
        gripper_state: Optional[torch.Tensor] = None,
        contact_xyz_target: Optional[torch.Tensor] = None,
        contact_xyz_valid: Optional[torch.Tensor] = None,
        goal_pose: Optional[torch.Tensor] = None,
        goal_valid: Optional[torch.Tensor] = None,
    ):
        gate_mode = str(gate_mode).lower()
        if gate_mode not in {"predicted", "oracle"}:
            raise ValueError(f"gate_mode must be predicted or oracle, got {gate_mode}")
        if gate_mode == "oracle" and not self.use_part_gate:
            raise ValueError("gate_mode=oracle requires use_part_gate=True")
        if gate_mode == "oracle" and gt_part_mask is None:
            raise ValueError("gate_mode=oracle requires gt_part_mask")

        text_tok = text_mask = None
        reusing_plan = self.use_hierarchy and hierarchy_mode != "direct" and plan_slots_override is not None
        if not reusing_plan:
            if instructions is None:
                raise ValueError("instructions required unless reusing a predicted plan")
            text_tok, text_mask = self.text(instructions)
            text_tok, text_mask = text_tok.to(rgb.device), text_mask.to(rgb.device)

        early_contact = self.contact_conditioning_stage == "early_temporal"
        visual_history = None
        if early_contact:
            visual_history = self._encode_visual(rgb, return_history=True)
            if visual_history.ndim != 4 or visual_history.shape[1] != self.n_obs_steps:
                raise ValueError("early_temporal requires the configured RGB frame history")
            # Localize contact before temporal action features are constructed.
            raw_visual_tok = visual_history[:, -1]
        else:
            raw_visual_tok = self._encode_visual(rgb)
        phase_visual_tok = raw_visual_tok
        gate_logits = None
        legacy_gated_visual_tok = None
        if self.use_part_gate and not self.use_skill_gate:
            legacy_gated_visual_tok, gate_logits = self.part_gate(raw_visual_tok)
            if gate_mode == "oracle":
                oracle_logits = self.mask_to_gate_logits(
                    gt_part_mask, raw_visual_tok.shape[1], oracle_gate_logit_scale
                ).to(device=raw_visual_tok.device, dtype=gate_logits.dtype)
                legacy_gated_visual_tok, _ = self.apply_part_gate(raw_visual_tok, oracle_logits)
            phase_visual_tok = legacy_gated_visual_tok
        pcd_embed = point_tokens = None
        if self.use_pcd:
            if agentview_pcd is None:
                raise ValueError("use_pcd=True requires full-scene agentview_pcd")
            encoded_pcd = self.pcd_encoder(
                agentview_pcd=agentview_pcd, tcp_pose=tcp_pose,
                return_tokens=(
                    self.use_contact_action_token or self.use_point_part_head
                ),
            )
            if self.use_contact_action_token or self.use_point_part_head:
                point_tokens, pcd_embed = encoded_pcd
            else:
                pcd_embed = encoded_pcd

        hierarchy_out = {}
        conditioning_skill = None
        if self.use_hierarchy and hierarchy_mode != "direct":
            if plan_slots_override is None:
                assert text_tok is not None and text_mask is not None
                slots, valid_logits = self.skill_planner(text_tok, text_mask)
            else:
                if slot_valid_logits_override is None:
                    raise ValueError("slot validity logits required with plan override")
                slots, valid_logits = plan_slots_override, slot_valid_logits_override
            if skill_valid_mask is None:
                counts = (valid_logits.sigmoid() >= 0.5).sum(-1).clamp(min=1)
                selector_valid = torch.arange(valid_logits.shape[1], device=rgb.device)[None] < counts[:, None]
            else:
                selector_valid = skill_valid_mask.bool()
            phase_logits = self.phase_selector(
                phase_visual_tok, slots, pcd_embed, selector_valid, current_phase,
                self.phase_selector_use_3d,
            )
            predicted_phase = phase_logits.argmax(-1)
            if force_current_phase and current_phase is not None:
                predicted_phase = current_phase.clamp(0, slots.shape[1] - 1)
            selected = slots.gather(
                1, predicted_phase[:, None, None].expand(-1, 1, slots.shape[-1])
            ).squeeze(1)
            if oracle_skill_instructions is not None:
                oracle = self.encode_pooled_skill_texts(list(oracle_skill_instructions), rgb.device)
                if hierarchy_mode == "oracle" or scheduled_sampling_prob <= 0:
                    selected = oracle
                elif self.training and scheduled_sampling_prob < 1:
                    choose = torch.rand(rgb.shape[0], device=rgb.device) < scheduled_sampling_prob
                    selected = torch.where(choose[:, None], selected, oracle)
            termination_logits = self.termination_head(phase_visual_tok, selected, pcd_embed)
            conditioning_skill = selected
            hierarchy_out = {
                "plan_slots": slots, "slot_valid_logits": valid_logits,
                "phase_logits": phase_logits, "predicted_phase": predicted_phase,
                "selected_skill": selected, "termination_logits": termination_logits,
            }
            if skill_plan is not None:
                hierarchy_out["skill_target_slots"] = self.encode_skill_plans(skill_plan, rgb.device)
        else:
            assert text_tok is not None and text_mask is not None
            conditioning_skill = F.normalize(self._masked_pool(text_tok, text_mask), dim=-1)

        grounded_visual_tok = (
            legacy_gated_visual_tok
            if legacy_gated_visual_tok is not None
            else raw_visual_tok
        )
        if self.use_part_gate:
            if self.use_skill_gate:
                assert conditioning_skill is not None
                grounded_visual_tok, gate_logits = self.part_gate(
                    raw_visual_tok, conditioning_skill
                )
                if gate_mode == "oracle":
                    oracle_logits = self.mask_to_gate_logits(
                        gt_part_mask, raw_visual_tok.shape[1], oracle_gate_logit_scale
                    ).to(device=raw_visual_tok.device, dtype=gate_logits.dtype)
                    grounded_visual_tok, _ = self.apply_part_gate(raw_visual_tok, oracle_logits)
        contact_out = {}
        if early_contact:
            assert point_tokens is not None and agentview_pcd is not None
            assert conditioning_skill is not None
            contact_xyz, contact_token, contact_point_logits = self.contact_attention(
                point_tokens, agentview_pcd, grounded_visual_tok,
                conditioning_skill, tcp_pose,
                world_xyz=camera_to_world_torch(agentview_pcd),
            )
            contact_out.update(
                contact_xyz=contact_xyz,
                contact_token=contact_token,
                contact_point_logits=contact_point_logits,
            )
        if early_contact:
            context, geometry = self._early_contact_context(
                contact_out["contact_xyz"], contact_out["contact_point_logits"],
                tcp_pose, gripper_state)
            contact_out["contact_conditioning_features"] = geometry
            contact_out["early_contact_context"] = context
            action_visual = self.temporal(visual_history, context_token=context)
            if gate_logits is not None:
                action_visual, _ = self.apply_part_gate(action_visual, gate_logits)
            grounded_visual_tok = action_visual

        visual_tok = grounded_visual_tok
        if pcd_embed is not None:
            visual_tok = visual_tok + pcd_embed[:, None, :].to(visual_tok.dtype)

        if self.use_hierarchy and hierarchy_mode != "direct":
            assert conditioning_skill is not None
            fused = self.fusion(
                visual_tok, conditioning_skill[:, None],
                text_attn_mask=torch.ones(rgb.shape[0], 1, device=rgb.device, dtype=torch.long),
            )
        else:
            assert text_tok is not None and text_mask is not None
            fused = self.fusion(visual_tok, text_tok, text_mask)
        action_fused = self._condition_on_gripper(fused, gripper_state)

        out = {"fused": fused, "gate_mode": gate_mode, **hierarchy_out, **contact_out}
        if self.use_action_deltas:
            out["action_representation"] = "tcp_xyz_delta"
        if self.use_contact_relative_actions:
            out["action_representation"] = "contact_relative_xyz"
        if gate_logits is not None:
            out["gate_logits"] = gate_logits
        if self.use_point_part_head:
            assert point_tokens is not None and conditioning_skill is not None
            out["point_part_logits"] = self.point_part_head(
                point_tokens, grounded_visual_tok, conditioning_skill
            )
        if "heatmap" in self.active_heads:
            out["heatmap_logits"] = self.heatmap_head(fused).squeeze(1)
        if "contact" in self.active_heads:
            out["contact_xy"] = self.contact_head(fused)
        if self.use_contact_action_token and not early_contact:
            assert point_tokens is not None and agentview_pcd is not None
            assert conditioning_skill is not None
            contact_xyz, contact_token, contact_point_logits = self.contact_attention(
                point_tokens, agentview_pcd, grounded_visual_tok,
                conditioning_skill, tcp_pose,
                world_xyz=camera_to_world_torch(agentview_pcd),
            )
            out.update(contact_xyz=contact_xyz, contact_token=contact_token,
                       contact_point_logits=contact_point_logits)
        if self.use_contact_xyz and not self.use_contact_action_token:
            out["contact_xyz"] = self.contact_xyz_head(fused)
        if "approach" in self.active_heads:
            out["approach_dir"] = self.approach_head(fused)
        if "action" in self.active_heads:
            if self.use_unified_contact_conditioning and not early_contact:
                if tcp_pose is None:
                    raise ValueError("unified contact conditioning requires tcp_pose")
                action_fused, geometry_features = self.contact_geometry_conditioner(
                    action_fused, out["contact_xyz"], tcp_pose,
                    enabled=self.contact_conditioning_enabled,
                )
                out["contact_conditioning_features"] = geometry_features
                if self.contact_geometry_conditioner.mode == "normalized_global":
                    out["contact_conditioning_weight"] = self.contact_geometry_conditioner.contact_weight
            # NEW V1: feature conditioning only; retain the original action decoder.
            action_fused, pose_features = self.pose_conditioner(
                action_fused, tcp_pose, goal_pose, goal_valid,
                enabled=self.pose_conditioning_enabled)
            out["pose_conditioning_features"] = pose_features
            if self.split_action_heads:
                out.update(
                    self.action_head(
                        action_fused, out["contact_token"], target_action=target_action
                    )
                )
            elif self.action_head_type == "diffusion":
                out.update(self.action_head(action_fused, target_action=target_action))
            else:
                baseline_action = self.action_head(action_fused)
                if self.use_contact_relative_actions:
                    if tcp_pose is None:
                        raise ValueError("contact-relative actions require tcp_pose")
                    action, offset = self.contact_relative_head(
                        baseline_action, action_fused, out["contact_token"],
                        out["contact_xyz"], tcp_pose,
                    )
                    out.update(
                        baseline_action_chunk=baseline_action,
                        contact_xyz_offset=offset,
                        action_chunk=action,
                    )
                elif self.use_near_contact_actions:
                    if tcp_pose is None:
                        raise ValueError("near-contact actions require tcp_pose")
                    if "contact_xyz" not in out or "contact_token" not in out:
                        raise ValueError("near-contact actions require a predicted contact")
                    if self.training and contact_xyz_target is None:
                        raise ValueError("near-contact training requires contact_xyz_target")
                    gate_contact = (
                        contact_xyz_target if self.training else out["contact_xyz"]
                    )
                    gate_valid = contact_xyz_valid if self.training else None
                    near = near_contact_mask(
                        tcp_pose, gate_contact, self.near_contact_distance, gate_valid
                    )
                    contact_action, offset = self.near_contact_head(
                        baseline_action, action_fused, out["contact_token"],
                        out["contact_xyz"].detach(), tcp_pose,
                    )
                    action = apply_near_contact_translation(
                        baseline_action, contact_action, near
                    )
                    out.update(
                        baseline_action_chunk=baseline_action,
                        contact_xyz_offset=offset,
                        near_contact=near,
                        action_chunk=action,
                    )
                elif self.use_contact_residual:
                    if tcp_pose is None:
                        raise ValueError("contact residual requires tcp_pose")
                    action, residual = self.contact_residual_head(
                        baseline_action, action_fused, out["contact_token"],
                        out["contact_xyz"], tcp_pose,
                    )
                    out.update(
                        baseline_action_chunk=baseline_action,
                        contact_xyz_residual=residual,
                        action_chunk=action,
                    )
                else:
                    out["action_chunk"] = baseline_action
        return out
