"""Part2Action with optional 3D-grounded hierarchical skill planning."""
from __future__ import annotations

from typing import Iterable, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.geometry import (
    apply_near_contact_translation, camera_to_world_torch, near_contact_mask,
)
from .backbone import FrozenDINOv2, FrozenT5
from .heads import (
    ActionChunkHead, ApproachHead, BoundedContactResidualHead, ContactHead2D, ContactHead3D,
    ContactRelativeOffsetHead, ContactGeometryConditioner,
    CrossAttentionFusion, DiffusionActionHead, HeatmapHead,
    GroundedHybridActionHead, PartMaskGate, PointCloudEncoder,
    PointwiseContactAttention, SkillConditionedPartGate,
    SkillConditionedPointPartHead,
)
from .hierarchical import PhaseSelector, SkillPlanDecoder, SkillTerminationHead
from .temporal import build_temporal_encoder

_VALID_HEADS = {"heatmap", "contact", "approach", "action"}


class Part2ActionModel(nn.Module):
    def __init__(
        self, heads: Iterable[str] = ("heatmap",), img_size: int = 252,
        out_size: int = 96, action_chunk: int = 8, action_dim: int = 7,
        hidden_dim: int = 256, num_fusion_layers: int = 2,
        text_device: str = "cpu", action_head_type: str = "mlp",
        diffusion_steps: int = 50, temporal_encoder_type: str = "none",
        n_obs_steps: int = 1, temporal_layers: int = 1, temporal_heads: int = 4,
        use_part_gate: bool = False, use_pcd: bool = False,
        use_contact_xyz: bool = False, use_hierarchy: bool = False,
        max_skill_slots: int = 4, hierarchy_layers: int = 2,
        phase_selector_use_3d: bool = True,
        use_skill_gate: bool = False,
        use_contact_action_token: bool = False,
        split_action_heads: bool = False,
        gripper_open_width: float = 0.04,
        gripper_closed_width: float = 0.0,
        use_contact_residual: bool = False,
        contact_residual_max_translation: float = 0.03,
        use_contact_relative_actions: bool = False,
        contact_relative_max_translation: float = 0.40,
        use_unified_contact_conditioning: bool = False,
        contact_conditioning_enabled: bool = True,
        contact_geometry_scale_m: float = 0.1,
        use_near_contact_actions: bool = False,
        near_contact_distance: float = 0.03,
        near_contact_max_translation: float = 0.03,
        use_action_deltas: bool = False,
        use_point_part_head: bool = False,
        use_gripper_state: bool = False,
    ) -> None:
        super().__init__()
        heads_set: Set[str] = set(heads)
        unknown = heads_set - _VALID_HEADS
        if unknown or not heads_set:
            raise ValueError(f"Invalid heads {heads_set}; valid: {_VALID_HEADS}")
        self.active_heads = heads_set
        self.action_head_type = action_head_type.lower()
        self.n_obs_steps = max(1, int(n_obs_steps))
        self.use_part_gate = bool(use_part_gate)
        self.use_pcd = bool(use_pcd)
        self.use_contact_xyz = bool(use_contact_xyz)
        self.use_hierarchy = bool(use_hierarchy)
        self.max_skill_slots = max(1, int(max_skill_slots))
        self.phase_selector_use_3d = bool(phase_selector_use_3d)
        self.use_skill_gate = bool(use_skill_gate)
        self.use_contact_action_token = bool(use_contact_action_token)
        self.split_action_heads = bool(split_action_heads)
        self.use_contact_residual = bool(use_contact_residual)
        self.use_contact_relative_actions = bool(use_contact_relative_actions)
        self.use_unified_contact_conditioning = bool(use_unified_contact_conditioning)
        self.contact_conditioning_enabled = bool(contact_conditioning_enabled)
        if self.use_unified_contact_conditioning:
            if not (self.use_contact_action_token and "action" in heads_set):
                raise ValueError("unified contact conditioning requires contact action token, TCP, and action head")
            if action_head_type != "mlp" or split_action_heads or use_near_contact_actions or use_contact_relative_actions or use_contact_residual or use_action_deltas:
                raise ValueError("unified conditioning requires monolithic world-action MLP without contact switching/residuals")
            self.contact_geometry_conditioner = ContactGeometryConditioner(hidden_dim, contact_geometry_scale_m)
        self.use_near_contact_actions = bool(use_near_contact_actions)
        self.near_contact_distance = float(near_contact_distance)
        self.near_contact_max_translation = float(near_contact_max_translation)
        self.use_action_deltas = bool(use_action_deltas)
        self.use_point_part_head = bool(use_point_part_head)
        self.use_gripper_state = bool(use_gripper_state)
        if self.use_skill_gate and not self.use_part_gate:
            raise ValueError("use_skill_gate=True requires use_part_gate=True")
        if self.use_contact_action_token and not (self.use_pcd and self.use_contact_xyz):
            raise ValueError("contact action tokens require use_pcd and use_contact_xyz")
        if self.split_action_heads and not self.use_contact_action_token:
            raise ValueError("split action heads require use_contact_action_token")
        if self.use_contact_residual and self.use_contact_relative_actions:
            raise ValueError("contact residual and contact-relative actions are mutually exclusive")
        if self.use_near_contact_actions and (
            self.use_contact_relative_actions or self.use_contact_residual or self.use_action_deltas
        ):
            raise ValueError(
                "near-contact actions are mutually exclusive with the full-path "
                "contact offset, the ±2 cm residual, and action deltas"
            )
        if self.use_action_deltas and self.use_contact_relative_actions:
            raise ValueError("action deltas and contact-relative actions are mutually exclusive")
        if self.use_near_contact_actions and not self.use_contact_action_token:
            raise ValueError("near-contact actions require use_contact_action_token")
        if self.use_near_contact_actions and self.action_head_type != "mlp":
            raise ValueError("near-contact actions currently require the MLP action head")
        if self.use_near_contact_actions and self.split_action_heads:
            raise ValueError("near-contact actions currently require the monolithic action head")
        if self.use_near_contact_actions and (
            self.near_contact_distance <= 0 or self.near_contact_max_translation <= 0
        ):
            raise ValueError("near-contact distance and offset cap must be positive")
        if self.use_near_contact_actions and self.near_contact_max_translation > 0.05:
            raise ValueError("near-contact offset cap must stay within 5 cm")
        if self.use_contact_residual and not self.use_contact_action_token:
            raise ValueError("contact residual requires use_contact_action_token")
        if self.use_contact_residual and self.action_head_type != "mlp":
            raise ValueError("contact residual currently requires the MLP action head")
        if self.use_contact_relative_actions and not self.use_contact_action_token:
            raise ValueError("contact-relative actions require use_contact_action_token")
        if self.use_contact_relative_actions and self.action_head_type != "mlp":
            raise ValueError("contact-relative actions currently require the MLP action head")
        if self.use_contact_relative_actions and self.split_action_heads:
            raise ValueError("contact-relative actions currently require the monolithic action head")
        if self.use_action_deltas and self.action_head_type == "diffusion":
            raise ValueError("use_action_deltas currently supports the MLP action head only")
        if self.use_action_deltas and self.split_action_heads:
            raise ValueError("use_action_deltas currently supports the monolithic action head")
        if self.use_point_part_head and not self.use_pcd:
            raise ValueError("point-part head requires use_pcd")

        self.visual = FrozenDINOv2(img_size=img_size)
        self.text = FrozenT5(device=text_device)
        self._skill_embedding_cache: dict[str, torch.Tensor] = {}
        self.temporal = build_temporal_encoder(
            encoder_type=temporal_encoder_type, embed_dim=self.visual.embed_dim,
            n_obs_steps=self.n_obs_steps, num_layers=temporal_layers,
            num_heads=temporal_heads,
        )
        if self.use_part_gate:
            self.part_gate = (
                SkillConditionedPartGate(self.visual.embed_dim, self.text.embed_dim)
                if self.use_skill_gate
                else PartMaskGate(self.visual.embed_dim)
            )
        if self.use_pcd:
            self.pcd_encoder = PointCloudEncoder(out_dim=self.visual.embed_dim)
        if self.use_point_part_head:
            self.point_part_head = SkillConditionedPointPartHead(
                point_dim=self.visual.embed_dim,
                visual_dim=self.visual.embed_dim,
                skill_dim=self.text.embed_dim,
                hidden_dim=hidden_dim,
            )
        if self.use_contact_action_token:
            self.contact_attention = PointwiseContactAttention(
                point_dim=self.visual.embed_dim,
                visual_dim=self.visual.embed_dim,
                skill_dim=self.text.embed_dim,
                hidden_dim=hidden_dim,
                token_dim=hidden_dim,
            )
        self.fusion = CrossAttentionFusion(
            visual_dim=self.visual.embed_dim, text_dim=self.text.embed_dim,
            hidden_dim=hidden_dim, num_layers=num_fusion_layers,
        )
        if self.use_hierarchy:
            self.skill_planner = SkillPlanDecoder(
                self.text.embed_dim, hidden_dim, self.max_skill_slots, hierarchy_layers
            )
            self.phase_selector = PhaseSelector(self.visual.embed_dim, self.text.embed_dim, hidden_dim)
            self.termination_head = SkillTerminationHead(
                self.visual.embed_dim, self.text.embed_dim, hidden_dim
            )
        if "heatmap" in heads_set:
            self.heatmap_head = HeatmapHead(hidden_dim, self.visual.grid, out_size)
        if "contact" in heads_set:
            self.contact_head = ContactHead2D(hidden_dim)
        if self.use_contact_xyz:
            self.contact_xyz_head = ContactHead3D(hidden_dim)
        if "approach" in heads_set:
            self.approach_head = ApproachHead(hidden_dim)
        if "action" in heads_set:
            if self.split_action_heads:
                self.action_head = GroundedHybridActionHead(
                    in_dim=hidden_dim, contact_dim=hidden_dim, chunk=action_chunk,
                    arm_head_type=self.action_head_type, diffusion_steps=diffusion_steps,
                    open_width=gripper_open_width, closed_width=gripper_closed_width,
                )
            else:
                self.action_head = (
                    DiffusionActionHead(hidden_dim, action_chunk, action_dim, num_steps=diffusion_steps)
                    if self.action_head_type == "diffusion"
                    else ActionChunkHead(hidden_dim, action_chunk, action_dim)
                )
            if self.use_contact_residual:
                self.contact_residual_head = BoundedContactResidualHead(
                    in_dim=hidden_dim, contact_dim=hidden_dim, chunk=action_chunk,
                    max_translation=contact_residual_max_translation,
                )
            if self.use_contact_relative_actions:
                self.contact_relative_head = ContactRelativeOffsetHead(
                    in_dim=hidden_dim, contact_dim=hidden_dim, chunk=action_chunk,
                    max_translation=contact_relative_max_translation,
                )
            if self.use_near_contact_actions:
                self.near_contact_head = ContactRelativeOffsetHead(
                    in_dim=hidden_dim, contact_dim=hidden_dim, chunk=action_chunk,
                    max_translation=near_contact_max_translation,
                )
        if self.use_gripper_state:
            self.gripper_state_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.text.model.to(self.text.device_str)
        return module

    def train(self, mode: bool = True):
        module = super().train(mode)
        self.visual.model.eval()
        self.text.model.eval()
        return module

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _masked_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)

    def encode_pooled_skill_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        missing = list(dict.fromkeys(t for t in texts if t not in self._skill_embedding_cache))
        if missing:
            tokens, mask = self.text(missing)
            pooled = F.normalize(self._masked_pool(tokens, mask), dim=-1)
            for text, embedding in zip(missing, pooled):
                self._skill_embedding_cache[text] = embedding.detach().cpu()
        return torch.stack([self._skill_embedding_cache[t] for t in texts]).to(device)

    def encode_skill_plans(self, plans: list[list[str]], device: torch.device) -> torch.Tensor:
        targets = torch.zeros(len(plans), self.max_skill_slots, self.text.embed_dim, device=device)
        flat, positions = [], []
        for b, plan in enumerate(plans):
            for s, text in enumerate(plan[: self.max_skill_slots]):
                flat.append(text)
                positions.append((b, s))
        if flat:
            for embedding, (b, s) in zip(self.encode_pooled_skill_texts(flat, device), positions):
                targets[b, s] = embedding
        return targets

    def _encode_visual(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if rgb.ndim == 5:
                b, t, c, h, w = rgb.shape
                flat = self.visual(rgb.reshape(b * t, c, h, w))
                tokens = flat.reshape(b, t, flat.shape[1], flat.shape[2])
            elif rgb.ndim == 4:
                tokens = self.visual(rgb)
            else:
                raise RuntimeError(f"Unexpected RGB shape {tuple(rgb.shape)}")
        return self.temporal(tokens)

    def _condition_on_gripper(
        self, fused: torch.Tensor, gripper_state: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Add the measured finger opening to the features used by the action head.

        The demo stores this width in meters. Open is about 0.04, so dividing
        by that width keeps the encoder input near 1 when the fingers are open.
        """
        if not self.use_gripper_state:
            return fused
        if gripper_state is None:
            raise ValueError("use_gripper_state=True requires gripper_state")
        opening = gripper_state.to(device=fused.device, dtype=fused.dtype).reshape(fused.shape[0], -1)
        if opening.shape[-1] != 1:
            raise ValueError(f"Expected gripper_state (B, 1), got {tuple(gripper_state.shape)}")
        embedded = self.gripper_state_encoder(opening / 0.04)
        return fused + embedded[:, None, :]

    @staticmethod
    def mask_to_gate_logits(
        part_mask: torch.Tensor, num_tokens: int, logit_scale: float = 8.0
    ) -> torch.Tensor:
        """Downsample a dense part mask to per-patch gate logits for oracle grounding."""
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            raise ValueError(f"num_tokens must be a square grid, got {num_tokens}")
        if part_mask.ndim != 3:
            raise ValueError(f"Expected part_mask (B,H,W), got {tuple(part_mask.shape)}")
        target = F.interpolate(
            part_mask[:, None].to(dtype=torch.float32),
            size=(side, side),
            mode="area",
        ).flatten(1)
        return (2.0 * target - 1.0) * float(logit_scale)

    @staticmethod
    def apply_part_gate(
        visual_tokens: torch.Tensor, gate_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)
        return visual_tokens * (1.0 + gate), gate_logits

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

        out = {"fused": fused, "gate_mode": gate_mode, **hierarchy_out}
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
        if self.use_contact_action_token:
            assert point_tokens is not None and agentview_pcd is not None
            assert conditioning_skill is not None
            contact_xyz, contact_token, contact_point_logits = self.contact_attention(
                point_tokens, agentview_pcd, grounded_visual_tok,
                conditioning_skill, tcp_pose,
                world_xyz=camera_to_world_torch(agentview_pcd),
            )
            out.update(
                contact_xyz=contact_xyz,
                contact_token=contact_token,
                contact_point_logits=contact_point_logits,
            )
        elif self.use_contact_xyz:
            out["contact_xyz"] = self.contact_xyz_head(fused)
        if "approach" in self.active_heads:
            out["approach_dir"] = self.approach_head(fused)
        if "action" in self.active_heads:
            if self.use_unified_contact_conditioning:
                if tcp_pose is None:
                    raise ValueError("unified contact conditioning requires tcp_pose")
                action_fused, geometry_features = self.contact_geometry_conditioner(
                    action_fused, out["contact_xyz"], tcp_pose,
                    enabled=self.contact_conditioning_enabled,
                )
                out["contact_conditioning_features"] = geometry_features
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
