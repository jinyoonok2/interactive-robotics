"""Part2Action with optional 3D-grounded hierarchical skill planning."""
from __future__ import annotations

from typing import Iterable, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FrozenDINOv2, FrozenT5
from .heads import (
    ActionChunkHead, ApproachHead, ContactHead2D, ContactHead3D,
    CrossAttentionFusion, DiffusionActionHead, HeatmapHead,
    PartMaskGate, PointCloudEncoder,
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

        self.visual = FrozenDINOv2(img_size=img_size)
        self.text = FrozenT5(device=text_device)
        self._skill_embedding_cache: dict[str, torch.Tensor] = {}
        self.temporal = build_temporal_encoder(
            encoder_type=temporal_encoder_type, embed_dim=self.visual.embed_dim,
            n_obs_steps=self.n_obs_steps, num_layers=temporal_layers,
            num_heads=temporal_heads,
        )
        if self.use_part_gate:
            self.part_gate = PartMaskGate(self.visual.embed_dim)
        if self.use_pcd:
            self.pcd_encoder = PointCloudEncoder(out_dim=self.visual.embed_dim)
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
            self.action_head = (
                DiffusionActionHead(hidden_dim, action_chunk, action_dim, num_steps=diffusion_steps)
                if self.action_head_type == "diffusion"
                else ActionChunkHead(hidden_dim, action_chunk, action_dim)
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
    ):
        text_tok = text_mask = None
        reusing_plan = self.use_hierarchy and hierarchy_mode != "direct" and plan_slots_override is not None
        if not reusing_plan:
            if instructions is None:
                raise ValueError("instructions required unless reusing a predicted plan")
            text_tok, text_mask = self.text(instructions)
            text_tok, text_mask = text_tok.to(rgb.device), text_mask.to(rgb.device)

        visual_tok = self._encode_visual(rgb)
        gate_logits = None
        if self.use_part_gate:
            visual_tok, gate_logits = self.part_gate(visual_tok)
        phase_visual_tok = visual_tok
        pcd_embed = None
        if self.use_pcd:
            if agentview_pcd is None:
                raise ValueError("use_pcd=True requires full-scene agentview_pcd")
            pcd_embed = self.pcd_encoder(agentview_pcd=agentview_pcd, tcp_pose=tcp_pose)
            visual_tok = visual_tok + pcd_embed[:, None, :].to(visual_tok.dtype)

        hierarchy_out = {}
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
            fused = self.fusion(
                visual_tok, selected[:, None],
                text_attn_mask=torch.ones(rgb.shape[0], 1, device=rgb.device, dtype=torch.long),
            )
            hierarchy_out = {
                "plan_slots": slots, "slot_valid_logits": valid_logits,
                "phase_logits": phase_logits, "predicted_phase": predicted_phase,
                "selected_skill": selected, "termination_logits": termination_logits,
            }
            if skill_plan is not None:
                hierarchy_out["skill_target_slots"] = self.encode_skill_plans(skill_plan, rgb.device)
        else:
            assert text_tok is not None and text_mask is not None
            fused = self.fusion(visual_tok, text_tok, text_mask)

        out = {"fused": fused, **hierarchy_out}
        if gate_logits is not None:
            out["gate_logits"] = gate_logits
        if "heatmap" in self.active_heads:
            out["heatmap_logits"] = self.heatmap_head(fused).squeeze(1)
        if "contact" in self.active_heads:
            out["contact_xy"] = self.contact_head(fused)
        if self.use_contact_xyz:
            out["contact_xyz"] = self.contact_xyz_head(fused)
        if "approach" in self.active_heads:
            out["approach_dir"] = self.approach_head(fused)
        if "action" in self.active_heads:
            if self.action_head_type == "diffusion":
                out.update(self.action_head(fused, target_action=target_action))
            else:
                out["action_chunk"] = self.action_head(fused)
        return out
