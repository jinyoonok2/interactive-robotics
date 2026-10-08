"""Trainable heads for part2action.

CrossAttentionFusion produces task-grounded features by letting per-token
text embeddings attend to DINOv2 patch tokens. The result is a
(B, N_patch, D) feature map shared by all heads.

Heads:
  HeatmapHead       -> (B, 1, H, W) part-affordance probability
  ContactHead2D     -> (B, 2)      normalized image coords in [0, 1]
  ApproachHead      -> (B, 3)      unit vector
  ActionChunkHead   -> (B, K, 7)   k-step end-effector deltas + gripper
  DiffusionActionHead -> train-time noise prediction, eval-time action chunks
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.geometry import (
    camera_to_world_torch,
    contact_relative_to_absolute_actions,
    tcp_position_torch,
)


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        visual_dim: int = 384,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v = self.visual_proj(visual_tokens)
        t = self.text_proj(text_tokens)

        kpm = None
        if text_attn_mask is not None:
            kpm = ~(text_attn_mask.bool())

        for attn, n1, ffn, n2 in zip(self.layers, self.norms, self.ffns, self.ffn_norms):
            attn_out, _ = attn(query=v, key=t, value=t, key_padding_mask=kpm, need_weights=False)
            v = n1(v + attn_out)
            v = n2(v + ffn(v))
        return v


class PartMaskGate(nn.Module):
    """Predict an early per-patch part gate over visual tokens."""

    def __init__(self, visual_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Linear(visual_dim // 2, 1),
        )

    def forward(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(visual_tokens).squeeze(-1)
        gate = torch.sigmoid(logits).unsqueeze(-1)
        return visual_tokens * (1.0 + gate), logits


class SkillConditionedPartGate(nn.Module):
    """Predict a per-patch gate conditioned on the active skill embedding."""

    def __init__(self, visual_dim: int = 384, skill_dim: int = 512, hidden: int = 192) -> None:
        super().__init__()
        self.visual = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, hidden))
        self.skill = nn.Sequential(nn.LayerNorm(skill_dim), nn.Linear(skill_dim, hidden))
        self.score = nn.Sequential(nn.GELU(), nn.Linear(hidden, 1))

    def forward(
        self, visual_tokens: torch.Tensor, skill_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if skill_embedding.ndim != 2:
            raise ValueError(f"Expected skill embedding (B,D), got {tuple(skill_embedding.shape)}")
        logits = self.score(
            self.visual(visual_tokens) + self.skill(skill_embedding)[:, None, :]
        ).squeeze(-1)
        gate = torch.sigmoid(logits).unsqueeze(-1)
        return visual_tokens * (1.0 + gate), logits


class PointCloudEncoder(nn.Module):
    """PointNet-style encoder using world-frame points and TCP-relative geometry."""

    def __init__(self, in_dim: int = 10, out_dim: int = 384, hidden: int = 128) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        agentview_pcd: torch.Tensor | None = None,
        agentview_part_pcd: torch.Tensor | None = None,
        tcp_pose: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if agentview_pcd is None and agentview_part_pcd is None:
            raise ValueError("PointCloudEncoder requires at least one point cloud")
        base = agentview_part_pcd if agentview_part_pcd is not None else agentview_pcd
        assert base is not None
        scene_world = (
            torch.zeros(base.shape[0], base.shape[1], 3, device=base.device, dtype=base.dtype)
            if agentview_pcd is None
            else camera_to_world_torch(agentview_pcd[..., :3])
        )
        if agentview_part_pcd is None:
            part_world = torch.zeros_like(scene_world)
            part_flag = torch.zeros(
                *scene_world.shape[:2], 1, device=scene_world.device, dtype=scene_world.dtype
            )
        else:
            part_world = camera_to_world_torch(agentview_part_pcd[..., :3])
            part_flag = (
                agentview_part_pcd[..., 3:4]
                if agentview_part_pcd.shape[-1] >= 4
                else torch.ones(
                    *part_world.shape[:2],
                    1,
                    device=part_world.device,
                    dtype=part_world.dtype,
                )
            )
        reference_world = scene_world if agentview_pcd is not None else part_world
        tcp_delta = (
            torch.zeros_like(reference_world)
            if tcp_pose is None
            else reference_world
            - tcp_position_torch(tcp_pose)[:, None, :].to(dtype=reference_world.dtype)
        )
        raw_tokens = self.point_mlp(
            torch.cat([scene_world, part_world, part_flag, tcp_delta], dim=-1)
        )
        pooled = self.out_norm(raw_tokens.max(dim=1).values)
        tokens = self.out_norm(raw_tokens)
        return (tokens, pooled) if return_tokens else pooled


class SkillConditionedPointPartHead(nn.Module):
    """Predict target-part membership for each scene point."""

    def __init__(
        self,
        point_dim: int = 384,
        visual_dim: int = 384,
        skill_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.skill_proj = nn.Linear(skill_dim, hidden_dim)
        self.score = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        point_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        skill_embedding: torch.Tensor,
    ) -> torch.Tensor:
        context = (
            self.visual_proj(visual_tokens.mean(dim=1))
            + self.skill_proj(skill_embedding)
        )
        return self.score(
            self.point_proj(point_tokens) + context[:, None, :]
        ).squeeze(-1)


class PointwiseContactAttention(nn.Module):
    """Select a deployable 3D contact from full-scene point features."""

    def __init__(
        self,
        point_dim: int = 384,
        visual_dim: int = 384,
        skill_dim: int = 512,
        hidden_dim: int = 256,
        token_dim: int = 256,
    ) -> None:
        super().__init__()
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.skill_proj = nn.Linear(skill_dim, hidden_dim)
        self.geometry_proj = nn.Linear(6, hidden_dim)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, 1))
        self.token = nn.Sequential(
            nn.Linear(point_dim + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        point_tokens: torch.Tensor,
        scene_xyz: torch.Tensor,
        visual_tokens: torch.Tensor,
        skill_embedding: torch.Tensor,
        tcp_pose: torch.Tensor | None = None,
        world_xyz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = scene_xyz[..., :3] if world_xyz is None else world_xyz[..., :3]
        tcp_xyz = (
            torch.zeros(xyz.shape[0], 3, device=xyz.device, dtype=xyz.dtype)
            if tcp_pose is None
            else tcp_position_torch(tcp_pose).to(xyz.dtype)
        )
        geometry = torch.cat([xyz, xyz - tcp_xyz[:, None, :]], dim=-1)
        context = (
            self.visual_proj(visual_tokens.mean(dim=1))
            + self.skill_proj(skill_embedding)
        )
        logits = self.score(
            self.point_proj(point_tokens)
            + self.geometry_proj(geometry)
            + context[:, None, :]
        ).squeeze(-1)
        valid = torch.isfinite(xyz).all(dim=-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(point_tokens.dtype)
        safe_xyz = torch.where(valid[..., None], xyz, torch.zeros_like(xyz))
        contact_xyz = (weights[..., None] * safe_xyz).sum(dim=1)
        point_feature = (weights[..., None] * point_tokens).sum(dim=1)
        contact_token = self.token(
            torch.cat([point_feature, contact_xyz, contact_xyz - tcp_xyz], dim=-1)
        )
        return contact_xyz, contact_token, logits


class HeatmapHead(nn.Module):
    def __init__(self, in_dim: int = 256, grid: int = 18, out_size: int = 96) -> None:
        super().__init__()
        self.grid = int(grid)
        self.out_size = int(out_size)
        self.proj = nn.Linear(in_dim, 64)
        self.up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        B, N, D = fused.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise RuntimeError(f"Fused feature length {N} is not a perfect square")
        x = self.proj(fused).reshape(B, side, side, 64).permute(0, 3, 1, 2)
        x = self.up(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return x


class _PooledMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        pooled = fused.mean(dim=1)
        return self.net(pooled)


class ContactHead2D(_PooledMLP):
    def __init__(self, in_dim: int = 256) -> None:
        super().__init__(in_dim=in_dim, out_dim=2)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(fused))


class ContactHead3D(_PooledMLP):
    def __init__(self, in_dim: int = 256) -> None:
        super().__init__(in_dim=in_dim, out_dim=3)


class ApproachHead(_PooledMLP):
    def __init__(self, in_dim: int = 256) -> None:
        super().__init__(in_dim=in_dim, out_dim=3)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        x = super().forward(fused)
        return F.normalize(x, dim=-1)


class ActionChunkHead(nn.Module):
    def __init__(self, in_dim: int = 256, chunk: int = 8, action_dim: int = 7) -> None:
        super().__init__()
        self.chunk = int(chunk)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.chunk * self.action_dim),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        pooled = fused.mean(dim=1)
        x = self.net(pooled)
        return x.reshape(-1, self.chunk, self.action_dim)


class BoundedContactResidualHead(nn.Module):
    """Apply a small contact-aware XYZ residual to a baseline 7D action chunk."""

    def __init__(
        self,
        in_dim: int = 256,
        contact_dim: int = 256,
        chunk: int = 8,
        max_translation: float = 0.03,
    ) -> None:
        super().__init__()
        self.chunk = int(chunk)
        self.max_translation = float(max_translation)
        self.net = nn.Sequential(
            nn.Linear(in_dim + contact_dim + 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.chunk * 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        baseline_action: torch.Tensor,
        fused: torch.Tensor,
        contact_token: torch.Tensor,
        contact_xyz: torch.Tensor,
        tcp_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_xyz = tcp_position_torch(tcp_pose).to(contact_xyz.dtype)
        context = torch.cat(
            [fused.mean(dim=1), contact_token, contact_xyz - tcp_xyz], dim=-1
        )
        residual = self.max_translation * torch.tanh(
            self.net(context).reshape(-1, self.chunk, 3)
        )
        action = torch.cat(
            [baseline_action[..., :3] + residual, baseline_action[..., 3:]], dim=-1
        )
        return action, residual


class ContactRelativeOffsetHead(nn.Module):
    """Replace world-XYZ cloning with contact-relative TCP offsets.

    Executed translation is ``contact_xyz + offset``. Orientation and gripper
    stay on the baseline 7D action head. Offset is bounded, but the cap is the
    TCP-to-contact workspace (tens of cm), not the old ±2 cm residual.
    """

    def __init__(
        self,
        in_dim: int = 256,
        contact_dim: int = 256,
        chunk: int = 8,
        max_translation: float = 0.40,
    ) -> None:
        super().__init__()
        self.chunk = int(chunk)
        self.max_translation = float(max_translation)
        self.net = nn.Sequential(
            nn.Linear(in_dim + contact_dim + 3, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.chunk * 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        baseline_action: torch.Tensor,
        fused: torch.Tensor,
        contact_token: torch.Tensor,
        contact_xyz: torch.Tensor,
        tcp_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_xyz = tcp_position_torch(tcp_pose).to(contact_xyz.dtype)
        context = torch.cat(
            [fused.mean(dim=1), contact_token, contact_xyz - tcp_xyz], dim=-1
        )
        offset = self.max_translation * torch.tanh(
            self.net(context).reshape(-1, self.chunk, 3)
        )
        relative = torch.cat([offset, baseline_action[..., 3:]], dim=-1)
        action = contact_relative_to_absolute_actions(relative, contact_xyz)
        return action, offset


class DiffusionActionHead(nn.Module):
    """Small conditional DDPM head for action chunks.

    This is not a standalone pretrained diffusion model. It is a lightweight
    action head conditioned on the shared part-grounded features.
    """

    def __init__(
        self,
        in_dim: int = 256,
        chunk: int = 8,
        action_dim: int = 7,
        hidden: int = 256,
        num_steps: int = 50,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        self.chunk = int(chunk)
        self.action_dim = int(action_dim)
        self.num_steps = int(num_steps)
        self.action_flat_dim = self.chunk * self.action_dim

        self.cond = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.time = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(self.action_flat_dim + hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.action_flat_dim),
        )

        betas = torch.linspace(beta_start, beta_end, self.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def _time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[1]))
        return emb

    def _denoise(self, fused: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pooled = fused.mean(dim=1)
        cond = self.cond(pooled)
        t_emb = self.time(self._time_embedding(t, cond.shape[-1]).to(cond.dtype))
        x = noisy_action.reshape(noisy_action.shape[0], -1)
        pred = self.net(torch.cat([x, cond, t_emb], dim=-1))
        return pred.reshape(-1, self.chunk, self.action_dim)

    def forward(self, fused: torch.Tensor, target_action: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        if target_action is None:
            return {"action_chunk": self.sample(fused)}

        bsz = target_action.shape[0]
        t = torch.randint(0, self.num_steps, (bsz,), device=target_action.device)
        noise = torch.randn_like(target_action)
        alpha_bar = self.alpha_bars[t].view(bsz, 1, 1).to(target_action.dtype)
        noisy = alpha_bar.sqrt() * target_action + (1.0 - alpha_bar).sqrt() * noise
        pred_noise = self._denoise(fused, noisy, t)
        return {
            "action_noise_pred": pred_noise,
            "action_noise": noise,
            "action_noisy": noisy,
            "diffusion_t": t,
        }

    @torch.no_grad()
    def sample(self, fused: torch.Tensor) -> torch.Tensor:
        bsz = fused.shape[0]
        x = torch.randn(bsz, self.chunk, self.action_dim, device=fused.device, dtype=fused.dtype)
        for step in reversed(range(self.num_steps)):
            t = torch.full((bsz,), step, device=fused.device, dtype=torch.long)
            pred_noise = self._denoise(fused, x, t)
            beta = self.betas[step].to(x.dtype)
            alpha = self.alphas[step].to(x.dtype)
            alpha_bar = self.alpha_bars[step].to(x.dtype)
            mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x


class GroundedHybridActionHead(nn.Module):
    """Predict arm trajectories and binary grasp timing with separate heads."""

    def __init__(
        self,
        in_dim: int = 256,
        contact_dim: int = 256,
        chunk: int = 8,
        arm_dim: int = 6,
        arm_head_type: str = "mlp",
        diffusion_steps: int = 50,
        open_width: float = 0.04,
        closed_width: float = 0.0,
    ) -> None:
        super().__init__()
        self.chunk = int(chunk)
        self.arm_head_type = arm_head_type.lower()
        self.open_width = float(open_width)
        self.closed_width = float(closed_width)
        self.contact_to_fused = nn.Linear(contact_dim, in_dim)
        self.arm_head = (
            DiffusionActionHead(in_dim, chunk, arm_dim, num_steps=diffusion_steps)
            if self.arm_head_type == "diffusion"
            else ActionChunkHead(in_dim, chunk, arm_dim)
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(in_dim + contact_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.chunk),
        )

    def forward(
        self,
        fused: torch.Tensor,
        contact_token: torch.Tensor,
        target_action: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        conditioned = fused + self.contact_to_fused(contact_token)[:, None, :]
        pooled = conditioned.mean(dim=1)
        gripper_logits = self.gripper_head(torch.cat([pooled, contact_token], dim=-1))
        arm_target = None if target_action is None else target_action[..., :6]
        if self.arm_head_type == "diffusion":
            arm_out = self.arm_head(conditioned, target_action=arm_target)
        else:
            arm_out = {"arm_action_chunk": self.arm_head(conditioned)}
        out = {**arm_out, "gripper_logits": gripper_logits}
        if "action_chunk" in out:
            out["arm_action_chunk"] = out.pop("action_chunk")
        if "arm_action_chunk" in out:
            closed = gripper_logits >= 0
            grip = torch.where(
                closed,
                torch.full_like(gripper_logits, self.closed_width),
                torch.full_like(gripper_logits, self.open_width),
            )
            out["action_chunk"] = torch.cat(
                [out["arm_action_chunk"], grip.unsqueeze(-1)], dim=-1
            )
        return out



class ContactGeometryConditioner(nn.Module):
    """Add detached predicted world contact relative to TCP to action features.

    Disabled conditioning feeds zeros through the same projection for a matched
    control. Nonfinite contact/TCP geometry likewise contributes zero inputs.
    """
    def __init__(self, hidden_dim: int, scale_m: float = 0.1):
        super().__init__()
        if scale_m <= 0:
            raise ValueError('contact_geometry_scale_m must be positive')
        self.scale_m = float(scale_m)
        self.projection = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, features, contact_xyz, tcp_pose, enabled=True):
        from data.geometry import tcp_position_torch
        relative = (contact_xyz.detach() - tcp_position_torch(tcp_pose).detach()).reshape(-1, 3)
        valid = torch.isfinite(relative).all(dim=-1, keepdim=True)
        relative = torch.where(valid, relative, torch.zeros_like(relative)) / self.scale_m
        if not enabled:
            relative = torch.zeros_like(relative)
        projected = self.projection(relative)
        if features.ndim == 3:
            projected = projected[:, None, :]
        return features + projected, relative
