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
