"""Learned task plans and state-grounded skill phase selection."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkillPlanDecoder(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int = 256, max_skill_slots: int = 4,
                 num_layers: int = 2, num_heads: int = 4) -> None:
        super().__init__()
        self.max_skill_slots = int(max_skill_slots)
        self.memory_proj = nn.Linear(text_dim, hidden_dim)
        self.slot_queries = nn.Parameter(torch.randn(self.max_skill_slots, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.slot_proj = nn.Linear(hidden_dim, text_dim)
        self.valid_head = nn.Linear(hidden_dim, 1)

    def forward(self, task_tokens: torch.Tensor, task_mask: torch.Tensor | None = None):
        queries = self.slot_queries.unsqueeze(0).expand(task_tokens.shape[0], -1, -1)
        decoded = self.decoder(
            tgt=queries,
            memory=self.memory_proj(task_tokens),
            memory_key_padding_mask=None if task_mask is None else ~task_mask.bool(),
        )
        return F.normalize(self.slot_proj(decoded), dim=-1), self.valid_head(decoded).squeeze(-1)


class PhaseSelector(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.slot = nn.Linear(text_dim, hidden_dim)

    def forward(self, visual_tokens: torch.Tensor, skill_slots: torch.Tensor,
                pcd_embedding: torch.Tensor | None = None,
                valid_mask: torch.Tensor | None = None,
                current_phase: torch.Tensor | None = None,
                use_3d: bool = True) -> torch.Tensor:
        visual_state = visual_tokens.mean(dim=1)
        if pcd_embedding is None or not use_3d:
            pcd_embedding = torch.zeros_like(visual_state)
        state = self.state(torch.cat([visual_state, pcd_embedding], dim=-1))
        logits = torch.einsum("bd,bkd->bk", state, self.slot(skill_slots)) / state.shape[-1] ** 0.5
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.bool(), torch.finfo(logits.dtype).min)
        if current_phase is not None:
            indices = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
            logits = logits.masked_fill(indices < current_phase[:, None], torch.finfo(logits.dtype).min)
        return logits


class SkillTerminationHead(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim * 2 + text_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, visual_tokens: torch.Tensor, selected_skill: torch.Tensor,
                pcd_embedding: torch.Tensor | None = None) -> torch.Tensor:
        visual_state = visual_tokens.mean(dim=1)
        if pcd_embedding is None:
            pcd_embedding = torch.zeros_like(visual_state)
        return self.net(torch.cat([visual_state, pcd_embedding, selected_skill], dim=-1)).squeeze(-1)
