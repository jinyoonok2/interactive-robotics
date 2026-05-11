"""Frozen visual + text backbones for part2action.

Visual: DINOv2 ViT-S/14 (frozen), 384-dim per patch token. We try to load
        from the local torch.hub cache first so the workstation can run
        offline; if that fails we fall back to torch.hub.load(...) which
        requires network access.
Text:   Flan-T5-small encoder (frozen), 512-dim per token. Runs on CPU
        by default to keep VRAM free for training (8 GB GPU budget).

Both backbones are loaded once and never updated. Only the cross-attention
fusion module + heads are trainable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel


_HUB_DIR_DEFAULT = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
_CKPT_DIR_DEFAULT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


class FrozenDINOv2(nn.Module):
    """ViT-S/14 from torch hub, frozen. Returns (B, N_patches, 384).

    The cached checkpoint on the workstation is the registers variant
    (`dinov2_vits14_reg4_pretrain.pth`), so the default model_name is
    `dinov2_vits14_reg`. Override via the `model_name` arg if a non-reg
    checkpoint is available.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14_reg",
        img_size: int = 252,
        local_hub_dir: Optional[str] = None,
        local_ckpt: Optional[str] = None,
    ) -> None:
        super().__init__()
        if img_size % 14 != 0:
            raise ValueError(f"img_size must be a multiple of 14, got {img_size}")
        self.img_size = int(img_size)
        self.grid = self.img_size // 14
        self.embed_dim = 384

        hub_dir = Path(local_hub_dir or os.environ.get("DINOV2_HUB_DIR", _HUB_DIR_DEFAULT))
        ckpt_path = local_ckpt or os.environ.get("DINOV2_CKPT")
        if ckpt_path is None:
            default_name = (
                "dinov2_vits14_reg4_pretrain.pth"
                if "reg" in model_name
                else "dinov2_vits14_pretrain.pth"
            )
            cand = _CKPT_DIR_DEFAULT / default_name
            if cand.exists():
                ckpt_path = str(cand)

        try:
            self.model = torch.hub.load(
                str(hub_dir), model_name, source="local", pretrained=False
            )
            if ckpt_path is not None and Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict(state, strict=True)
            else:
                raise FileNotFoundError(
                    "No DINOv2 checkpoint found locally; set DINOV2_CKPT env var."
                )
        except Exception as e_local:
            print(f"[FrozenDINOv2] local load failed ({e_local}); trying online torch.hub")
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-2:] != (self.img_size, self.img_size):
            rgb = F.interpolate(rgb, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        out = self.model.forward_features(rgb)
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            return out["x_norm_patchtokens"]
        if isinstance(out, torch.Tensor):
            return out[:, 1:, :]
        raise RuntimeError(f"Unexpected DINOv2 output type: {type(out)}")


class FrozenT5(nn.Module):
    """Flan-T5-small encoder, frozen. Returns (B, T_text, 512)."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = "cpu",
        max_length: int = 32,
    ) -> None:
        super().__init__()
        self.device_str = device
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().to(self.device_str)
        self.embed_dim = int(self.model.config.d_model)

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        toks = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device_str)
        out = self.model(input_ids=toks.input_ids, attention_mask=toks.attention_mask)
        return out.last_hidden_state, toks.attention_mask
