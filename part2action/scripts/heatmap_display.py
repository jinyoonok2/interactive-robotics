"""Shared qualitative heatmap rendering helpers."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_heatmap_to_image(heatmap: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Resize a single heatmap tensor to image size and return it as numpy."""
    if heatmap.shape != (h, w):
        heatmap = F.interpolate(
            heatmap.view(1, 1, *heatmap.shape),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return heatmap.detach().cpu().float().numpy()


def normalize_heatmap_for_display(heatmap: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    """Contrast-normalize a heatmap for visualization only.

    The model still emits absolute probabilities. This helper makes offline
    figures and rollout videos use the same qualitative display scale.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        return np.zeros_like(heatmap, dtype=np.float32)

    lo = float(np.percentile(finite, lower_pct))
    hi = float(np.percentile(finite, upper_pct))
    if hi <= lo + 1e-6:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo + 1e-6:
        return np.zeros_like(heatmap, dtype=np.float32)

    normalized = (heatmap - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)
