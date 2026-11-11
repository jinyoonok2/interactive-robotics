"""Matched feature masks for the existing point-to-interaction-pose head.

All variants keep the same 52-dimensional input and ``InteractionPoseHead``
architecture. Masking removes selected information, without changing parameter
counts. The separate ``point`` argument to that head must stay identical across
variants: each predicts ``point + bounded_offset`` around the same oracle-selected
surface point. In particular, ``object_geometry`` is NOT an unlocalized or
oracle-free baseline.

The indices below follow ``interaction_pose_head.pose_features`` exactly:

    0:3   point / .5                    3:6   (point - TCP position) / .1
    6:12  current TCP orientation6d     12:13  current opening / .04
    13:16 object centroid - point      16:19  part centroid - point
    19:22 local part mean              22:25  local part std
    25:28 local part minimum           28:31  local part maximum
    31:37 local covariance upper triangle, NumPy triu order
    37:40 object extent                40:43  part extent
    43:52 semantic part one-hot, in interaction_pose_head.PARTS order

Centroids are scaled by .1, local statistics by .05, covariance by .0025, and
extents by .1. The local neighborhood comprises up to 64 nearest part points.
``part_geometry`` keeps geometric descriptors but excludes semantic identity;
``part_geometry_semantics`` additionally retains the semantic part one-hot.
"""
from __future__ import annotations

import numpy as np
import torch

from .interaction_pose_head import FEATURE_DIM


FEATURE_GROUPS = {
    'point': slice(0, 3),
    'point_relative_tcp': slice(3, 6),
    'tcp_orientation': slice(6, 12),
    'opening': slice(12, 13),
    'object_centroid': slice(13, 16),
    'part_centroid': slice(16, 19),
    'local_mean': slice(19, 22),
    'local_std': slice(22, 25),
    'local_min': slice(25, 28),
    'local_max': slice(28, 31),
    'local_covariance': slice(31, 37),
    'object_extent': slice(37, 40),
    'part_extent': slice(40, 43),
    'part_semantic': slice(43, 52),
}

_COMMON_GROUPS = ('point', 'point_relative_tcp', 'tcp_orientation', 'opening')
FEATURE_VARIANTS = {
    'point_state': _COMMON_GROUPS,
    'object_geometry': _COMMON_GROUPS + ('object_centroid', 'object_extent'),
    'part_geometry': tuple(group for group in FEATURE_GROUPS if group != 'part_semantic'),
    'part_geometry_semantics': tuple(FEATURE_GROUPS),
}


def feature_mask(variant: str) -> np.ndarray:
    """Return a fresh 52D Boolean mask for one declared feature variant."""
    if not isinstance(variant, str) or variant not in FEATURE_VARIANTS:
        raise ValueError(f'Unknown pose feature variant {variant!r}; '
                         f'choose from {tuple(FEATURE_VARIANTS)}')
    mask = np.zeros(FEATURE_DIM, dtype=bool)
    for group in FEATURE_VARIANTS[variant]:
        mask[FEATURE_GROUPS[group]] = True
    return mask


def apply_feature_variant(features, variant: str):
    """Zero excluded channels without changing shape, dtype, or the input.

    Accept a NumPy ndarray or PyTorch Tensor with final dimension 52; arbitrary
    batch dimensions are retained. Tensor outputs stay on the same device and
    preserve gradients on retained channels. Even the all-features variant
    returns independent storage, so callers cannot mutate the original input.
    This function never changes or supplies the head's separate point anchor.
    """
    mask = feature_mask(variant)
    if not isinstance(features, (np.ndarray, torch.Tensor)):
        raise TypeError('features must be a NumPy ndarray or PyTorch Tensor')
    if features.ndim < 1 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(f'Expected final feature dimension {FEATURE_DIM}, '
                         f'got shape {tuple(features.shape)}')
    if isinstance(features, torch.Tensor):
        keep = torch.as_tensor(mask, device=features.device)
        return features.masked_fill(~keep, 0)
    return np.where(mask, features, np.zeros((), dtype=features.dtype))
