"""Unit tests for oracle part-mask gate override and grounding helpers."""
from __future__ import annotations

import inspect

import torch

from models.part2action_model import Part2ActionModel


def test_mask_to_gate_logits_square_grid():
    mask = torch.zeros(2, 64, 64)
    mask[:, 16:48, 16:48] = 1.0
    logits = Part2ActionModel.mask_to_gate_logits(mask, num_tokens=16, logit_scale=8.0)
    assert logits.shape == (2, 16)
    probs = torch.sigmoid(logits)
    assert float(probs.max()) > 0.9
    assert float(probs.min()) < 0.1


def test_apply_part_gate_residual_modulation():
    tokens = torch.ones(1, 4, 8)
    logits = torch.tensor([[8.0, -8.0, 8.0, -8.0]])
    gated, out_logits = Part2ActionModel.apply_part_gate(tokens, logits)
    assert torch.equal(out_logits, logits)
    # High gate ≈ 2x token; low gate ≈ 1x token.
    assert torch.allclose(gated[0, 0], tokens[0, 0] * 2.0, atol=1e-3)
    assert torch.allclose(gated[0, 1], tokens[0, 1] * 1.0, atol=1e-2)


def test_forward_exposes_oracle_gate_args():
    sig = inspect.signature(Part2ActionModel.forward)
    assert "gate_mode" in sig.parameters
    assert "gt_part_mask" in sig.parameters
