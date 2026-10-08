"""Tests for contact-relative XYZ offsets replacing the ±2 cm residual."""
from __future__ import annotations

import torch

from data.geometry import (
    absolute_actions_to_contact_relative,
    contact_relative_to_absolute_actions,
)
from models.heads import ContactRelativeOffsetHead
from models.part2action_model import Part2ActionModel


def test_contact_relative_roundtrip():
    contact = torch.tensor([[0.10, -0.05, 0.18]])
    absolute = torch.tensor(
        [
            [
                [0.12, -0.04, 0.20, 0.1, 0.0, -0.2, 0.04],
                [0.11, -0.05, 0.19, 0.1, 0.0, -0.2, 0.00],
            ]
        ]
    )
    relative = absolute_actions_to_contact_relative(absolute, contact)
    assert torch.allclose(relative[..., :3], absolute[..., :3] - contact[:, None, :])
    assert torch.allclose(relative[..., 3:], absolute[..., 3:])
    recovered = contact_relative_to_absolute_actions(relative, contact)
    assert torch.allclose(recovered, absolute, atol=1e-6)


def test_zero_init_offset_executes_at_predicted_contact():
    torch.manual_seed(0)
    batch, patches, chunk = 2, 8, 8
    head = ContactRelativeOffsetHead(chunk=chunk, max_translation=0.40)
    baseline = torch.randn(batch, chunk, 7)
    fused = torch.randn(batch, patches, 256)
    contact_token = torch.randn(batch, 256)
    contact_xyz = torch.tensor([[0.02, -0.03, 0.16], [0.05, 0.01, 0.14]])
    tcp = torch.randn(batch, 7)
    action, offset = head(baseline, fused, contact_token, contact_xyz, tcp)
    assert torch.allclose(offset, torch.zeros_like(offset))
    assert torch.allclose(action[..., :3], contact_xyz[:, None, :].expand_as(action[..., :3]))
    assert torch.allclose(action[..., 3:], baseline[..., 3:])
    assert offset.abs().max() <= 0.40


def test_contact_relative_exclusive_with_residual():
    try:
        Part2ActionModel(
            heads=["action"],
            use_pcd=True,
            use_contact_xyz=True,
            use_contact_action_token=True,
            use_contact_residual=True,
            use_contact_relative_actions=True,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutual exclusion error")


if __name__ == "__main__":
    test_contact_relative_roundtrip()
    test_zero_init_offset_executes_at_predicted_contact()
    test_contact_relative_exclusive_with_residual()
    print("contact-relative tests passed")
