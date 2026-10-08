"""Last-centimeter contact: cloned XYZ while far, contact plus a 3 cm offset when near."""
from __future__ import annotations

import torch

from data.geometry import apply_near_contact_translation, near_contact_mask
from models.heads import ContactRelativeOffsetHead
from models.part2action_model import Part2ActionModel


def test_far_hand_keeps_cloned_xyz_and_near_hand_uses_contact():
    baseline = torch.tensor(
        [
            [[0.50, 0.10, 0.20, 0.1, 0.2, 0.3, 0.04]],
            [[0.08, 0.02, 0.16, 0.4, 0.5, 0.6, 0.00]],
        ]
    )
    contact_action = torch.tensor(
        [
            [[0.02, 0.00, 0.15, 9.0, 9.0, 9.0, 9.0]],
            [[0.11, 0.01, 0.18, 9.0, 9.0, 9.0, 9.0]],
        ]
    )
    tcp = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.80, 0.10, 0.20],
            [0.0, 0.0, 0.0, 1.0, 0.10, 0.00, 0.16],
        ]
    )
    contact = torch.tensor(
        [
            [0.02, 0.00, 0.15],
            [0.11, 0.01, 0.18],
        ]
    )
    near = near_contact_mask(tcp, contact, max_distance=0.03)
    assert torch.equal(near, torch.tensor([False, True]))
    action = apply_near_contact_translation(baseline, contact_action, near)
    assert torch.allclose(action[0, :, :3], baseline[0, :, :3])
    assert torch.allclose(action[1, :, :3], contact_action[1, :, :3])
    assert torch.allclose(action[..., 3:], baseline[..., 3:])


def test_invalid_contact_stays_on_the_cloned_command():
    tcp = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.10, 0.00, 0.16]])
    contact = torch.tensor([[0.11, 0.01, 0.18]])
    near = near_contact_mask(
        tcp, contact, max_distance=0.03, valid=torch.tensor([0.0])
    )
    assert torch.equal(near, torch.tensor([False]))


def test_zero_init_near_offset_is_capped_at_3cm():
    torch.manual_seed(0)
    head = ContactRelativeOffsetHead(chunk=8, max_translation=0.03)
    baseline = torch.randn(2, 8, 7)
    fused = torch.randn(2, 4, 256)
    token = torch.randn(2, 256)
    contact = torch.tensor([[0.02, -0.01, 0.15], [0.04, 0.02, 0.12]])
    tcp = torch.zeros(2, 7)
    action, offset = head(baseline, fused, token, contact, tcp)
    assert torch.allclose(offset, torch.zeros_like(offset))
    assert offset.abs().max() <= 0.03
    assert torch.allclose(action[..., :3], contact[:, None, :].expand_as(action[..., :3]))
    assert torch.allclose(action[..., 3:], baseline[..., 3:])


def test_near_contact_rejects_the_full_path_offset():
    try:
        Part2ActionModel(
            heads=["action"],
            use_pcd=True,
            use_contact_xyz=True,
            use_contact_action_token=True,
            use_contact_relative_actions=True,
            use_near_contact_actions=True,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected mutual exclusion error")


if __name__ == "__main__":
    test_far_hand_keeps_cloned_xyz_and_near_hand_uses_contact()
    test_invalid_contact_stays_on_the_cloned_command()
    test_zero_init_near_offset_is_capped_at_3cm()
    test_near_contact_rejects_the_full_path_offset()
    print("near-contact tests passed")
