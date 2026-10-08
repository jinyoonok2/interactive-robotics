"""Measured finger opening should change the predicted action."""
from __future__ import annotations

import torch

from models.part2action_model import Part2ActionModel


def test_open_and_closed_fingers_change_the_action():
    torch.manual_seed(0)
    model = Part2ActionModel(
        heads=["action"],
        text_device="cpu",
        use_hierarchy=False,
        use_part_gate=False,
        use_gripper_state=True,
    )
    model.eval()
    rgb = torch.rand(2, 3, 252, 252)
    instructions = ["grasp the handle", "touch the cap"]
    with torch.no_grad():
        closed = model(rgb, instructions, gripper_state=torch.zeros(2, 1))["action_chunk"]
        opened = model(
            rgb, instructions, gripper_state=torch.full((2, 1), 0.04)
        )["action_chunk"]
    assert closed.shape == (2, 8, 7)
    assert not torch.allclose(closed, opened)


def test_gripper_state_is_required_when_enabled():
    model = Part2ActionModel(
        heads=["action"],
        text_device="cpu",
        use_hierarchy=False,
        use_part_gate=False,
        use_gripper_state=True,
    )
    model.eval()
    try:
        model(torch.rand(1, 3, 252, 252), ["grasp the handle"])
    except ValueError as exc:
        assert "gripper_state" in str(exc)
    else:
        raise AssertionError("expected missing gripper_state to fail")


if __name__ == "__main__":
    test_open_and_closed_fingers_change_the_action()
    test_gripper_state_is_required_when_enabled()
    print("gripper-state input tests passed")
