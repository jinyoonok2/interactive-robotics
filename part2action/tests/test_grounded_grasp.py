import numpy as np
import torch

from data.targets import (
    _detect_contact_step,
    derive_contact_xyz,
    derive_gripper_targets,
)
from data.geometry import (
    camera_to_world_numpy,
    camera_to_world_torch,
    tcp_position_torch,
)
from models.heads import (
    BoundedContactResidualHead,
    GroundedHybridActionHead,
    PointCloudEncoder,
    PointwiseContactAttention,
    SkillConditionedPartGate,
    SkillConditionedPointPartHead,
)


def test_contact_and_gripper_targets_use_action_commands():
    actions = np.zeros((6, 7), dtype=np.float32)
    actions[:3, 6] = 0.04
    assert _detect_contact_step(actions, 0) == 3
    assert _detect_contact_step(actions, 4) == 4

    closed, weights = derive_gripper_targets(
        actions[1:5], previous_command=0.04, transition_weight=6.0
    )
    np.testing.assert_array_equal(closed, [0, 0, 1, 1])
    np.testing.assert_array_equal(weights, [1, 1, 6, 1])


def test_contact_target_is_selected_from_current_frame():
    part_pcd = np.zeros((2, 3, 4), dtype=np.float32)
    part_pcd[..., 3] = 1
    part_pcd[0, :, :3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    part_pcd[1, :, :3] = [[10, 0, 0], [11, 0, 0], [12, 0, 0]]
    tcp = np.zeros((2, 7), dtype=np.float32)
    tcp[1, 4:7] = camera_to_world_numpy(
        np.array([[1.1, 0, 0]], dtype=np.float32)
    )[0]
    xyz, valid = derive_contact_xyz(part_pcd, tcp, contact_t=1, current_t=0)
    assert valid == 1.0
    np.testing.assert_allclose(
        xyz, camera_to_world_numpy(np.array([[1, 0, 0]], dtype=np.float32))[0]
    )


def test_pointcloud_encoder_uses_world_frame_tcp_delta():
    scene_camera = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.1, -0.2, 1.2]]], dtype=torch.float32
    )
    tcp_pose = torch.tensor(
        [[0.25, 0.5, 0.75, 1.0, 0.55, 0.60, 0.35]], dtype=torch.float32
    )
    encoder = PointCloudEncoder()
    captured = {}

    def capture_features(_module, inputs):
        captured["features"] = inputs[0].detach()

    handle = encoder.point_mlp[0].register_forward_pre_hook(capture_features)
    try:
        encoder(scene_camera, tcp_pose=tcp_pose)
    finally:
        handle.remove()

    features = captured["features"]
    world = camera_to_world_torch(scene_camera)
    torch.testing.assert_close(features[..., :3], world)
    torch.testing.assert_close(features[..., 3:7], torch.zeros_like(features[..., 3:7]))
    torch.testing.assert_close(
        features[..., 7:10],
        world - tcp_position_torch(tcp_pose)[:, None, :],
    )


def test_grounded_modules_preserve_shapes_and_physical_gripper_values():
    torch.manual_seed(0)
    batch, patches, points = 2, 16, 32
    visual = torch.randn(batch, patches, 384)
    skill = torch.randn(batch, 512)
    scene = torch.randn(batch, points, 3)
    tcp = torch.randn(batch, 7)

    gate = SkillConditionedPartGate()
    gated, gate_logits = gate(visual, skill)
    assert gated.shape == visual.shape
    assert gate_logits.shape == (batch, patches)

    pcd_encoder = PointCloudEncoder()
    point_tokens, pooled = pcd_encoder(scene, tcp_pose=tcp, return_tokens=True)
    assert point_tokens.shape == (batch, points, 384)
    assert pooled.shape == (batch, 384)

    point_part = SkillConditionedPointPartHead()
    point_part_logits = point_part(point_tokens, gated, skill)
    assert point_part_logits.shape == (batch, points)

    contact = PointwiseContactAttention()
    world = camera_to_world_torch(scene)
    xyz, token, logits = contact(
        point_tokens, scene, gated, skill, tcp, world_xyz=world
    )
    assert xyz.shape == (batch, 3)
    assert token.shape == (batch, 256)
    assert logits.shape == (batch, points)
    torch.testing.assert_close(torch.softmax(logits, -1).sum(-1), torch.ones(batch))

    action_head = GroundedHybridActionHead(chunk=8)
    out = action_head(torch.randn(batch, patches, 256), token)
    assert out["arm_action_chunk"].shape == (batch, 8, 6)
    assert out["gripper_logits"].shape == (batch, 8)
    assert out["action_chunk"].shape == (batch, 8, 7)
    values = out["action_chunk"][..., 6]
    assert torch.logical_or(torch.isclose(values, torch.tensor(0.0)), torch.isclose(values, torch.tensor(0.04))).all()

    baseline = torch.randn(batch, 8, 7)
    residual_head = BoundedContactResidualHead(chunk=8, max_translation=0.02)
    corrected, residual = residual_head(
        baseline, torch.randn(batch, patches, 256), token, xyz, tcp
    )
    torch.testing.assert_close(corrected, baseline)
    assert residual.abs().max() <= 0.02


if __name__ == "__main__":
    test_contact_and_gripper_targets_use_action_commands()
    test_contact_target_is_selected_from_current_frame()
    test_pointcloud_encoder_uses_world_frame_tcp_delta()
    test_grounded_modules_preserve_shapes_and_physical_gripper_values()
    print("grounded grasp tests passed")
