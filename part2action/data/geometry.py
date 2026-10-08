"""Coordinate helpers for PartInstruct's fixed agent-view camera."""
from __future__ import annotations

import numpy as np
import torch

# Inverse of PyBullet's fixed agent-view matrix from PartGym config.yaml:
# target=[-0.25, 0.05, 0.5], distance=0.8, yaw=-90, pitch=-20.
_AGENTVIEW_TO_WORLD = np.array(
    [
        [-1.1886688805285457e-07, 0.3420201314511609, -0.9396926363933252, -1.0017542047461594],
        [-0.9999999999999851, -1.490115808782253e-08, 1.2107193606680588e-07, 0.050000096319126594],
        [2.7406526051636005e-08, 0.9396926363933399, 0.34202013145115934, 0.7736161645470457],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def camera_to_world_numpy(points: np.ndarray) -> np.ndarray:
    """Transform Open3D camera points (x right, y down, z forward) to world."""
    points = np.asarray(points, dtype=np.float32)
    gl_points = points[..., :3].copy()
    gl_points[..., 1:3] *= -1
    ones = np.ones((*gl_points.shape[:-1], 1), dtype=np.float32)
    homogeneous = np.concatenate([gl_points, ones], axis=-1)
    return (homogeneous @ _AGENTVIEW_TO_WORLD.T)[..., :3]


def camera_to_world_torch(points: torch.Tensor) -> torch.Tensor:
    """Torch version of :func:`camera_to_world_numpy`."""
    gl_points = torch.cat(
        [points[..., :1], -points[..., 1:2], -points[..., 2:3]], dim=-1
    )
    ones = torch.ones_like(gl_points[..., :1])
    homogeneous = torch.cat([gl_points, ones], dim=-1)
    transform = torch.as_tensor(
        _AGENTVIEW_TO_WORLD, device=points.device, dtype=points.dtype
    )
    return torch.matmul(homogeneous, transform.transpose(0, 1))[..., :3]


def tcp_position_numpy(tcp_pose: np.ndarray) -> np.ndarray:
    """PartInstruct stores TCP as quaternion (xyzw) followed by world XYZ."""
    pose = np.asarray(tcp_pose, dtype=np.float32)
    if pose.shape[-1] < 7:
        raise ValueError(f"Expected TCP pose (...,7), got {pose.shape}")
    return pose[..., 4:7]


def tcp_position_torch(tcp_pose: torch.Tensor) -> torch.Tensor:
    if tcp_pose.shape[-1] < 7:
        raise ValueError(f"Expected TCP pose (...,7), got {tuple(tcp_pose.shape)}")
    return tcp_pose[..., 4:7]


def absolute_actions_to_tcp_deltas(
    action_chunk: torch.Tensor, tcp_pose: torch.Tensor
) -> torch.Tensor:
    """Convert absolute EE pose chunks to XYZ deltas relative to current TCP.

    Orientation (dims 3:6) and gripper (dim 6) are left unchanged. This is the
    minimal action-representation change used for the delta-action ablation.
    """
    if action_chunk.ndim != 3 or action_chunk.shape[-1] < 3:
        raise ValueError(
            f"Expected action_chunk (B,T,>=3), got {tuple(action_chunk.shape)}"
        )
    tcp_xyz = tcp_position_torch(tcp_pose).to(
        device=action_chunk.device, dtype=action_chunk.dtype
    )
    deltas = action_chunk.clone()
    deltas[..., :3] = action_chunk[..., :3] - tcp_xyz[:, None, :]
    return deltas


def tcp_deltas_to_absolute_actions(
    delta_chunk: torch.Tensor, tcp_pose: torch.Tensor
) -> torch.Tensor:
    """Convert XYZ TCP deltas (+ absolute orientation/gripper) back to EE poses."""
    if delta_chunk.ndim != 3 or delta_chunk.shape[-1] < 3:
        raise ValueError(
            f"Expected delta_chunk (B,T,>=3), got {tuple(delta_chunk.shape)}"
        )
    tcp_xyz = tcp_position_torch(tcp_pose).to(
        device=delta_chunk.device, dtype=delta_chunk.dtype
    )
    absolute = delta_chunk.clone()
    absolute[..., :3] = delta_chunk[..., :3] + tcp_xyz[:, None, :]
    return absolute


def _as_contact_xyz(contact_xyz: torch.Tensor) -> torch.Tensor:
    if contact_xyz.ndim != 2 or contact_xyz.shape[-1] != 3:
        raise ValueError(f"Expected contact_xyz (B,3), got {tuple(contact_xyz.shape)}")
    return contact_xyz


def absolute_actions_to_contact_relative(
    action_chunk: torch.Tensor, contact_xyz: torch.Tensor
) -> torch.Tensor:
    """Convert absolute EE pose chunks to XYZ offsets relative to a contact point.

    Orientation (dims 3:6) and gripper (dim 6) are left unchanged. This is the
    last-centimeter action parameterization: the policy aims at contact, then
    learns the TCP-to-contact offset instead of a ±2 cm residual on world XYZ.
    """
    if action_chunk.ndim != 3 or action_chunk.shape[-1] < 3:
        raise ValueError(
            f"Expected action_chunk (B,T,>=3), got {tuple(action_chunk.shape)}"
        )
    contact = _as_contact_xyz(contact_xyz).to(
        device=action_chunk.device, dtype=action_chunk.dtype
    )
    relative = action_chunk.clone()
    relative[..., :3] = action_chunk[..., :3] - contact[:, None, :]
    return relative


def contact_relative_to_absolute_actions(
    relative_chunk: torch.Tensor, contact_xyz: torch.Tensor
) -> torch.Tensor:
    """Convert contact-relative XYZ offsets (+ absolute ori/gripper) to EE poses."""
    if relative_chunk.ndim != 3 or relative_chunk.shape[-1] < 3:
        raise ValueError(
            f"Expected relative_chunk (B,T,>=3), got {tuple(relative_chunk.shape)}"
        )
    contact = _as_contact_xyz(contact_xyz).to(
        device=relative_chunk.device, dtype=relative_chunk.dtype
    )
    absolute = relative_chunk.clone()
    absolute[..., :3] = relative_chunk[..., :3] + contact[:, None, :]
    return absolute


def near_contact_mask(
    tcp_pose: torch.Tensor,
    contact_xyz: torch.Tensor,
    max_distance: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """True where the hand is already within ``max_distance`` meters of contact."""
    contact = _as_contact_xyz(contact_xyz)
    tcp_xyz = tcp_position_torch(tcp_pose).to(device=contact.device, dtype=contact.dtype)
    if tcp_xyz.ndim == 1:
        tcp_xyz = tcp_xyz.unsqueeze(0)
    near = torch.linalg.norm(contact - tcp_xyz, dim=-1) <= float(max_distance)
    if valid is not None:
        near = near & (valid.reshape(-1).to(device=near.device) > 0.5)
    return near


def apply_near_contact_translation(
    baseline_action: torch.Tensor,
    contact_action: torch.Tensor,
    near: torch.Tensor,
) -> torch.Tensor:
    """Use contact XYZ only on near samples. Wrist and gripper stay on the baseline."""
    if baseline_action.shape != contact_action.shape:
        raise ValueError(
            f"baseline {tuple(baseline_action.shape)} and contact action "
            f"{tuple(contact_action.shape)} must match"
        )
    action = baseline_action.clone()
    action[..., :3] = torch.where(
        near.reshape(-1)[:, None, None],
        contact_action[..., :3],
        baseline_action[..., :3],
    )
    return action
