"""Derive part-to-action supervision targets from PartInstruct demos.

PartInstruct does not store an explicit "contact point" or "approach
direction" per timestep, but expert demonstrations make them recoverable:

- Contact event:  the first timestep where the gripper transitions from
  open to closed after the current step. Action[:3] is the delta position
  the planner commanded; integrating it from the current frame gives an
  estimate of the EE position at contact, in EE/world frame coordinates.
- Approach direction: average of action[:3] in a small window before
  contact, normalized.

For the heatmap-only track this module is unused; for part-action tracks it
produces the extra targets the contact, approach, and action heads regress.

We also project the contact position into normalized image coordinates
using the static agentview camera intrinsics from PartInstruct's
env_config.yaml. The projection is approximate (we use the action-space
delta integration rather than ground-truth EE pose) but is consistent
across all demos so it's a fair learning signal.
"""
from __future__ import annotations

import numpy as np

from .geometry import camera_to_world_numpy, tcp_position_numpy

GRIPPER_OPEN_WIDTH = 0.04
GRIPPER_CLOSED_WIDTH = 0.0
GRIPPER_CLOSED_THRESHOLD = 0.02

# PartInstruct agentview camera (see env_config.yaml).
# 300x300, fx=fy=259.8, cx=cy=150.
_FX = 259.80761647
_FY = 259.80761647
_CX = 150.0
_CY = 150.0
_IMG_H = 300
_IMG_W = 300


def classify_skill_instruction(instruction: str) -> str:
    """Map PartInstruct's natural-language skill text to a contact policy."""
    normalized = " ".join(str(instruction).strip().lower().split())
    if normalized.startswith("grasp "):
        return "grasp_obj"
    if normalized.startswith("touch "):
        return "touch_obj"
    if normalized.startswith(("move ", "release", "reorient ")):
        return "non_contact"
    return "unknown"


def _segment_bounds(length: int, start: int, end: int) -> tuple[int, int] | None:
    if length <= 0:
        return None
    start = int(np.clip(start, 0, length - 1))
    end = int(np.clip(end, 0, length - 1))
    return (start, end) if start <= end else None


def _detect_grasp_contact_step(
    actions: np.ndarray, segment_start: int, segment_end: int
) -> int | None:
    """Return the first open-to-closed command inside one grasp segment."""
    commands = np.asarray(actions, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[-1] < 7:
        return None
    bounds = _segment_bounds(len(commands), segment_start, segment_end)
    if bounds is None:
        return None
    start, end = bounds
    previous = (
        float(commands[start - 1, 6])
        if start > 0
        else float(GRIPPER_OPEN_WIDTH)
    )
    for t in range(start, end + 1):
        current = float(commands[t, 6])
        if previous > GRIPPER_CLOSED_THRESHOLD >= current:
            return t
        previous = current
    return None


def _part_points_world(part_pcd: np.ndarray, t: int) -> np.ndarray:
    cloud = np.asarray(part_pcd, dtype=np.float32)
    if cloud.ndim != 3 or cloud.shape[-1] < 3 or not len(cloud):
        return np.empty((0, 3), dtype=np.float32)
    t = int(np.clip(t, 0, len(cloud) - 1))
    points = cloud[t, :, :3]
    valid = np.isfinite(points).all(axis=-1)
    if cloud.shape[-1] >= 4:
        valid &= cloud[t, :, 3] > 0.5
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)
    return camera_to_world_numpy(points[valid]).astype(np.float32)


def _detect_touch_contact_step(
    part_pcd: np.ndarray,
    tcp_pose: np.ndarray,
    segment_start: int,
    segment_end: int,
) -> tuple[int | None, float | None]:
    """Select the closest TCP-to-target-part timestep inside a touch segment."""
    poses = np.asarray(tcp_pose, dtype=np.float32)
    clouds = np.asarray(part_pcd, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[-1] < 7 or clouds.ndim != 3:
        return None, None
    length = min(len(poses), len(clouds))
    bounds = _segment_bounds(length, segment_start, segment_end)
    if bounds is None:
        return None, None
    best_t, best_distance = None, float("inf")
    for t in range(bounds[0], bounds[1] + 1):
        points = _part_points_world(clouds, t)
        tcp_xyz = tcp_position_numpy(poses[t])
        if not len(points) or not np.isfinite(tcp_xyz).all():
            continue
        distance = float(np.linalg.norm(points - tcp_xyz[None, :], axis=-1).min())
        if distance < best_distance:
            best_t, best_distance = t, distance
    if best_t is None:
        return None, None
    return int(best_t), float(best_distance)


def _detect_contact_step(actions: np.ndarray, t_start: int) -> int:
    """Return the first commanded close at or after ``t_start``.

    PartInstruct actions use finger targets of 0.04 (open) and 0.00
    (closed). Observed gripper state moves gradually and never exhibits the
    old 0.05 single-step drop, so command labels are the reliable signal.
    """
    commands = np.asarray(actions, dtype=np.float32)
    if commands.ndim != 2 or commands.shape[-1] < 7:
        return max(0, int(t_start))
    g = commands[:, 6]
    n = len(g)
    if n == 0:
        return t_start
    start = int(np.clip(t_start, 0, n - 1))
    if g[start] <= GRIPPER_CLOSED_THRESHOLD:
        return start
    for t in range(start + 1, n):
        if g[t] <= GRIPPER_CLOSED_THRESHOLD:
            return t
    return n - 1


def derive_gripper_targets(
    action_chunk: np.ndarray,
    previous_command: float | None = None,
    transition_weight: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary closed labels and weights emphasizing close onset."""
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] < 7:
        raise ValueError(f"Expected action chunk (K, >=7), got {actions.shape}")
    closed = (actions[:, 6] <= GRIPPER_CLOSED_THRESHOLD).astype(np.float32)
    weights = np.ones_like(closed, dtype=np.float32)
    previous_closed = (
        bool(closed[0])
        if previous_command is None
        else bool(float(previous_command) <= GRIPPER_CLOSED_THRESHOLD)
    )
    for i, is_closed in enumerate(closed.astype(bool)):
        if is_closed and not previous_closed:
            weights[i] = float(transition_weight)
        previous_closed = bool(is_closed)
    return closed, weights


def _integrate_actions(actions: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """Sum the position deltas in actions[t0:t1, :3]."""
    if t1 <= t0:
        return np.zeros(3, dtype=np.float32)
    return actions[t0:t1, :3].sum(axis=0).astype(np.float32)


def _project_to_image(world_xyz: np.ndarray) -> np.ndarray:
    """Project a (3,) world point into normalized image coords [0,1]^2.

    We use a pinhole model centered on the agentview principal point.
    The world frame and the camera frame are different in PyBullet, but
    we only need a *learnable* image-space target that is consistent
    across demos. Treat (x, y, z_for_depth) as (du, dv, depth) deltas
    relative to the image center; this is an approximation but gives a
    stable signal proportional to true 2D location.
    """
    z = max(1e-3, float(world_xyz[2]) + 1.0)
    u = _CX + _FX * float(world_xyz[0]) / z
    v = _CY + _FY * float(world_xyz[1]) / z
    return np.array([np.clip(u / _IMG_W, 0.0, 1.0), np.clip(v / _IMG_H, 0.0, 1.0)], dtype=np.float32)


def derive_contact_and_approach(
    actions: np.ndarray,
    gripper: np.ndarray,
    t: int,
    window: int = 4,
):
    """Return (contact_xy_norm, approach_dir_unit, contact_t).

    Args:
        actions: (T, 7) full demo actions.
        gripper: (T, 1) full demo gripper states.
        t:       current timestep within the demo.
        window:  number of frames just before contact to average for the
                 approach direction.

    Returns:
        contact_xy_norm: (2,) projected image coords in [0,1].
        approach_dir:    (3,) unit vector in world delta-action space.
        contact_t:       int.
    """
    actions = np.asarray(actions, dtype=np.float32)
    gripper = np.asarray(gripper, dtype=np.float32)

    contact_t = _detect_contact_step(actions, t_start=t)

    delta_xyz = _integrate_actions(actions, t0=t, t1=contact_t)
    contact_xy_norm = _project_to_image(delta_xyz)

    a = max(0, contact_t - int(window))
    if contact_t > a:
        approach = actions[a:contact_t, :3].mean(axis=0)
    else:
        approach = actions[max(0, contact_t - 1) : contact_t + 1, :3].mean(axis=0)
    n = float(np.linalg.norm(approach))
    if n < 1e-6:
        approach_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        approach_dir = (approach / n).astype(np.float32)

    return contact_xy_norm, approach_dir, int(contact_t)


def derive_contact_xyz(
    part_pcd: np.ndarray,
    tcp_pose: np.ndarray,
    contact_t: int,
    current_t: int | None = None,
) -> tuple[np.ndarray, float]:
    """Return a current-frame target-part point nearest the contact TCP."""
    part_pcd = np.asarray(part_pcd, dtype=np.float32)
    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    if part_pcd.ndim != 3 or part_pcd.shape[-1] < 3 or tcp_pose.ndim != 2 or tcp_pose.shape[-1] < 3:
        return np.zeros(3, dtype=np.float32), 0.0
    last = min(part_pcd.shape[0], tcp_pose.shape[0]) - 1
    contact_t = int(np.clip(contact_t, 0, last))
    point_t = contact_t if current_t is None else int(np.clip(current_t, 0, last))
    points = part_pcd[point_t, :, :3]
    finite = np.isfinite(points).all(axis=-1)
    if part_pcd.shape[-1] >= 4:
        finite = finite & (part_pcd[point_t, :, 3] > 0.5)
    valid_points = camera_to_world_numpy(points[finite])
    if valid_points.size == 0:
        return np.zeros(3, dtype=np.float32), 0.0
    tcp_xyz = tcp_position_numpy(tcp_pose[contact_t])
    nearest = int(np.argmin(np.linalg.norm(valid_points - tcp_xyz[None, :], axis=-1)))
    return valid_points[nearest].astype(np.float32), 1.0


def derive_skill_contact_target(
    *,
    actions: np.ndarray,
    part_pcd: np.ndarray,
    tcp_pose: np.ndarray,
    skill_kind: str,
    segment_start: int,
    segment_end: int,
    current_t: int,
    window: int = 4,
    contact_event: dict | None = None,
) -> dict:
    """Derive contact supervision without crossing the active skill segment."""
    zero_xyz = np.zeros(3, dtype=np.float32)
    zero_xy = np.zeros(2, dtype=np.float32)
    default = {
        "contact_xyz": zero_xyz,
        "contact_xy": zero_xy,
        "approach_dir": zero_xyz.copy(),
        "valid": 0.0,
        "contact_t": -1,
        "method": "invalid",
        "reason": "",
        "tcp_to_part_distance_m": None,
    }
    if contact_event is None:
        if skill_kind not in {"grasp_obj", "touch_obj"}:
            return {
                **default,
                "reason": (
                    "non-contact skill"
                    if skill_kind == "non_contact"
                    else f"unknown skill kind: {skill_kind}"
                ),
            }

        if skill_kind == "grasp_obj":
            contact_t = _detect_grasp_contact_step(
                actions, segment_start, segment_end
            )
            method = "gripper_close_in_segment"
            if contact_t is None:
                return {**default, "reason": "no gripper close inside grasp segment"}
            points = _part_points_world(part_pcd, contact_t)
            poses = np.asarray(tcp_pose, dtype=np.float32)
            if (
                not len(points)
                or poses.ndim != 2
                or poses.shape[-1] < 7
                or contact_t >= len(poses)
            ):
                distance = None
            else:
                tcp_xyz = tcp_position_numpy(poses[contact_t])
                distance = float(
                    np.linalg.norm(points - tcp_xyz[None, :], axis=-1).min()
                )
        else:
            contact_t, distance = _detect_touch_contact_step(
                part_pcd, tcp_pose, segment_start, segment_end
            )
            method = "minimum_tcp_part_distance"
            if contact_t is None:
                return {
                    **default,
                    "reason": "no finite TCP/target-part geometry in touch segment",
                }
    else:
        contact_t = int(contact_event.get("contact_t", -1))
        method = str(contact_event.get("method", "invalid"))
        distance = contact_event.get("tcp_to_part_distance_m")
        if contact_t < 0:
            return {
                **default,
                "method": method,
                "reason": str(contact_event.get("reason", "invalid contact event")),
                "tcp_to_part_distance_m": distance,
            }

    contact_xyz, valid = derive_contact_xyz(
        part_pcd, tcp_pose, contact_t, current_t=current_t
    )
    if not valid:
        return {
            **default,
            "contact_t": int(contact_t),
            "method": method,
            "tcp_to_part_distance_m": distance,
            "reason": "no finite target-part points in current frame",
        }

    actions_array = np.asarray(actions, dtype=np.float32)
    contact_xy = _project_to_image(
        _integrate_actions(actions_array, current_t, contact_t)
    )
    approach_start = max(int(segment_start), int(contact_t) - int(window))
    approach_slice = actions_array[approach_start : int(contact_t), :3]
    if not len(approach_slice):
        approach_slice = actions_array[int(contact_t) : int(contact_t) + 1, :3]
    approach = approach_slice.mean(axis=0)
    norm = float(np.linalg.norm(approach))
    approach_dir = (
        np.array([0.0, 0.0, -1.0], dtype=np.float32)
        if norm < 1e-6
        else (approach / norm).astype(np.float32)
    )
    return {
        "contact_xyz": contact_xyz.astype(np.float32),
        "contact_xy": contact_xy.astype(np.float32),
        "approach_dir": approach_dir,
        "valid": 1.0,
        "contact_t": int(contact_t),
        "method": method,
        "reason": "",
        "tcp_to_part_distance_m": distance,
    }
