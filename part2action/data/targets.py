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

# PartInstruct agentview camera (see env_config.yaml).
# 300x300, fx=fy=259.8, cx=cy=150.
_FX = 259.80761647
_FY = 259.80761647
_CX = 150.0
_CY = 150.0
_IMG_H = 300
_IMG_W = 300


def _detect_contact_step(gripper: np.ndarray, t_start: int) -> int:
    """Return first index >= t_start where gripper transitions toward closed.

    PartInstruct gripper_state is a single scalar per step. We treat any
    decrease >= 0.05 (open -> closed) after t_start as the contact event.
    Falls back to the last step if no transition is found.
    """
    g = gripper.reshape(-1)
    n = len(g)
    if n == 0:
        return t_start
    for t in range(max(0, t_start), n - 1):
        if (g[t] - g[t + 1]) >= 0.05:
            return t + 1
    return n - 1


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

    contact_t = _detect_contact_step(gripper, t_start=t)

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
