"""Geometry for a temporary oracle-pose approach diagnostic; no learned policy."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CONDITIONS = ("recorded_replay", "commanded_pose_hold", "measured_pose_hold",
              "measured_pose_bridge", "shifted_pose_bridge")


def pose_error(actual_xyzw_xyz, target_xyzw_xyz):
    actual, target = np.asarray(actual_xyzw_xyz), np.asarray(target_xyzw_xyz)
    return {
        "position_mm": float(np.linalg.norm(actual[4:7] - target[4:7]) * 1000),
        "rotation_deg": float(np.rad2deg((Rotation.from_quat(target[:4]).inv()
                                          * Rotation.from_quat(actual[:4])).magnitude())),
    }


def approach_window(actions, instructions):
    """Use the last distinct open-gripper waypoint before the first closure.

    This isolates the existing pregrasp-to-grasp segment, rather than replacing
    the entire collision-avoiding approach by a straight line from the reset pose.
    Unsupported sources fail explicitly instead of inventing an interaction event.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("Expected finite (T,7) source actions")
    if len(instructions) != len(actions) or not instructions[0].lower().startswith("grasp "):
        raise ValueError("An aligned initial grasp segment is required")
    end = next((i for i, s in enumerate(instructions) if s != instructions[0]), len(actions))
    closed = np.flatnonzero(actions[:end, 6] <= .02)
    if actions[0, 6] <= .02 or not len(closed):
        raise ValueError("Need an initially open gripper and an observed closure command")
    close = int(closed[0])
    if end - close < 5:
        raise ValueError("Too little recorded closure duration")
    changes = np.flatnonzero(np.linalg.norm(np.diff(actions[:close, :3], axis=0), axis=1) > .002) + 1
    if not len(changes):
        raise ValueError("No distinct preclosure approach waypoint")
    start = int(changes[-1])
    if close - start < 5:
        raise ValueError("Too little approach duration")
    return {"start": start, "close": close, "end": end}


def pose_to_action(pose_xyzw_xyz, opening):
    pose = np.asarray(pose_xyzw_xyz, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite quaternion-XYZ pose")
    if not np.isfinite(opening) or not 0 <= opening <= .04 + 1e-7:
        raise ValueError("Invalid per-finger opening")
    return np.r_[pose[4:7], Rotation.from_quat(pose[:4]).as_euler("xyz"), opening]


def guided_commands(actions, window, start_pose, measured_target, condition):
    """Replace only approach/closure arm commands, preserving closure times.

    The original prefix and full-task continuation are unchanged. No extra dwell,
    lift, controller change or teleportation is introduced. The target is an
    oracle measured pose or the independently logged successful command pose.
    """
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition: " + condition)
    original = np.asarray(actions)
    result = original.copy()
    target = np.asarray(measured_target, dtype=float).copy()
    start, close, end = (int(window[k]) for k in ("start", "close", "end"))
    if not 0 <= start < close < end <= len(result):
        raise ValueError("Invalid approach/closure window")
    if condition == "recorded_replay":
        return result, target
    if condition == "commanded_pose_hold":
        command = original[close]
        target = np.r_[Rotation.from_euler("xyz", command[3:6]).as_quat(), command[:3]]
    if condition == "shifted_pose_bridge":
        target[4] += .05
    target_action = pose_to_action(target, original[start, 6])
    result[start:end, :6] = target_action[:6]
    if condition.endswith("bridge"):
        start_pose = np.asarray(start_pose, dtype=float)
        u = np.arange(1, close - start + 1, dtype=float) / (close - start)
        smooth = u * u * (3 - 2 * u)
        xyz = start_pose[4:7] + smooth[:, None] * (target[4:7] - start_pose[4:7])
        rotations = Slerp([0., 1.], Rotation.from_quat([start_pose[:4], target[:4]]))(smooth)
        result[start:close, :3] = xyz
        result[start:close, 3:6] = rotations.as_euler("xyz")
    assert np.array_equal(result[:start], original[:start])
    assert np.array_equal(result[end:], original[end:])
    assert np.array_equal(result[:, 6], original[:, 6])
    return result, target
