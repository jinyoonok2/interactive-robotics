"""Privileged PartGym target-part distance diagnostics.

These measurements are logging-only and must never enter policy tensors.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from data.geometry import camera_to_world_numpy
from .partgym_skill_oracle import PartGymSkillOracle


_PART_PARAM_BY_SKILL = {
    "grasp_obj": "part_grasp",
    "touch_obj": "part_touch",
}


def target_part_for_skill(
    skill_name: str, params: Mapping[str, Any]
) -> tuple[str | None, str]:
    """Resolve the semantic target part for touch/grasp skills."""
    param_name = _PART_PARAM_BY_SKILL.get(skill_name)
    if param_name is None:
        return None, f"no target-part distance mapping for {skill_name}"
    part_name = str(params.get(param_name, "")).strip()
    return part_name, ""


def _target_part_points(
    env: Any, skill_index: int
) -> tuple[dict[str, Any], np.ndarray | None]:
    entry, reason = PartGymSkillOracle.skill_entry(env, skill_index)
    if entry is None:
        return {"supported": False, "reason": reason}, None

    skill_name = str(entry["skill_name"])
    params = entry.get("params", {}) or {}
    if not isinstance(params, Mapping):
        return {
            "supported": False,
            "skill_name": skill_name,
            "reason": "skill params must be a mapping",
        }, None
    part_name, reason = target_part_for_skill(skill_name, params)
    if part_name is None:
        return {
            "supported": False,
            "skill_name": skill_name,
            "reason": reason,
        }, None

    parser = getattr(env, "parser", None)
    if parser is None or not callable(getattr(parser, "update_part_pcds", None)):
        return {
            "supported": False,
            "skill_name": skill_name,
            "part_name": part_name,
            "reason": "env parser or update_part_pcds unavailable",
        }, None
    parser.update_part_pcds()
    lookup_name = part_name or str(getattr(parser, "obj_name", ""))
    part_pcds = getattr(parser, "part_pcds", {})
    if lookup_name not in part_pcds:
        return {
            "supported": False,
            "skill_name": skill_name,
            "part_name": lookup_name,
            "reason": f"target part not found: {lookup_name}",
        }, None

    points = np.asarray(part_pcds[lookup_name], dtype=np.float32)
    points = points[..., :3].reshape(-1, 3)
    finite = np.isfinite(points).all(axis=-1)
    points = points[finite]
    if not len(points):
        return {
            "supported": False,
            "skill_name": skill_name,
            "part_name": lookup_name,
            "reason": "target part has no finite points",
        }, None
    return {
        "supported": True,
        "skill_name": skill_name,
        "skill_index": int(skill_index),
        "part_name": lookup_name,
        "target_part_num_points": int(len(points)),
        "reason": "",
    }, points


def _point_distance(
    points: np.ndarray, query_xyz: Any
) -> tuple[float, list[float], list[float]]:
    query = np.asarray(query_xyz, dtype=np.float32).reshape(-1)
    if query.size != 3 or not np.isfinite(query).all():
        raise ValueError(f"invalid query point shape/value: {query.shape}")
    distances = np.linalg.norm(points - query[None, :], axis=-1)
    nearest_index = int(np.argmin(distances))
    return (
        float(distances[nearest_index]),
        query.astype(float).tolist(),
        points[nearest_index].astype(float).tolist(),
    )


def measure_tcp_to_target_part(
    env: Any, tcp_pose: Any, skill_index: int
) -> dict[str, Any]:
    """Measure world-frame TCP distance to the active skill's target part."""
    metadata, points = _target_part_points(env, skill_index)
    if points is None:
        return metadata

    pose = np.asarray(tcp_pose, dtype=np.float32).reshape(-1)
    if pose.size < 7 or not np.isfinite(pose[4:7]).all():
        return {
            **metadata,
            "supported": False,
            "reason": f"invalid tcp_pose shape/value: {pose.shape}",
        }

    tcp_xyz = pose[4:7]
    distance, tcp_list, nearest = _point_distance(points, tcp_xyz)
    return {
        **metadata,
        "tcp_xyz": tcp_list,
        "nearest_part_xyz": nearest,
        "tcp_to_part_min_distance_m": distance,
    }


def measure_policy_target_alignment(
    env: Any,
    tcp_pose: Any,
    skill_index: int,
    *,
    predicted_contact_xyz: Any,
    action_target_xyz: Any,
) -> dict[str, Any]:
    """Compare policy contact/action targets with privileged part geometry."""
    metadata, points = _target_part_points(env, skill_index)
    if points is None:
        return metadata
    pose = np.asarray(tcp_pose, dtype=np.float32).reshape(-1)
    if pose.size < 7 or not np.isfinite(pose[4:7]).all():
        return {
            **metadata,
            "supported": False,
            "reason": f"invalid tcp_pose shape/value: {pose.shape}",
        }
    try:
        tcp_to_part, tcp_xyz, _ = _point_distance(points, pose[4:7])
        contact_to_part, contact_xyz, nearest_contact_part = _point_distance(
            points, predicted_contact_xyz
        )
        action_to_part, action_xyz, nearest_action_part = _point_distance(
            points, action_target_xyz
        )
    except ValueError as exc:
        return {**metadata, "supported": False, "reason": str(exc)}

    tcp_arr = np.asarray(tcp_xyz, dtype=np.float32)
    contact_arr = np.asarray(contact_xyz, dtype=np.float32)
    action_arr = np.asarray(action_xyz, dtype=np.float32)
    return {
        **metadata,
        "tcp_xyz": tcp_xyz,
        "predicted_contact_xyz": contact_xyz,
        "action_target_xyz": action_xyz,
        "nearest_part_to_contact_xyz": nearest_contact_part,
        "nearest_part_to_action_xyz": nearest_action_part,
        "tcp_to_part_min_distance_m": tcp_to_part,
        "predicted_contact_to_part_min_distance_m": contact_to_part,
        "action_target_to_part_min_distance_m": action_to_part,
        "tcp_to_predicted_contact_distance_m": float(
            np.linalg.norm(tcp_arr - contact_arr)
        ),
        "action_target_to_predicted_contact_distance_m": float(
            np.linalg.norm(action_arr - contact_arr)
        ),
    }


def measure_point_part_prediction(
    env: Any,
    scene_pcd: Any,
    point_logits: Any,
    skill_index: int,
    *,
    radii_m: tuple[float, ...] = (0.003, 0.005, 0.01),
) -> dict[str, Any]:
    """Score predicted scene-point membership against privileged part geometry."""
    metadata, target_points = _target_part_points(env, skill_index)
    if target_points is None:
        return metadata
    scene = np.asarray(scene_pcd, dtype=np.float32)
    scene = scene[..., :3].reshape(-1, 3)
    logits = np.asarray(point_logits, dtype=np.float32).reshape(-1)
    if len(scene) != len(logits):
        return {
            **metadata,
            "supported": False,
            "reason": (
                f"scene/logit count mismatch: {len(scene)} versus {len(logits)}"
            ),
        }
    finite = np.isfinite(scene).all(axis=-1) & np.isfinite(logits)
    if not finite.any():
        return {**metadata, "supported": False, "reason": "no finite scene points"}
    scene_world = camera_to_world_numpy(scene[finite])
    finite_logits = logits[finite]
    nearest_distance = cKDTree(target_points).query(scene_world, k=1)[0]
    predicted = finite_logits >= 0.0
    radius_metrics: dict[str, dict[str, Any]] = {}
    for radius in radii_m:
        target = nearest_distance <= float(radius)
        true_positive = int(np.logical_and(predicted, target).sum())
        false_positive = int(np.logical_and(predicted, ~target).sum())
        false_negative = int(np.logical_and(~predicted, target).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        radius_metrics[f"{float(radius):g}"] = {
            "radius_m": float(radius),
            "target_positive_points": int(target.sum()),
            "predicted_positive_points": int(predicted.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(
                2.0 * precision * recall / max(1e-8, precision + recall)
            ),
        }
    return {
        **metadata,
        "scene_point_count": int(finite.sum()),
        "prediction_threshold": 0.5,
        "radius_metrics": radius_metrics,
    }
