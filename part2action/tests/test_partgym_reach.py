"""Tests for privileged PartGym target-part distance diagnostics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from data.geometry import camera_to_world_numpy
from diagnostics.partgym_reach import (
    measure_point_part_prediction,
    measure_policy_target_alignment,
    measure_tcp_to_target_part,
)


class _FakeParser:
    obj_name = "mug"

    def __init__(self):
        self.part_pcds = {
            "handle": np.array(
                [[0.50, 0.00, 0.20], [0.55, 0.00, 0.20]],
                dtype=np.float32,
            )
        }
        self.updated = 0

    def update_part_pcds(self):
        self.updated += 1
        return self.part_pcds


def test_measure_tcp_to_semantic_target_part():
    parser = _FakeParser()
    env = SimpleNamespace(
        chain_params=[
            {
                "skill_name": "touch_obj",
                "params": {"part_touch": "handle"},
            }
        ],
        parser=parser,
    )
    tcp_pose = [0.0, 0.0, 0.0, 1.0, 0.51, 0.0, 0.20]
    result = measure_tcp_to_target_part(env, tcp_pose, 0)
    assert result["supported"]
    assert result["part_name"] == "handle"
    assert abs(result["tcp_to_part_min_distance_m"] - 0.01) < 1e-5
    assert parser.updated == 1


def test_move_skill_has_no_target_part_distance_mapping():
    env = SimpleNamespace(
        chain_params=[{"skill_name": "move_gripper", "params": {}}],
        parser=_FakeParser(),
    )
    result = measure_tcp_to_target_part(
        env, [0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.2], 0
    )
    assert not result["supported"]
    assert "no target-part distance mapping" in result["reason"]


def test_measure_policy_contact_and_action_alignment():
    env = SimpleNamespace(
        chain_params=[
            {
                "skill_name": "grasp_obj",
                "params": {"part_grasp": "handle"},
            }
        ],
        parser=_FakeParser(),
    )
    result = measure_policy_target_alignment(
        env,
        [0.0, 0.0, 0.0, 1.0, 0.40, 0.0, 0.20],
        0,
        predicted_contact_xyz=[0.51, 0.0, 0.20],
        action_target_xyz=[0.52, 0.0, 0.20],
    )
    assert result["supported"]
    assert result["predicted_contact_to_part_min_distance_m"] < 0.011
    assert result["action_target_to_part_min_distance_m"] < 0.021
    assert result["action_target_to_predicted_contact_distance_m"] < 0.011


def test_measure_point_part_prediction_against_geometry():
    scene = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.5, 0.0, 1.0]],
        dtype=np.float32,
    )
    parser = _FakeParser()
    parser.part_pcds["handle"] = camera_to_world_numpy(scene[:2])
    env = SimpleNamespace(
        chain_params=[
            {
                "skill_name": "touch_obj",
                "params": {"part_touch": "handle"},
            }
        ],
        parser=parser,
    )
    result = measure_point_part_prediction(
        env, scene, [2.0, -2.0, 2.0], 0, radii_m=(0.001,)
    )
    metrics = result["radius_metrics"]["0.001"]
    assert result["supported"]
    assert metrics["target_positive_points"] == 2
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1


if __name__ == "__main__":
    test_measure_tcp_to_semantic_target_part()
    test_move_skill_has_no_target_part_distance_mapping()
    test_measure_policy_contact_and_action_alignment()
    test_measure_point_part_prediction_against_geometry()
    print("PartGym reach diagnostic tests passed")
