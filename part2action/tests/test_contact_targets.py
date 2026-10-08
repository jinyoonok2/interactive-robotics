import tempfile
from pathlib import Path

import h5py
import numpy as np

from data.geometry import camera_to_world_numpy
from data.partinstruct_loader import PartInstructDataset, collate_part2action
from data.targets import (
    classify_skill_instruction,
    derive_skill_contact_target,
)


def _geometry(length: int = 7, points: int = 3):
    part_pcd = np.zeros((length, points, 4), dtype=np.float32)
    part_pcd[..., 3] = 1.0
    tcp_pose = np.zeros((length, 7), dtype=np.float32)
    part_world = camera_to_world_numpy(part_pcd[0, :, :3])
    tcp_pose[:, :4] = [0.0, 0.0, 0.0, 1.0]
    tcp_pose[:, 4:7] = part_world[0] + [0.20, 0.0, 0.0]
    return part_pcd, tcp_pose, part_world[0]


def test_skill_instruction_classification():
    assert classify_skill_instruction("Grasp the mug at its handle") == "grasp_obj"
    assert classify_skill_instruction(" Touch the bottle at its lid ") == "touch_obj"
    assert classify_skill_instruction("Move upwards") == "non_contact"
    assert classify_skill_instruction("Release") == "non_contact"
    assert classify_skill_instruction("Unexpected primitive") == "unknown"


def test_grasp_uses_only_in_segment_close_transition():
    actions = np.zeros((7, 7), dtype=np.float32)
    actions[:, 6] = 0.04
    actions[2:, 6] = 0.0
    part_pcd, tcp_pose, _ = _geometry()
    target = derive_skill_contact_target(
        actions=actions,
        part_pcd=part_pcd,
        tcp_pose=tcp_pose,
        skill_kind="grasp_obj",
        segment_start=0,
        segment_end=3,
        current_t=0,
    )
    assert target["valid"] == 1.0
    assert target["contact_t"] == 2
    assert target["method"] == "gripper_close_in_segment"

    actions[:, 6] = 0.04
    actions[5:, 6] = 0.0
    target = derive_skill_contact_target(
        actions=actions,
        part_pcd=part_pcd,
        tcp_pose=tcp_pose,
        skill_kind="grasp_obj",
        segment_start=0,
        segment_end=3,
        current_t=0,
    )
    assert target["valid"] == 0.0
    assert target["contact_t"] == -1


def test_touch_selects_minimum_tcp_part_distance():
    actions = np.zeros((7, 7), dtype=np.float32)
    part_pcd, tcp_pose, part_world = _geometry()
    tcp_pose[0, 4:7] = part_world + [0.20, 0.0, 0.0]
    tcp_pose[1, 4:7] = part_world + [0.08, 0.0, 0.0]
    tcp_pose[2, 4:7] = part_world + [0.01, 0.0, 0.0]
    tcp_pose[3, 4:7] = part_world + [0.12, 0.0, 0.0]
    target = derive_skill_contact_target(
        actions=actions,
        part_pcd=part_pcd,
        tcp_pose=tcp_pose,
        skill_kind="touch_obj",
        segment_start=0,
        segment_end=3,
        current_t=0,
    )
    assert target["valid"] == 1.0
    assert target["contact_t"] == 2
    assert target["tcp_to_part_distance_m"] < 0.011
    np.testing.assert_allclose(target["contact_xyz"], part_world, atol=1e-6)


def test_non_contact_and_empty_geometry_are_invalid():
    actions = np.zeros((7, 7), dtype=np.float32)
    part_pcd, tcp_pose, _ = _geometry()
    non_contact = derive_skill_contact_target(
        actions=actions,
        part_pcd=part_pcd,
        tcp_pose=tcp_pose,
        skill_kind="non_contact",
        segment_start=0,
        segment_end=3,
        current_t=1,
    )
    assert non_contact["valid"] == 0.0
    assert non_contact["contact_t"] == -1

    part_pcd[..., :3] = np.nan
    empty = derive_skill_contact_target(
        actions=actions,
        part_pcd=part_pcd,
        tcp_pose=tcp_pose,
        skill_kind="touch_obj",
        segment_start=0,
        segment_end=3,
        current_t=1,
    )
    assert empty["valid"] == 0.0
    assert "geometry" in empty["reason"]


def test_contact_remains_defined_after_event_in_same_segment():
    actions = np.zeros((7, 7), dtype=np.float32)
    actions[:2, 6] = 0.04
    part_pcd, tcp_pose, _ = _geometry()
    target = derive_skill_contact_target(
        actions=actions,
        part_pcd=part_pcd,
        tcp_pose=tcp_pose,
        skill_kind="grasp_obj",
        segment_start=0,
        segment_end=4,
        current_t=4,
    )
    assert target["valid"] == 1.0
    assert target["contact_t"] == 2


def test_loader_phase_boundaries_and_collation():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.hdf5"
        length, points = 6, 3
        part_pcd, tcp_pose, part_world = _geometry(length, points)
        tcp_pose[1, 4:7] = part_world + [0.01, 0.0, 0.0]
        actions = np.zeros((length, 7), dtype=np.float32)
        actions[:, 6] = 0.04
        instructions = [
            "Touch the mug at its handle",
            "Touch the mug at its handle",
            "Touch the mug at its handle",
            "Move upwards",
            "Move upwards",
            "Move upwards",
        ]
        with h5py.File(path, "w") as handle:
            demo = handle.create_group("data/demo_0")
            demo.create_dataset("actions", data=actions)
            text_dtype = h5py.string_dtype("utf-8")
            demo.create_dataset(
                "skill_instructions",
                data=np.asarray(instructions, dtype=object),
                dtype=text_dtype,
            )
            demo.create_dataset(
                "task_instructions",
                data=np.asarray(["touch then move"], dtype=object),
                dtype=text_dtype,
            )
            obs = demo.create_group("obs")
            obs.create_dataset(
                "agentview_rgb",
                data=np.zeros((length, 8, 8, 3), dtype=np.uint8),
            )
            obs.create_dataset(
                "agentview_part_mask",
                data=np.ones((length, 8, 8, 1), dtype=np.uint8),
            )
            obs.create_dataset(
                "agentview_pcd",
                data=np.zeros((length, points, 3), dtype=np.float32),
            )
            obs.create_dataset("agentview_part_pcd", data=part_pcd)
            obs.create_dataset("tcp_pose", data=tcp_pose)
            obs.create_dataset(
                "gripper_state", data=np.zeros((length, 1), dtype=np.float32)
            )
            obs.create_dataset(
                "joint_states", data=np.zeros((length, 7), dtype=np.float32)
            )

        dataset = PartInstructDataset(
            [str(path)],
            action_chunk=2,
            use_pcd=True,
            use_tcp_pose=True,
            use_contact_xyz=True,
            contact_label_mode="skill_specific",
            use_point_part_labels=True,
            use_hierarchy=True,
            max_skill_slots=4,
        )
        touch = dataset[0]
        move = dataset[3]
        assert touch["meta"]["skill_start"] == 0
        assert touch["meta"]["skill_end"] == 2
        assert touch["meta"]["contact_t"] == 1
        assert touch["contact_xyz_valid"].item() == 1.0
        assert touch["point_part_labels"].shape == (points,)
        assert touch["point_part_valid"].item() == 1.0
        assert move["meta"]["skill_start"] == 3
        assert move["meta"]["skill_end"] == 5
        assert move["contact_xyz_valid"].item() == 0.0
        assert move["point_part_valid"].item() == 0.0
        batch = collate_part2action([touch, move])
        assert batch["contact_xyz"].shape == (2, 3)
        assert batch["contact_xyz_valid"].shape == (2,)
        assert len(batch["meta"]) == 2
        dataset.close()


if __name__ == "__main__":
    test_skill_instruction_classification()
    test_grasp_uses_only_in_segment_close_transition()
    test_touch_selects_minimum_tcp_part_distance()
    test_non_contact_and_empty_geometry_are_invalid()
    test_contact_remains_defined_after_event_in_same_segment()
    test_loader_phase_boundaries_and_collation()
    print("contact target tests passed")
