"""PartInstruct HDF5 dataset adapter.

The PartInstruct HuggingFace dataset stores one HDF5 file per object category
(e.g. scissors.hdf5, pliers.hdf5). Schema (verified from upstream config):

    data/demo_{i}/
        actions                (T, 7)   pos(3) + axis_angle(3) + gripper(1)
        skill_instructions     (T,)     bytes utf-8 per timestep
        obs/agentview_rgb      (T, H, W, 3)  uint8        H=W=300
        obs/agentview_part_mask(T, H, W, 1)  uint8 binary part mask
        obs/agentview_pcd      (T, 1024, 3) float32       scene point cloud
        obs/agentview_part_pcd (T, 1024, 4) float32       part pcd + flag
        obs/wrist_rgb          (T, H, W, 3)  uint8        (optional)
        obs/wrist_pcd          (T, 1024, 3)  float32      (optional)
        obs/gripper_state      (T, 1)
        obs/joint_states       (T, 7)

This adapter yields per-step samples used by the heatmap and part-action
tracks. It does NOT depend on the upstream PartInstruct python package or its diffusion_policy fork;
we only need h5py + numpy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import (
    classify_skill_instruction,
    derive_contact_and_approach,
    derive_contact_xyz,
    derive_gripper_targets,
    derive_skill_contact_target,
)


@dataclass
class SampleSpec:
    """One training sample = one timestep inside one demo of one HDF5 file."""

    file_path: str
    demo_key: str
    t: int
    n_steps: int


class PartInstructDataset(Dataset):
    """Per-timestep dataset over one or more PartInstruct HDF5 files.

    Each __getitem__ returns a dict of tensors with the keys required by
    the training tracks. Heads that a config doesn't use can simply ignore the
    corresponding key.
    """

    def __init__(
        self,
        hdf5_paths: Sequence[str],
        action_chunk: int = 8,
        rgb_key: str = "agentview_rgb",
        mask_key: str = "agentview_part_mask",
        gripper_key: str = "gripper_state",
        joints_key: str = "joint_states",
        max_demos_per_file: Optional[int] = None,
        sample_stride: int = 1,
        require_part_mask: bool = True,
        n_obs_steps: int = 1,
        use_pcd: bool = False,
        use_part_pcd: bool = False,
        use_tcp_pose: bool = False,
        use_contact_xyz: bool = False,
        contact_label_mode: str = "legacy",
        use_point_part_labels: bool = False,
        use_hierarchy: bool = False,
        max_skill_slots: int = 8,
        gripper_transition_weight: float = 4.0,
        pcd_key: str = "agentview_pcd",
        part_pcd_key: str = "agentview_part_pcd",
        tcp_pose_key: str = "tcp_pose",
    ) -> None:
        super().__init__()
        if not hdf5_paths:
            raise ValueError("PartInstructDataset requires at least one HDF5 path")
        self.hdf5_paths = [str(Path(p).expanduser().resolve()) for p in hdf5_paths]
        self.action_chunk = int(action_chunk)
        self.rgb_key = rgb_key
        self.mask_key = mask_key
        self.gripper_key = gripper_key
        self.joints_key = joints_key
        self.require_part_mask = bool(require_part_mask)
        self.n_obs_steps = max(1, int(n_obs_steps))
        self.use_pcd = bool(use_pcd)
        self.use_part_pcd = bool(use_part_pcd)
        self.use_tcp_pose = bool(use_tcp_pose)
        self.use_contact_xyz = bool(use_contact_xyz)
        self.use_point_part_labels = bool(use_point_part_labels)
        self.contact_label_mode = str(contact_label_mode).strip().lower()
        if self.contact_label_mode not in {"legacy", "skill_specific"}:
            raise ValueError(
                "contact_label_mode must be legacy or skill_specific, got "
                f"{contact_label_mode}"
            )
        self.use_hierarchy = bool(use_hierarchy)
        if self.use_point_part_labels and not self.use_hierarchy:
            raise ValueError("point-part labels require hierarchy/skill metadata")
        self.max_skill_slots = max(1, int(max_skill_slots))
        self.gripper_transition_weight = float(gripper_transition_weight)
        self.pcd_key = pcd_key
        self.part_pcd_key = part_pcd_key
        self.tcp_pose_key = tcp_pose_key

        self._files: dict[str, h5py.File] = {}
        self._hierarchy_cache: dict[tuple[str, str], dict] = {}
        self._contact_event_cache: dict[tuple[str, str, int], dict] = {}
        self._samples: List[SampleSpec] = []
        self._index_demos(max_demos_per_file=max_demos_per_file, sample_stride=sample_stride)

        if not self._samples:
            raise RuntimeError(
                f"No usable samples found in: {self.hdf5_paths}. "
                f"Check that the HDF5 schema matches the expected PartInstruct layout."
            )

    def _open(self, path: str) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def _index_demos(self, max_demos_per_file: Optional[int], sample_stride: int) -> None:
        for path in self.hdf5_paths:
            f = self._open(path)
            if "data" not in f:
                raise RuntimeError(f"{path} has no top-level 'data' group")
            demos = f["data"]
            demo_keys = sorted(
                [k for k in demos.keys() if k.startswith("demo_")],
                key=lambda k: int(k.split("_")[-1]),
            )
            if max_demos_per_file is not None:
                demo_keys = demo_keys[: int(max_demos_per_file)]

            for dk in demo_keys:
                demo = demos[dk]
                if "actions" not in demo:
                    continue
                if self.require_part_mask:
                    if "obs" not in demo or self.mask_key not in demo["obs"]:
                        continue
                if self.use_pcd and ("obs" not in demo or self.pcd_key not in demo["obs"]):
                    continue
                if (
                    self.use_part_pcd
                    or self.use_contact_xyz
                    or self.use_point_part_labels
                ) and (
                    "obs" not in demo or self.part_pcd_key not in demo["obs"]
                ):
                    continue
                if (self.use_tcp_pose or self.use_contact_xyz) and (
                    "obs" not in demo or self.tcp_pose_key not in demo["obs"]
                ):
                    continue
                n_steps = int(demo["actions"].shape[0])
                if n_steps < self.action_chunk + 1:
                    continue
                last_valid = n_steps - self.action_chunk
                for t in range(0, last_valid, max(1, int(sample_stride))):
                    self._samples.append(
                        SampleSpec(file_path=path, demo_key=dk, t=t, n_steps=n_steps)
                    )

    def __len__(self) -> int:
        return len(self._samples)

    @staticmethod
    def _decode_text(raw) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        if isinstance(raw, np.ndarray):
            try:
                return raw.tobytes().decode("utf-8", errors="ignore")
            except Exception:
                return str(raw)
        return str(raw)

    def _decode_instruction(self, demo: h5py.Group, t: int, key: str = "skill_instructions") -> str:
        if key not in demo:
            return ""
        return self._decode_text(demo[key][t])

    def _hierarchy_labels(self, spec: SampleSpec, demo: h5py.Group) -> dict:
        cache_key = (spec.file_path, spec.demo_key)
        if cache_key not in self._hierarchy_cache:
            if "skill_instructions" not in demo or len(demo["skill_instructions"]) == 0:
                raise ValueError(f"{spec.file_path}:{spec.demo_key} has no skill_instructions")
            skill_texts = [self._decode_text(raw).strip() for raw in demo["skill_instructions"][:]]
            task_instruction = (
                self._decode_text(demo["task_instructions"][0]).strip()
                if "task_instructions" in demo and len(demo["task_instructions"]) > 0
                else skill_texts[0]
            )
            skill_plan: list[str] = []
            phase_indices: list[int] = []
            phase_starts: list[int] = []
            phase_ends: list[int] = []
            for t, text in enumerate(skill_texts):
                if not skill_plan or text != skill_plan[-1]:
                    if skill_plan:
                        phase_ends[-1] = t - 1
                    skill_plan.append(text)
                    phase_starts.append(t)
                    phase_ends.append(len(skill_texts) - 1)
                phase_indices.append(len(skill_plan) - 1)
            if len(skill_plan) > self.max_skill_slots:
                raise ValueError(
                    f"{spec.file_path}:{spec.demo_key} has {len(skill_plan)} phases, "
                    f"exceeding max_skill_slots={self.max_skill_slots}"
                )
            self._hierarchy_cache[cache_key] = {
                "task_instruction": task_instruction,
                "skill_plan": skill_plan,
                "phase_indices": phase_indices,
                "phase_starts": phase_starts,
                "phase_ends": phase_ends,
                "skill_kinds": [
                    classify_skill_instruction(text) for text in skill_plan
                ],
            }
        labels = self._hierarchy_cache[cache_key]
        phase_index = int(labels["phase_indices"][spec.t])
        valid = np.zeros(self.max_skill_slots, dtype=np.float32)
        valid[: len(labels["skill_plan"])] = 1.0
        return {
            "task_instruction": labels["task_instruction"],
            "skill_plan": list(labels["skill_plan"]),
            "phase_index": phase_index,
            "phase_start": int(labels["phase_starts"][phase_index]),
            "phase_end": int(labels["phase_ends"][phase_index]),
            "skill_kind": str(labels["skill_kinds"][phase_index]),
            "skill_instruction": str(labels["skill_plan"][phase_index]),
            "phase_termination": float(spec.t >= int(labels["phase_ends"][phase_index])),
            "skill_valid_mask": valid,
        }

    def __getitem__(self, idx: int) -> dict:
        spec = self._samples[idx]
        f = self._open(spec.file_path)
        demo = f["data"][spec.demo_key]
        obs = demo["obs"]

        frame_indices = [max(0, spec.t - i) for i in reversed(range(self.n_obs_steps))]
        rgb_frames = np.stack([np.asarray(obs[self.rgb_key][i]) for i in frame_indices], axis=0)
        if rgb_frames.dtype != np.uint8:
            rgb_frames = rgb_frames.astype(np.uint8)
        if rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3:
            raise RuntimeError(f"Unexpected RGB history shape {rgb_frames.shape} in {spec.file_path}")
        rgb_current = rgb_frames[-1]
        if self.n_obs_steps == 1:
            rgb_tensor = torch.from_numpy(rgb_current).permute(2, 0, 1).float() / 255.0
        else:
            rgb_tensor = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0

        if self.mask_key in obs:
            mask = np.asarray(obs[self.mask_key][spec.t])
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros(rgb_current.shape[:2], dtype=np.float32)

        instruction = self._decode_instruction(demo, spec.t)
        hierarchy = (
            self._hierarchy_labels(spec, demo)
            if self.use_hierarchy
            or (self.use_contact_xyz and self.contact_label_mode == "skill_specific")
            else None
        )

        actions_full = np.asarray(demo["actions"][spec.t : spec.t + self.action_chunk])
        if actions_full.shape[0] < self.action_chunk:
            pad = self.action_chunk - actions_full.shape[0]
            actions_full = np.concatenate([actions_full, np.tile(actions_full[-1:], (pad, 1))], axis=0)

        full_actions_demo = np.asarray(demo["actions"][:])
        previous_gripper_command = (
            float(full_actions_demo[spec.t - 1, 6]) if spec.t > 0 else float(actions_full[0, 6])
        )
        gripper_closed, gripper_weights = derive_gripper_targets(
            actions_full,
            previous_command=previous_gripper_command,
            transition_weight=self.gripper_transition_weight,
        )
        gripper_demo = (
            np.asarray(obs[self.gripper_key][:])
            if self.gripper_key in obs
            else np.zeros((spec.n_steps, 1), dtype=np.float32)
        )
        joints_demo = (
            np.asarray(obs[self.joints_key][:])
            if self.joints_key in obs
            else np.zeros((spec.n_steps, 7), dtype=np.float32)
        )

        part_pcd_demo = None
        tcp_pose_demo = None
        contact_target = None
        if self.use_contact_xyz and self.contact_label_mode == "skill_specific":
            assert hierarchy is not None
            part_pcd_demo = np.asarray(obs[self.part_pcd_key][:]).astype(np.float32)
            tcp_pose_demo = np.asarray(obs[self.tcp_pose_key][:]).astype(np.float32)
            event_key = (
                spec.file_path,
                spec.demo_key,
                int(hierarchy["phase_index"]),
            )
            contact_target = derive_skill_contact_target(
                actions=full_actions_demo,
                part_pcd=part_pcd_demo,
                tcp_pose=tcp_pose_demo,
                skill_kind=hierarchy["skill_kind"],
                segment_start=hierarchy["phase_start"],
                segment_end=hierarchy["phase_end"],
                current_t=spec.t,
                window=4,
                contact_event=self._contact_event_cache.get(event_key),
            )
            if event_key not in self._contact_event_cache:
                self._contact_event_cache[event_key] = {
                    "contact_t": int(contact_target["contact_t"]),
                    "method": contact_target["method"],
                    "reason": (
                        contact_target["reason"]
                        if int(contact_target["contact_t"]) < 0
                        else ""
                    ),
                    "tcp_to_part_distance_m": contact_target[
                        "tcp_to_part_distance_m"
                    ],
                }
            contact_xy_norm = contact_target["contact_xy"]
            approach_dir = contact_target["approach_dir"]
            contact_t = int(contact_target["contact_t"])
        else:
            contact_xy_norm, approach_dir, contact_t = derive_contact_and_approach(
                actions=full_actions_demo,
                gripper=gripper_demo,
                t=spec.t,
                window=4,
            )

        sample = {
            "rgb": rgb_tensor,
            "part_mask": torch.from_numpy(mask).float(),
            "instruction": instruction,
            "action_chunk": torch.from_numpy(actions_full).float(),
            "gripper_closed": torch.from_numpy(gripper_closed).float(),
            "gripper_loss_weight": torch.from_numpy(gripper_weights).float(),
            "gripper_state": torch.from_numpy(gripper_demo[spec.t].astype(np.float32)),
            "joint_states": torch.from_numpy(joints_demo[spec.t].astype(np.float32)),
            "contact_xy": torch.from_numpy(contact_xy_norm).float(),
            "approach_dir": torch.from_numpy(approach_dir).float(),
            "meta": {
                "file": os.path.basename(spec.file_path),
                "demo": spec.demo_key,
                "t": int(spec.t),
                "contact_t": int(contact_t),
                "contact_label_mode": self.contact_label_mode,
            },
        }
        if contact_target is not None:
            sample["meta"].update(
                skill_kind=hierarchy["skill_kind"],
                skill_instruction=hierarchy["skill_instruction"],
                skill_start=int(hierarchy["phase_start"]),
                skill_end=int(hierarchy["phase_end"]),
                contact_method=contact_target["method"],
                contact_xyz_valid=float(contact_target["valid"]),
                contact_invalid_reason=contact_target["reason"],
                contact_tcp_to_part_distance_m=contact_target[
                    "tcp_to_part_distance_m"
                ],
            )
        if self.use_pcd:
            sample["agentview_pcd"] = torch.from_numpy(
                np.asarray(obs[self.pcd_key][spec.t]).astype(np.float32)
            )
        if self.use_part_pcd or self.use_contact_xyz or self.use_point_part_labels:
            part_pcd_current = np.asarray(obs[self.part_pcd_key][spec.t]).astype(np.float32)
            if self.use_part_pcd:
                sample["agentview_part_pcd"] = torch.from_numpy(part_pcd_current)
            if self.use_point_part_labels:
                if self.use_pcd and part_pcd_current.shape[0] != sample[
                    "agentview_pcd"
                ].shape[0]:
                    raise RuntimeError(
                        "point-part labels require aligned scene/part point counts"
                    )
                point_labels = (
                    (part_pcd_current[:, 3] > 0.5).astype(np.float32)
                    if part_pcd_current.shape[-1] >= 4
                    else np.zeros(part_pcd_current.shape[0], dtype=np.float32)
                )
                sample["point_part_labels"] = torch.from_numpy(point_labels)
                sample["point_part_valid"] = torch.tensor(
                    float(
                        point_labels.any()
                        and hierarchy is not None
                        and hierarchy["skill_kind"] in {"grasp_obj", "touch_obj"}
                    ),
                    dtype=torch.float32,
                )
        if self.use_tcp_pose or self.use_contact_xyz:
            if tcp_pose_demo is None:
                tcp_pose_demo = np.asarray(obs[self.tcp_pose_key][:]).astype(np.float32)
            if self.use_tcp_pose:
                sample["tcp_pose"] = torch.from_numpy(tcp_pose_demo[spec.t])
        if self.use_contact_xyz:
            if contact_target is not None:
                contact_xyz = contact_target["contact_xyz"]
                contact_valid = contact_target["valid"]
            else:
                if part_pcd_demo is None:
                    part_pcd_demo = np.asarray(
                        obs[self.part_pcd_key][:]
                    ).astype(np.float32)
                assert tcp_pose_demo is not None
                contact_xyz, contact_valid = derive_contact_xyz(
                    part_pcd_demo, tcp_pose_demo, contact_t, current_t=spec.t
                )
            sample["contact_xyz"] = torch.from_numpy(contact_xyz).float()
            sample["contact_xyz_valid"] = torch.tensor(
                contact_valid, dtype=torch.float32
            )
        if self.use_hierarchy:
            assert hierarchy is not None
            sample["task_instruction"] = hierarchy["task_instruction"]
            sample["skill_plan"] = hierarchy["skill_plan"]
            sample["phase_index"] = torch.tensor(hierarchy["phase_index"], dtype=torch.long)
            sample["phase_termination"] = torch.tensor(hierarchy["phase_termination"], dtype=torch.float32)
            sample["skill_valid_mask"] = torch.from_numpy(hierarchy["skill_valid_mask"])
        return sample

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def collate_part2action(batch: List[dict]) -> dict:
    """Custom collate: stacks tensors but keeps instructions as a list of str."""
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        if k in {"instruction", "task_instruction", "skill_plan"}:
            out[k] = [b[k] for b in batch]
        elif k == "meta":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def list_demos(hdf5_path: str) -> List[str]:
    """Helper: list demo keys in a PartInstruct HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        return sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_")[-1]),
        )


def write_split_json(
    hdf5_paths: Sequence[str],
    out_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Deterministic train/val split at the demo level (not timestep level)."""
    rng = np.random.RandomState(seed)
    splits = {"train": {}, "val": {}}
    for p in hdf5_paths:
        demos = list_demos(p)
        rng.shuffle(demos)
        n_val = max(1, int(len(demos) * val_ratio))
        splits["val"][p] = demos[:n_val]
        splits["train"][p] = demos[n_val:]
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
