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

from .targets import derive_contact_and_approach


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

        self._files: dict[str, h5py.File] = {}
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

    def _decode_instruction(self, demo: h5py.Group, t: int) -> str:
        if "skill_instructions" not in demo:
            return ""
        raw = demo["skill_instructions"][t]
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        if isinstance(raw, np.ndarray):
            try:
                return raw.tobytes().decode("utf-8", errors="ignore")
            except Exception:
                return str(raw)
        return str(raw)

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

        actions_full = np.asarray(demo["actions"][spec.t : spec.t + self.action_chunk])
        if actions_full.shape[0] < self.action_chunk:
            pad = self.action_chunk - actions_full.shape[0]
            actions_full = np.concatenate([actions_full, np.tile(actions_full[-1:], (pad, 1))], axis=0)

        full_actions_demo = np.asarray(demo["actions"][:])
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

        contact_xy_norm, approach_dir, contact_t = derive_contact_and_approach(
            actions=full_actions_demo,
            gripper=gripper_demo,
            t=spec.t,
            window=4,
        )

        return {
            "rgb": rgb_tensor,
            "part_mask": torch.from_numpy(mask).float(),
            "instruction": instruction,
            "action_chunk": torch.from_numpy(actions_full).float(),
            "gripper_state": torch.from_numpy(gripper_demo[spec.t].astype(np.float32)),
            "joint_states": torch.from_numpy(joints_demo[spec.t].astype(np.float32)),
            "contact_xy": torch.from_numpy(contact_xy_norm).float(),
            "approach_dir": torch.from_numpy(approach_dir).float(),
            "meta": {
                "file": os.path.basename(spec.file_path),
                "demo": spec.demo_key,
                "t": int(spec.t),
                "contact_t": int(contact_t),
            },
        }

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
        if k == "instruction":
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
