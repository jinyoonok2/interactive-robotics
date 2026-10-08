"""Verify correspondence between PartInstruct scene and target-part point clouds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial import cKDTree


_THRESHOLDS_M = (1e-5, 1e-4, 1e-3, 5e-3, 1e-2)


def _summary(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.concatenate(values).astype(np.float64)
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean_m": float(array.mean()),
        "median_m": float(np.median(array)),
        "p95_m": float(np.quantile(array, 0.95)),
        "max_m": float(array.max()),
    }
    for threshold in _THRESHOLDS_M:
        result[f"within_{threshold:g}m_rate"] = float((array <= threshold).mean())
    return result


def audit_file(
    path: Path,
    *,
    max_demos: int | None,
    frame_stride: int,
) -> dict[str, Any]:
    same_index_distances: list[np.ndarray] = []
    part_to_scene_nn_distances: list[np.ndarray] = []
    scene_to_part_nn_distances: list[np.ndarray] = []
    frames = demos = flagged_points = 0
    shape_mismatches = empty_part_frames = 0

    with h5py.File(path, "r") as handle:
        demo_keys = sorted(
            handle["data"].keys(),
            key=lambda key: int(key.split("_")[-1]),
        )
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        for demo_key in demo_keys:
            obs = handle["data"][demo_key].get("obs")
            if (
                obs is None
                or "agentview_pcd" not in obs
                or "agentview_part_pcd" not in obs
            ):
                continue
            scene_ds = obs["agentview_pcd"]
            part_ds = obs["agentview_part_pcd"]
            length = min(len(scene_ds), len(part_ds))
            demos += 1
            for t in range(0, length, max(1, frame_stride)):
                scene = np.asarray(scene_ds[t], dtype=np.float32)[..., :3]
                part = np.asarray(part_ds[t], dtype=np.float32)
                if scene.ndim != 2 or part.ndim != 2 or part.shape[-1] < 3:
                    shape_mismatches += 1
                    continue
                scene_valid = np.isfinite(scene).all(axis=-1)
                part_valid = np.isfinite(part[:, :3]).all(axis=-1)
                if part.shape[-1] >= 4:
                    part_valid &= part[:, 3] > 0.5
                scene_points = scene[scene_valid]
                part_points = part[part_valid, :3]
                frames += 1
                flagged_points += int(len(part_points))
                if not len(part_points) or not len(scene_points):
                    empty_part_frames += int(not len(part_points))
                    continue

                if len(scene) == len(part):
                    aligned_valid = part_valid & scene_valid
                    same_index_distances.append(
                        np.linalg.norm(
                            scene[aligned_valid] - part[aligned_valid, :3],
                            axis=-1,
                        )
                    )
                else:
                    shape_mismatches += 1

                scene_tree = cKDTree(scene_points)
                part_to_scene_nn_distances.append(
                    scene_tree.query(part_points, k=1)[0].astype(np.float32)
                )
                part_tree = cKDTree(part_points)
                scene_to_part_nn_distances.append(
                    part_tree.query(scene_points, k=1)[0].astype(np.float32)
                )

    return {
        "file": str(path),
        "demos_scanned": demos,
        "frames_sampled": frames,
        "flagged_part_points": flagged_points,
        "shape_mismatches": shape_mismatches,
        "empty_part_frames": empty_part_frames,
        "same_index_part_to_scene_distance": _summary(same_index_distances),
        "nearest_part_to_scene_distance": _summary(
            part_to_scene_nn_distances
        ),
        "nearest_scene_to_part_distance": _summary(
            scene_to_part_nn_distances
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, action="append", required=True)
    parser.add_argument("--max-demos-per-file", type=int, default=100)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = [path.expanduser().resolve() for path in args.hdf5]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HDF5 files: {missing}")
    report = {
        "max_demos_per_file": args.max_demos_per_file,
        "frame_stride": args.frame_stride,
        "thresholds_m": list(_THRESHOLDS_M),
        "files": [
            audit_file(
                path,
                max_demos=args.max_demos_per_file,
                frame_stride=args.frame_stride,
            )
            for path in paths
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[pointcloud-audit] wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
