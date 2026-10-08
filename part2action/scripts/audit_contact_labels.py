"""Audit legacy and skill-specific PartInstruct contact pseudo-labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from data.targets import (
    _detect_contact_step,
    classify_skill_instruction,
    derive_skill_contact_target,
)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _segments(raw_instructions: np.ndarray) -> list[tuple[int, int, str]]:
    texts = [_decode(value).strip() for value in raw_instructions]
    if not texts:
        return []
    segments: list[tuple[int, int, str]] = []
    start, current = 0, texts[0]
    for t, text in enumerate(texts[1:], start=1):
        if text != current:
            segments.append((start, t - 1, current))
            start, current = t, text
    segments.append((start, len(texts) - 1, current))
    return segments


def _new_stats() -> dict[str, Any]:
    return {
        "segments": 0,
        "legacy_final_timestep": 0,
        "legacy_segment_start": 0,
        "legacy_cross_phase": 0,
        "corrected_valid": 0,
        "corrected_invalid": 0,
        "corrected_selected_final_timestep": 0,
        "distance_sum_m": 0.0,
        "distance_count": 0,
        "methods": Counter(),
        "invalid_reasons": Counter(),
    }


def audit(paths: list[Path], max_demos_per_file: int | None) -> dict[str, Any]:
    by_skill: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    instruction_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    demos_scanned = 0

    for path in paths:
        with h5py.File(path, "r") as handle:
            demo_keys = sorted(
                handle["data"].keys(),
                key=lambda key: int(key.split("_")[-1]),
            )
            if max_demos_per_file is not None:
                demo_keys = demo_keys[:max_demos_per_file]
            for demo_key in demo_keys:
                demo = handle["data"][demo_key]
                obs = demo["obs"]
                required = {
                    "skill_instructions": demo.get("skill_instructions"),
                    "actions": demo.get("actions"),
                    "tcp_pose": obs.get("tcp_pose"),
                    "part_pcd": obs.get("agentview_part_pcd"),
                }
                if any(value is None for value in required.values()):
                    continue
                actions = np.asarray(required["actions"][:], dtype=np.float32)
                tcp_pose = np.asarray(required["tcp_pose"][:], dtype=np.float32)
                part_pcd = np.asarray(required["part_pcd"][:], dtype=np.float32)
                demos_scanned += 1

                for start, end, instruction in _segments(
                    required["skill_instructions"][:]
                ):
                    skill_kind = classify_skill_instruction(instruction)
                    stats = by_skill[skill_kind]
                    stats["segments"] += 1
                    instruction_counts[instruction] += 1

                    legacy_t = _detect_contact_step(actions, start)
                    stats["legacy_final_timestep"] += int(
                        legacy_t == len(actions) - 1
                    )
                    stats["legacy_segment_start"] += int(legacy_t == start)
                    stats["legacy_cross_phase"] += int(legacy_t > end)

                    corrected = derive_skill_contact_target(
                        actions=actions,
                        part_pcd=part_pcd,
                        tcp_pose=tcp_pose,
                        skill_kind=skill_kind,
                        segment_start=start,
                        segment_end=end,
                        current_t=start,
                    )
                    valid = bool(corrected["valid"])
                    stats["corrected_valid" if valid else "corrected_invalid"] += 1
                    stats["methods"][corrected["method"]] += 1
                    if corrected["reason"]:
                        stats["invalid_reasons"][corrected["reason"]] += 1
                    selected_t = int(corrected["contact_t"])
                    stats["corrected_selected_final_timestep"] += int(
                        selected_t == len(actions) - 1
                    )
                    distance = corrected["tcp_to_part_distance_m"]
                    if distance is not None:
                        stats["distance_sum_m"] += float(distance)
                        stats["distance_count"] += 1

                    if len(examples) < 20 and (
                        legacy_t > end
                        or (skill_kind == "touch_obj" and valid)
                        or not valid
                    ):
                        examples.append(
                            {
                                "file": path.name,
                                "demo": demo_key,
                                "instruction": instruction,
                                "skill_kind": skill_kind,
                                "segment": [start, end],
                                "legacy_contact_t": int(legacy_t),
                                "corrected_contact_t": selected_t,
                                "corrected_method": corrected["method"],
                                "corrected_valid": valid,
                                "corrected_reason": corrected["reason"],
                                "tcp_to_part_distance_m": distance,
                            }
                        )

    finalized = {}
    for skill_kind, raw in sorted(by_skill.items()):
        count = int(raw["distance_count"])
        finalized[skill_kind] = {
            **raw,
            "methods": dict(raw["methods"]),
            "invalid_reasons": dict(raw["invalid_reasons"]),
            "mean_selected_tcp_to_part_distance_m": (
                float(raw["distance_sum_m"]) / count if count else None
            ),
        }
    return {
        "files": [str(path) for path in paths],
        "max_demos_per_file": max_demos_per_file,
        "demos_scanned": demos_scanned,
        "instruction_counts": dict(instruction_counts.most_common()),
        "by_skill": finalized,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, action="append", required=True)
    parser.add_argument("--max-demos-per-file", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for path in args.hdf5]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HDF5 files: {missing}")
    report = audit(paths, args.max_demos_per_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[contact-audit] wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
