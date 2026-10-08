"""Aggregate held-out PartGym point-part diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _counts() -> dict[str, int]:
    return {
        "trials": 0,
        "target_positive_points": 0,
        "predicted_positive_points": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }


def _finalize(raw: dict[str, int]) -> dict[str, Any]:
    precision = raw["true_positive"] / max(
        1, raw["true_positive"] + raw["false_positive"]
    )
    recall = raw["true_positive"] / max(
        1, raw["true_positive"] + raw["false_negative"]
    )
    return {
        **raw,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(1e-8, precision + recall),
    }


def analyze(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    overall: dict[str, dict[str, int]] = defaultdict(_counts)
    by_skill: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(_counts)
    )
    unsupported: Counter[str] = Counter()
    diagnosed = 0
    for result in payload.get("results", []):
        diagnostic = result.get("initial_point_part_diagnostic") or {}
        if not diagnostic.get("supported", False):
            unsupported[str(diagnostic.get("reason", "missing diagnostic"))] += 1
            continue
        diagnosed += 1
        skill = str(diagnostic.get("skill_name", "unknown"))
        for radius, metrics in diagnostic.get("radius_metrics", {}).items():
            for bucket in (overall[radius], by_skill[skill][radius]):
                bucket["trials"] += 1
                for key in (
                    "target_positive_points",
                    "predicted_positive_points",
                    "true_positive",
                    "false_positive",
                    "false_negative",
                ):
                    bucket[key] += int(metrics.get(key, 0))
    return {
        "source": str(path),
        "rollout_trials": len(payload.get("results", [])),
        "diagnosed_trials": diagnosed,
        "unsupported_reasons": dict(unsupported),
        "overall_by_radius_m": {
            radius: _finalize(counts)
            for radius, counts in sorted(overall.items(), key=lambda item: float(item[0]))
        },
        "by_skill_and_radius_m": {
            skill: {
                radius: _finalize(counts)
                for radius, counts in sorted(radii.items(), key=lambda item: float(item[0]))
            }
            for skill, radii in sorted(by_skill.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input.expanduser().resolve())
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[point-part-analysis] wrote {out}")


if __name__ == "__main__":
    main()
