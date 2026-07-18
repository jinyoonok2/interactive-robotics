"""Validate the maintained hierarchical Part2Action configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "configs" / "architecture_update" / "hierarchical_gated_pcd_contact3d_all_demos.yaml"


def validate(path: Path, check_files: bool = False) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    errors = []
    data, model, losses = cfg.get("data", {}), cfg.get("model", {}), cfg.get("losses", {})
    if not model.get("use_hierarchy"):
        errors.append("model.use_hierarchy must be true")
    if not data.get("use_hierarchy"):
        errors.append("data.use_hierarchy must be true")
    if model.get("max_skill_slots", 4) != data.get("max_skill_slots", 4):
        errors.append("model/data max_skill_slots must match")
    if model.get("use_pcd") and not data.get("use_pcd"):
        errors.append("3D policy requires deployable full-scene PCD")
    if data.get("use_part_pcd"):
        errors.append("target-part PCD must not be a policy input")
    for key in ("skill_embedding_weight", "slot_validity_weight", "phase_weight", "termination_weight"):
        if float(losses.get(key, 0)) <= 0:
            errors.append(f"{key} must be positive")
    if check_files:
        for item in data.get("train_hdf5", []):
            candidate = Path(item) if Path(item).is_absolute() else ROOT / item
            if not candidate.resolve().exists():
                errors.append(f"missing dataset: {candidate.resolve()}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path, default=[DEFAULT])
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    failed = False
    for path in args.configs:
        errors = validate(path.expanduser().resolve(), args.check_files)
        print(f"[{'ERROR' if errors else 'OK'}] {path}")
        for error in errors:
            print(f"  {error}")
        failed |= bool(errors)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
