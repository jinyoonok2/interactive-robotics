"""Generate real-data configs for a selected PartInstruct object subset.

Example:
    python scripts/make_object_configs.py --objects mug bottle scissors --tag mug_bottle_scissors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def normalize_objects(raw_objects: list[str]) -> list[str]:
    objects: list[str] = []
    for item in raw_objects:
        objects.extend(part.strip() for part in item.replace(",", " ").split())
    objects = [obj for obj in objects if obj]
    if not objects:
        raise ValueError("At least one object is required")
    return objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", required=True, help="Object names matching demos/<object>.hdf5")
    parser.add_argument("--tag", default=None, help="Suffix for generated config names")
    parser.add_argument("--out-dir", type=Path, default=CONFIG_DIR / "generated")
    args = parser.parse_args()

    objects = normalize_objects(args.objects)
    tag = args.tag or "_".join(objects)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_hdf5 = [f"../datasets/PartInstruct/demos/{obj}.hdf5" for obj in objects]
    for src in sorted(CONFIG_DIR.glob("*_real.yaml")):
        cfg = yaml.safe_load(src.read_text())
        cfg.setdefault("data", {})["train_hdf5"] = train_hdf5
        cfg["name"] = f"{cfg.get('name', src.stem)}_{tag}"
        if "output_dir" in cfg:
            cfg["output_dir"] = str(Path(cfg["output_dir"]) / tag)

        dst = out_dir / f"{src.stem}_{tag}.yaml"
        dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
