"""Shared utilities for the train/eval scripts."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | os.PathLike, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("[device] CUDA requested but unavailable; falling back to CPU.")
    return torch.device("cpu")


def resolve_paths(cfg: dict, root: Path) -> dict:
    """Make relative paths in `data` and `output_dir` absolute."""
    data = cfg.setdefault("data", {})
    for key in ("train_hdf5", "val_hdf5"):
        if key in data and data[key]:
            data[key] = [str((root / p).resolve()) if not os.path.isabs(p) else p for p in data[key]]
    out = cfg.get("output_dir")
    if out and not os.path.isabs(out):
        cfg["output_dir"] = str((root / out).resolve())
    return cfg
