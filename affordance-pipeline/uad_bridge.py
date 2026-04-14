#!/usr/bin/env python3
"""
UAD Bridge — Run unsupervised affordance detection from the habitat-grasp env.

Since UAD requires Python 3.10 + numpy 2.x (the `uad` conda env) while the
main pipeline uses Python 3.9 + numpy 1.26.x (`habitat-grasp` env), this
module calls UAD inference as a subprocess and communicates via .npy files.

Usage from habitat-grasp env:
    from uad_bridge import predict_affordance
    mask = predict_affordance(rgb, "grasp the handle", threshold=0.5)
"""

import os
import sys
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UAD_DIR = PROJECT_ROOT / "unsup-affordance"
UAD_SRC = UAD_DIR / "src"
UAD_CONFIG = UAD_DIR / "configs" / "st_emb.yaml"
UAD_CHECKPOINT = UAD_DIR / "checkpoints" / "st_emb.pth"

# The standalone inference script we'll create next to this bridge
_WORKER_SCRIPT = Path(__file__).resolve().parent / "_uad_worker.py"


def _find_conda_python(env_name: str = "uad") -> str:
    """Find the Python executable for the uad conda env."""
    # Try common conda locations
    for base in [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
    ]:
        python_path = os.path.join(base, "envs", env_name, "bin", "python")
        if os.path.isfile(python_path):
            return python_path

    # Fallback: try to resolve via conda
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "which", "python"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    raise FileNotFoundError(
        f"Could not find Python for conda env '{env_name}'. "
        f"Make sure the uad environment is set up (see ENVIRONMENT_STATUS.md)."
    )


def predict_affordance(
    rgb: np.ndarray,
    text_query: str,
    threshold: float = 0.5,
    config_path: str = None,
    checkpoint_path: str = None,
    conda_env: str = "uad",
) -> np.ndarray:
    """
    Run UAD affordance prediction on an RGB image.

    Spawns a subprocess in the `uad` conda environment, runs inference,
    and returns the binary affordance mask.

    Args:
        rgb:             RGB image (H, W, 3) uint8
        text_query:      Natural language affordance query (e.g. "grasp the handle")
        threshold:       Binarization threshold for affordance map (0-1)
        config_path:     Path to UAD config YAML (default: configs/st_emb.yaml)
        checkpoint_path: Path to UAD checkpoint (default: checkpoints/st_emb.pth)
        conda_env:       Name of the conda environment with UAD installed

    Returns:
        Binary mask (H, W) as bool numpy array — True where affordance is detected
    """
    config_path = config_path or str(UAD_CONFIG)
    checkpoint_path = checkpoint_path or str(UAD_CHECKPOINT)

    python_bin = _find_conda_python(conda_env)

    with tempfile.TemporaryDirectory(prefix="uad_bridge_") as tmpdir:
        # Write input image
        img_path = os.path.join(tmpdir, "input_rgb.npy")
        np.save(img_path, rgb)

        # Write request
        request = {
            "image_path": img_path,
            "text_query": text_query,
            "threshold": threshold,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "output_path": os.path.join(tmpdir, "affordance_mask.npy"),
            "output_raw_path": os.path.join(tmpdir, "affordance_raw.npy"),
        }
        request_path = os.path.join(tmpdir, "request.json")
        with open(request_path, "w") as f:
            json.dump(request, f)

        # Run UAD worker in uad env
        print(f"  Running UAD inference (env={conda_env}, threshold={threshold})...")
        print(f"  Text query: \"{text_query}\"")

        result = subprocess.run(
            [python_bin, str(_WORKER_SCRIPT), request_path],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(UAD_SRC),
        )

        if result.stdout:
            # Print UAD worker output with indent
            for line in result.stdout.strip().split("\n"):
                print(f"    [uad] {line}")

        if result.returncode != 0:
            err_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(
                f"UAD inference failed (exit code {result.returncode}):\n{err_msg}"
            )

        # Load result
        mask_path = request["output_path"]
        raw_path = request["output_raw_path"]

        if not os.path.exists(mask_path):
            raise RuntimeError("UAD worker did not produce output mask")

        mask = np.load(mask_path).astype(bool)
        n_pixels = int(mask.sum())
        H, W = mask.shape
        pct = 100.0 * n_pixels / (H * W)
        print(f"  UAD affordance: {n_pixels} pixels ({pct:.1f}% of image)")

        # Optionally load raw heatmap for debugging
        if os.path.exists(raw_path):
            raw = np.load(raw_path)
            print(f"  Raw heatmap range: [{raw.min():.3f}, {raw.max():.3f}]")

        return mask


def predict_affordance_raw(
    rgb: np.ndarray,
    text_query: str,
    config_path: str = None,
    checkpoint_path: str = None,
    conda_env: str = "uad",
) -> np.ndarray:
    """
    Like predict_affordance() but returns the continuous heatmap (0-1)
    instead of a binary mask. Useful for visualization or custom thresholding.
    """
    config_path = config_path or str(UAD_CONFIG)
    checkpoint_path = checkpoint_path or str(UAD_CHECKPOINT)

    python_bin = _find_conda_python(conda_env)

    with tempfile.TemporaryDirectory(prefix="uad_bridge_") as tmpdir:
        img_path = os.path.join(tmpdir, "input_rgb.npy")
        np.save(img_path, rgb)

        request = {
            "image_path": img_path,
            "text_query": text_query,
            "threshold": None,  # no thresholding — return raw
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "output_path": os.path.join(tmpdir, "affordance_mask.npy"),
            "output_raw_path": os.path.join(tmpdir, "affordance_raw.npy"),
        }
        request_path = os.path.join(tmpdir, "request.json")
        with open(request_path, "w") as f:
            json.dump(request, f)

        result = subprocess.run(
            [python_bin, str(_WORKER_SCRIPT), request_path],
            capture_output=True, text=True, timeout=120,
            cwd=str(UAD_SRC),
        )

        if result.returncode != 0:
            err_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"UAD inference failed:\n{err_msg}")

        raw_path = request["output_raw_path"]
        if not os.path.exists(raw_path):
            raise RuntimeError("UAD worker did not produce raw heatmap")

        return np.load(raw_path)
