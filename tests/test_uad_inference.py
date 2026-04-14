#!/usr/bin/env python3
"""
Quick test: Run UAD inference on the example image.
Will download DINOv2 weights (~85MB) on first run.

Usage:
    conda activate uad
    cd unsup-affordance/src
    python ../../tests/test_uad_inference.py
"""
import sys
import os
import numpy as np
from pathlib import Path

# Setup paths
PROJECT = Path(__file__).resolve().parent.parent
UAD_DIR = PROJECT / "unsup-affordance"
UAD_SRC = UAD_DIR / "src"
sys.path.insert(0, str(UAD_SRC))
os.chdir(UAD_SRC)

from PIL import Image

# Load the example image
img_path = UAD_DIR / "examples" / "example_image.png"
img = Image.open(img_path).convert("RGB")
img_np = np.array(img)
print(f"Image: {img_np.shape} {img_np.dtype}")

# Load UAD
from inference import AffordanceInference
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import load_config

config_path = str(UAD_DIR / "configs" / "st_emb.yaml")
checkpoint_path = str(UAD_DIR / "checkpoints" / "st_emb.pth")

cfg = load_config(config_path)
text_embedding_func = get_text_embedding_options(cfg.get("text_embedding", "embeddings_st"))

print("Loading UAD model (DINOv2 weights will download on first run)...")
inference = AffordanceInference(config_path, checkpoint_path, text_embedding_func)
print("Model loaded!")

# Test multiple queries
queries = ["twist open", "grasp the handle", "pick up"]
out_dir = PROJECT / "affordance-pipeline" / "results" / "uad_test"
out_dir.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for query in queries:
    print(f'\nQuery: "{query}"')
    heatmap = inference.predict(img_np, query, thresh=None)
    print(f"  Heatmap: shape={heatmap.shape}, range=[{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # Threshold stats
    for t in [0.3, 0.5, 0.7]:
        n = int((heatmap > t).sum())
        pct = 100.0 * n / heatmap.size
        print(f"  thresh={t}: {n} pixels ({pct:.1f}%)")

    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f'UAD Heatmap: "{query}"', fontsize=14)
    axes[1].axis("off")

    overlay = img_np.copy().astype(float) / 255.0
    heat_rgb = plt.cm.hot(heatmap)[:, :, :3]
    blended = overlay * 0.5 + heat_rgb * 0.5
    axes[2].imshow(blended)
    axes[2].set_title("Overlay", fontsize=14)
    axes[2].axis("off")

    plt.tight_layout()
    safe_name = query.replace(" ", "_")
    out_path = out_dir / f"uad_test_{safe_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

print("\n✓ UAD inference test PASSED!")
