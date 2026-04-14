#!/usr/bin/env python3
"""
CLIPSeg Part Detection Test

Tests CLIPSeg's ability to localize object parts from text queries.
Runs TWO modes:
  - PURE:    Full scene RGB, no depth mask, no cropping
  - CROPPED: Depth-based crop to object bounding box (15% padding),
             same preprocessing as the pipeline (core/clipseg_detector.py)

For each object (mug, hammer, drill):
  1. Load RGB + depth from output/
  2. Run CLIPSeg with part-name queries (e.g. "handle", "rim", "body")
  3. Visualize heatmaps, threshold sweeps, and discrimination maps

Usage (from habitat-lab/):
  /home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/clipseg/test_part_detection.py --object mug
  /home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/clipseg/test_part_detection.py --object hammer
  /home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/clipseg/test_part_detection.py --object drill

Outputs to: affordance-pipeline/results/diagnosis/clipseg/{object_name}/
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

import torch
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ── Paths ──
PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PIPELINE_DIR / "output"
DIAG_BASE = PIPELINE_DIR / "results" / "diagnosis" / "clipseg"


def get_object_config(obj_name):
    """Load object info from objects.json."""
    cfg_path = PIPELINE_DIR / "config" / "objects.json"
    with open(cfg_path) as f:
        catalog = json.load(f)
    if obj_name not in catalog:
        available = [k for k in catalog if not k.startswith('_')]
        raise ValueError(f"Object '{obj_name}' not in {cfg_path}. Available: {available}")
    return catalog[obj_name]


def load_rgb():
    rgb = np.array(Image.open(OUTPUT_DIR / "rgb.png"))
    print(f"  RGB loaded: {rgb.shape}")
    return rgb


def load_depth_mask():
    """Load depth-based object mask (same logic as pipeline)."""
    dep = np.load(OUTPUT_DIR / "depth_raw.npy")
    valid = dep[dep > 0]
    near = float(np.percentile(valid, 2))
    cutoff = near * 1.3
    mask = (dep > 0) & (dep <= cutoff)
    print(f"  Depth mask: near={near:.2f}m, cutoff={cutoff:.2f}m, {int(mask.sum())} pixels")
    return mask


def crop_to_object(rgb, obj_mask, padding=0.15):
    """Crop RGB to object bounding box with padding (same as pipeline)."""
    ys, xs = np.where(obj_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    H, W = rgb.shape[:2]
    pad_y = max(1, int((y1 - y0) * padding))
    pad_x = max(1, int((x1 - x0) * padding))
    y0c = max(0, y0 - pad_y)
    y1c = min(H, y1 + pad_y)
    x0c = max(0, x0 - pad_x)
    x1c = min(W, x1 + pad_x)
    crop = rgb[y0c:y1c, x0c:x1c]
    bbox = (y0c, x0c, y1c, x1c)
    print(f"  Crop: ({y0c},{x0c})->({y1c},{x1c})  {crop.shape[1]}x{crop.shape[0]}px")
    return crop, bbox


def load_clipseg():
    """Load CLIPSeg model and processor."""
    model_name = "CIDAS/clipseg-rd64-refined"
    print(f"  Loading CLIPSeg: {model_name}")
    processor = CLIPSegProcessor.from_pretrained(model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  Device: {device}")
    return processor, model, device


def run_clipseg_on_image(img, queries, processor, model, device):
    """
    Run CLIPSeg on an image (RGB array) with text queries.
    Returns dict: query -> heatmap (H, W) in [0, 1] at img's resolution.
    """
    pil_img = Image.fromarray(img)
    results = {}

    for query in queries:
        inputs = processor(
            text=[query],
            images=[pil_img],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        logits_up = torch.nn.functional.interpolate(
            logits.unsqueeze(0).unsqueeze(0),
            size=(img.shape[0], img.shape[1]),
            mode='bilinear',
            align_corners=False
        )[0, 0]

        heatmap = torch.sigmoid(logits_up).cpu().numpy()
        results[query] = heatmap

    return results


def run_clipseg_pure(rgb, queries, processor, model, device):
    """Run CLIPSeg on full scene (no crop)."""
    print("  Mode: PURE (full scene, no crop)")
    heatmaps = run_clipseg_on_image(rgb, queries, processor, model, device)
    for q in queries:
        h = heatmaps[q]
        print(f"    '{q}': min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f}")
    return heatmaps


def run_clipseg_cropped(rgb, queries, obj_mask, processor, model, device):
    """Run CLIPSeg on depth-cropped object region, map back to full image."""
    print("  Mode: CROPPED (depth-based object crop)")
    crop, bbox = crop_to_object(rgb, obj_mask)
    y0c, x0c, y1c, x1c = bbox

    crop_heatmaps = run_clipseg_on_image(crop, queries, processor, model, device)

    # Map crop heatmaps back to full image coordinates
    full_heatmaps = {}
    for q in queries:
        full = np.zeros(rgb.shape[:2], dtype=np.float32)
        full[y0c:y1c, x0c:x1c] = crop_heatmaps[q]
        full_heatmaps[q] = full
        h = crop_heatmaps[q]
        print(f"    '{q}': min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f} (on crop)")

    return full_heatmaps


def run_clipseg_pipeline(rgb, queries, obj_mask, processor, model, device):
    """
    Run CLIPSeg with full pipeline post-processing:
    crop + depth mask intersection + fallback thresholds.
    Reproduces the behavior of core/clipseg_detector.py.
    Returns dict: query -> binary mask (H, W) bool.
    """
    print("  Mode: PIPELINE (crop + depth mask + fallback thresholds)")
    crop, bbox = crop_to_object(rgb, obj_mask)
    y0c, x0c, y1c, x1c = bbox

    crop_heatmaps = run_clipseg_on_image(crop, queries, processor, model, device)

    masks = {}
    for q in queries:
        # Map crop heatmap back to full image
        full_heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
        full_heatmap[y0c:y1c, x0c:x1c] = crop_heatmaps[q]

        # Threshold at 0.4 (pipeline default), then intersect with depth mask
        mask = full_heatmap > 0.4
        mask = mask & obj_mask
        n_pixels = int(mask.sum())

        # Fallback thresholds (same as pipeline)
        if n_pixels == 0:
            for t in [0.3, 0.2, 0.1, 0.05]:
                fallback = full_heatmap > t
                fallback = fallback & obj_mask
                n_pixels = int(fallback.sum())
                if n_pixels > 0:
                    mask = fallback
                    print(f"    '{q}': fallback threshold={t} -> {n_pixels} pixels")
                    break

        if n_pixels == 0:
            print(f"    '{q}': NO pixels even at threshold=0.05")
        else:
            crop_max = crop_heatmaps[q].max()
            print(f"    '{q}': {n_pixels} pixels (crop max={crop_max:.3f})")

        masks[q] = mask

    return masks


def visualize_pipeline_masks(rgb, masks, obj_mask, obj_name, diag_dir):
    """Visualize pipeline mode binary masks (what the old pipeline produced)."""
    queries = list(masks.keys())
    n = len(queries)

    colors = [
        [228, 26, 28],    # red
        [55, 126, 184],   # blue
        [77, 175, 74],    # green
        [152, 78, 163],   # purple
        [255, 127, 0],    # orange
    ]

    fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 10))

    # Row 0: individual part masks
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    for qi, query in enumerate(queries):
        mask = masks[query]
        col = qi + 1
        ax = axes[0, col]
        overlay = rgb.copy()
        if mask.sum() > 0:
            overlay[mask] = (0.4 * rgb[mask] + 0.6 * np.array(colors[qi])).astype(np.uint8)
        n_pix = int(mask.sum())
        obj_pix = int(obj_mask.sum())
        pct = 100.0 * n_pix / obj_pix if obj_pix > 0 else 0
        ax.imshow(overlay)
        ax.set_title(f'"{query}"\n{n_pix:,} pixels ({pct:.1f}% of obj)',
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    # Row 1: all parts combined (winner-take-all on detected pixels)
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title("Depth mask\n(object region)", fontsize=12, fontweight='bold')
    # Show depth mask outline
    obj_vis = rgb.copy()
    obj_vis[obj_mask] = (0.6 * rgb[obj_mask] + 0.4 * np.array([200, 200, 200])).astype(np.uint8)
    axes[1, 0].imshow(obj_vis)
    axes[1, 0].axis('off')

    combined = rgb.copy()
    for qi, query in enumerate(queries):
        mask = masks[query]
        if mask.sum() > 0:
            combined[mask] = (0.4 * rgb[mask] + 0.6 * np.array(colors[qi])).astype(np.uint8)

    axes[1, 1].imshow(combined)
    axes[1, 1].set_title("All parts combined", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    for qi, q in enumerate(queries):
        axes[1, 1].plot([], [], 's', color=np.array(colors[qi]) / 255.0,
                        markersize=10, label=q)
    axes[1, 1].legend(loc='lower right', fontsize=9)

    # Hide remaining axes in row 1
    for ci in range(2, n + 1):
        axes[1, ci].axis('off')

    plt.suptitle(
        f"CLIPSeg Pipeline Mode — {obj_name}\n"
        f"(Crop + depth mask intersection + fallback thresholds)",
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    out_path = diag_dir / "pipeline_masks.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def visualize_heatmaps(rgb, heatmaps, obj_name, mode_label, diag_dir, filename):
    """
    Visualize raw CLIPSeg heatmaps overlaid on RGB.
    Row 0: raw heatmap overlay per query
    Row 1: thresholded masks at 0.5
    """
    queries = list(heatmaps.keys())
    n = len(queries)

    fig, axes = plt.subplots(2, n + 1, figsize=(5 * (n + 1), 10))

    # First column: original RGB
    for row in range(2):
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Original RGB", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')

    for qi, query in enumerate(queries):
        heatmap = heatmaps[query]
        col = qi + 1

        # Row 0: raw heatmap overlay
        ax = axes[0, col]
        ax.imshow(rgb)
        im = ax.imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f'"{query}"\n(max={heatmap.max():.2f})', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Row 1: binary mask at threshold 0.5
        ax2 = axes[1, col]
        mask = heatmap > 0.5
        overlay = rgb.copy()
        overlay[mask] = (0.4 * rgb[mask] + 0.6 * np.array([255, 50, 50])).astype(np.uint8)
        ax2.imshow(overlay)
        pct = 100.0 * mask.sum() / mask.size
        ax2.set_title(f'threshold=0.5\n({pct:.1f}% pixels)', fontsize=11)
        ax2.axis('off')

    axes[0, 0].set_ylabel("Raw heatmap", fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel("Thresh @ 0.5", fontsize=13, fontweight='bold')

    plt.suptitle(
        f"CLIPSeg Part Detection — {obj_name}\n({mode_label})",
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    out_path = diag_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def visualize_threshold_sweep(rgb, heatmaps, obj_name, mode_label, diag_dir, filename):
    """
    Show how thresholding affects detection at 0.3, 0.5, 0.7 for each query.
    """
    queries = list(heatmaps.keys())
    thresholds = [0.3, 0.5, 0.7]
    n_q = len(queries)
    n_t = len(thresholds)

    fig, axes = plt.subplots(n_q, n_t + 1, figsize=(5 * (n_t + 1), 5 * n_q))
    if n_q == 1:
        axes = axes[np.newaxis, :]

    for qi, query in enumerate(queries):
        heatmap = heatmaps[query]

        # First col: raw heatmap
        ax = axes[qi, 0]
        ax.imshow(rgb)
        ax.imshow(heatmap, cmap='jet', alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f'"{query}" raw', fontsize=11, fontweight='bold')
        ax.axis('off')

        for ti, thresh in enumerate(thresholds):
            ax = axes[qi, ti + 1]
            mask = heatmap > thresh
            overlay = rgb.copy()
            overlay[mask] = (0.4 * rgb[mask] + 0.6 * np.array([255, 50, 50])).astype(np.uint8)
            pct = 100.0 * mask.sum() / mask.size
            ax.imshow(overlay)
            ax.set_title(f'thresh={thresh} ({pct:.1f}%)', fontsize=11)
            ax.axis('off')

    plt.suptitle(
        f"CLIPSeg Threshold Sweep — {obj_name} ({mode_label})",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    out_path = diag_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def visualize_discrimination(rgb, heatmaps, obj_name, mode_label, diag_dir, filename):
    """
    Test whether CLIPSeg can discriminate between parts:
    - For each pixel, which query has the highest activation?
    - Show a winner-take-all map (like the k-means cluster map from DINOv2)
    """
    queries = list(heatmaps.keys())
    n = len(queries)

    # Stack heatmaps: (n_queries, H, W)
    stacked = np.stack([heatmaps[q] for q in queries], axis=0)
    winner = np.argmax(stacked, axis=0)  # (H, W) — index of winning query
    max_val = np.max(stacked, axis=0)    # (H, W) — confidence of winner

    # Colors for each query
    colors = [
        [228, 26, 28],    # red
        [55, 126, 184],   # blue
        [77, 175, 74],    # green
        [152, 78, 163],   # purple
        [255, 127, 0],    # orange
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: winner-take-all (all pixels)
    ax = axes[0]
    winner_img = np.zeros((*winner.shape, 3), dtype=np.uint8)
    for qi in range(n):
        winner_img[winner == qi] = colors[qi]
    blended = (0.4 * rgb + 0.6 * winner_img).astype(np.uint8)
    ax.imshow(blended)
    ax.set_title("Winner-take-all\n(all pixels)", fontsize=12, fontweight='bold')
    ax.axis('off')
    # Legend
    for qi, q in enumerate(queries):
        pct = 100.0 * (winner == qi).sum() / winner.size
        ax.plot([], [], 's', color=np.array(colors[qi]) / 255.0, markersize=10,
                label=f'{q} ({pct:.0f}%)')
    ax.legend(loc='lower right', fontsize=9)

    # Plot 2: winner-take-all with confidence threshold (only pixels > 0.5)
    ax = axes[1]
    confident_mask = max_val > 0.5
    winner_img2 = np.zeros((*winner.shape, 3), dtype=np.uint8)
    for qi in range(n):
        mask_qi = (winner == qi) & confident_mask
        winner_img2[mask_qi] = colors[qi]
    blended2 = rgb.copy()
    blended2[confident_mask] = (
        0.4 * rgb[confident_mask] + 0.6 * winner_img2[confident_mask]
    ).astype(np.uint8)
    ax.imshow(blended2)
    pct_conf = 100.0 * confident_mask.sum() / confident_mask.size
    ax.set_title(f"Winner-take-all\n(confidence > 0.5, {pct_conf:.1f}% pixels)", fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot 3: per-query activation comparison (bar chart of mean activation)
    ax = axes[2]
    means = [heatmaps[q].mean() for q in queries]
    maxes = [heatmaps[q].max() for q in queries]
    x = np.arange(n)
    width = 0.35
    bars1 = ax.bar(x - width / 2, means, width, label='Mean activation',
                   color=[np.array(c) / 255.0 for c in colors[:n]])
    bars2 = ax.bar(x + width / 2, maxes, width, label='Max activation',
                   color=[np.array(c) / 255.0 for c in colors[:n]], alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(queries, fontsize=11)
    ax.set_ylabel("Activation", fontsize=11)
    ax.set_title("Per-query activation stats", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)

    plt.suptitle(
        f"CLIPSeg Part Discrimination — {obj_name} ({mode_label})",
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    out_path = diag_dir / filename
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Print discrimination metrics
    print(f"\n  DISCRIMINATION ANALYSIS ({mode_label}):")
    print(f"  {'Query':<12} {'Mean':>8} {'Max':>8} {'Pixels>0.5':>12}")
    print(f"  {'-'*44}")
    for q in queries:
        h = heatmaps[q]
        above = (h > 0.5).sum()
        print(f"  {q:<12} {h.mean():>8.3f} {h.max():>8.3f} {above:>12,}")


def main():
    parser = argparse.ArgumentParser(description="CLIPSeg Pure Part Detection Test")
    parser.add_argument("--object", type=str, required=True,
                        help="Object name (mug, hammer, drill)")
    args = parser.parse_args()
    obj_name = args.object

    print(f"\n{'='*60}")
    print(f"  CLIPSeg Pure Part Detection — {obj_name}")
    print(f"{'='*60}\n")

    # Load object config
    obj_cfg = get_object_config(obj_name)
    parts = obj_cfg["parts"]
    queries = [part_name for part_name in parts.keys()]
    print(f"  Object: {obj_cfg['display_name']}")
    print(f"  Parts: {queries}")

    # Setup output directory
    diag_dir = DIAG_BASE / obj_name
    diag_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {diag_dir}")

    # Load image + depth
    print("\n[1/7] Loading RGB image + depth mask...")
    rgb = load_rgb()
    obj_mask = load_depth_mask()

    # Load model
    print("\n[2/7] Loading CLIPSeg model...")
    processor, model, device = load_clipseg()

    display = obj_cfg['display_name']

    # ── Mode 1: Pure (full scene) ──
    print(f"\n[3/7] Running CLIPSeg — PURE (full scene)...")
    heatmaps_pure = run_clipseg_pure(rgb, queries, processor, model, device)

    print(f"\n  Generating pure visualizations...")
    visualize_heatmaps(rgb, heatmaps_pure, display,
                       "Pure — no crop", diag_dir, "pure_heatmaps.png")
    visualize_threshold_sweep(rgb, heatmaps_pure, display,
                              "Pure", diag_dir, "pure_threshold_sweep.png")
    visualize_discrimination(rgb, heatmaps_pure, display,
                             "Pure", diag_dir, "pure_discrimination.png")

    # ── Mode 2: Cropped (depth zoom) ──
    print(f"\n[4/7] Running CLIPSeg — CROPPED (depth-based object crop)...")
    heatmaps_crop = run_clipseg_cropped(rgb, queries, obj_mask, processor, model, device)

    print(f"\n  Generating cropped visualizations...")
    visualize_heatmaps(rgb, heatmaps_crop, display,
                       "Cropped — depth-based object zoom", diag_dir, "cropped_heatmaps.png")
    visualize_threshold_sweep(rgb, heatmaps_crop, display,
                              "Cropped", diag_dir, "cropped_threshold_sweep.png")
    visualize_discrimination(rgb, heatmaps_crop, display,
                             "Cropped", diag_dir, "cropped_discrimination.png")

    # ── Mode 3: Pipeline (crop + depth mask + fallback thresholds) ──
    print(f"\n[5/7] Running CLIPSeg — PIPELINE (full post-processing)...")
    pipeline_masks = run_clipseg_pipeline(rgb, queries, obj_mask, processor, model, device)

    print(f"\n[6/7] Generating pipeline visualizations...")
    visualize_pipeline_masks(rgb, pipeline_masks, obj_mask, display, diag_dir)

    # ── Summary comparison ──
    print(f"\n[7/7] COMPARISON: Pure vs Cropped vs Pipeline")
    print(f"  {'Query':<12} {'Pure Max':>10} {'Crop Max':>10} {'Pure>0.5':>10} {'Crop>0.5':>10} {'Pipeline':>10}")
    print(f"  {'-'*64}")
    for q in queries:
        hp = heatmaps_pure[q]
        hc = heatmaps_crop[q]
        pm = pipeline_masks[q]
        print(f"  {q:<12} {hp.max():>10.3f} {hc.max():>10.3f} {int((hp>0.5).sum()):>10,} {int((hc>0.5).sum()):>10,} {int(pm.sum()):>10,}")

    print(f"\n{'='*60}")
    print(f"  DONE — Results in: {diag_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
