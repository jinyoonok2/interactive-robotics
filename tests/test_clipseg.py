#!/usr/bin/env python3
"""CLIPSeg affordance visualization — shows per-part heatmaps on objects.

Usage:
    cd interactive-robotics
    conda run -n uad python tests/test_clipseg.py --object hammer
    conda run -n uad python tests/test_clipseg.py --object mug
    conda run -n uad python tests/test_clipseg.py --object hammer --queries "handle" "head" "grip"
"""
import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PIPELINE_DIR = Path("affordance-pipeline")
INPUT_DIR = PIPELINE_DIR / "output"


def load_model():
    """Load CLIPSeg model onto GPU."""
    print("Loading CLIPSeg model...")
    t0 = time.time()
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}  Loaded in {time.time()-t0:.1f}s")
    return processor, model, device


def clipseg_heatmap(processor, model, device, img_pil, query):
    """Run CLIPSeg and return probability heatmap at original image resolution."""
    inputs = processor(text=[query], images=[img_pil], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits).cpu().numpy()
    # Resize to original image size
    w, h = img_pil.size
    prob_resized = np.array(Image.fromarray(probs).resize((w, h), Image.BILINEAR))
    return prob_resized


def get_object_mask(depth_path):
    """Build depth-based object mask (nearest object in scene)."""
    dep = np.load(depth_path)
    valid = dep[dep > 0]
    if len(valid) == 0:
        return None
    near = float(np.percentile(valid, 2))
    cutoff = near * 1.3
    return (dep > 0) & (dep <= cutoff)


def crop_to_object(img_np, obj_mask, pad_frac=0.15):
    """Crop image to object bounding box with padding. Returns crop, (y0,x0,y1,x1)."""
    if obj_mask is None or obj_mask.sum() == 0:
        return img_np, (0, 0, img_np.shape[0], img_np.shape[1])
    ys, xs = np.where(obj_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    H, W = img_np.shape[:2]
    pad_y = max(1, int((y1 - y0) * pad_frac))
    pad_x = max(1, int((x1 - x0) * pad_frac))
    y0c, y1c = max(0, y0 - pad_y), min(H, y1 + pad_y)
    x0c, x1c = max(0, x0 - pad_x), min(W, x1 + pad_x)
    return img_np[y0c:y1c, x0c:x1c], (y0c, x0c, y1c, x1c)


def main():
    parser = argparse.ArgumentParser(description="CLIPSeg affordance visualization")
    parser.add_argument("--object", required=True, help="Object name (must match objects.json)")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom queries (default: use objects.json part queries)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Heatmap overlay threshold for binary mask")
    args = parser.parse_args()

    obj_name = args.object

    # Load object config
    cfg_path = PIPELINE_DIR / "config" / "objects.json"
    with open(cfg_path) as f:
        objects_cfg = json.load(f)
    if obj_name not in objects_cfg:
        print(f"ERROR: '{obj_name}' not in {cfg_path}. Available: {[k for k in objects_cfg if not k.startswith('_')]}")
        return
    obj_cfg = objects_cfg[obj_name]

    # Build query list: either custom or from objects.json parts
    if args.queries:
        queries = args.queries
        labels = args.queries
    else:
        queries = []
        labels = []
        for part_name, part_cfg in obj_cfg["parts"].items():
            q = part_cfg.get("query", f"{part_name} of the {obj_name}")
            q = q.replace("{obj}", obj_name)
            queries.append(q)
            labels.append(f"{part_name}: {q}")
            # Also add a simple part-name-only query for comparison
            queries.append(part_name)
            labels.append(f"{part_name} (simple)")

    # Load image and depth mask
    rgb_path = INPUT_DIR / "rgb.png"
    if not rgb_path.exists():
        print(f"ERROR: {rgb_path} not found. Run scene_capture.py first.")
        return
    img = Image.open(rgb_path).convert("RGB")
    img_np = np.array(img)
    print(f"Image: {img.size[0]}x{img.size[1]}")

    # Depth mask
    depth_path = INPUT_DIR / "depth_raw.npy"
    obj_mask = get_object_mask(depth_path) if depth_path.exists() else None
    if obj_mask is not None:
        print(f"Object mask: {int(obj_mask.sum())} pixels")

    # Crop to object region
    crop_np, bbox = crop_to_object(img_np, obj_mask)
    crop_pil = Image.fromarray(crop_np)
    y0c, x0c, y1c, x1c = bbox
    print(f"Object crop: ({x0c},{y0c})→({x1c},{y1c})  {crop_np.shape[1]}x{crop_np.shape[0]}px")

    # Load model
    processor, model, device = load_model()

    # Run all queries
    n_queries = len(queries)
    # Layout: top row = full scene heatmaps, bottom row = object crop heatmaps
    n_cols = max(n_queries + 1, 3)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    # Row 0: full-scene view
    axes[0, 0].imshow(img_np)
    if obj_mask is not None:
        # Draw object bbox
        import matplotlib.patches as patches
        rect = patches.Rectangle((x0c, y0c), x1c - x0c, y1c - y0c,
                                  linewidth=2, edgecolor='lime', facecolor='none')
        axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f"Scene ({obj_name})", fontsize=11)
    axes[0, 0].axis('off')

    # Row 1: cropped object view
    axes[1, 0].imshow(crop_np)
    axes[1, 0].set_title("Object crop", fontsize=11)
    axes[1, 0].axis('off')

    print(f"\n{'#':<3} {'Query':<50} {'Mean':>6} {'Max':>6} {'Cov>0.3':>8} {'ObjCov':>8}")
    print("-" * 90)

    for i, (query, label) in enumerate(zip(queries, labels)):
        col = i + 1

        # Full-scene heatmap
        heatmap_full = clipseg_heatmap(processor, model, device, img, query)

        # Crop heatmap — run CLIPSeg on the cropped object image for better resolution
        heatmap_crop = clipseg_heatmap(processor, model, device, crop_pil, query)

        # Stats (within object mask)
        if obj_mask is not None:
            obj_vals = heatmap_full[obj_mask]
            mean_val = obj_vals.mean()
            max_val = obj_vals.max()
            cov = (obj_vals > args.threshold).sum() / len(obj_vals) * 100
            obj_cov_str = f"{cov:.1f}%"
        else:
            mean_val = heatmap_full.mean()
            max_val = heatmap_full.max()
            obj_cov_str = "n/a"
        full_cov = (heatmap_full > args.threshold).sum() / heatmap_full.size * 100

        print(f"{col:<3} {label:<50} {mean_val:>6.3f} {max_val:>6.3f} {full_cov:>7.1f}% {obj_cov_str:>8}")

        # Row 0: full scene with heatmap overlay
        if col < n_cols:
            axes[0, col].imshow(img_np)
            im = axes[0, col].imshow(heatmap_full, alpha=0.6, cmap='jet', vmin=0, vmax=1)
            axes[0, col].set_title(f'"{label}"', fontsize=9, wrap=True)
            axes[0, col].axis('off')

        # Row 1: cropped object with heatmap overlay
        if col < n_cols:
            axes[1, col].imshow(crop_np)
            axes[1, col].imshow(heatmap_crop, alpha=0.6, cmap='jet', vmin=0, vmax=1)
            # Also draw threshold contour
            from matplotlib.colors import Normalize
            binary = (heatmap_crop > args.threshold).astype(np.uint8)
            if binary.sum() > 0:
                axes[1, col].contour(binary, levels=[0.5], colors='white', linewidths=1.5)
            axes[1, col].set_title(f'crop: "{label}"', fontsize=9, wrap=True)
            axes[1, col].axis('off')

    # Hide unused axes
    for row in range(2):
        for col in range(n_queries + 1, n_cols):
            axes[row, col].axis('off')

    plt.suptitle(f"CLIPSeg Affordance: {obj_cfg.get('display_name', obj_name)}", fontsize=14, y=1.01)
    plt.tight_layout()

    out_dir = PIPELINE_DIR / "results" / obj_name / "clipseg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clipseg_parts_{obj_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")

    # Also save individual per-part heatmaps (crop only)
    for i, (query, label) in enumerate(zip(queries, labels)):
        heatmap_crop = clipseg_heatmap(processor, model, device, crop_pil, query)
        safe_name = label.split(":")[0].strip().replace(" ", "_")

        # Heatmap overlay
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        ax2.imshow(crop_np)
        ax2.imshow(heatmap_crop, alpha=0.65, cmap='jet', vmin=0, vmax=1)
        binary = (heatmap_crop > args.threshold).astype(np.uint8)
        if binary.sum() > 0:
            ax2.contour(binary, levels=[0.5], colors='white', linewidths=2)
        ax2.set_title(f'{label}\nmax={heatmap_crop.max():.3f}', fontsize=12)
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / f"{safe_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Individual heatmaps saved to: {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
