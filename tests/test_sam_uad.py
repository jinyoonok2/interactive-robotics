#!/usr/bin/env python3
"""
SAM2 + UAD Integration Test
============================
Tests whether having SAM2 tightly crop the target object (keeping natural
background) helps UAD produce part-aware affordance heatmaps.

Hypothesis: UAD was trained on white-background studio renders where the
object fills the frame. In our Habitat scenes the object is small in the
full 512×512 image. SAM2 can give us a tight crop of just the object so
DINOv2 patches see more object detail → possibly better part-awareness.

Pipeline:
  1. Load existing RGB + depth from affordance-pipeline/output/
  2. Build depth-based mask → find object center for SAM prompt
  3. SAM2 segments the object → tight bounding box
  4. Crop RGB to SAM bbox (natural background kept) → upscale to 224×224
  5. Run UAD on (a) full image, (b) depth crop, (c) SAM crop
  6. Visualize and compare heatmaps

Runs entirely in the `uad` conda env (no Habitat needed).

Usage:
    conda activate uad
    cd ~/workspace/interactive-robotics
    python tests/test_sam_uad.py --object hammer --part handle
    python tests/test_sam_uad.py --object mug --part handle
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import time
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT  = Path(__file__).resolve().parent.parent
PIPELINE = PROJECT / "affordance-pipeline"
UAD_DIR  = PROJECT / "unsup-affordance"
UAD_SRC  = UAD_DIR / "src"
SAM2_DIR = PROJECT.parent / "sam2"   # /home/jinyoon/workspace/sam2

# Make sure we can import UAD
sys.path.insert(0, str(UAD_SRC))

# ── Configuration ──────────────────────────────────────────────────────
OBJECTS_JSON = PIPELINE / "config" / "objects.json"
SAM2_CHECKPOINT = SAM2_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"
UAD_CONFIG      = str(UAD_DIR / "configs" / "st_emb.yaml")
UAD_CHECKPOINT  = str(UAD_DIR / "checkpoints" / "st_emb.pth")


def header(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def load_objects_config():
    with open(OBJECTS_JSON) as f:
        return json.load(f)


# ── Depth-based object mask ───────────────────────────────────────────
def build_depth_mask(depth: np.ndarray, percentile=2, margin=1.3):
    """Mask the nearest object via depth percentile."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return None
    near = float(np.percentile(valid, percentile))
    far  = near * margin
    mask = (depth > 0) & (depth <= far)
    n = int(mask.sum())
    print(f"  [depth mask] near={near:.3f}m  cutoff={far:.3f}m  → {n} px")
    return mask


# ── SAM2 segmentation ─────────────────────────────────────────────────
def segment_with_sam2(rgb: np.ndarray, depth_mask: np.ndarray):
    """
    Use SAM2 with a point prompt at the depth-mask centroid to segment
    the target object. Returns (sam_mask, sam_bbox, sam_iou).
    """
    header("SAM2 Object Segmentation")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    import torch

    t0 = time.time()

    # Build SAM2 predictor
    print(f"  Loading SAM2 tiny from: {SAM2_CHECKPOINT}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_model = build_sam2(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)

    # Set image
    predictor.set_image(rgb)

    # Compute centroid of depth mask as the point prompt
    ys, xs = np.where(depth_mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    print(f"  Point prompt: ({cx}, {cy})  [center of depth mask]")

    # Predict with single foreground point
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,  # returns 3 masks
    )

    # Pick the best mask by IoU score
    best_idx = int(np.argmax(scores))
    sam_mask = masks[best_idx].astype(bool)
    sam_iou = float(scores[best_idx])

    # Compute bbox
    ys, xs = np.where(sam_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    sam_bbox = (y0, x0, y1, x1)

    n_px = int(sam_mask.sum())
    dt = time.time() - t0
    print(f"  SAM2 result: {n_px} px, IoU={sam_iou:.3f}, "
          f"bbox=({x0},{y0})→({x1},{y1}), took {dt:.1f}s")
    print(f"  All mask scores: {scores}")

    # Also return all 3 masks for debugging
    del predictor, sam2_model
    torch.cuda.empty_cache()

    return sam_mask, sam_bbox, sam_iou, masks, scores


# ── Cropping helpers ──────────────────────────────────────────────────
def crop_to_bbox(rgb: np.ndarray, mask: np.ndarray, pad_frac: float = 0.10,
                 target_size: int = 224):
    """Crop RGB to mask bounding box with padding, resize to target_size."""
    from PIL import Image as PILImage

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    H, W = rgb.shape[:2]

    # Add padding
    ph = max(1, int((y1 - y0) * pad_frac))
    pw = max(1, int((x1 - x0) * pad_frac))
    y0p = max(0, y0 - ph); y1p = min(H, y1 + ph)
    x0p = max(0, x0 - pw); x1p = min(W, x1 + pw)

    patch = rgb[y0p:y1p, x0p:x1p]
    crop_h, crop_w = patch.shape[:2]
    resized = np.array(PILImage.fromarray(patch).resize(
        (target_size, target_size), PILImage.BILINEAR))

    print(f"  Crop: ({x0p},{y0p})→({x1p},{y1p}) "
          f"patch={crop_w}×{crop_h}px → {target_size}×{target_size}")
    return resized, (y0p, x0p, y1p, x1p)


def unproject_heatmap(heatmap_crop: np.ndarray, bbox: tuple,
                      full_shape: tuple) -> np.ndarray:
    """Map cropped heatmap back to full image coordinates."""
    from PIL import Image as PILImage
    y0, x0, y1, x1 = bbox
    patch_h, patch_w = y1 - y0, x1 - x0
    hmap_patch = np.array(
        PILImage.fromarray((heatmap_crop * 255).astype(np.uint8))
            .resize((patch_w, patch_h), PILImage.BILINEAR)
    ).astype(np.float32) / 255.0
    full = np.zeros(full_shape[:2], dtype=np.float32)
    full[y0:y1, x0:x1] = hmap_patch
    return full


# ── UAD inference ─────────────────────────────────────────────────────
_uad_model = None

def get_uad_model():
    """Lazily initialize UAD model (singleton)."""
    global _uad_model
    if _uad_model is not None:
        return _uad_model

    # UAD expects to run from unsup-affordance/src/ (torch cache path is relative)
    import os
    old_cwd = os.getcwd()
    os.chdir(str(UAD_SRC))

    from inference import AffordanceInference
    from utils.file_utils import load_config
    from utils.vlm_utils import get_text_embedding_options

    cfg = load_config(UAD_CONFIG)
    text_embedding_option = cfg.get("text_embedding", "embeddings_oai")
    text_embedding_func = get_text_embedding_options(text_embedding_option)

    _uad_model = AffordanceInference(UAD_CONFIG, UAD_CHECKPOINT, text_embedding_func)

    os.chdir(old_cwd)
    return _uad_model


def run_uad(rgb: np.ndarray, text_query: str) -> np.ndarray:
    """Run UAD inference on an RGB image. Returns raw heatmap (0-1)."""
    model = get_uad_model()
    # predict with thresh=None → returns continuous heatmap
    heatmap = model.predict(rgb, text_query, thresh=None)
    return heatmap


def pixel_stats(heatmap: np.ndarray, mask: np.ndarray = None) -> str:
    """Summary stats for heatmap values in a region."""
    vals = heatmap[mask] if mask is not None else heatmap.ravel()
    if len(vals) == 0:
        return "(empty)"
    return (f"mean={vals.mean():.3f} std={vals.std():.3f} "
            f"max={vals.max():.3f} "
            f">0.5:{(vals>0.5).sum()} >0.7:{(vals>0.7).sum()} "
            f"/ {len(vals)} px")


# ── Visualization ────────────────────────────────────────────────────
def visualize(rgb, depth_mask, sam_mask, heatmaps, stats, obj_name, part_name,
              out_dir, sam_bbox, sam_iou):
    """Create comparison figure: 3 columns × 2 rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"SAM2 + UAD Test — {obj_name}/{part_name}", fontsize=14, y=0.98)

    # Row 0: Masks and crops
    # (0,0) RGB with depth mask overlay
    axes[0,0].imshow(rgb)
    if depth_mask is not None:
        overlay = np.zeros((*rgb.shape[:2], 4))
        overlay[depth_mask] = [0, 0, 1, 0.3]
        axes[0,0].imshow(overlay)
    axes[0,0].set_title("RGB + Depth Mask (blue)")
    axes[0,0].axis("off")

    # (0,1) RGB with SAM mask overlay
    axes[0,1].imshow(rgb)
    if sam_mask is not None:
        overlay = np.zeros((*rgb.shape[:2], 4))
        overlay[sam_mask] = [0, 1, 0, 0.35]
        axes[0,1].imshow(overlay)
        y0, x0, y1, x1 = sam_bbox
        rect = Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2,
                         edgecolor='lime', facecolor='none')
        axes[0,1].add_patch(rect)
    axes[0,1].set_title(f"SAM2 Mask (IoU={sam_iou:.3f})")
    axes[0,1].axis("off")

    # (0,2) Depth mask vs SAM mask comparison
    comparison = np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    if depth_mask is not None:
        comparison[depth_mask] = [0, 0, 255]  # blue = depth only
    if sam_mask is not None:
        both = depth_mask & sam_mask if depth_mask is not None else sam_mask
        sam_only = sam_mask & (~depth_mask if depth_mask is not None else np.ones_like(sam_mask, dtype=bool))
        comparison[sam_only] = [0, 255, 0]    # green = SAM only
        comparison[both] = [255, 255, 0]      # yellow = overlap
    axes[0,2].imshow(comparison)
    axes[0,2].set_title("Blue=depth  Green=SAM  Yellow=both")
    axes[0,2].axis("off")

    # (0,3) The SAM crop that gets fed to UAD
    if "sam_crop" in heatmaps:
        axes[0,3].imshow(heatmaps["sam_crop"])
        axes[0,3].set_title("SAM Crop → UAD input (224×224)")
    else:
        axes[0,3].axis("off")
    axes[0,3].axis("off")

    # Row 1: UAD heatmaps comparison
    heatmap_keys = ["full_image", "depth_crop", "sam_crop_heatmap"]
    titles = [
        "UAD on full 512×512",
        "UAD on depth crop (224×224)",
        "UAD on SAM crop (224×224)",
    ]
    for col, (key, title) in enumerate(zip(heatmap_keys, titles)):
        if key in heatmaps and heatmaps[key] is not None:
            axes[1,col].imshow(rgb)
            # Unproject heatmap if it has a bbox
            hm = heatmaps[key]
            axes[1,col].imshow(hm, cmap="hot", alpha=0.7, vmin=0, vmax=1)
            stat_key = key + "_stats"
            stat_str = stats.get(stat_key, "")
            axes[1,col].set_title(f"{title}\n{stat_str}", fontsize=9)
        else:
            axes[1,col].set_title(f"{title}\n(not available)")
        axes[1,col].axis("off")

    # (1,3) Difference: SAM_crop minus full_image (where SAM helps)
    if "sam_crop_heatmap" in heatmaps and "full_image" in heatmaps:
        diff = heatmaps["sam_crop_heatmap"] - heatmaps["full_image"]
        axes[1,3].imshow(rgb)
        axes[1,3].imshow(diff, cmap="RdBu_r", alpha=0.7, vmin=-0.5, vmax=0.5)
        axes[1,3].set_title("Diff: SAM_crop - full\n(red=SAM higher, blue=lower)")
    else:
        axes[1,3].set_title("Diff (not available)")
    axes[1,3].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"sam_uad_{obj_name}_{part_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM2 + UAD integration test")
    parser.add_argument("--object", default="hammer",
                        help="Object name (hammer, mug, pitcher, etc.)")
    parser.add_argument("--part", default="handle",
                        help="Part to query for")
    parser.add_argument("--uad-size", type=int, default=224,
                        help="Crop resize target for UAD")
    args = parser.parse_args()

    obj_name = args.object
    part_name = args.part

    # Output directory
    out_dir = PIPELINE / "results" / obj_name / "sam_uad_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──
    config = load_objects_config()
    if obj_name not in config:
        print(f"ERROR: {obj_name} not in objects.json")
        sys.exit(1)
    obj_cfg = config[obj_name]
    parts = obj_cfg.get("parts", {})
    if part_name not in parts:
        print(f"ERROR: {part_name} not in {obj_name} parts: {list(parts.keys())}")
        sys.exit(1)
    part_cfg = parts[part_name]
    query = part_cfg["query"].replace("{obj}", obj_name)
    print(f"\nObject: {obj_name}  Part: {part_name}")
    print(f"Query: \"{query}\"")

    # ── Load existing capture ──
    header("Loading Existing Capture")
    from PIL import Image as PILImage
    output_dir = PIPELINE / "output"
    rgb_path = output_dir / "rgb.png"
    depth_path = output_dir / "depth_raw.npy"

    if not rgb_path.exists() or not depth_path.exists():
        print(f"ERROR: No capture found in {output_dir}")
        print("Run test_integration.py first to capture a scene")
        sys.exit(1)

    rgb = np.array(PILImage.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path)
    with open(output_dir / "metadata.json") as f:
        metadata = json.load(f)

    print(f"  RGB: {rgb.shape}, Depth: {depth.shape}")
    spawned = metadata.get("spawned_objects", [{}])[0]
    captured_obj = spawned.get("name", spawned.get("object_name", ""))
    print(f"  Object: {captured_obj} at {spawned.get('position', '?')}")

    # Verify the right object is captured
    if obj_name.replace("_", "") not in captured_obj.replace("_", "").lower() and \
       captured_obj.replace("_", "").lower() not in obj_name.replace("_", ""):
        print(f"\n  WARNING: Captured object is '{captured_obj}', "
              f"but you requested '{obj_name}'")
        print(f"  You may need to re-run capture for {obj_name} first.")

    # ── Step 1: Depth mask ──
    header("Step 1: Depth Mask")
    depth_mask = build_depth_mask(depth)
    if depth_mask is None or depth_mask.sum() == 0:
        print("ERROR: No depth mask")
        sys.exit(1)

    # ── Step 2: SAM2 segmentation ──
    sam_mask, sam_bbox, sam_iou, all_masks, all_scores = \
        segment_with_sam2(rgb, depth_mask)

    # ── Step 3: Crop images ──
    header("Step 3: Prepare UAD Inputs")

    # 3a: Depth-based crop (what we had before)
    print("\n  [Depth crop]")
    depth_crop, depth_bbox = crop_to_bbox(rgb, depth_mask,
                                           pad_frac=0.10,
                                           target_size=args.uad_size)

    # 3b: SAM-based crop (new — tighter, follows actual object boundary)
    print("\n  [SAM crop]")
    sam_crop, sam_crop_bbox = crop_to_bbox(rgb, sam_mask,
                                            pad_frac=0.10,
                                            target_size=args.uad_size)

    # Save crops for inspection
    PILImage.fromarray(depth_crop).save(out_dir / "crop_depth.png")
    PILImage.fromarray(sam_crop).save(out_dir / "crop_sam.png")
    PILImage.fromarray(rgb).save(out_dir / "full_image.png")
    print(f"\n  Saved crops to {out_dir}")

    # ── Step 4: Run UAD on all three inputs ──
    header(f"Step 4: UAD Inference (query=\"{query}\")")

    print("\n  [A] UAD on full 512×512 image...")
    hm_full = run_uad(rgb, query)
    print(f"    Full image stats (on depth mask): {pixel_stats(hm_full, depth_mask)}")
    print(f"    Full image stats (on SAM mask):   {pixel_stats(hm_full, sam_mask)}")

    print("\n  [B] UAD on depth crop (224×224)...")
    hm_depth_crop_raw = run_uad(depth_crop, query)
    hm_depth_crop = unproject_heatmap(hm_depth_crop_raw, depth_bbox, rgb.shape)
    print(f"    Depth crop stats (on depth mask): {pixel_stats(hm_depth_crop, depth_mask)}")

    print("\n  [C] UAD on SAM crop (224×224)...")
    hm_sam_crop_raw = run_uad(sam_crop, query)
    hm_sam_crop = unproject_heatmap(hm_sam_crop_raw, sam_crop_bbox, rgb.shape)
    print(f"    SAM crop stats (on SAM mask):     {pixel_stats(hm_sam_crop, sam_mask)}")
    print(f"    SAM crop stats (on depth mask):   {pixel_stats(hm_sam_crop, depth_mask)}")

    # ── Step 5: Detailed comparison ──
    header("Step 5: Comparison")

    # Compare variance — higher variance = more differentiation between parts
    for label, hm, mask in [
        ("Full image",  hm_full,       depth_mask),
        ("Depth crop",  hm_depth_crop, depth_mask),
        ("SAM crop",    hm_sam_crop,   sam_mask),
    ]:
        vals = hm[mask]
        if len(vals) > 0:
            print(f"  {label:15s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"range=[{vals.min():.3f}, {vals.max():.3f}]  "
                  f"variance={vals.var():.4f}")

    # Save raw heatmaps
    np.save(out_dir / "heatmap_full.npy", hm_full)
    np.save(out_dir / "heatmap_depth_crop.npy", hm_depth_crop)
    np.save(out_dir / "heatmap_sam_crop.npy", hm_sam_crop)
    np.save(out_dir / "heatmap_sam_crop_raw.npy", hm_sam_crop_raw)

    # ── Step 6: Visualize ──
    header("Step 6: Visualization")
    heatmaps = {
        "full_image": hm_full,
        "depth_crop": hm_depth_crop,
        "sam_crop_heatmap": hm_sam_crop,
        "sam_crop": sam_crop,  # the actual crop image
    }
    stats = {
        "full_image_stats": pixel_stats(hm_full, depth_mask),
        "depth_crop_stats": pixel_stats(hm_depth_crop, depth_mask),
        "sam_crop_heatmap_stats": pixel_stats(hm_sam_crop, sam_mask),
    }
    visualize(rgb, depth_mask, sam_mask, heatmaps, stats,
              obj_name, part_name, out_dir, sam_bbox, sam_iou)

    print("\n" + "="*60)
    print("  DONE — Check results in:")
    print(f"  {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
