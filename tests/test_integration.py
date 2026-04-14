#!/usr/bin/env python3
"""
Integration Test: HSSD Scene Capture → UAD Affordance → Grasp Planning

Full end-to-end pipeline in the `habitat-grasp` conda environment:
  Stage 1: Habitat-Sim renders HSSD scene + spawns YCB object → RGB, Depth, Semantic
  Stage 2: AffordanceDetector runs UAD + semantic masking → object-only heatmap + 3D points
  Stage 3: GraspPlanner proposes a grasp on the affordance region → 6-DoF grasp pose
  Stage 4: Visualization of all stages

Usage:
    conda activate habitat-grasp
    python tests/test_integration.py                          # default: mug/handle
    python tests/test_integration.py --object mug --part handle
    python tests/test_integration.py --object hammer --part handle
    python tests/test_integration.py --skip-capture           # reuse previous capture
    python tests/test_integration.py --grasp-method geometric # or neural (GraspNet)
"""

from __future__ import annotations
import sys
import argparse
import json
import numpy as np
from pathlib import Path

# Setup paths
PROJECT = Path(__file__).resolve().parent.parent
PIPELINE = PROJECT / "affordance-pipeline"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(PIPELINE / "core"))


def parse_args():
    p = argparse.ArgumentParser(description="UAD + Habitat-Sim full pipeline test")
    p.add_argument("--object", default="mug",
                   help="Object name (mug, pitcher, hammer)")
    p.add_argument("--part", default="handle",
                   help="Part name (handle, body, rim, etc.)")
    p.add_argument("--scene", default="102344250", help="HSSD scene ID")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="UAD affordance threshold")
    p.add_argument("--skip-capture", action="store_true",
                   help="Reuse existing capture data")
    p.add_argument("--grasp-method", default="neural",
                   choices=["geometric", "neural"],
                   help="Grasp planning method (neural=GraspNet, geometric=PCA heuristic)")
    return p.parse_args()


def header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ════════════════════════════════════════════════════════════════
# Stage 1: Scene Capture
# ════════════════════════════════════════════════════════════════

def stage1_capture(obj_name: str, scene_id: str):
    """Render HSSD scene with a YCB object using Habitat-Sim."""
    header("Stage 1: HSSD Scene Capture")

    from scene_capture import SceneCapture
    capture = SceneCapture(scene_id=scene_id)

    try:
        capture.setup()
        agent_pos, obj_info = capture.spawn_object(obj_name)
        rgb, depth, semantic = capture.capture()
        capture.save(rgb, depth, semantic, obj_info)
        metadata = capture.get_metadata()
        header("Stage 1 Complete")
        return rgb, depth, metadata
    finally:
        capture.close()


# ════════════════════════════════════════════════════════════════
# Stage 2: Affordance Detection (UAD + semantic masking)
# ════════════════════════════════════════════════════════════════

def stage2_affordance(rgb, depth, metadata, obj_name, part_name, threshold):
    """
    Run AffordanceDetector.detect() which:
      1. Calls UAD via subprocess bridge → raw heatmap
      2. Thresholds the heatmap → binary mask
      3. Intersects with semantic mask → only target object pixels survive
      4. Lifts masked pixels to 3D world coordinates
    """
    header("Stage 2: Affordance Detection (UAD + Semantic Mask)")

    from affordance_detector import AffordanceDetector
    detector = AffordanceDetector(threshold=threshold, obj_name=obj_name)

    query = detector.build_query(obj_name, part_name)
    print(f"  Object:    {obj_name}")
    print(f"  Part:      {part_name}")
    print(f"  Query:     \"{query}\"")
    print(f"  Threshold: {threshold}")

    result = detector.detect(obj_name, part_name, rgb, depth, metadata)

    if result is None:
        print("  ERROR: Affordance detection returned no result")
        return None, detector

    print(f"\n  --- Detection Result ---")
    print(f"  Mask pixels:      {result['mask_pixels']}")
    print(f"  3D world points:  {result['num_3d_points']}")
    print(f"  Confidence:       {result['detection_confidence']:.4f}")
    print(f"  Heatmap range:    [{result['raw_heatmap'].min():.4f}, "
          f"{result['raw_heatmap'].max():.4f}]")

    header("Stage 2 Complete")
    return result, detector


# ════════════════════════════════════════════════════════════════
# Stage 3: Grasp Planning
# ════════════════════════════════════════════════════════════════

def stage3_grasp(seg_result, metadata, obj_name, part_name, method):
    """
    Plan a grasp on the detected affordance region.
      - geometric: PCA-based heuristic (fast, no GPU needed)
      - neural: GraspNet inference (slower, needs checkpoint)
    """
    header(f"Stage 3: Grasp Planning ({method})")

    from grasp_planner import GraspPlanner
    planner = GraspPlanner()

    world_points = seg_result["world_points"]
    part_indices = np.arange(len(world_points))  # all points are part points

    print(f"  Method:      {method}")
    print(f"  Part points: {len(world_points)}")

    grasp = planner.plan(
        obj_name, part_name,
        world_points, part_indices,
        metadata, method=method,
    )

    print(f"\n  --- Grasp Result ---")
    print(f"  Position:    [{grasp.position[0]:.4f}, {grasp.position[1]:.4f}, "
          f"{grasp.position[2]:.4f}]")
    print(f"  Approach:    [{grasp.approach_dir[0]:.4f}, {grasp.approach_dir[1]:.4f}, "
          f"{grasp.approach_dir[2]:.4f}]")
    print(f"  Grasp type:  {grasp.grasp_type}")
    print(f"  Confidence:  {grasp.confidence:.2%}")
    print(f"  Description: {grasp.description}")

    header("Stage 3 Complete")
    return grasp


# ════════════════════════════════════════════════════════════════
# Stage 4: Visualization
# ════════════════════════════════════════════════════════════════

def stage4_visualize(rgb, depth, metadata, seg_result, grasp, detector,
                     obj_name, part_name, threshold):
    """Generate multi-panel visualizations showing all pipeline stages."""
    header("Stage 4: Visualization")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = PIPELINE / "results" / obj_name / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_heatmap = seg_result["raw_heatmap"]
    mask = seg_result["mask"]
    n_mask = int(mask.sum())
    n_3d = seg_result["num_3d_points"]

    # Load semantic mask for comparison
    sem_path = PIPELINE / "output" / "semantic_raw.npy"
    semantic_raw = np.load(sem_path) if sem_path.exists() else None
    sem_id = metadata["spawned_objects"][0].get("semantic_id") if metadata.get("spawned_objects") else None
    obj_mask = (semantic_raw == sem_id) if (semantic_raw is not None and sem_id is not None) else None

    # ── 6-panel figure ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # (0,0) RGB scene
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB: {obj_name} in HSSD scene {metadata['scene_id']}", fontsize=13)
    axes[0, 0].axis("off")

    # (0,1) Raw UAD heatmap (before semantic masking)
    axes[0, 1].imshow(rgb)
    axes[0, 1].imshow(raw_heatmap, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[0, 1].set_title(f'Raw UAD Heatmap: "{seg_result["text_query"]}"', fontsize=11)
    axes[0, 1].axis("off")

    # (0,2) Semantic object mask from Habitat-Sim
    if obj_mask is not None:
        sem_vis = rgb.copy()
        sem_vis[obj_mask] = (
            sem_vis[obj_mask].astype(float) * 0.4 +
            np.array([100, 180, 255], dtype=float) * 0.6
        ).astype(np.uint8)
        axes[0, 2].imshow(sem_vis)
        axes[0, 2].set_title(f"Semantic Mask: {obj_name} (id={sem_id})", fontsize=13)
    else:
        axes[0, 2].imshow(rgb)
        axes[0, 2].set_title("Semantic mask unavailable", fontsize=13)
    axes[0, 2].axis("off")

    # (1,0) Heatmap masked to object only
    masked_heatmap = raw_heatmap.copy()
    if obj_mask is not None:
        masked_heatmap[~obj_mask] = 0
    axes[1, 0].imshow(rgb)
    axes[1, 0].imshow(masked_heatmap, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[1, 0].set_title(f"Heatmap (object-only, semantic masked)", fontsize=12)
    axes[1, 0].axis("off")

    # (1,1) Final binary affordance mask
    overlay = rgb.copy()
    overlay[mask] = (
        overlay[mask].astype(float) * 0.4 +
        np.array([100, 255, 150], dtype=float) * 0.6
    ).astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        f"Affordance: {part_name} (>{threshold})\n"
        f"{n_mask} px → {n_3d} 3D pts",
        fontsize=12)
    axes[1, 1].axis("off")

    # (1,2) Grasp visualization
    img_grasp = detector.visualize(rgb, seg_result, obj_name, part_name,
                                   grasp=grasp, metadata=metadata)
    axes[1, 2].imshow(img_grasp)
    axes[1, 2].set_title(
        f"Grasp: {grasp.grasp_type} ({grasp.confidence:.0%})\n"
        f"pos=[{grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f}]",
        fontsize=11)
    axes[1, 2].axis("off")

    fig.suptitle(
        f"Full Pipeline: {obj_name}/{part_name}  |  "
        f"Scene → UAD → Semantic Mask → Grasp",
        fontsize=16, fontweight="bold")
    plt.tight_layout()

    pipeline_path = out_dir / f"pipeline_{obj_name}_{part_name}.png"
    plt.savefig(pipeline_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {pipeline_path}")

    # ── Save annotated image + grasp data ──
    detector.save_results(rgb, img_grasp, seg_result, grasp,
                          obj_name, part_name, metadata)

    # Save raw heatmap
    np.save(out_dir / f"heatmap_{obj_name}_{part_name}.npy", raw_heatmap)
    print(f"  Saved: heatmap_{obj_name}_{part_name}.npy")

    header("Stage 4 Complete")
    return out_dir


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    obj_name = args.object
    part_name = args.part

    header(f"Integration Test: {obj_name}/{part_name}")
    print(f"  Scene:        {args.scene}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Grasp method: {args.grasp_method}")
    print(f"  Skip capture: {args.skip_capture}")

    # Stage 1: Capture
    output_dir = PIPELINE / "output"
    if args.skip_capture and (output_dir / "rgb.png").exists():
        header("Stage 1: Reusing existing capture")
        from PIL import Image as PILImage
        rgb = np.array(PILImage.open(output_dir / "rgb.png"))
        depth = np.load(output_dir / "depth_raw.npy")
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)
        print(f"  RGB: {rgb.shape}, Depth: {depth.shape}")
    else:
        rgb_raw, depth, metadata = stage1_capture(obj_name, args.scene)
        from PIL import Image as PILImage
        rgb = np.array(PILImage.open(output_dir / "rgb.png"))
        depth = np.load(output_dir / "depth_raw.npy")

    # Stage 2: Affordance (UAD + semantic masking)
    seg_result, detector = stage2_affordance(
        rgb, depth, metadata, obj_name, part_name, args.threshold)

    if seg_result is None:
        print("\n  PIPELINE FAILED: No affordance detected")
        return

    # Stage 3: Grasp planning
    grasp = stage3_grasp(seg_result, metadata, obj_name, part_name, args.grasp_method)

    # Stage 4: Visualization
    out_dir = stage4_visualize(
        rgb, depth, metadata, seg_result, grasp, detector,
        obj_name, part_name, args.threshold)

    # Summary
    header("INTEGRATION TEST COMPLETE")
    print(f"  Object:       {obj_name}")
    print(f"  Part:         {part_name}")
    print(f"  Threshold:    {args.threshold}")
    print(f"  Grasp method: {args.grasp_method}")
    print(f"  Grasp type:   {grasp.grasp_type}")
    print(f"  Grasp conf:   {grasp.confidence:.2%}")
    if out_dir:
        print(f"  Output:       {out_dir}/")
        for p in sorted(out_dir.iterdir()):
            if p.is_file():
                sz = p.stat().st_size / 1024
                print(f"    {p.name:45s} ({sz:.1f} KB)")
    # Also list results/{obj_name}/affordance/ from detector.save_results
    uad_dir = PIPELINE / "results" / obj_name / "affordance"
    if uad_dir.exists():
        print(f"  Detector:     {uad_dir}/")
        for p in sorted(uad_dir.iterdir()):
            if p.is_file():
                sz = p.stat().st_size / 1024
                print(f"    {p.name:45s} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()
