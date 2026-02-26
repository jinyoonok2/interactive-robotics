"""
Visualize Affordance — Stage 2 of the Affordance Pipeline
==========================================================
Entry point for Stage 2. Uses AffordanceDetector + GraspPlanner classes.

Usage:
    cd habitat-lab
    python ../affordance-pipeline/visualize_affordance.py --object mug --part handle
    python ../affordance-pipeline/visualize_affordance.py --object power_drill --part chuck
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import open3d as o3d

from core.affordance_detector import AffordanceDetector
from core.grasp_planner import GraspPlanner
from objects import get_object_names, get_object, get_parts, validate_part, print_object_parts


PIPELINE_DIR = Path(__file__).resolve().parent
INPUT_DIR    = PIPELINE_DIR / "output"


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 2: Visualize affordance for a specific object part",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object mug --part handle\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object hammer --part head"
        ),
    )
    parser.add_argument("--object", required=True, choices=get_object_names(),
                        help="Object placed in the scene (from Stage 1)")
    parser.add_argument("--part", required=True,
                        help="Part to highlight (e.g., handle, body, rim)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom text prompt for DINO detection")
    parser.add_argument("--box-threshold", type=float, default=0.25,
                        help="Grounding DINO detection confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO text matching threshold")
    args = parser.parse_args()

    try:
        validate_part(args.object, args.part)
    except ValueError as e:
        parser.error(str(e))

    return args


def main():
    args = parse_args()
    obj_name  = args.object
    part_name = args.part

    print_header("Visualize Affordance — Stage 2")
    print(f"  Object: {obj_name}")
    print(f"  Part:   {part_name}")

    # Load Stage 1 data
    print_header("Loading Stage 1 data")
    meta_path = INPUT_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found. Run Stage 1 first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    captured_obj = metadata.get("object_name", "")
    if captured_obj != obj_name:
        print(f"  WARNING: Stage 1 captured '{captured_obj}' but you specified '{obj_name}'")
        sys.exit(1)

    rgb = np.array(Image.open(INPUT_DIR / "rgb.png"))
    depth = np.load(INPUT_DIR / "depth_raw.npy")
    height, width = rgb.shape[:2]
    print(f"  RGB: {rgb.shape}")
    print(f"  Depth: {depth.shape}")

    # Load point cloud
    ply_files = list(INPUT_DIR.glob(f"object_{obj_name}_sem*.ply"))
    if not ply_files:
        print(f"  ERROR: No point cloud found for '{obj_name}'")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(str(ply_files[0]))
    points = np.asarray(pcd.points)
    print(f"  Point cloud: {len(points)} points")

    # Detect affordance
    detector = AffordanceDetector(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    print_header(f"Language-guided segmentation")
    seg_result = detector.detect(obj_name, part_name, rgb, depth, metadata, prompt=args.prompt)

    if seg_result is None:
        print(f"\n  ERROR: Segmentation failed for '{part_name}'")
        sys.exit(1)

    # Plan grasp
    print_header(f"Proposing grasp for {part_name}")
    planner = GraspPlanner()

    part_3d_points = seg_result["world_points"]
    part_indices = np.arange(len(part_3d_points))

    grasp = planner.plan(obj_name, part_name, part_3d_points, part_indices, metadata)

    print(f"  Type:       {grasp.grasp_type}")
    print(f"  Position:   ({grasp.position[0]:.4f}, {grasp.position[1]:.4f}, {grasp.position[2]:.4f})")
    print(f"  Approach:   ({grasp.approach_dir[0]:.3f}, {grasp.approach_dir[1]:.3f}, {grasp.approach_dir[2]:.3f})")
    print(f"  Confidence: {grasp.confidence:.0%}")
    print(f"  {grasp.description}")

    # Visualize
    print_header("Rendering visualization")
    img_annotated = detector.visualize(
        rgb, seg_result, obj_name, part_name,
        grasp=grasp, metadata=metadata,
    )

    # Save
    text_prompt = args.prompt or f"{part_name} of the {obj_name}"
    detector.save_results(
        rgb, img_annotated, seg_result, grasp,
        obj_name, part_name, metadata, text_prompt,
    )

    # Summary
    print_header("STAGE 2 COMPLETE")
    obj_cfg = get_object(obj_name)
    print(f"  Object: {obj_cfg['display_name']}")
    print(f"  Part:   {part_name} ({obj_cfg['parts'][part_name]})")
    print(f"  Grasp:  {grasp.grasp_type} @ confidence {grasp.confidence:.0%}")
    print(f"  Output: {detector.results_dir}/")

    for p in sorted(detector.results_dir.iterdir()):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name:45s} ({size_kb:.1f} KB)")

    other_parts = [p for p in get_parts(obj_name) if p != part_name]
    if other_parts:
        print(f"\n  Try other parts:")
        for p in other_parts:
            print(f"    python ../affordance-pipeline/visualize_affordance.py "
                  f"--object {obj_name} --part {p}")


if __name__ == "__main__":
    main()
