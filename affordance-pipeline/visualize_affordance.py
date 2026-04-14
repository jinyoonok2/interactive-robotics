"""
Visualize Affordance — Stage 2 of the Affordance Pipeline
==========================================================
Runs affordance detection + grasp planning using either UAD or CLIPSeg.
Each method outputs to its own directory so results can be compared.

Usage:
    cd habitat-lab

    # CLIPSeg (default — runs in habitat-grasp env directly):
    python ../affordance-pipeline/visualize_affordance.py --object hammer --part handle
    python ../affordance-pipeline/visualize_affordance.py --object mug --part handle --method clipseg

    # UAD (requires uad env subprocess bridge):
    python ../affordance-pipeline/visualize_affordance.py --object hammer --part handle --method uad

Output:
    results/{object}/clipseg/   — CLIPSeg affordance + grasp
    results/{object}/uad/       — UAD affordance + grasp
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import open3d as o3d

from core.grasp_planner import GraspPlanner
from objects import get_object_names, get_object, get_parts, validate_part


PIPELINE_DIR = Path(__file__).resolve().parent
INPUT_DIR    = PIPELINE_DIR / "output"
METHODS      = ["clipseg", "uad"]


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def make_detector(method: str, obj_name: str):
    """Factory: create the right detector for the chosen method."""
    if method == "clipseg":
        from core.clipseg_detector import CLIPSegDetector
        return CLIPSegDetector(obj_name=obj_name)
    elif method == "uad":
        from core.affordance_detector import AffordanceDetector
        return AffordanceDetector(obj_name=obj_name)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 2: Visualize affordance for a specific object part",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object hammer --part handle\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object mug --part handle --method clipseg\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object hammer --part head --method uad"
        ),
    )
    parser.add_argument("--object", required=True, choices=get_object_names(),
                        help="Object placed in the scene (from Stage 1)")
    parser.add_argument("--part", required=True,
                        help="Part to highlight (e.g., handle, body, rim)")
    parser.add_argument("--method", choices=METHODS, default="clipseg",
                        help="Affordance detection method (default: clipseg)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom text prompt (overrides objects.json query)")
    args = parser.parse_args()

    try:
        validate_part(args.object, args.part)
    except ValueError as e:
        parser.error(str(e))

    return args


def load_stage1(obj_name: str):
    """Load Stage 1 capture data. Returns (rgb, depth, metadata, points)."""
    meta_path = INPUT_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found. Run Stage 1 first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    captured_obj = metadata.get("object_name", "")
    if captured_obj != obj_name:
        print(f"  ERROR: Stage 1 captured '{captured_obj}' but you specified '{obj_name}'")
        sys.exit(1)

    rgb = np.array(Image.open(INPUT_DIR / "rgb.png"))
    depth = np.load(INPUT_DIR / "depth_raw.npy")

    ply_files = list(INPUT_DIR.glob(f"object_{obj_name}_sem*.ply"))
    if not ply_files:
        print(f"  ERROR: No point cloud found for '{obj_name}'")
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(str(ply_files[0]))
    points = np.asarray(pcd.points)

    print(f"  RGB: {rgb.shape}  Depth: {depth.shape}  Points: {len(points)}")
    return rgb, depth, metadata, points


def run_detection(detector, obj_name, part_name, rgb, depth, metadata, prompt=None):
    """Run affordance detection and return seg_result."""
    seg_result = detector.detect(obj_name, part_name, rgb, depth, metadata, query=prompt)
    if seg_result is None:
        print(f"\n  ERROR: Detection failed for '{part_name}'")
        sys.exit(1)
    return seg_result


def run_grasp_planning(obj_name, part_name, seg_result, metadata):
    """Plan grasp from detected affordance region."""
    planner = GraspPlanner()
    part_3d_points = seg_result["world_points"]
    part_indices = np.arange(len(part_3d_points))
    grasp = planner.plan(obj_name, part_name, part_3d_points, part_indices, metadata)

    print(f"  Type:       {grasp.grasp_type}")
    print(f"  Position:   ({grasp.position[0]:.4f}, {grasp.position[1]:.4f}, {grasp.position[2]:.4f})")
    print(f"  Approach:   ({grasp.approach_dir[0]:.3f}, {grasp.approach_dir[1]:.3f}, {grasp.approach_dir[2]:.3f})")
    print(f"  Confidence: {grasp.confidence:.0%}")
    print(f"  {grasp.description}")
    return grasp


def main():
    args = parse_args()
    obj_name  = args.object
    part_name = args.part
    method    = args.method

    print_header(f"Visualize Affordance — Stage 2 [{method.upper()}]")
    print(f"  Object: {obj_name}")
    print(f"  Part:   {part_name}")
    print(f"  Method: {method}")

    # Load Stage 1
    print_header("Loading Stage 1 data")
    rgb, depth, metadata, points = load_stage1(obj_name)

    # Detect affordance
    print_header(f"Affordance detection [{method}]")
    detector = make_detector(method, obj_name)
    seg_result = run_detection(detector, obj_name, part_name, rgb, depth, metadata, args.prompt)

    # Plan grasp
    print_header(f"Grasp planning for {part_name}")
    grasp = run_grasp_planning(obj_name, part_name, seg_result, metadata)

    # Visualize
    print_header("Rendering visualization")
    img_annotated = detector.visualize(
        rgb, seg_result, obj_name, part_name,
        grasp=grasp, metadata=metadata,
    )

    # Save
    detector.save_results(
        rgb, img_annotated, seg_result, grasp,
        obj_name, part_name, metadata,
    )

    # Summary
    print_header(f"STAGE 2 COMPLETE [{method.upper()}]")
    obj_cfg = get_object(obj_name)
    print(f"  Object: {obj_cfg['display_name']}")
    print(f"  Part:   {part_name}")
    print(f"  Method: {method}")
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
                  f"--object {obj_name} --part {p} --method {method}")


if __name__ == "__main__":
    main()
