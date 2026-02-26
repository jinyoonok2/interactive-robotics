"""
Scene Capture — Stage 1 of the Affordance Pipeline
====================================================
Entry point for Stage 1. Uses the modular SceneCapture class.

Usage:
    cd habitat-lab
    python ../affordance-pipeline/scene_capture.py --object mug
    python ../affordance-pipeline/scene_capture.py --object power_drill
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.scene_capture import SceneCapture
from objects import get_object_names, get_object, print_object_parts


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 1: Capture scene with a single object",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python ../affordance-pipeline/scene_capture.py --object mug",
    )
    parser.add_argument("--object", required=True, choices=get_object_names(),
                        help="Object to place in the scene")
    return parser.parse_args()


def main():
    args = parse_args()
    obj_name = args.object
    obj_cfg = get_object(obj_name)

    print_header("Scene Capture — Stage 1")
    print(f"  Object: {obj_name}")

    capture = SceneCapture()
    print(f"  Output: {capture.output_dir}")

    try:
        capture.setup()
        agent_pos, obj_info = capture.spawn_object(obj_name)
        rgb, depth, semantic = capture.capture()
        capture.save(rgb, depth, semantic, obj_info)

        print_header("STAGE 1 COMPLETE")
        print(f"  Object captured: {obj_cfg['display_name']}")
        print(f"  Output:          {capture.output_dir}/")
        for p in sorted(capture.output_dir.iterdir()):
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                print(f"    {p.name:40s} ({size_kb:.1f} KB)")

        print_object_parts(obj_name)
        parts = list(obj_cfg["parts"].keys())
        print(f"\n  Next step:")
        print(f"    python ../affordance-pipeline/visualize_affordance.py "
              f"--object {obj_name} --part {parts[0]}")
    finally:
        capture.close()


if __name__ == "__main__":
    main()
