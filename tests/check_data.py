#!/usr/bin/env python3
"""
Check that all required datasets, checkpoints, and model files exist.
Run from any environment — only checks file/directory presence.

Usage:
    python tests/check_data.py
"""

import os
import sys

# ── Formatting helpers ──────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []

# Project root: one level up from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name: str, passed: bool, detail: str = "", warn: bool = False):
    tag = PASS if passed else (WARN if warn else FAIL)
    results.append((name, passed, warn))
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def check_path(label: str, rel_path: str, is_file: bool = False, required: bool = True,
               min_size_mb: float = 0):
    """Check if a path exists under PROJECT_ROOT."""
    full = os.path.join(PROJECT_ROOT, rel_path)
    exists = os.path.isfile(full) if is_file else os.path.isdir(full)

    detail = ""
    if exists and is_file and min_size_mb > 0:
        size_mb = os.path.getsize(full) / (1024 * 1024)
        detail = f"{size_mb:.1f} MB"
        if size_mb < min_size_mb:
            check(label, False, f"{detail} (expected >= {min_size_mb} MB)")
            return

    if exists:
        # For directories, show rough count
        if not is_file:
            try:
                count = sum(1 for _ in os.scandir(full))
                detail = f"{count} entries"
            except PermissionError:
                detail = "exists"

    if exists:
        check(label, True, detail)
    else:
        check(label, False, f"missing: {rel_path}", warn=not required)


# ── Checks ──────────────────────────────────────────────────────────────────

def check_repos():
    """Verify cloned repositories exist."""
    print(f"\n  {BOLD}Cloned repositories:{RESET}")
    check_path("habitat-lab", "habitat-lab")
    check_path("graspnet-baseline", "graspnet-baseline")
    check_path("unsup-affordance", "unsup-affordance")
    check_path("affordance-pipeline", "affordance-pipeline")


def check_habitat_datasets():
    """Verify Habitat datasets are downloaded."""
    print(f"\n  {BOLD}Habitat datasets (habitat-lab/data/):{RESET}")
    check_path("HSSD scenes", "habitat-lab/data/scene_datasets/hssd-hab")
    check_path("hab3-episodes", "habitat-lab/data/datasets/hssd")
    check_path("Humanoid models", "habitat-lab/data/humanoids")
    check_path("Spot robot", "habitat-lab/data/robots/hab_spot_arm")
    check_path("Fetch robot", "habitat-lab/data/robots/hab_fetch")
    check_path("YCB objects", "habitat-lab/data/objects/ycb")


def check_graspnet_checkpoint():
    """GraspNet checkpoint file."""
    print(f"\n  {BOLD}GraspNet checkpoint:{RESET}")
    check_path("checkpoint-rs.tar", "graspnet-baseline/checkpoints/checkpoint-rs.tar",
               is_file=True, min_size_mb=5)


def check_uad_checkpoints():
    """UAD model checkpoints."""
    print(f"\n  {BOLD}UAD checkpoints (unsup-affordance/checkpoints/):{RESET}")
    check_path("st_emb.pth", "unsup-affordance/checkpoints/st_emb.pth", is_file=True)
    check_path("oai_emb.pth", "unsup-affordance/checkpoints/oai_emb.pth", is_file=True)
    check_path("eval_agd.pth", "unsup-affordance/checkpoints/eval_agd.pth", is_file=True)


def check_graspnet_extensions():
    """GraspNet CUDA extension source dirs (build artifacts may vary)."""
    print(f"\n  {BOLD}GraspNet CUDA extension sources:{RESET}")
    check_path("knn/ source", "graspnet-baseline/knn")
    check_path("pointnet2/ source", "graspnet-baseline/pointnet2")
    # Check for build artifacts
    knn_build = os.path.join(PROJECT_ROOT, "graspnet-baseline", "knn", "build")
    pn2_build = os.path.join(PROJECT_ROOT, "graspnet-baseline", "pointnet2", "build")
    check("knn built", os.path.isdir(knn_build),
          "build/ exists" if os.path.isdir(knn_build) else "not built yet — run setup.py install",
          warn=not os.path.isdir(knn_build))
    check("pointnet2 built", os.path.isdir(pn2_build),
          "build/ exists" if os.path.isdir(pn2_build) else "not built yet — run setup.py install",
          warn=not os.path.isdir(pn2_build))


def check_pipeline_config():
    """Affordance pipeline config files."""
    print(f"\n  {BOLD}Pipeline config:{RESET}")
    check_path("objects.json", "affordance-pipeline/config/objects.json", is_file=True)
    check_path("prompts.json", "affordance-pipeline/config/prompts.json", is_file=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}═══ Data & Assets Check ═══{RESET}\n")
    print(f"  Project root: {PROJECT_ROOT}\n")

    check_repos()
    check_habitat_datasets()
    check_graspnet_checkpoint()
    check_uad_checkpoints()
    check_graspnet_extensions()
    check_pipeline_config()

    # Summary
    total = len(results)
    passed = sum(1 for _, p, w in results if p)
    failed = sum(1 for _, p, w in results if not p and not w)
    warned = sum(1 for _, p, w in results if not p and w)

    print(f"\n{BOLD}── Summary: {passed}/{total} passed", end="")
    if warned:
        print(f", {warned} warnings", end="")
    if failed:
        print(f", {failed} FAILED", end="")
    print(f" ──{RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    rc = main()
    result_file = os.environ.get("CHECK_RESULT_FILE")
    if result_file:
        with open(result_file, "w") as f:
            f.write(str(rc))
    sys.exit(rc)
