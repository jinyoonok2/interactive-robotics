#!/usr/bin/env python3
"""
Verify the `uad` conda environment has all required packages and correct versions.

Usage:
    conda activate uad
    python tests/check_uad.py
"""

import sys
import os

# ── Formatting helpers ──────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

results = []


def check(name: str, passed: bool, detail: str = "", warn: bool = False):
    tag = PASS if passed else (WARN if warn else FAIL)
    results.append((name, passed, warn))
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ── Environment check ──────────────────────────────────────────────────────

def check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    check("Conda env = uad", env == "uad",
          f"current: '{env}'" if env != "uad" else "")


def check_python_version():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    check("Python >= 3.10", ok, f"{v.major}.{v.minor}.{v.micro}")


# ── Package imports ─────────────────────────────────────────────────────────

def try_import(module_name: str, display_name: str = None, version_attr: str = "__version__",
               expected_version: str = None):
    display = display_name or module_name
    try:
        mod = __import__(module_name)
        ver = getattr(mod, version_attr, None) if version_attr else None
        detail = f"v{ver}" if ver else "OK"

        if expected_version and ver:
            if not ver.startswith(expected_version):
                check(display, False, f"v{ver} (expected {expected_version})")
                return mod
        check(display, True, detail)
        return mod
    except ImportError as e:
        check(display, False, f"ImportError: {e}")
        return None
    except Exception as e:
        check(display, False, f"Error: {e}")
        return None


def check_core_packages():
    print(f"\n  {BOLD}Core packages:{RESET}")
    torch = try_import("torch", "PyTorch", expected_version="2.")
    try_import("numpy", "numpy", expected_version="2.")
    try_import("PIL", "Pillow")

    print(f"\n  {BOLD}Vision:{RESET}")
    try_import("cv2", "OpenCV")
    try_import("open3d", "Open3D")
    try_import("transformers", "transformers")
    try_import("sentence_transformers", "sentence-transformers")
    try_import("scipy", "scipy")

    return torch


def check_torch_cuda(torch):
    print(f"\n  {BOLD}CUDA:{RESET}")
    if torch is None:
        check("PyTorch CUDA", False, "torch not imported")
        return
    cuda_ok = torch.cuda.is_available()
    check("PyTorch CUDA available", cuda_ok)
    if cuda_ok:
        dev_name = torch.cuda.get_device_name(0)
        check("GPU via PyTorch", True, dev_name)
        try:
            t = torch.randn(2, 2, device="cuda")
            _ = t @ t
            check("CUDA tensor ops", True)
        except Exception as e:
            check("CUDA tensor ops", False, str(e))


def check_uad_network():
    """Try to import the UAD model architecture."""
    print(f"\n  {BOLD}UAD model:{RESET}")

    # Add unsup-affordance/src to path if needed
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    uad_src = os.path.join(project_root, "unsup-affordance", "src")
    if uad_src not in sys.path:
        sys.path.insert(0, uad_src)

    try:
        from model.network import Conv2DFiLMNet
        check("UAD Conv2DFiLMNet import", True)
    except ImportError as e:
        check("UAD Conv2DFiLMNet import", False, str(e))
    except Exception as e:
        check("UAD Conv2DFiLMNet import", False, str(e))


def check_dinov2_available():
    """Check if DINOv2 can be loaded (may require network on first run)."""
    print(f"\n  {BOLD}DINOv2:{RESET}")
    try:
        import torch
        # Just check torch.hub is functional — don't actually download
        hub_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "unsup-affordance", "None", "hub"
        )
        if os.path.isdir(hub_dir):
            check("DINOv2 weights cached", True, hub_dir)
        else:
            # Also check default torch hub cache
            default_hub = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
            if os.path.isdir(default_hub):
                check("DINOv2 weights cached", True, default_hub)
            else:
                check("DINOv2 weights cached", False,
                      "not yet downloaded — will auto-download on first inference run", warn=True)
    except Exception as e:
        check("DINOv2 weights cached", False, str(e), warn=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}═══ uad Environment Check ═══{RESET}\n")

    check_conda_env()
    check_python_version()
    torch = check_core_packages()
    check_torch_cuda(torch)
    check_uad_network()
    check_dinov2_available()

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
