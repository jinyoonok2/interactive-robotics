#!/usr/bin/env python3
"""
Verify the `habitat-grasp` conda environment has all required packages,
correct versions, and working CUDA extensions.

Usage:
    conda activate habitat-grasp
    python tests/check_habitat_grasp.py
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
    """Verify we're inside the habitat-grasp conda env."""
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    check("Conda env = habitat-grasp", env == "habitat-grasp",
          f"current: '{env}'" if env != "habitat-grasp" else "")


def check_python_version():
    """Python 3.9.x required."""
    v = sys.version_info
    ok = v.major == 3 and v.minor == 9
    check("Python 3.9", ok, f"{v.major}.{v.minor}.{v.micro}")


# ── Package imports ─────────────────────────────────────────────────────────

def try_import(module_name: str, display_name: str = None, version_attr: str = "__version__",
               expected_version: str = None):
    """Try to import a module and optionally check its version."""
    display = display_name or module_name
    try:
        mod = __import__(module_name)
        ver = getattr(mod, version_attr, None) if version_attr else None
        detail = f"v{ver}" if ver else "OK"

        if expected_version and ver:
            ver_ok = ver.startswith(expected_version)
            if not ver_ok:
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
    """Import all required packages and verify key versions."""
    print(f"\n  {BOLD}ML / Numerics:{RESET}")
    # Import torch FIRST — habitat-sim initializes CUDA differently than
    # PyTorch, and loading graspnet CUDA extensions (knn, pointnet2) after
    # habitat-sim without torch causes heap corruption.  Pre-importing torch
    # ensures consistent CUDA context initialization.
    torch = try_import("torch", "PyTorch", expected_version="2.")
    try_import("numpy", "numpy", expected_version="1.26")
    try_import("PIL", "Pillow", version_attr="__version__")
    try_import("scipy", "scipy")

    print(f"\n  {BOLD}Core packages:{RESET}")
    try_import("habitat_sim", "habitat-sim", expected_version="0.3")
    try_import("habitat", "habitat-lab")
    try_import("habitat_baselines", "habitat-baselines")

    print(f"\n  {BOLD}Vision / Pipeline:{RESET}")
    try_import("cv2", "OpenCV")
    try_import("open3d", "Open3D")

    print(f"\n  {BOLD}Grasp:{RESET}")
    try_import("graspnetAPI", "graspnetAPI")
    try_import("transforms3d", "transforms3d")

    print(f"\n  {BOLD}HITL / Misc:{RESET}")
    try_import("hydra", "hydra-core")
    try_import("websockets", "websockets", version_attr=None)
    try_import("aiohttp", "aiohttp")
    try_import("pygame", "pygame")

    return torch


def check_torch_cuda(torch):
    """PyTorch can see the GPU."""
    print(f"\n  {BOLD}CUDA:{RESET}")
    if torch is None:
        check("PyTorch CUDA", False, "torch not imported")
        return
    cuda_ok = torch.cuda.is_available()
    check("PyTorch CUDA available", cuda_ok)
    if cuda_ok:
        dev_name = torch.cuda.get_device_name(0)
        check("GPU via PyTorch", True, dev_name)
        # Quick tensor test
        try:
            t = torch.randn(2, 2, device="cuda")
            _ = t @ t
            check("CUDA tensor ops", True)
        except Exception as e:
            check("CUDA tensor ops", False, str(e))


def check_graspnet_cuda_extensions():
    """GraspNet CUDA extensions (knn_pytorch, pointnet2) compiled and loadable."""
    print(f"\n  {BOLD}GraspNet CUDA extensions:{RESET}")
    try:
        import knn_pytorch
        check("knn_pytorch", True)
    except ImportError:
        check("knn_pytorch", False,
              "cd graspnet-baseline/knn && python setup.py install")

    try:
        import pointnet2
        check("pointnet2", True)
    except ImportError:
        check("pointnet2", False,
              "cd graspnet-baseline/pointnet2 && python setup.py install")


def check_transforms3d_patch():
    """transforms3d patched for numpy >= 1.24 (np.float removed)."""
    print(f"\n  {BOLD}Patches:{RESET}")
    try:
        import transforms3d.quaternions as q
        # quat2mat uses np.float internally if unpatched
        import numpy as np
        mat = q.quat2mat([1, 0, 0, 0])
        check("transforms3d np.float patch", True, "quat2mat works")
    except AttributeError as e:
        if "np.float" in str(e):
            check("transforms3d np.float patch", False,
                  "np.float still in use — run patch from ENVIRONMENT_STATUS.md step 7")
        else:
            check("transforms3d np.float patch", False, str(e))
    except ImportError:
        check("transforms3d np.float patch", False, "transforms3d not installed", warn=True)
    except Exception as e:
        check("transforms3d np.float patch", False, str(e))


def check_nvcc():
    """CUDA compiler available (for building extensions)."""
    import shutil
    import subprocess
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            for line in out.stdout.split("\n"):
                if "release" in line:
                    ver = line.split("release")[-1].strip().rstrip(",")
                    check("nvcc", True, f"release {ver}")
                    return
            check("nvcc", True)
        except Exception:
            check("nvcc", True, nvcc)
    else:
        check("nvcc", False,
              "not found — conda install -c 'nvidia/label/cuda-12.8.0' cuda-nvcc cuda-cudart-dev",
              warn=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}═══ habitat-grasp Environment Check ═══{RESET}\n")

    check_conda_env()
    check_python_version()
    torch = check_core_packages()
    check_torch_cuda(torch)
    check_graspnet_cuda_extensions()
    check_transforms3d_patch()
    check_nvcc()

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
    # Write exit code to a file if CHECK_RESULT_FILE is set.
    # This allows check_all.sh to read the result even if habitat-sim crashes during cleanup.
    result_file = os.environ.get("CHECK_RESULT_FILE")
    if result_file:
        with open(result_file, "w") as f:
            f.write(str(rc))
    sys.exit(rc)
