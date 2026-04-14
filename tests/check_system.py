#!/usr/bin/env python3
"""
System-level checks: NVIDIA driver, GPU, CUDA toolkit.
Run from any environment (no conda env required).

Usage:
    python tests/check_system.py
"""

from __future__ import annotations

import subprocess
import shutil
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


def run_cmd(cmd: list[str], timeout: int = 10) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


# ── Checks ──────────────────────────────────────────────────────────────────

def check_nvidia_smi():
    """NVIDIA driver visible via nvidia-smi."""
    out = run_cmd(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                   "--format=csv,noheader,nounits"])
    if out:
        parts = [p.strip() for p in out.split("\n")[0].split(",")]
        driver, gpu_name, vram_mb = parts[0], parts[1], parts[2]
        check("NVIDIA driver", True, f"v{driver}")
        check("GPU detected", True, f"{gpu_name}, {int(vram_mb)} MiB VRAM")
    else:
        check("NVIDIA driver", False, "nvidia-smi not found or failed")
        check("GPU detected", False)


def check_nvcc():
    """CUDA compiler (nvcc) available."""
    out = run_cmd(["nvcc", "--version"])
    if out:
        # Extract version like "release 12.8"
        for line in out.split("\n"):
            if "release" in line:
                ver = line.split("release")[-1].strip().rstrip(",")
                check("nvcc (CUDA compiler)", True, f"release {ver}")
                return
        check("nvcc (CUDA compiler)", True, out.split("\n")[-1])
    else:
        # nvcc may only be available inside conda env — that's OK at system level
        check("nvcc (CUDA compiler)", False,
              "not in PATH (expected if not inside habitat-grasp env)", warn=True)


def check_git_lfs():
    """git-lfs installed (required for Habitat dataset downloads)."""
    out = run_cmd(["git", "lfs", "version"])
    if out:
        check("git-lfs", True, out.split()[0] if out else "")
    else:
        check("git-lfs", False, "sudo apt install git-lfs && git lfs install")


def check_opengl_libs():
    """OpenGL libraries present (required by habitat-sim)."""
    libs = ["libOpenGL.so.0", "libGL.so.1", "libGLX.so.0"]
    missing = []
    for lib in libs:
        found = run_cmd(["ldconfig", "-p"])
        if found and lib in found:
            continue
        missing.append(lib)
    if not missing:
        check("OpenGL libraries", True, ", ".join(libs))
    else:
        check("OpenGL libraries", False,
              f"missing: {', '.join(missing)} — sudo apt install libopengl0 libgl1-mesa-dev libglx0 libglx-mesa0")


def check_conda():
    """Conda available."""
    conda = shutil.which("conda")
    if conda:
        out = run_cmd(["conda", "--version"])
        check("Conda", True, out or "")
    else:
        check("Conda", False, "conda not found in PATH")


def check_disk_space():
    """Enough free disk space (datasets need ~15GB)."""
    st = os.statvfs(os.path.expanduser("~"))
    free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
    check("Disk space", free_gb > 5,
          f"{free_gb:.1f} GB free" + (" (< 5 GB — may be tight)" if free_gb <= 5 else ""),
          warn=(free_gb <= 5))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}═══ System Environment Check ═══{RESET}\n")

    check_nvidia_smi()
    check_nvcc()
    check_conda()
    check_git_lfs()
    check_opengl_libs()
    check_disk_space()

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
