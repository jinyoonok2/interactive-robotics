# Complete Setup Guide for Interactive Robotics

This guide provides step-by-step instructions for setting up the interactive robotics environment using Habitat-Lab and Human-in-the-Loop (HITL) framework.

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
- [Testing HITL Examples](#testing-hitl-examples)
- [Troubleshooting](#troubleshooting)
- [Additional Datasets](#additional-datasets-optional)
- [Quick Reference](#quick-reference)

---

## Prerequisites

Before starting, ensure you have:

- **Conda/Miniconda** installed ([Install Guide](https://docs.conda.io/en/latest/miniconda.html))
- **Git** installed and configured
- **Ubuntu/Linux** system (tested on Ubuntu with GNOME desktop)
- **GPU with OpenGL support** (tested with AMD Radeon 780M, NVIDIA GPUs also work)
- **At least 15GB free disk space** (13GB for datasets + 2GB for environment)

### System Requirements

- Python 3.9
- cmake 3.14+
- OpenGL 4.6+ support
- X11 display server (for GUI examples)

---

## Step-by-Step Setup

### 1. Clone the Repository

```bash
cd ~/Jinyoon_Projects  # or your preferred directory
git clone git@github.com:jinyoonok2/interactive-robotics.git
cd interactive-robotics
```

### 2. Clone Habitat-Lab Submodule

Since habitat-lab is included but data is excluded, you need to clone it:

```bash
# If habitat-lab folder doesn't exist or is empty
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.3.3  # Use stable version 0.3.3
cd ..
```

### 3. Create Conda Environment

```bash
# Create environment with Python 3.9 and cmake
conda create -n interactive-robotics python=3.9 cmake=3.14.0 -y

# Activate the environment
conda activate interactive-robotics
```

### 4. Install Habitat-Sim

**IMPORTANT:** Install the **display version** (not headless) for HITL GUI examples:

```bash
# Install habitat-sim with Bullet physics and display support
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
```

**Verify installation:**
```bash
python -c "import habitat_sim; print(f'Version: {habitat_sim.__version__}')"
```

Expected output: `Version: 0.3.3`

**Check for display support:**
```bash
conda list | grep habitat-sim-mutex
```

Expected output should show: `habitat-sim-mutex-1.0-display_bullet` (NOT `headless_bullet`)

### 5. Install Habitat Packages

Navigate to habitat-lab and install all three packages in editable mode:

```bash
cd habitat-lab

# Install habitat-lab
cd habitat-lab
pip install -e .
cd ..

# Install habitat-baselines
cd habitat-baselines
pip install -e .
cd ..

# Install habitat-hitl
cd habitat-hitl
pip install -e .
cd ..
```

**Fix pillow version if needed:**
```bash
pip install pillow==10.4.0
```

### 5.5. Install HITL Dependencies

The HITL framework requires additional Python packages:

```bash
# Make sure you're in the interactive-robotics conda environment
conda activate interactive-robotics

# Install HITL dependencies
pip install hydra-core websockets aiohttp pygame
```

These packages provide:
- **hydra-core**: Configuration management for HITL apps
- **websockets & aiohttp**: Networking for VR client/server communication
- **pygame**: Input handling and window management for interactive applications

### 6. Verify Installation

```bash
python -c "import habitat; import habitat_sim; import habitat_baselines; import habitat_hitl; import hydra; import pygame; print('✓ All packages imported successfully')"
```

### 7. Download Required Datasets

**⚠️ WARNING:** Datasets are large (~13GB total). Ensure you have enough disk space.

Navigate to the habitat-lab directory:
```bash
cd habitat-lab
```

**Download all required datasets at once:**
```bash
python -m habitat_sim.utils.datasets_download \
    --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm ycb \
    --data-path data/
```

This downloads:
- **hssd-hab** (~12GB): HSSD scene dataset with realistic 3D environments
- **hab3-episodes** (~50MB): Episode configurations for tasks
- **habitat_humanoids** (~58MB): 12 SMPL-X humanoid avatars with motions
- **hab_spot_arm** (~6MB): Boston Dynamics Spot robot with arm
- **ycb** (~479MB): YCB object dataset for manipulation

**Download time:** 10-30 minutes depending on your internet connection.

**Alternative: Download individually**
```bash
# Scene dataset only
python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path data/

# Add humanoids
python -m habitat_sim.utils.datasets_download --uids habitat_humanoids --data-path data/

# Add robot
python -m habitat_sim.utils.datasets_download --uids hab_spot_arm --data-path data/

# Add objects
python -m habitat_sim.utils.datasets_download --uids ycb --data-path data/

# Add episodes
python -m habitat_sim.utils.datasets_download --uids hab3-episodes --data-path data/
```

**View all available datasets:**
```bash
python -m habitat_sim.utils.datasets_download --list
```

### 8. Verify Data Structure

After download, your data directory should look like this:

```bash
cd data
ls -lh
```

Expected structure:
```
data/
├── versioned_data/          # Main storage
│   ├── hssd-hab/           # 12GB - Scene dataset
│   ├── hab3-episodes/      # 50MB - Episode configs
│   ├── habitat_humanoids/  # 58MB - Humanoid models
│   ├── hab_spot_arm/       # 6MB - Spot robot
│   └── ycb/                # 479MB - YCB objects
├── scene_datasets/          # Symlinks
│   └── hssd-hab -> ../versioned_data/hssd-hab
├── datasets/
│   ├── hssd/
│   │   └── rearrange -> ../../versioned_data/hab3-episodes
│   └── replica_cad/
├── humanoids/
│   └── humanoid_data -> ../versioned_data/habitat_humanoids
├── robots/
│   └── hab_spot_arm -> ../versioned_data/hab_spot_arm
└── objects/
    └── ycb -> ../versioned_data/ycb
```

**Check for broken symlinks:**
```bash
find . -type l -exec test ! -e {} \; -print
```

If any broken symlinks are found, fix them:
```bash
# Fix scene_datasets symlink
rm scene_datasets/hssd-hab
ln -s "$PWD/versioned_data/hssd-hab" scene_datasets/hssd-hab

# Fix datasets/hssd/rearrange symlink
rm datasets/hssd/rearrange
ln -s "$PWD/versioned_data/hab3-episodes" datasets/hssd/rearrange

# Fix humanoids symlink
rm humanoids/humanoid_data
ln -s "$PWD/versioned_data/habitat_humanoids" humanoids/humanoid_data

# Fix robots symlink
rm robots/hab_spot_arm
ln -s "$PWD/versioned_data/hab_spot_arm" robots/hab_spot_arm

# Fix objects symlink
rm objects/ycb
ln -s "$PWD/versioned_data/ycb" objects/ycb
```

## 🧪 Testing HITL Examples

**IMPORTANT:** Always run HITL examples from the `habitat-lab/` directory with the conda environment activated.

Return to habitat-lab root:

```bash
cd ..  # Back to habitat-lab/
pwd    # Should show: .../interactive-robotics/habitat-lab
```

### Test Rearrange Example (Recommended)

This is the best example for testing pick-up motions:

```bash
# Make sure environment is activated
conda activate interactive-robotics

# Run from habitat-lab directory
python examples/hitl/rearrange/rearrange.py
```

**Features demonstrated:**
- ✅ Spot robot with arm performing autonomous pick-up and manipulation
- ✅ Humanoid avatar that you control with keyboard/mouse
- ✅ Collaborative rearrangement tasks in realistic HSSD scenes
- ✅ Pick & place motions with grasping and object manipulation

**Controls (once the window opens):**

| Key | Action |
|-----|--------|
| `I`, `J`, `K`, `L` | Move the humanoid avatar base |
| `SPACEBAR` | Grasp/release nearby objects |
| `Z` | Toggle free camera mode |
| `W`, `S`, `A`, `D`, `Q`, `E` | Move camera (in free camera mode) |
| `H` | Show/hide help text |
| `ESC` | Exit application |

### Other HITL Examples

```bash
# Pick and throw VR (desktop mode, no VR headset required)
conda activate interactive-robotics
cd habitat-lab
python examples/hitl/pick_throw_vr/pick_throw_vr.py

# Rearrange v2
conda activate interactive-robotics
cd habitat-lab
python examples/hitl/rearrange_v2/rearrange_v2.py
```

### Common Issues

- **FileNotFoundError: Could not find /data** → Make sure you're running from `habitat-lab/` directory
- **ModuleNotFoundError: No module named 'hydra'** → Install HITL dependencies: `pip install hydra-core websockets aiohttp pygame`
- **Window doesn't open** → Check your display: `echo $DISPLAY`

**If a GUI window opens showing the 3D environment with a robot and humanoid, setup is complete!** ✅

---

## Environment Configuration Details

### Display Support

The environment requires **display support** for HITL GUI examples. Verify:

```bash
echo $DISPLAY  # Should show :0 or similar
glxinfo | grep "OpenGL version"  # Should show OpenGL 4.x+
```

### GPU Detection

Verify GPU is detected:
```bash
python -c "import habitat_sim; print(habitat_sim.built_with_cuda if hasattr(habitat_sim, 'built_with_cuda') else 'N/A')"
```

### Package Versions

Expected versions:
```bash
conda activate interactive-robotics
conda list | grep habitat
```

Output should include:
- `habitat-sim 0.3.3`
- `habitat-sim-mutex 1.0 display_bullet` (or `main_display_bullet`)

```bash
pip list | grep habitat
```

Output should include:
- `habitat-lab 0.3.3`
- `habitat-baselines 0.3.3`
- `habitat-hitl 0.3.3`

---

## Troubleshooting

### Issue: "No module named 'magnum.platform.glfw'"

**Cause:** Headless version of habitat-sim was installed instead of display version.

**Solution:**

```bash
# Remove environment
conda deactivate
conda env remove -n interactive-robotics -y

# Recreate with display version
conda create -n interactive-robotics python=3.9 cmake=3.14.0 -y
conda activate interactive-robotics
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y  # WITHOUT --headless!

# Verify display version
conda list | grep habitat-sim-mutex  # Should show "display_bullet"

# Reinstall habitat packages
cd habitat-lab
pip install -e habitat-lab/ habitat-baselines/ habitat-hitl/
```

### Issue: "FileNotFoundError: data/datasets/hssd/rearrange/train/..."

**Cause:** Missing or broken symlinks to dataset files.

**Solution:**

```bash
cd habitat-lab/data

# Check for broken symlinks
find . -type l -exec test ! -e {} \; -print

# Fix scene_datasets/hssd-hab
rm scene_datasets/hssd-hab
ln -s "$PWD/versioned_data/hssd-hab" scene_datasets/hssd-hab

# Fix datasets/hssd/rearrange
rm datasets/hssd/rearrange
ln -s "$PWD/versioned_data/hab3-episodes" datasets/hssd/rearrange
```

### Issue: "FileNotFoundError: data/humanoids/humanoid_data/..."

**Cause:** Broken humanoid symlink.

**Solution:**

```bash
cd habitat-lab/data
rm humanoids/humanoid_data
ln -s "$PWD/versioned_data/habitat_humanoids" humanoids/humanoid_data
```

### Issue: PIL/Pillow version conflict

**Error:** `ImportError: cannot import name 'PILLOW_VERSION'`

**Solution:**

```bash
pip install pillow==10.4.0
```

### Issue: Gym deprecation warnings

**Warning:** `Gym has been unmaintained since 2022...`

This is a **warning only** and does not affect functionality. The Habitat team is working on migrating to Gymnasium.

### Issue: Display not available / DISPLAY not set

**Error:** `Could not initialize OpenGL context`

**Solution:**

```bash
# Check display
echo $DISPLAY  # Should show :0 or :1

# If empty, set it
export DISPLAY=:0

# Verify X11 is running
ps aux | grep X
```

### Issue: Git LFS files not downloaded

If dataset downloads fail with git errors:

**Solution:**

```bash
# Install git-lfs if not available
sudo apt install git-lfs
git lfs install

# Or use --no-prune flag
python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path data/ --no-prune
```

---

## Additional Datasets (Optional)

### Replica CAD Dataset

For rearrange task examples:

```bash
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/
```

### HM3D Dataset

For navigation tasks (requires registration):

```bash
python -m habitat_sim.utils.datasets_download --uids hm3d_minival_v0.2 --data-path data/
```

### Full Rearrange Assets

```bash
python -m habitat_sim.utils.datasets_download --uids rearrange_dataset_v2 --data-path data/
```

---

## Quick Reference

### Every Time You Start Working

```bash
# 1. Activate the conda environment
conda activate interactive-robotics

# 2. Navigate to habitat-lab directory
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab  # Adjust path as needed

# 3. Run HITL examples
python examples/hitl/rearrange/rearrange.py
```

### Complete One-Line Command

For convenience, run everything in one command:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate interactive-robotics && \
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab && \
python examples/hitl/rearrange/rearrange.py
```

### Environment Summary

Your current setup includes:

- **Conda Environment:** `interactive-robotics` (Python 3.9)
- **Core Packages:**
  - habitat-sim 0.3.3 (with Bullet physics and display support)
  - habitat-lab 0.3.3
  - habitat-baselines 0.3.3
  - habitat-hitl 0.3.3
- **HITL Dependencies:**
  - hydra-core (configuration management)
  - websockets & aiohttp (networking)
  - pygame (input/window management)
- **Downloaded Datasets (~13GB):**
  - HSSD scenes (hssd-hab)
  - Episode configurations (hab3-episodes)
  - Humanoid models (habitat_humanoids)
  - Spot robot with arm (hab_spot_arm)
  - YCB objects (ycb)

### Uninstall

To completely remove the environment:

```bash
# Deactivate if active
conda deactivate

# Remove conda environment
conda env remove -n interactive-robotics -y

# Remove data (if desired)
rm -rf ~/Jinyoon_Projects/interactive-robotics/habitat-lab/data/versioned_data
```

---

## 📖 Resources

- **Habitat-Lab GitHub:** https://github.com/facebookresearch/habitat-lab
- **Habitat-Sim GitHub:** https://github.com/facebookresearch/habitat-sim
- **Documentation:** https://aihabitat.org/docs/habitat-lab/
- **HITL Framework:** https://github.com/facebookresearch/habitat-lab/tree/main/habitat-hitl
- **Habitat 3.0 Paper:** https://arxiv.org/abs/2310.13724
- **Community Forum:** https://github.com/facebookresearch/habitat-lab/discussions

## 🆘 Support

For issues:

1. Check this troubleshooting guide
2. Check [habitat-lab issues](https://github.com/facebookresearch/habitat-lab/issues)
3. Create a new issue on the [project repository](https://github.com/jinyoonok2/interactive-robotics/issues)

---

**Last Updated:** February 17, 2026  
**Habitat Version:** 0.3.3  
**Tested On:** Ubuntu with GNOME, AMD Radeon 780M Graphics
