# Interactive Robotics Project

A research project for interactive robotics using Habitat-Lab and Human-in-the-Loop (HITL) framework.

## 📚 Documentation

- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** - Complete installation and setup guide
- **[HITL_QUICKSTART.md](HITL_QUICKSTART.md)** - Quick reference for running HITL examples
- **[PICK_ANIMATION_SUMMARY.md](PICK_ANIMATION_SUMMARY.md)** - Technical notes on humanoid pick animation

## 🚀 Quick Start

**For first-time setup**, follow the detailed guide in [SETUP_COMPLETE.md](SETUP_COMPLETE.md).

**TL;DR for experienced users:**

```bash
# 1. Create conda environment
conda create -n interactive-robotics python=3.9 cmake=3.14.0 -y
conda activate interactive-robotics

# 2. Install habitat-sim with display support (NOT headless)
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y

# 3. Install habitat packages
cd habitat-lab
pip install -e habitat-lab/
pip install -e habitat-baselines/
pip install -e habitat-hitl/

# 4. Install HITL dependencies
pip install hydra-core websockets aiohttp pygame

# 5. Download datasets (13GB, see SETUP_COMPLETE.md for details)
python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm ycb --data-path data/
```

## 🎮 Running HITL Examples

Once setup is complete, see [HITL_QUICKSTART.md](HITL_QUICKSTART.md) for detailed instructions.

**Quick command:**

```bash
conda activate interactive-robotics
cd habitat-lab
python examples/hitl/rearrange/rearrange.py
```

## 🛠️ Environment Details

**Conda Environment:** `interactive-robotics` (Python 3.9)

**Key Packages:**
- habitat-sim 0.3.3 (with Bullet physics & display support)
- habitat-lab 0.3.3
- habitat-baselines 0.3.3
- habitat-hitl 0.3.3
- hydra-core, websockets, aiohttp, pygame

**Datasets (~13GB):**
- HSSD realistic indoor scenes
- Habitat 3.0 episode configurations
- SMPL-X humanoid avatars with motions
- Boston Dynamics Spot robot with arm
- YCB object dataset for manipulation

## 📁 Project Structure

```
interactive-robotics/
├── README.md                  # This file - project overview
├── SETUP_COMPLETE.md          # Detailed setup instructions
├── HITL_QUICKSTART.md         # Quick reference for running examples
├── PICK_ANIMATION_SUMMARY.md  # Technical notes
├── .gitignore                 # Excludes large data files
├── affordance-pipeline/       # Affordance visualization tools
├── graspnet-baseline/         # GraspNet implementation
├── vposer-tools/              # VPoser motion generation tools
└── habitat-lab/               # Habitat-Lab framework (cloned from GitHub)
    ├── habitat-lab/           # Core library
    ├── habitat-baselines/     # RL training algorithms
    ├── habitat-hitl/          # Human-in-the-Loop framework
    ├── examples/hitl/         # HITL example applications
    └── data/                  # Datasets (not in git, download separately)
        ├── versioned_data/    # Main data storage (13GB)
        ├── scene_datasets/    # Scene files (symlinked)
        ├── humanoids/         # Humanoid models (symlinked)
        ├── robots/            # Robot models (symlinked)
        └── objects/           # Object models (symlinked)
```

## 🎯 Features

This project demonstrates:
- ✅ **Autonomous robot manipulation** - Spot robot with arm performing pick & place
- ✅ **Human-in-the-Loop control** - Direct control of humanoid avatars
- ✅ **Realistic environments** - HSSD photorealistic indoor scenes
- ✅ **Collaborative tasks** - Humans and robots working together
- ✅ **Physics simulation** - Bullet physics for realistic interactions
- ✅ **VR support** - Optional VR headset integration

## 🔧 Development

**Verify installation:**

```bash
conda activate interactive-robotics
cd habitat-lab
python -c "import habitat; import habitat_sim; import habitat_hitl; import hydra; print('✓ All packages imported successfully')"
```

**List available datasets:**

```bash
python -m habitat_sim.utils.datasets_download --list
```

**Check habitat-sim version:**

```bash
python -c "import habitat_sim; print(f'Version: {habitat_sim.__version__}')"
```

## 📖 Resources

- **Habitat-Lab:** https://github.com/facebookresearch/habitat-lab
- **Documentation:** https://aihabitat.org/docs/habitat-lab/
- **Habitat 3.0 Paper:** https://arxiv.org/abs/2310.13724
- **Community Forum:** https://github.com/facebookresearch/habitat-lab/discussions

## 🐛 Troubleshooting

For common issues and solutions, see [SETUP_COMPLETE.md](SETUP_COMPLETE.md#troubleshooting).

## 📝 License

This project follows the licensing of its dependencies:
- Habitat-Lab: MIT License
- Habitat-Sim: MIT License

---

**Last Updated:** February 2026  
**Habitat Version:** 0.3.3

**Test pick-up motions:**
```bash
conda activate interactive-robotics
cd habitat-lab
# Interactive play with manual control of robot arm
python examples/interactive_play.py
conda activate interactive-robotics
cd habitat-lab
python examples/hitl/rearrange_v2/rearrange_v2.py
```

## Development Tips

**Activate environment:**
```bash
conda activate interactive-robotics
```

**Check available datasets:**
```bash
cd habitat-lab
python -m habitat_sim.utils.datasets_download --list
```

**Verify GPU support:**
```bash
python -c "import habitat_sim; print(f'Habitat-sim version: {habitat_sim.__version__}')"
```

## Git Repository

**Repository:** https://github.com/jinyoonok2/interactive-robotics

```bash
git status              # Check changes
git add .               # Stage changes
git commit -m "msg"     # Commit
git push                # Push to GitHub
```

## Resources

- **Habitat-Lab Docs:** https://aihabitat.org/docs/habitat-lab/
- **HITL Framework:** `habitat-lab/habitat-hitl/README.md`
- **Habitat Examples:** `habitat-lab/examples/`
- **Setup Guide:** [SETUP_COMPLETE.md](SETUP_COMPLETE.md)

## Troubleshooting

See [SETUP_COMPLETE.md](SETUP_COMPLETE.md) for detailed troubleshooting steps.

**Quick checks:**
```bash
# Verify environment
conda activate interactive-robotics
python -c "import habitat; import habitat_sim; import habitat_hitl; print('✓ OK')"

# Check installed packages
conda list | grep habitat

# Verify data symlinks
cd habitat-lab/data && find . -type l -exec test ! -e {} \; -print
```
cd habitat-lab/habitat-lab && pip install -e .
cd ../habitat-baselines && pip install -e .
cd ../habitat-hitl && pip install -e .
```

## Notes

- Dataset downloads are separate from the environment setup
- HITL apps run at 30 FPS by default
- For VR support, additional setup may be required
- The project uses SSH authentication for GitHub
