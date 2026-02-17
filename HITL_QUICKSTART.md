# HITL Quick Start Guide

Quick reference for running Human-in-the-Loop (HITL) examples with Habitat-Lab.

> **Note:** For initial setup, see [SETUP_COMPLETE.md](SETUP_COMPLETE.md)

---

## 🚀 Running HITL Examples

### Step by Step

```bash
# 1. Activate environment
conda activate interactive-robotics

# 2. Navigate to habitat-lab
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab

# 3. Run the rearrange example
python examples/hitl/rearrange/rearrange.py
```

### One-Line Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate interactive-robotics && \
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab && \
python examples/hitl/rearrange/rearrange.py
```

---

## 📺 What to Expect

When you run the rearrange example:

1. **Loading messages** - Scene, robot, and humanoid models load (~30 seconds)
2. **3D window opens** - Showing an HSSD indoor scene
3. **Spot robot** - Boston Dynamics Spot with arm, performs autonomous actions
4. **Humanoid avatar** - Your character that you control
5. **Objects** - YCB objects scattered in the scene to manipulate

---

## 🎮 Controls

## 🎮 Controls

### Basic Movement

| Key | Action |
|-----|--------|
| `I` | Move humanoid forward |
| `K` | Move humanoid backward |
| `J` | Turn humanoid left |
| `L` | Turn humanoid right |
| `SPACEBAR` | Grasp/release nearby objects |
| `H` | Show/hide on-screen help |
| `ESC` | Exit application |

### Free Camera Mode (Press `Z` to toggle)

| Key | Action |
|-----|--------|
| `W`, `S`, `A`, `D` | Move camera |
| `Q`, `E` | Move camera up/down |
| `I`, `J`, `K`, `L`, `U`, `O` | Rotate camera |

---

## ✨ Features Demonstrated

- ✅ **Autonomous robot manipulation** - Spot robot picks up and places objects
- ✅ **Human control** - You control the humanoid avatar
- ✅ **Grasping** - Press spacebar near objects to pick them up
- ✅ **Collaborative tasks** - Work with the robot to rearrange objects
- ✅ **Realistic environments** - HSSD photorealistic indoor scenes
- ✅ **Human-in-the-loop** - Real-time interaction with simulated agents

---

## 📦 Installed Packages

Your environment includes:

- **habitat-sim 0.3.3** - Physics simulation (with Bullet and display support)
- **habitat-lab 0.3.3** - Task framework
- **habitat-baselines 0.3.3** - RL policies
- **habitat-hitl 0.3.3** - Human-in-the-loop framework
- **hydra-core** - Configuration management
- **websockets, aiohttp** - Networking (for VR support)
- **pygame** - Input and window management

## 💾 Downloaded Datasets (~13GB)

- **hssd-hab** - HSSD realistic indoor scenes
- **hab3-episodes** - Task episode configurations
- **habitat_humanoids** - 12 SMPL-X humanoid avatars
- **hab_spot_arm** - Boston Dynamics Spot robot with arm
- **ycb** - YCB object dataset for manipulation

---

## 🧪 Other HITL Examples

### Pick and Throw VR

```bash
conda activate interactive-robotics
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab
python examples/hitl/pick_throw_vr/pick_throw_vr.py
```

### Rearrange V2

```bash
conda activate interactive-robotics
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab
python examples/hitl/rearrange_v2/rearrange_v2.py
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'hydra'"

```bash
conda activate interactive-robotics
pip install hydra-core websockets aiohttp pygame
```

### "FileNotFoundError: Could not find /data"

Make sure you're running from the `habitat-lab/` directory:

```bash
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab
python examples/hitl/rearrange/rearrange.py
```

### Window doesn't open

Check your display:

```bash
echo $DISPLAY  # Should show :0 or :1
```

### "No module named 'habitat_hitl'"

Reinstall habitat-hitl:

```bash
conda activate interactive-robotics
cd ~/Jinyoon_Projects/interactive-robotics/habitat-lab/habitat-hitl
pip install -e .
```

---

## 🔗 Next Steps

- **Test pick-up motions** - Run the rearrange example and use SPACEBAR to grasp objects
- **Explore collaborative tasks** - Watch how the Spot robot interacts with objects
- **Try other examples** - Check `habitat-lab/examples/hitl/` for more applications
- **Read documentation** - See `habitat-lab/habitat-hitl/README.md` for advanced features
- **For VR setup** - See `habitat-lab/examples/hitl/pick_throw_vr/README.md`

## 📖 Resources

- Full setup guide: [SETUP_COMPLETE.md](SETUP_COMPLETE.md)
- Project overview: [README.md](README.md)
- HITL documentation: `habitat-lab/habitat-hitl/README.md`
- Habitat-Lab docs: https://aihabitat.org/docs/habitat-lab/

---

**Ready to test!** Run the command above and start exploring human-robot interaction in simulation.

**Last Updated:** February 17, 2026
