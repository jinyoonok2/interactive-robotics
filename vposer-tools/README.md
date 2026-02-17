# VPoser Tools for Habitat Lab

This directory contains tools for generating custom humanoid poses using VPoser to enhance pick-up motions in Habitat Lab.

## Files

- **VPOSER_CUSTOM_POSES_GUIDE.md** - Complete guide on using VPoser with Habitat Lab
- **analyze_motion_workspace.py** - Analyze existing motion data to understand workspace coverage
- **generate_detailed_pick_motions.py** - Generate new custom poses with VPoser

## Quick Start

### 1. Analyze Existing Poses

```bash
cd /home/jinyoon-kim/Jinyoon_Projects/interactive-robotics/vposer-tools

python analyze_motion_workspace.py ../habitat-lab/data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl
```

This will show you:
- Current workspace bounds
- Pose density
- Gaps in coverage
- 3D visualizations

### 2. Generate Custom Poses

First, install VPoser:

```bash
conda activate interactive-robotics
pip install git+https://github.com/nghorbani/human_body_prior
```

Download VPoser model from https://smpl-x.is.tue.mpg.de/ (registration required) and extract to `../habitat-lab/data/vposer/V02_05/`

Then generate:

```bash
python generate_detailed_pick_motions.py \
    --grid-size 8 8 6 \
    --output ../habitat-lab/data/humanoids/custom_motions/detailed_pick_female_2.pkl
```

### 3. Use in Your Demo

```python
# In simple_pick_demo.py or other scripts
controller = HumanoidRearrangeController(
    "data/humanoids/custom_motions/detailed_pick_female_2.pkl"
)
```

## Why Separate Directory?

These tools are **external pose generation utilities** that work with Habitat Lab but aren't part of the core framework. Keeping them separate:

- Avoids cluttering habitat-lab directory
- Makes it clear these are custom additions
- Easier to version control separately
- Can be shared across different Habitat projects

## Output Location

Generated motion files should be saved to:
```
habitat-lab/data/humanoids/custom_motions/
```

This keeps them with other humanoid data but clearly marked as custom.
