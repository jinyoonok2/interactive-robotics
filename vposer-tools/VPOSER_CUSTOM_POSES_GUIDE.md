# VPoser Custom Pose Generation Guide

## Overview

This guide explains how to generate **custom detailed pick-up motions** using VPoser and integrate them into Habitat Lab's interactive environment.

## Current System Limitations

The existing motion data has **only 48 pre-generated poses** per hand:
- Grid size: typically 4×4×3 = 48 poses
- Limited workspace coverage (~0.7m³ around humanoid)
- Single reaching style (direct reach)
- No grasp variation or detailed pick sequences

## Goal: Generate Detailed Pick Motions

We want to create:
1. **Multi-phase pick sequences**: Approach → Pre-grasp → Grasp → Retract
2. **Varied grasp types**: Power grip, precision grip, pinch
3. **Denser pose coverage**: More poses for smoother interpolation
4. **Ground-level pickups**: Squatting/bending poses
5. **Custom reach trajectories**: Curved paths, obstacle avoidance

## Architecture Overview

```
VPoser Model → Generate Poses → Package as Motion Data → Load in Habitat
     ↓                ↓                    ↓                      ↓
  body_prior    optimize.py      custom_motion.pkl    HumanoidRearrangeController
```

## Step 1: Install VPoser

```bash
conda activate interactive-robotics

# Install human_body_prior (contains VPoser)
pip install git+https://github.com/nghorbani/human_body_prior

# Download pre-trained VPoser model
mkdir -p data/vposer
cd data/vposer

# Download from: https://smpl-x.is.tue.mpg.de/
# You'll need to register and download "VPoser v2.0"
# Extract to: data/vposer/V02_05/
```

## Step 2: Understanding the Motion Data Format

Each motion file (`.pkl`) contains:

```python
{
    'walk_motion': {...},       # Walking animation
    'stop_pose': {...},         # Standing pose
    'left_hand': {              # Left hand reaching poses
        'pose_motion': {
            'joints_array': np.array,      # (N*J) × 4 quaternions
            'transform_array': np.array     # N × 4 × 4 transforms
        },
        'coord_info': {
            'min': [x_min, y_min, z_min],
            'max': [x_max, y_max, z_max],
            'num_bins': [nx, ny, nz]       # Grid dimensions
        }
    },
    'right_hand': {...}         # Same structure
}
```

## Step 3: Generate Custom Poses with VPoser

Key files to create:
- `generate_custom_poses.py` - Main pose generation script
- `custom_motion_builder.py` - Package poses into Habitat format

### Understanding VPoser Parameters

```python
# VPoser latent space (32D vector) controls pose
# Optimize latent code to reach target 3D positions

import torch
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

# Load VPoser
vposer, _ = load_model(
    'data/vposer/V02_05',
    model_code=VPoser,
    remove_words_in_model_weights='vp_model.',
    disable_grad=True
)

# Generate pose from latent code
latent_code = torch.randn(1, 32)  # Random pose
body_pose = vposer.decode(latent_code)['pose_body']
```

## Step 4: Optimization for Target Reaching

```python
# Optimize VPoser latent code to reach specific 3D hand position

import torch.nn as nn
import torch.optim as optim

def optimize_reach_pose(target_3d_position, hand='right', iterations=500):
    """
    Generate a humanoid pose that reaches to target_3d_position.
    
    Args:
        target_3d_position: (x, y, z) in humanoid local coordinates
        hand: 'left' or 'right'
        iterations: Number of optimization steps
    
    Returns:
        body_pose: 21 joint rotations (SMPL-X body)
        hand_pose: Hand joint configuration
    """
    
    # Initialize latent code
    latent = torch.randn(1, 32, requires_grad=True)
    
    # Optimizer
    optimizer = optim.Adam([latent], lr=0.01)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Decode pose
        pose_out = vposer.decode(latent)
        body_pose = pose_out['pose_body']
        
        # Forward kinematics to get hand position
        # (You'll need SMPL-X model for this)
        hand_pos = forward_kinematics(body_pose, hand_idx)
        
        # Loss: distance to target
        loss = torch.norm(hand_pos - target_3d_position)
        
        # Additional constraints
        loss += 0.01 * torch.norm(latent)  # Regularization
        loss += collision_loss(body_pose)  # Self-collision avoidance
        
        loss.backward()
        optimizer.step()
    
    return body_pose.detach()
```

## Step 5: Create Denser Pose Grids

Instead of 4×4×3 = 48 poses, generate **8×8×6 = 384 poses** for smoother motion:

```python
# Define workspace bounds (in humanoid local frame)
x_range = (-0.8, 0.8)   # Left/right
y_range = (0.0, 1.6)    # Height
z_range = (-0.5, 0.8)   # Front/back

# Dense grid
nx, ny, nz = 8, 8, 6

# Generate target positions
target_positions = []
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            x = x_range[0] + (ix / (nx-1)) * (x_range[1] - x_range[0])
            y = y_range[0] + (iy / (ny-1)) * (y_range[1] - y_range[0])
            z = z_range[0] + (iz / (nz-1)) * (z_range[1] - z_range[0])
            target_positions.append((x, y, z))

# Optimize VPoser for each target
poses = []
for target in target_positions:
    pose = optimize_reach_pose(target, hand='right')
    poses.append(pose)
```

## Step 6: Add Multi-Phase Pick Sequences

Generate poses for detailed pick-up motion:

```python
def generate_detailed_pick_sequence(object_position):
    """
    Generate 4-phase pick sequence:
    1. Approach (hand 20cm above object)
    2. Pre-grasp (hand at object, fingers open)
    3. Grasp (fingers closed)
    4. Retract (lift object 30cm)
    """
    
    phases = {
        'approach': object_position + np.array([0, 0.2, 0]),
        'pregrasp': object_position,
        'grasp': object_position,  # Different hand config
        'retract': object_position + np.array([0, 0.3, 0])
    }
    
    sequence_poses = {}
    for phase_name, target in phases.items():
        pose = optimize_reach_pose(target, hand='right')
        sequence_poses[phase_name] = pose
    
    return sequence_poses
```

## Step 7: Package for Habitat

```python
import pickle as pkl

def create_custom_motion_file(poses, grid_info, output_path):
    """
    Package custom VPoser poses into Habitat-compatible format.
    """
    
    # Load base motion file
    with open('data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl', 'rb') as f:
        base_data = pkl.load(f)
    
    # Replace right_hand data
    custom_data = base_data.copy()
    custom_data['right_hand'] = {
        'pose_motion': {
            'joints_array': poses.reshape(-1, 4),  # Flatten to (N*J) × 4
            'transform_array': transforms           # N × 4 × 4
        },
        'coord_info': {
            'min': grid_info['min'],
            'max': grid_info['max'],
            'num_bins': grid_info['num_bins']
        }
    }
    
    # Save
    with open(output_path, 'wb') as f:
        pkl.dump(custom_data, f)
    
    print(f"✓ Custom motion file saved: {output_path}")
```

## Step 8: Use Custom Motions in Your Demo

```python
# In simple_pick_demo.py or your interactive app

# Load humanoid with CUSTOM motion data
controller = HumanoidRearrangeController(
    "data/humanoids/custom_motions/detailed_pick_female_2.pkl"  # Your new file!
)

# Now you have access to detailed pick motions
controller.calculate_reach_pose(ground_object_pos, index_hand=1)
```

## Complete Workflow Script

I'll create a complete script that does all this: `generate_detailed_pick_motions.py`

## Ground-Level Pickup Poses

For ground pickups, you need poses with **bent knees/waist**:

```python
def optimize_ground_pickup_pose(ground_position, bend_type='squat'):
    """
    Generate ground pickup poses with body bending.
    
    Args:
        bend_type: 'squat' (knees bent) or 'lean' (waist bent)
    """
    
    latent = torch.randn(1, 32, requires_grad=True)
    optimizer = optim.Adam([latent], lr=0.01)
    
    for i in range(500):
        optimizer.zero_grad()
        pose_out = vposer.decode(latent)
        body_pose = pose_out['pose_body']
        
        # Get joint positions
        joints = forward_kinematics(body_pose)
        hand_pos = joints['right_hand']
        pelvis_height = joints['pelvis'][1]
        
        # Loss terms
        loss = torch.norm(hand_pos - ground_position)  # Reach target
        
        if bend_type == 'squat':
            # Encourage low pelvis
            loss += 0.5 * torch.relu(pelvis_height - 0.6)
        elif bend_type == 'lean':
            # Encourage bent spine
            spine_bend = joints['spine3_angle']
            loss += 0.5 * torch.relu(1.0 - spine_bend)
        
        loss.backward()
        optimizer.step()
    
    return body_pose.detach()
```

## Integration with Interactive Environment

For HITL apps like `rearrange.py`:

```python
# In habitat-hitl/environment/controllers/gui_controller.py

# Load custom motions
self._humanoid_controller = HumanoidRearrangeController(
    walk_pose_path="data/humanoids/custom_motions/detailed_pick_female_2.pkl"
)

# In update_pick_pose() method
def update_pick_pose(self, target_obj):
    # Use multi-phase sequence
    if self._pick_phase == 'approach':
        reach_pos = target_obj.translation + mn.Vector3(0, 0.2, 0)
    elif self._pick_phase == 'grasp':
        reach_pos = target_obj.translation
    elif self._pick_phase == 'retract':
        reach_pos = target_obj.translation + mn.Vector3(0, 0.3, 0)
    
    self._humanoid_controller.calculate_reach_pose(reach_pos, index_hand=1)
```

## Next Steps

1. **Run the generator script** (I'll create this next)
2. **Test custom poses** with your simple_pick_demo.py
3. **Integrate with HITL apps** for interactive picking
4. **Generate task-specific pose sets** (e.g., shelf reaching, tool grasping)

## Resources

- VPoser Paper: https://arxiv.org/abs/1909.10462
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- human_body_prior: https://github.com/nghorbani/human_body_prior
- Habitat Humanoids README: `data/humanoids/humanoid_data/README.md`

---

**Ready to proceed?** I can create the complete pose generation script next!
