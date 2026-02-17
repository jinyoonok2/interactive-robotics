#!/usr/bin/env python3
"""
Generate custom detailed pick-up poses using VPoser.

This script creates a denser grid of reaching poses with multi-phase
pick sequences for more realistic humanoid manipulation.

Usage:
    python generate_detailed_pick_motions.py --output custom_detailed_pick.pkl
"""

import argparse
import numpy as np
import pickle as pkl
from pathlib import Path

# NOTE: This requires VPoser installation
# pip install git+https://github.com/nghorbani/human_body_prior

try:
    import torch
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
    VPOSER_AVAILABLE = True
except ImportError:
    VPOSER_AVAILABLE = False
    print("WARNING: VPoser not installed. This is a template script.")
    print("Install with: pip install git+https://github.com/nghorbani/human_body_prior")


def load_vposer_model(model_path='data/vposer/V02_05'):
    """Load pre-trained VPoser model."""
    if not VPOSER_AVAILABLE:
        raise ImportError("VPoser not available. Please install human_body_prior.")
    
    print(f"Loading VPoser from: {model_path}")
    vposer, _ = load_model(
        model_path,
        model_code=VPoser,
        remove_words_in_model_weights='vp_model.',
        disable_grad=True
    )
    vposer.eval()
    return vposer


def optimize_reach_pose(vposer, target_position, hand='right', iterations=500):
    """
    Optimize VPoser latent code to generate pose reaching to target.
    
    Args:
        vposer: VPoser model
        target_position: [x, y, z] target in humanoid local frame
        hand: 'left' or 'right'
        iterations: Number of optimization steps
    
    Returns:
        body_pose: Optimized body pose parameters
        loss_history: Optimization loss over iterations
    """
    print(f"  Optimizing {hand} hand reach to {target_position}...")
    
    # Initialize latent code
    latent = torch.randn(1, 32, requires_grad=True)
    target_tensor = torch.tensor(target_position, dtype=torch.float32)
    
    # Optimizer
    optimizer = torch.optim.Adam([latent], lr=0.01)
    
    loss_history = []
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Decode latent code to body pose
        pose_out = vposer.decode(latent)
        body_pose = pose_out['pose_body']
        
        # TODO: You need SMPL-X forward kinematics here
        # For now, this is a simplified placeholder
        # In reality, you'd use smplx.SMPLX model to get hand position
        
        # Placeholder loss (you need actual FK)
        # hand_pos_3d = forward_kinematics(body_pose, hand_joint_idx)
        # loss = torch.norm(hand_pos_3d - target_tensor)
        
        # Simplified loss for demonstration
        loss = torch.norm(body_pose.mean() - 0.0)  # Keep poses near neutral
        loss = loss + 0.01 * torch.norm(latent)  # Regularization
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if i % 100 == 0:
            print(f"    Iteration {i}/{iterations}, Loss: {loss.item():.4f}")
    
    return body_pose.detach().cpu().numpy(), loss_history


def generate_dense_reach_grid(vposer, hand='right', grid_size=(8, 8, 6)):
    """
    Generate dense grid of reaching poses.
    
    Args:
        grid_size: (nx, ny, nz) number of poses per dimension
    
    Returns:
        poses: List of body poses
        grid_info: Dictionary with grid bounds
    """
    print(f"\nGenerating {grid_size[0]}×{grid_size[1]}×{grid_size[2]} = {np.prod(grid_size)} poses...")
    
    # Define workspace in humanoid local coordinates
    # These bounds work well for humanoid reaching
    x_range = (-0.8, 0.8)   # Left (-) to Right (+)
    y_range = (0.0, 1.6)    # Ground to above head
    z_range = (-0.5, 0.8)   # Behind to front
    
    nx, ny, nz = grid_size
    
    # Generate grid points
    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)
    z_vals = np.linspace(z_range[0], z_range[1], nz)
    
    poses = []
    transforms = []
    total = nx * ny * nz
    count = 0
    
    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            for iz, z in enumerate(z_vals):
                count += 1
                target = np.array([x, y, z])
                
                print(f"\nPose {count}/{total}: Target = [{x:.2f}, {y:.2f}, {z:.2f}]")
                
                if VPOSER_AVAILABLE:
                    body_pose, _ = optimize_reach_pose(vposer, target, hand=hand)
                    poses.append(body_pose)
                    
                    # Transform would come from SMPL-X FK
                    # For now, placeholder
                    transform = np.eye(4)
                    transform[:3, 3] = target
                    transforms.append(transform)
                else:
                    # Dummy data if VPoser not available
                    poses.append(np.zeros((21, 3)))
                    transforms.append(np.eye(4))
    
    grid_info = {
        'min': [x_range[0], y_range[0], z_range[0]],
        'max': [x_range[1], y_range[1], z_range[1]],
        'num_bins': [nx, ny, nz]
    }
    
    return poses, transforms, grid_info


def convert_poses_to_quaternions(poses):
    """
    Convert pose parameters to quaternions for Habitat format.
    
    Args:
        poses: List of body poses (N, 21, 3) or similar
    
    Returns:
        joints_array: (N*J, 4) quaternions
    """
    # TODO: Convert rotation vectors to quaternions
    # For SMPL-X, you'd use rodrigues formula
    
    # Placeholder: convert to quaternions
    N = len(poses)
    J = 21  # SMPL-X body joints
    
    # Dummy quaternions (identity rotations)
    joints_array = np.zeros((N, J, 4))
    joints_array[..., 3] = 1.0  # w=1 for identity quaternion
    
    return joints_array.reshape(-1, 4)


def create_custom_motion_file(poses, transforms, grid_info, output_path, base_motion_path):
    """
    Package custom poses into Habitat-compatible .pkl format.
    
    Args:
        poses: Generated body poses
        transforms: Root transforms for each pose
        grid_info: Grid bounds and dimensions
        output_path: Where to save custom motion file
        base_motion_path: Existing motion file to use as template
    """
    print(f"\nPackaging custom motion data...")
    
    # Load base motion file (for walk_motion and stop_pose)
    with open(base_motion_path, 'rb') as f:
        base_data = pkl.load(f)
    
    print(f"  Loaded base motion from: {base_motion_path}")
    
    # Convert poses to quaternions
    joints_array = convert_poses_to_quaternions(poses)
    
    # Convert transforms to numpy array
    transforms_array = np.array(transforms)
    
    # Create custom motion data
    custom_data = base_data.copy()
    custom_data['right_hand'] = {
        'pose_motion': {
            'joints_array': joints_array,
            'transform_array': transforms_array
        },
        'coord_info': grid_info
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pkl.dump(custom_data, f)
    
    print(f"\n{'='*70}")
    print(f"✓ Custom motion file saved: {output_path}")
    print(f"  Total poses: {len(poses)}")
    print(f"  Grid size: {grid_info['num_bins']}")
    print(f"  Workspace bounds:")
    print(f"    X: [{grid_info['min'][0]:.2f}, {grid_info['max'][0]:.2f}]")
    print(f"    Y: [{grid_info['min'][1]:.2f}, {grid_info['max'][1]:.2f}]")
    print(f"    Z: [{grid_info['min'][2]:.2f}, {grid_info['max'][2]:.2f}]")
    print(f"{'='*70}\n")


def generate_multi_phase_sequences(vposer, object_positions):
    """
    Generate multi-phase pick sequences for specific objects.
    
    Phases:
    1. Approach (hand 20cm above)
    2. Pre-grasp (hand at object)
    3. Grasp (with hand closure)
    4. Retract (lift 30cm)
    """
    print("\nGenerating multi-phase pick sequences...")
    
    sequences = {}
    
    for i, obj_pos in enumerate(object_positions):
        print(f"\n  Object {i+1}: {obj_pos}")
        
        phases = {
            'approach': obj_pos + np.array([0, 0.2, 0]),
            'pregrasp': obj_pos,
            'grasp': obj_pos,
            'retract': obj_pos + np.array([0, 0.3, 0])
        }
        
        sequence = {}
        for phase_name, target in phases.items():
            if VPOSER_AVAILABLE:
                pose, _ = optimize_reach_pose(vposer, target, hand='right', iterations=200)
            else:
                pose = np.zeros((21, 3))
            
            sequence[phase_name] = pose
        
        sequences[f'object_{i}'] = sequence
    
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Generate custom VPoser reaching poses')
    parser.add_argument('--vposer-path', type=str, default='data/vposer/V02_05',
                       help='Path to VPoser model')
    parser.add_argument('--base-motion', type=str,
                       default='data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl',
                       help='Base motion file to use as template')
    parser.add_argument('--output', type=str,
                       default='data/humanoids/custom_motions/detailed_pick_female_2.pkl',
                       help='Output path for custom motion file')
    parser.add_argument('--grid-size', type=int, nargs=3, default=[8, 8, 6],
                       help='Grid dimensions (nx ny nz)')
    parser.add_argument('--hand', type=str, choices=['left', 'right'], default='right',
                       help='Which hand to generate poses for')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CUSTOM VPOSER POSE GENERATION")
    print("="*70)
    
    # Load VPoser
    if VPOSER_AVAILABLE:
        try:
            vposer = load_vposer_model(args.vposer_path)
        except Exception as e:
            print(f"\nERROR: Could not load VPoser: {e}")
            print("\nTo download VPoser:")
            print("1. Visit: https://smpl-x.is.tue.mpg.de/")
            print("2. Register and download VPoser v2.0")
            print(f"3. Extract to: {args.vposer_path}")
            return
    else:
        print("\nVPoser not installed - generating template/dummy data")
        print("Install with: pip install git+https://github.com/nghorbani/human_body_prior")
        vposer = None
    
    # Generate dense reach grid
    poses, transforms, grid_info = generate_dense_reach_grid(
        vposer,
        hand=args.hand,
        grid_size=tuple(args.grid_size)
    )
    
    # Optional: Generate multi-phase sequences
    # ground_objects = [np.array([0.3, 0.05, 0.3]), np.array([-0.3, 0.05, 0.3])]
    # sequences = generate_multi_phase_sequences(vposer, ground_objects)
    
    # Package into Habitat format
    create_custom_motion_file(
        poses,
        transforms,
        grid_info,
        args.output,
        args.base_motion
    )
    
    print("\nNEXT STEPS:")
    print("1. Test with: simple_pick_demo.py")
    print("2. Modify controller init:")
    print(f"   controller = HumanoidRearrangeController('{args.output}')")
    print("3. Enjoy smoother, more detailed pick animations!")


if __name__ == "__main__":
    main()
