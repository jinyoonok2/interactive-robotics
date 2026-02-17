#!/usr/bin/env python3
"""
Analyze existing VPoser poses in motion data files.
Shows workspace coverage, pose distribution, and identifies gaps.

This helps you understand what new poses to generate!
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_motion_data(motion_path):
    """Load and parse motion data file."""
    print(f"Loading: {motion_path}")
    with open(motion_path, 'rb') as f:
        data = pkl.load(f)
    return data


def analyze_hand_workspace(hand_data, hand_name='right_hand'):
    """Analyze the reachable workspace from hand poses."""
    
    if hand_data is None:
        print(f"  No {hand_name} data found!")
        return
    
    coord_info = hand_data['coord_info']
    
    # coord_info might be a numpy array wrapping a dict
    if hasattr(coord_info, 'item'):
        coord_info = coord_info.item()
    
    print(f"\n  {hand_name.upper()} WORKSPACE:")
    print(f"  ─────────────────────")
    print(f"  Grid dimensions: {coord_info['num_bins']}")
    print(f"  Total poses: {np.prod(coord_info['num_bins'])}")
    print(f"\n  Workspace bounds (in humanoid local frame):")
    print(f"    X (left/right): [{coord_info['min'][0]:.3f}, {coord_info['max'][0]:.3f}] = {coord_info['max'][0] - coord_info['min'][0]:.3f}m range")
    print(f"    Y (height):     [{coord_info['min'][1]:.3f}, {coord_info['max'][1]:.3f}] = {coord_info['max'][1] - coord_info['min'][1]:.3f}m range")
    print(f"    Z (depth):      [{coord_info['min'][2]:.3f}, {coord_info['max'][2]:.3f}] = {coord_info['max'][2] - coord_info['min'][2]:.3f}m range")
    
    # Calculate workspace volume
    volume = (coord_info['max'][0] - coord_info['min'][0]) * \
             (coord_info['max'][1] - coord_info['min'][1]) * \
             (coord_info['max'][2] - coord_info['min'][2])
    
    print(f"\n  Workspace volume: {volume:.3f} m³")
    
    # Pose density
    density = np.prod(coord_info['num_bins']) / volume
    print(f"  Pose density: {density:.1f} poses/m³")
    
    # Resolution per dimension
    nx, ny, nz = coord_info['num_bins']
    dx = (coord_info['max'][0] - coord_info['min'][0]) / (nx - 1)
    dy = (coord_info['max'][1] - coord_info['min'][1]) / (ny - 1)
    dz = (coord_info['max'][2] - coord_info['min'][2]) / (nz - 1)
    
    print(f"\n  Spacing between poses:")
    print(f"    X: {dx*100:.1f}cm")
    print(f"    Y: {dy*100:.1f}cm")
    print(f"    Z: {dz*100:.1f}cm")
    
    return coord_info


def visualize_workspace_3d(coord_info, hand_name='right_hand'):
    """Create 3D visualization of pose workspace."""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid points
    nx, ny, nz = coord_info['num_bins']
    x_vals = np.linspace(coord_info['min'][0], coord_info['max'][0], nx)
    y_vals = np.linspace(coord_info['min'][1], coord_info['max'][1], ny)
    z_vals = np.linspace(coord_info['min'][2], coord_info['max'][2], nz)
    
    # Create 3D grid
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # Plot points
    ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
               c='blue', marker='o', alpha=0.3, s=20, label='Reach poses')
    
    # Draw workspace bounds
    mins = coord_info['min']
    maxs = coord_info['max']
    
    # Draw bounding box
    corners = [
        [mins[0], mins[1], mins[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], maxs[1], mins[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], maxs[2]],
        [mins[0], maxs[1], maxs[2]],
    ]
    
    # Draw edges
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom
        (4,5), (5,6), (6,7), (7,4),  # Top
        (0,4), (1,5), (2,6), (3,7)   # Vertical
    ]
    
    for edge in edges:
        points = [corners[edge[0]], corners[edge[1]]]
        ax.plot3D(*zip(*points), 'r-', linewidth=2)
    
    # Plot humanoid origin
    ax.scatter([0], [0.9], [0], c='red', marker='*', s=500, label='Humanoid base')
    
    # Ground plane
    ground_x = np.linspace(mins[0], maxs[0], 10)
    ground_z = np.linspace(mins[2], maxs[2], 10)
    Ground_X, Ground_Z = np.meshgrid(ground_x, ground_z)
    Ground_Y = np.zeros_like(Ground_X)
    ax.plot_surface(Ground_X, Ground_Y, Ground_Z, alpha=0.1, color='gray')
    
    ax.set_xlabel('X (Left/Right) [m]')
    ax.set_ylabel('Y (Height) [m]')
    ax.set_zlabel('Z (Front/Back) [m]')
    ax.set_title(f'{hand_name.upper()} Reachable Workspace ({np.prod(coord_info["num_bins"])} poses)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{hand_name}_workspace_3d.png', dpi=150)
    print(f"\n  ✓ Saved visualization: {hand_name}_workspace_3d.png")


def identify_gaps(coord_info):
    """Identify areas where more poses would be beneficial."""
    
    print(f"\n  RECOMMENDATIONS:")
    print(f"  ────────────────")
    
    nx, ny, nz = coord_info['num_bins']
    total_poses = np.prod(coord_info['num_bins'])
    
    # Check if ground level is covered
    if coord_info['min'][1] > 0.1:
        print(f"  ⚠ Ground pickups NOT well covered (min Y = {coord_info['min'][1]:.2f}m)")
        print(f"    → Add poses with Y starting from 0.0m")
    else:
        print(f"  ✓ Ground level covered (Y from {coord_info['min'][1]:.2f}m)")
    
    # Check pose density
    dx = (coord_info['max'][0] - coord_info['min'][0]) / (nx - 1)
    dy = (coord_info['max'][1] - coord_info['min'][1]) / (ny - 1)
    dz = (coord_info['max'][2] - coord_info['min'][2]) / (nz - 1)
    avg_spacing = (dx + dy + dz) / 3
    
    if avg_spacing > 0.15:
        print(f"  ⚠ Low pose density (avg spacing: {avg_spacing*100:.1f}cm)")
        print(f"    → Increase grid to 8×8×6 or higher for smoother motion")
    else:
        print(f"  ✓ Good pose density (avg spacing: {avg_spacing*100:.1f}cm)")
    
    # Suggest improvements
    print(f"\n  SUGGESTED IMPROVEMENTS:")
    if total_poses < 200:
        suggested_grid = (8, 8, 6)
        print(f"  • Increase to {suggested_grid} = {np.prod(suggested_grid)} poses")
    
    if coord_info['min'][1] > 0.1:
        print(f"  • Extend Y range to [0.0, {coord_info['max'][1]:.2f}] for ground pickups")
    
    if coord_info['max'][0] - coord_info['min'][0] < 1.2:
        print(f"  • Widen X range to [-0.8, 0.8] for side reaching")


def compare_hands(data):
    """Compare left and right hand workspaces."""
    
    print(f"\n{'='*70}")
    print("COMPARING LEFT vs RIGHT HAND")
    print(f"{'='*70}")
    
    left_info = data.get('left_hand', {}).get('coord_info') if 'left_hand' in data else None
    right_info = data.get('right_hand', {}).get('coord_info') if 'right_hand' in data else None
    
    # Handle numpy array wrapping
    if left_info is not None and hasattr(left_info, 'item'):
        left_info = left_info.item()
    if right_info is not None and hasattr(right_info, 'item'):
        right_info = right_info.item()
    
    if left_info is None:
        print("  Left hand: NOT AVAILABLE")
    else:
        print(f"  Left hand:  {left_info['num_bins']} grid = {np.prod(left_info['num_bins'])} poses")
    
    if right_info is None:
        print("  Right hand: NOT AVAILABLE")
    else:
        print(f"  Right hand: {right_info['num_bins']} grid = {np.prod(right_info['num_bins'])} poses")
    
    if left_info and right_info:
        left_bins = np.array(left_info["num_bins"])
        right_bins = np.array(right_info["num_bins"])
        if np.array_equal(left_bins, right_bins):
            print("  ✓ Symmetric workspaces")
        else:
            print("  ⚠ Asymmetric workspaces - may cause unbalanced reaching")


def main():
    import sys
    
    if len(sys.argv) < 2:
        motion_file = "data/humanoids/humanoid_data/female_2/female_2_motion_data_smplx.pkl"
        print(f"Using default: {motion_file}")
    else:
        motion_file = sys.argv[1]
    
    print("="*70)
    print("MOTION DATA WORKSPACE ANALYZER")
    print("="*70)
    
    # Load data
    try:
        data = load_motion_data(motion_file)
    except FileNotFoundError:
        print(f"\nERROR: File not found: {motion_file}")
        print("\nUsage: python analyze_motion_workspace.py [motion_file.pkl]")
        return
    
    # Check what's in the file
    print(f"\nData keys: {list(data.keys())}")
    
    # Analyze each hand
    for hand_name in ['left_hand', 'right_hand']:
        if hand_name in data:
            print(f"\n{'='*70}")
            coord_info = analyze_hand_workspace(data[hand_name], hand_name)
            identify_gaps(coord_info)
            
            # Visualize
            try:
                visualize_workspace_3d(coord_info, hand_name)
            except Exception as e:
                print(f"  Could not create visualization: {e}")
    
    # Compare hands
    compare_hands(data)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("─"*70)
    print("1. Review the workspace bounds above")
    print("2. Check the visualization images (*.png)")
    print("3. Decide if you need:")
    print("   • Denser grid (more poses)")
    print("   • Larger workspace (extend bounds)")
    print("   • Ground-level poses (lower Y)")
    print("   • Multi-phase sequences (approach, grasp, retract)")
    print("\n4. Generate custom poses:")
    print("   python generate_detailed_pick_motions.py --grid-size 8 8 6")
    print("="*70)


if __name__ == "__main__":
    main()
