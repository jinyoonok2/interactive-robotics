"""
Visualize Affordance — Stage 2 of the Affordance Pipeline
==========================================================
Segments a specific part of the chosen object and visualizes the
affordance region on the captured RGB image with a grasp proposal.

Usage:
    cd habitat-lab
    python ../affordance-pipeline/visualize_affordance.py --object mug --part handle
    python ../affordance-pipeline/visualize_affordance.py --object power_drill --part chuck
    python ../affordance-pipeline/visualize_affordance.py --object hammer --part head
    python ../affordance-pipeline/visualize_affordance.py --object pitcher --part spout

    # With GraspNet (neural mode):
    python ../affordance-pipeline/visualize_affordance.py --object mug --part handle --method neural

Requires Stage 1 output in affordance-pipeline/output/
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False
    print("WARNING: open3d not installed. Install with: pip install open3d")

# ── Local imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from objects import (
    get_object, get_parts, validate_part,
    print_object_parts, get_object_names,
)

# ── Paths ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
INPUT_DIR    = PIPELINE_DIR / "output"
RESULTS_BASE = PIPELINE_DIR / "results"


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Visualize affordance for a specific object part",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object mug --part handle\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object hammer --part head\n"
            "  python ../affordance-pipeline/visualize_affordance.py --object pitcher --part spout --method neural"
        ),
    )
    parser.add_argument(
        "--object", required=True, choices=get_object_names(),
        help="Object placed in the scene (from Stage 1)",
    )
    parser.add_argument(
        "--part", required=True,
        help="Part to highlight (e.g., handle, body, rim, chuck, head)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Custom text prompt for language method (e.g., 'handle of the cup'). "
             "If not provided, uses the part name.",
    )
    parser.add_argument(
        "--box-threshold", type=float, default=0.25,
        help="Grounding DINO detection confidence threshold (language method)",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.25,
        help="Grounding DINO text matching threshold (language method)",
    )
    args = parser.parse_args()

    # Validate part against object
    try:
        validate_part(args.object, args.part)
    except ValueError as e:
        parser.error(str(e))

    return args


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraspPose:
    """A predicted grasp point with approach direction and orientation."""
    position: list
    approach_dir: list
    grasp_type: str
    confidence: float
    object_name: str
    part_name: str
    description: str
    grasp_rotation: list = None  # 3x3 rotation matrix [[r00..r02],[r10..r12],[r20..r22]]


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def compute_pca(points):
    """Compute PCA. Returns centroid, eigenvalues (desc), eigenvectors (rows)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return centroid, eigenvalues[idx], eigenvectors[:, idx].T


def find_vertical_axis(eigenvectors):
    """Find which PCA axis is most aligned with Y-up."""
    y_up = np.array([0, 1, 0])
    alignments = [abs(np.dot(eigenvectors[i], y_up)) for i in range(3)]
    vert_idx = np.argmax(alignments)
    vert_axis = eigenvectors[vert_idx].copy()
    if np.dot(vert_axis, y_up) < 0:
        vert_axis = -vert_axis
    return vert_idx, vert_axis


# ═══════════════════════════════════════════════════════════════════════════
# PART SEGMENTATION — one function per object type
# ═══════════════════════════════════════════════════════════════════════════

def segment_mug(points, pcd):
    """Segment mug into: handle, body, rim."""
    centroid, eigenvalues, eigenvectors = compute_pca(points)
    vert_idx, vert_axis = find_vertical_axis(eigenvectors)

    centered = points - centroid

    # Project out vertical component → horizontal plane
    horiz_proj = centered - np.outer(centered @ vert_axis, vert_axis)
    horiz_dist = np.linalg.norm(horiz_proj, axis=1)
    heights = centered @ vert_axis

    # Define rim FIRST (top 12% of points) - more conservative threshold
    rim_threshold = np.percentile(heights, 88)
    rim_mask = heights > rim_threshold

    # Define handle search region: middle 70% of height (exclude top rim and bottom)
    # This prevents rim contamination
    height_min = np.percentile(heights, 15)
    height_max = np.percentile(heights, 85)
    handle_search_mask = (heights >= height_min) & (heights <= height_max) & ~rim_mask

    # Step 1: Find candidate protrusion points (far from center axis)
    # Only search in the middle region, not in rim area
    median_dist = np.median(horiz_dist[handle_search_mask])
    std_dist = np.std(horiz_dist[handle_search_mask])
    
    # More aggressive threshold to only get strong protrusions
    handle_threshold = median_dist + 1.0 * std_dist
    candidate_mask = (horiz_dist > handle_threshold) & handle_search_mask

    # Step 2: Find the dominant direction of the protrusion
    # Stricter alignment to avoid rim spread
    if candidate_mask.sum() > 5:
        handle_dir = horiz_proj[candidate_mask].mean(axis=0)
        handle_dir_norm = np.linalg.norm(handle_dir)
        if handle_dir_norm > 1e-6:
            handle_dir /= handle_dir_norm
            # Stricter alignment threshold (0.5 instead of 0.3)
            # This ensures only points strongly aligned with handle direction
            horiz_unit = horiz_proj / (horiz_dist[:, None] + 1e-8)
            alignment = horiz_unit @ handle_dir
            
            # Also require minimum distance from center (avoid body points)
            min_handle_dist = median_dist + 0.5 * std_dist
            handle_mask = (candidate_mask & 
                          (alignment > 0.5) & 
                          (horiz_dist > min_handle_dist))
        else:
            handle_mask = candidate_mask
    else:
        handle_mask = candidate_mask

    # Additional cleanup: Use DBSCAN-style spatial clustering to separate handle
    # Remove isolated points that don't form a coherent handle cluster
    if HAS_O3D and handle_mask.sum() > 20:
        handle_indices = np.where(handle_mask)[0]
        handle_pcd_temp = pcd.select_by_index(handle_indices)
        
        # Cluster with DBSCAN (eps=0.015m = 1.5cm, min 10 points)
        labels = np.array(handle_pcd_temp.cluster_dbscan(eps=0.015, min_points=10))
        
        if len(labels) > 0 and labels.max() >= 0:
            # Keep only the largest cluster (the actual handle)
            largest_cluster = np.argmax(np.bincount(labels[labels >= 0]))
            valid_handle_local = (labels == largest_cluster)
            
            # Update handle mask to only include largest cluster
            handle_mask_new = np.zeros_like(handle_mask)
            handle_mask_new[handle_indices[valid_handle_local]] = True
            handle_mask = handle_mask_new

    # Body: everything not rim or handle
    body_mask = ~handle_mask & ~rim_mask

    return {
        "handle": np.where(handle_mask)[0],
        "body":   np.where(body_mask)[0],
        "rim":    np.where(rim_mask)[0],
    }


def segment_power_drill(points, pcd):
    """
    Segment power drill into: handle, body, chuck.

    The drill has a pistol-grip handle extending downward,
    a barrel-shaped body, and a chuck at the tip.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    y_up = np.array([0, 1, 0])
    heights = centered @ y_up

    # Handle: bottom portion (the grip hangs below the body)
    # The handle is the lower ~30% of the drill
    height_30 = np.percentile(heights, 30)
    handle_mask = heights < height_30

    # For the upper portion (body + chuck), find the barrel axis via PCA
    upper_mask = ~handle_mask
    upper_points = centered[upper_mask]

    chuck_local_mask = np.zeros(upper_mask.sum(), dtype=bool)

    if len(upper_points) > 20:
        upper_centroid = upper_points.mean(axis=0)
        upper_centered = upper_points - upper_centroid
        cov = np.cov(upper_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        barrel_axis = eigvecs[:, np.argmax(eigvals)]

        # Chuck: front 20% along barrel axis
        barrel_proj = upper_centered @ barrel_axis
        chuck_threshold = np.percentile(barrel_proj, 80)
        chuck_local_mask = barrel_proj > chuck_threshold

    upper_indices = np.where(upper_mask)[0]
    chuck_indices = upper_indices[chuck_local_mask]
    body_indices  = upper_indices[~chuck_local_mask]
    handle_indices = np.where(handle_mask)[0]

    return {
        "handle": handle_indices,
        "body":   body_indices,
        "chuck":  chuck_indices,
    }


def segment_pitcher(points, pcd):
    """
    Segment pitcher into: handle, spout, body, rim.

    The pitcher has a handle (lateral protrusion like a mug),
    a spout on the opposite side near the top, a body, and rim.
    """
    centroid, eigenvalues, eigenvectors = compute_pca(points)
    vert_idx, vert_axis = find_vertical_axis(eigenvectors)

    centered = points - centroid

    # Handle: lateral protrusion (same strategy as mug)
    horiz_proj = centered - np.outer(centered @ vert_axis, vert_axis)
    horiz_dist = np.linalg.norm(horiz_proj, axis=1)
    median_dist = np.median(horiz_dist)
    handle_threshold = median_dist + 0.7 * np.std(horiz_dist)
    protrusion_mask = horiz_dist > handle_threshold

    # Among protrusions, separate handle from spout
    # Handle is typically lower/mid-height, spout is near the top
    heights = centered @ vert_axis

    if protrusion_mask.sum() > 10:
        prot_heights = heights[protrusion_mask]
        prot_horiz = horiz_proj[protrusion_mask]

        # Find the dominant direction of protrusion
        prot_dir_mean = prot_horiz.mean(axis=0)
        prot_dir_norm = np.linalg.norm(prot_dir_mean)

        if prot_dir_norm > 1e-6:
            prot_dir = prot_dir_mean / prot_dir_norm
            # Points on the same side as mean direction → handle
            # Points on opposite side → spout
            dot_products = prot_horiz @ prot_dir
            handle_side = dot_products > 0
            spout_side = dot_products <= 0

            prot_indices = np.where(protrusion_mask)[0]
            handle_indices = prot_indices[handle_side]
            spout_indices  = prot_indices[spout_side]
        else:
            handle_indices = np.where(protrusion_mask)[0]
            spout_indices  = np.array([], dtype=int)
    else:
        handle_indices = np.where(protrusion_mask)[0]
        spout_indices  = np.array([], dtype=int)

    # If no spout found in protrusions, use top-front heuristic
    if len(spout_indices) < 5:
        # Spout is at the top, opposite side from handle
        handle_dir = np.zeros(3)
        if len(handle_indices) > 0:
            handle_pts = centered[handle_indices]
            handle_dir = handle_pts.mean(axis=0)
            handle_dir[1] = 0  # horizontal only
            handle_dir_norm = np.linalg.norm(handle_dir)
            if handle_dir_norm > 1e-6:
                handle_dir /= handle_dir_norm

        opp_dir = -handle_dir
        top_mask = heights > np.percentile(heights, 70)
        if np.linalg.norm(opp_dir) > 0.5:
            horiz_centered = centered.copy()
            horiz_centered[:, 1] = 0
            opp_scores = horiz_centered @ opp_dir
            spout_mask = top_mask & (opp_scores > np.percentile(opp_scores[top_mask], 70)) & ~protrusion_mask
        else:
            spout_mask = np.zeros(len(points), dtype=bool)
        spout_indices = np.where(spout_mask)[0]

    # Rim: topmost 12% not in handle/spout
    rim_threshold = np.percentile(heights, 88)
    assigned = np.zeros(len(points), dtype=bool)
    assigned[handle_indices] = True
    assigned[spout_indices] = True
    rim_mask = (heights > rim_threshold) & ~assigned
    rim_indices = np.where(rim_mask)[0]

    # Body: everything else
    assigned[rim_indices] = True
    body_indices = np.where(~assigned)[0]

    return {
        "handle": handle_indices,
        "spout":  spout_indices,
        "body":   body_indices,
        "rim":    rim_indices,
    }


def segment_hammer(points, pcd):
    """
    Segment hammer into: head, handle.

    The hammer has a long thin handle and a wider head at one end.
    """
    centroid, eigenvalues, eigenvectors = compute_pca(points)

    # Long axis = first PC (largest eigenvalue)
    long_axis = eigenvectors[0]
    centered = points - centroid

    # Project onto long axis
    long_proj = centered @ long_axis

    # Split into slices along the long axis to find the head
    n_bins = 10
    bin_edges = np.linspace(long_proj.min(), long_proj.max(), n_bins + 1)

    # Perpendicular spread per slice
    perp_spreads = []
    for b in range(n_bins):
        mask = (long_proj >= bin_edges[b]) & (long_proj < bin_edges[b + 1])
        if mask.sum() < 3:
            perp_spreads.append(0)
            continue
        bin_pts = centered[mask]
        perp = bin_pts - np.outer(bin_pts @ long_axis, long_axis)
        perp_spreads.append(np.std(np.linalg.norm(perp, axis=1)))

    perp_spreads = np.array(perp_spreads)

    # Head: slices with above-average perpendicular spread (wider cross-section)
    # Typically at one end of the long axis
    mean_spread = perp_spreads.mean()
    head_bins = perp_spreads > mean_spread * 1.2

    # Find head end (contiguous group of wide bins at one end)
    # Check first 3 bins vs last 3 bins
    start_spread = perp_spreads[:3].mean()
    end_spread = perp_spreads[-3:].mean()

    if start_spread > end_spread:
        # Head is at the start (low projection values)
        head_threshold = np.percentile(long_proj, 35)
        head_mask = long_proj < head_threshold
    else:
        # Head is at the end (high projection values)
        head_threshold = np.percentile(long_proj, 65)
        head_mask = long_proj > head_threshold

    handle_mask = ~head_mask

    return {
        "head":   np.where(head_mask)[0],
        "handle": np.where(handle_mask)[0],
    }


# Dispatcher
SEGMENTERS = {
    "mug":         segment_mug,
    "power_drill": segment_power_drill,
    "pitcher":     segment_pitcher,
    "hammer":      segment_hammer,
}


def segment_object(obj_name, points, pcd):
    """Segment an object into its parts. Returns {part_name: indices_array}."""
    segmenter = SEGMENTERS.get(obj_name)
    if segmenter is None:
        raise ValueError(f"No segmenter for '{obj_name}'")
    return segmenter(points, pcd)


# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRIC GRASP PROPOSAL
# ═══════════════════════════════════════════════════════════════════════════

def propose_grasp_geometric(obj_name, part_name, points, part_indices):
    """Propose a grasp pose for the selected part using geometric heuristics."""
    part_points = points[part_indices]
    all_centroid = points.mean(axis=0)
    part_centroid = part_points.mean(axis=0)

    # Default: approach from body centroid toward part center (side grasp)
    approach_dir = part_centroid - all_centroid
    approach_dir[1] = 0  # keep horizontal
    norm = np.linalg.norm(approach_dir)
    if norm > 1e-8:
        approach_dir /= norm
    else:
        approach_dir = np.array([1.0, 0.0, 0.0])

    grasp_type = "side"
    confidence = 0.7

    # Part-specific overrides
    if part_name == "rim":
        # Approach from above for rim grasps
        approach_dir = np.array([0.0, -1.0, 0.0])
        grasp_type = "top_down"
        confidence = 0.8

    elif part_name == "head" and obj_name == "hammer":
        # Approach from above for hammer head
        approach_dir = np.array([0.0, -1.0, 0.0])
        grasp_type = "top_down"
        confidence = 0.75

    elif part_name == "chuck":
        # Approach along barrel axis toward chuck
        _, eigenvalues, eigenvectors = compute_pca(points)
        barrel_axis = eigenvectors[0]
        # Orient toward chuck
        if np.dot(barrel_axis, part_centroid - all_centroid) < 0:
            barrel_axis = -barrel_axis
        approach_dir = barrel_axis
        grasp_type = "axial"
        confidence = 0.7

    elif part_name == "handle":
        grasp_type = "handle"
        confidence = 0.8

    elif part_name == "body":
        # Side grasp
        _, _, eigenvectors = compute_pca(points)
        # Use a horizontal PCA axis perpendicular to the vertical
        y_up = np.array([0, 1, 0])
        for i in range(3):
            if abs(np.dot(eigenvectors[i], y_up)) < 0.5:
                approach_dir = eigenvectors[i].copy()
                approach_dir[1] = 0
                norm = np.linalg.norm(approach_dir)
                if norm > 1e-8:
                    approach_dir /= norm
                break
        grasp_type = "side"
        confidence = 0.65

    elif part_name == "spout":
        grasp_type = "pinch"
        confidence = 0.6

    # Build full rotation matrix: col0=binormal, col1=baseline(fingers), col2=approach
    z_axis = approach_dir.copy()
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 1e-8:
        z_axis /= z_norm
    # Pick Y-axis (finger opening) perpendicular to approach
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_axis, world_up)) > 0.9:
        world_up = np.array([1.0, 0.0, 0.0])
    y_axis = np.cross(z_axis, world_up)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    grasp_rot = np.column_stack([x_axis, y_axis, z_axis])

    return GraspPose(
        position=part_centroid.tolist(),
        approach_dir=approach_dir.tolist(),
        grasp_type=grasp_type,
        confidence=confidence,
        object_name=obj_name,
        part_name=part_name,
        description=f"Geometric {grasp_type} grasp at {part_name} center "
                    f"({len(part_indices)} part points)",
        grasp_rotation=grasp_rot.tolist(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# NEURAL GRASP PROPOSAL (GraspNet) — Optional
# ═══════════════════════════════════════════════════════════════════════════

def propose_grasp_neural(obj_name, part_name, points, part_indices, metadata):
    """
    Run GraspNet inference and filter grasps to the selected part.
    Falls back to geometric if GraspNet is not available.
    """
    import torch

    GRASPNET_DIR = PIPELINE_DIR.parent / "graspnet-baseline"
    CHECKPOINT_PATH = GRASPNET_DIR / "checkpoints" / "checkpoint-rs.tar"

    if not CHECKPOINT_PATH.exists():
        print(f"  WARNING: GraspNet checkpoint not found at {CHECKPOINT_PATH}")
        print(f"  Falling back to geometric method")
        return propose_grasp_geometric(obj_name, part_name, points, part_indices)

    # Add GraspNet to path
    sys.path.insert(0, str(GRASPNET_DIR / "models"))
    sys.path.insert(0, str(GRASPNET_DIR / "dataset"))
    sys.path.insert(0, str(GRASPNET_DIR / "utils"))

    # Load depth and build cloud
    depth = np.load(INPUT_DIR / "depth_raw.npy")
    rgb = np.array(Image.open(INPUT_DIR / "rgb.png"), dtype=np.float32) / 255.0

    print("  Building workspace-cropped point cloud for GraspNet...")
    end_points, o3d_cloud = _build_graspnet_cloud(rgb, depth, metadata)

    print("  Loading GraspNet model...")
    from graspnet import GraspNet, pred_decode
    net = GraspNet(
        input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    print("  Running inference...")
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    from graspnetAPI import GraspGroup
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    print(f"  Raw predictions: {len(gg)} grasps")

    # Collision filter + NMS
    from collision_detector import ModelFreeCollisionDetector
    cloud_pts = np.array(o3d_cloud.points)
    mfcdetector = ModelFreeCollisionDetector(cloud_pts, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    gg.nms()
    gg.sort_by_score()
    print(f"  After filtering: {len(gg)} grasps")

    if len(gg) == 0:
        print("  WARNING: No grasps survived filtering, falling back to geometric")
        return propose_grasp_geometric(obj_name, part_name, points, part_indices)

    # Transform grasps to world frame
    world_positions, world_rotations = _grasps_to_world(gg, metadata)

    # ── Affordance-weighted grasp selection ────────────────────────
    # Instead of picking the highest GraspNet score near the part
    # centroid, we use a combined score that balances:
    #   1) GraspNet confidence (is the grasp physically feasible?)
    #   2) Proximity to the nearest affordance surface point
    #      (is the grasp actually ON the identified part?)
    #
    # This is object-agnostic: no per-object tuning needed.  The
    # part point cloud from DINO+SAM defines exactly where the
    # affordance surface is.

    part_points = points[part_indices]
    part_centroid = part_points.mean(axis=0)
    part_extent = part_points.max(axis=0) - part_points.min(axis=0)
    # Search radius: generous enough to find candidates, then let
    # the combined score handle fine selection.
    search_radius = max(part_extent.max() * 0.75, 0.08)

    # Find grasps near the part region
    dists_to_centroid = np.linalg.norm(world_positions - part_centroid, axis=1)
    near_mask = dists_to_centroid < search_radius
    near_indices = np.where(near_mask)[0]

    if len(near_indices) == 0:
        # Expand search
        search_radius *= 2
        near_mask = dists_to_centroid < search_radius
        near_indices = np.where(near_mask)[0]

    if len(near_indices) == 0:
        print(f"  WARNING: No GraspNet grasps near {part_name}, falling back to geometric")
        return propose_grasp_geometric(obj_name, part_name, points, part_indices)

    # ── Compute per-grasp proximity to nearest part surface point ──
    # For each candidate grasp position, find the distance to the
    # closest point in the part point cloud.  This avoids centroid
    # bias — a C-shaped handle's centroid is in mid-air, but we
    # want the grasp ON the surface.
    from scipy.spatial import cKDTree
    part_tree = cKDTree(part_points)
    candidate_positions = world_positions[near_indices]
    min_dists, _ = part_tree.query(candidate_positions)

    # Proximity score: exponential decay — sharply penalises grasps
    # that are far from any part surface point.
    SIGMA = 0.015  # 1.5 cm characteristic distance (universal, not per-object)
    proximity_scores = np.exp(-min_dists / SIGMA)

    # Normalised GraspNet scores (0–1 range)
    graspnet_scores = gg.scores[near_indices]
    gs_max = graspnet_scores.max()
    graspnet_norm = graspnet_scores / gs_max if gs_max > 0 else graspnet_scores

    # Combined score: proximity is weighted MORE than GraspNet score
    # so the grasp lands on the affordance region.  GraspNet still
    # prevents choosing infeasible grasps.
    ALPHA = 0.3  # weight for GraspNet score (0.3 = 30% feasibility, 70% proximity)
    combined = ALPHA * graspnet_norm + (1 - ALPHA) * proximity_scores
    best_local = np.argmax(combined)
    best_idx = near_indices[best_local]

    # Diagnostics
    print(f"  Affordance-weighted selection ({len(near_indices)} candidates):")
    print(f"    Search radius:       {search_radius:.3f}m")
    print(f"    Part surface points: {len(part_points)}")
    print(f"    Best GraspNet score: {graspnet_norm[best_local]:.4f}"
          f" (raw {graspnet_scores[best_local]:.4f})")
    print(f"    Nearest part dist:   {min_dists[best_local]*100:.1f}cm")
    print(f"    Proximity score:     {proximity_scores[best_local]:.4f}")
    print(f"    Combined score:      {combined[best_local]:.4f}")
    # Show runner-up for comparison
    if len(near_indices) > 1:
        runner_up = np.argsort(combined)[-2]
        print(f"    Runner-up dist:      {min_dists[runner_up]*100:.1f}cm, "
              f"GraspNet={graspnet_scores[runner_up]:.4f}, "
              f"combined={combined[runner_up]:.4f}")

    best_pos = world_positions[best_idx]
    best_rot = world_rotations[best_idx]
    approach_dir = best_rot[:, 2]  # Z-axis of grasp frame

    grasp_type = "top_down" if abs(approach_dir[1]) > 0.7 else "side"

    return GraspPose(
        position=best_pos.tolist(),
        approach_dir=approach_dir.tolist(),
        grasp_type=grasp_type,
        confidence=float(gg.scores[best_idx]),
        object_name=obj_name,
        part_name=part_name,
        description=f"GraspNet {grasp_type} grasp at {part_name} "
                    f"(combined={combined[best_local]:.4f}, "
                    f"dist={min_dists[best_local]*100:.1f}cm, "
                    f"{len(near_indices)} candidates)",
        grasp_rotation=best_rot.tolist(),
    )


def _build_graspnet_cloud(rgb, depth, metadata):
    """Build workspace-cropped point cloud for GraspNet (camera frame)."""
    import torch

    H, W = depth.shape
    fx = metadata["focal_length_px"]
    fy = fx
    cx, cy = metadata["principal_point"]

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    cloud_organized = np.stack([x, y, z], axis=-1)

    max_depth = metadata.get("max_depth_m", 10.0)
    valid_mask = (depth > 0) & (depth < max_depth)

    # Workspace crop around object
    sensor_R = np.array(metadata["sensor_rotation_matrix"], dtype=np.float32)
    sensor_t = np.array(metadata["sensor_position"], dtype=np.float32)
    opencv_to_habitat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)

    obj_cam_positions = []
    for obj in metadata["spawned_objects"]:
        wp = np.array(obj["position"], dtype=np.float32)
        cam_hab = sensor_R.T @ (wp - sensor_t)
        cam_cv = opencv_to_habitat.T @ cam_hab
        obj_cam_positions.append(cam_cv)

    obj_cam_positions = np.array(obj_cam_positions)
    margin = 0.4
    bounds = {
        ax: (obj_cam_positions[:, i].min() - margin, obj_cam_positions[:, i].max() + margin)
        for i, ax in enumerate("xyz")
    }

    workspace_mask = valid_mask
    for i, ax in enumerate("xyz"):
        lo, hi = bounds[ax]
        workspace_mask = workspace_mask & (cloud_organized[:, :, i] >= lo) & (cloud_organized[:, :, i] <= hi)

    cloud_masked = cloud_organized[workspace_mask]
    color_masked = rgb[workspace_mask]
    print(f"  Workspace points: {len(cloud_masked)} (cropped from {valid_mask.sum()})")

    NUM_POINTS = 20000
    np.random.seed(42)
    n = len(cloud_masked)
    if n >= NUM_POINTS:
        idxs = np.random.choice(n, NUM_POINTS, replace=False)
    else:
        idxs = np.concatenate([np.arange(n), np.random.choice(n, NUM_POINTS - n, replace=True)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_tensor = torch.from_numpy(cloud_masked[idxs][np.newaxis].astype(np.float32)).to(device)

    end_points = {"point_clouds": cloud_tensor, "cloud_colors": color_masked[idxs]}

    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    o3d_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    return end_points, o3d_cloud


def _grasps_to_world(gg, metadata):
    """Transform GraspNet grasps from camera (OpenCV) to world frame."""
    opencv_to_habitat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    sensor_R = np.array(metadata["sensor_rotation_matrix"], dtype=np.float32)
    sensor_t = np.array(metadata["sensor_position"], dtype=np.float32)
    R_combined = sensor_R @ opencv_to_habitat

    world_positions = (R_combined @ gg.translations.T).T + sensor_t
    world_rotations = np.array([R_combined @ r for r in gg.rotation_matrices])

    return world_positions, world_rotations


# ═══════════════════════════════════════════════════════════════════════════
# PROJECTION: 3D → 2D
# ═══════════════════════════════════════════════════════════════════════════

def project_3d_to_2d(point_3d, sensor_R, sensor_t, fx, cx, cy, width, height):
    """
    Project a world 3D point to 2D image coordinates.

    Args:
        point_3d: [x, y, z] world position
        sensor_R: 3x3 camera-to-world rotation matrix
        sensor_t: [x, y, z] camera world position
        fx: focal length in pixels
        cx, cy: principal point
        width, height: image dimensions

    Returns:
        (u, v) pixel coordinates, or None if behind camera
    """
    # World → camera (Habitat frame)
    R_w2c = np.array(sensor_R).T
    p_cam = R_w2c @ (np.array(point_3d) - np.array(sensor_t))

    x_c, y_c, z_c = p_cam

    # Camera looks along -Z; points with z > 0 are behind camera
    if z_c > -0.01:
        return None

    u = fx * (x_c / (-z_c)) + cx
    v = fx * (-y_c / (-z_c)) + cy

    if 0 <= u < width and 0 <= v < height:
        return (int(u), int(v))
    return None


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Part colors (BGR for OpenCV)
PART_COLORS = {
    "handle": (100, 255, 150),   # green
    "body":   (255, 200, 100),   # blue
    "rim":    (100, 200, 255),   # orange
    "chuck":  (150, 100, 255),   # pink/magenta
    "head":   (100, 255, 255),   # yellow
    "spout":  (255, 150, 200),   # light purple
}


def draw_affordance(img, points, part_indices, grasp, sensor_R, sensor_t,
                    fx, cx, cy, width, height, obj_name, part_name):
    """
    Draw affordance visualization on the RGB image:
      - Dim grey overlay for non-selected points (object outline)
      - Bright color overlay for selected part
      - Crosshair at grasp point
      - Arrow for approach direction
    """
    color = PART_COLORS.get(part_name, (200, 200, 200))
    all_indices = set(range(len(points)))
    part_set = set(part_indices.tolist())
    non_part_indices = np.array(list(all_indices - part_set))

    # Draw non-part points as dim outline
    overlay = img.copy()
    for idx in non_part_indices:
        px = project_3d_to_2d(points[idx], sensor_R, sensor_t, fx, cx, cy, width, height)
        if px is not None:
            cv2.circle(overlay, px, 3, (100, 100, 100), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # Draw selected part as bright overlay
    overlay = img.copy()
    drawn_part = 0
    for idx in part_indices:
        px = project_3d_to_2d(points[idx], sensor_R, sensor_t, fx, cx, cy, width, height)
        if px is not None:
            cv2.circle(overlay, px, 5, color, -1)
            drawn_part += 1
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Draw grasp crosshair
    grasp_pos = np.array(grasp.position)
    grasp_px = project_3d_to_2d(grasp_pos, sensor_R, sensor_t, fx, cx, cy, width, height)

    if grasp_px is not None:
        cv2.drawMarker(img, grasp_px, (255, 255, 255), cv2.MARKER_CROSS, 16, 2)

        # Draw approach arrow
        approach_end = grasp_pos + np.array(grasp.approach_dir) * 0.15
        end_px = project_3d_to_2d(approach_end, sensor_R, sensor_t, fx, cx, cy, width, height)
        if end_px is not None:
            cv2.arrowedLine(img, grasp_px, end_px, (255, 255, 255), 2, tipLength=0.3)

    # Draw label
    label = f"{obj_name}/{part_name}: {grasp.grasp_type} (conf={grasp.confidence:.0%})"
    if grasp_px is not None:
        lx, ly = grasp_px[0] + 15, grasp_px[1] - 15
    else:
        lx, ly = 10, 30

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0, 0, 0), -1)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    print(f"  Projected {drawn_part}/{len(part_indices)} part points onto image")

    return img


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    obj_name  = args.object
    part_name = args.part
    method    = "language"  # DINO + SAM + GraspNet

    # Results go into results/language/
    RESULTS_DIR = RESULTS_BASE / method
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_header("Visualize Affordance — Stage 2")
    print(f"  Object: {obj_name}")
    print(f"  Part:   {part_name}")
    print(f"  Output: {RESULTS_DIR}/")

    # ── Load Stage 1 outputs ────────────────────────────────────────
    print_header("Loading Stage 1 data")

    meta_path = INPUT_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"  ERROR: {meta_path} not found")
        print(f"  Run Stage 1 first: python ../affordance-pipeline/scene_capture.py --object {obj_name}")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    # Validate that the same object was captured in Stage 1
    captured_obj = metadata.get("object_name", "")
    if captured_obj != obj_name:
        print(f"  WARNING: Stage 1 captured '{captured_obj}' but you specified '{obj_name}'")
        print(f"  Re-run Stage 1: python ../affordance-pipeline/scene_capture.py --object {obj_name}")
        sys.exit(1)

    # Load RGB
    rgb_path = INPUT_DIR / "rgb.png"
    rgb = np.array(Image.open(rgb_path))
    height, width = rgb.shape[:2]
    print(f"  RGB: {rgb.shape}")

    # Load object point cloud
    ply_files = list(INPUT_DIR.glob(f"object_{obj_name}_sem*.ply"))
    if not ply_files:
        print(f"  ERROR: No point cloud found for '{obj_name}' in {INPUT_DIR}")
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(str(ply_files[0]))
    points = np.asarray(pcd.points)
    print(f"  Point cloud: {len(points)} points from {ply_files[0].name}")

    if len(points) < 20:
        print(f"  ERROR: Too few points ({len(points)}). Object may not be visible.")
        sys.exit(1)

    # Camera parameters
    sensor_R = metadata["sensor_rotation_matrix"]
    sensor_t = metadata["sensor_position"]
    fx = metadata["focal_length_px"]
    cx, cy = metadata["principal_point"]

    # ══════════════════════════════════════════════════════════════════
    # LANGUAGE METHOD: Grounding DINO + SAM
    # ══════════════════════════════════════════════════════════════════
    if method == "language":
        from language_segment import segment_part_by_language, draw_segmentation_overlay

        # Build text prompt
        text_prompt = args.prompt
        if text_prompt is None:
            # Auto-generate prompt from object + part
            obj_cfg = get_object(obj_name)
            part_desc = obj_cfg["parts"][part_name]
            text_prompt = f"{part_name} of the {obj_name}"
        print(f"  Text prompt: \"{text_prompt}\"")

        # Load depth
        depth = np.load(INPUT_DIR / "depth_raw.npy")
        print(f"  Depth: {depth.shape}")

        # Load raw semantic mask (if available)
        semantic_raw = None
        semantic_id = None
        sem_path = INPUT_DIR / "semantic_raw.npy"
        if sem_path.exists():
            semantic_raw = np.load(sem_path)
            # Get semantic ID from metadata
            if metadata.get("spawned_objects"):
                semantic_id = metadata["spawned_objects"][0].get("semantic_id")
            print(f"  Semantic mask: loaded (object id={semantic_id})")
        else:
            print(f"  Semantic mask: not found — using SAM-only mode")

        # ── Segment with language model ─────────────────────────────
        print_header(f"Language-guided segmentation: '{text_prompt}'")

        seg_result = segment_part_by_language(
            rgb, depth, text_prompt, metadata,
            part_name=part_name,
            obj_name=obj_name,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            semantic_raw=semantic_raw,
            semantic_id=semantic_id,
        )

        if seg_result is None:
            print(f"\n  ERROR: Language segmentation failed for '{text_prompt}'")
            print(f"  Try a different prompt with --prompt, e.g.:")
            print(f"    --prompt 'handle'")
            print(f"    --prompt '{part_name}'")
            print(f"    --prompt '{part_name} of the {obj_name}'")
            sys.exit(1)

        part_3d_points = seg_result["world_points"]
        mask = seg_result["mask"]
        bbox = seg_result["bbox"]

        print(f"  Segmented {len(part_3d_points)} 3D points for '{part_name}'")

        # ── Propose grasp with GraspNet ──────────────────────────────
        print_header(f"Proposing grasp for {part_name}")

        part_indices = np.arange(len(part_3d_points))
        grasp = propose_grasp_neural(obj_name, part_name, part_3d_points, part_indices, metadata)

        print(f"  Type:       {grasp.grasp_type}")
        print(f"  Position:   ({grasp.position[0]:.4f}, {grasp.position[1]:.4f}, {grasp.position[2]:.4f})")
        print(f"  Approach:   ({grasp.approach_dir[0]:.3f}, {grasp.approach_dir[1]:.3f}, {grasp.approach_dir[2]:.3f})")
        print(f"  Confidence: {grasp.confidence:.0%}")
        print(f"  {grasp.description}")

        # ── Visualize ───────────────────────────────────────────────
        print_header("Rendering visualization")

        part_color = PART_COLORS.get(part_name, (100, 255, 150))

        # Draw SAM mask overlay with contours
        img_annotated = draw_segmentation_overlay(
            rgb, mask, bbox, part_name, color=part_color,
        )

        # Draw grasp crosshair on the annotated image
        grasp_pos = np.array(grasp.position)
        grasp_px = project_3d_to_2d(grasp_pos, sensor_R, sensor_t, fx, cx, cy, width, height)

        if grasp_px is not None:
            cv2.drawMarker(img_annotated, grasp_px, (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            approach_end = grasp_pos + np.array(grasp.approach_dir) * 0.15
            end_px = project_3d_to_2d(approach_end, sensor_R, sensor_t,
                                       fx, cx, cy, width, height)
            if end_px is not None:
                cv2.arrowedLine(img_annotated, grasp_px, end_px,
                                (255, 255, 255), 2, tipLength=0.3)

        # Add grasp info label
        label = f"{obj_name}/{part_name}: {grasp.grasp_type} (conf={grasp.confidence:.0%})"
        if grasp_px is not None:
            lx, ly = grasp_px[0] + 15, grasp_px[1] - 15
        else:
            lx, ly = 10, height - 40
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_annotated, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0, 0, 0), -1)
        cv2.putText(img_annotated, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, part_color, 2, cv2.LINE_AA)

        # Save single image
        out_name = f"affordance_{obj_name}_{part_name}.png"
        out_path = RESULTS_DIR / out_name
        Image.fromarray(img_annotated).save(out_path)
        print(f"  Saved: {out_path}")

        # Save comparison: original | SAM mask | overlay
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Scene: {obj_name}", fontsize=16)
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f"SAM Mask: \"{text_prompt}\"", fontsize=14)
        axes[1].axis('off')

        axes[2].imshow(img_annotated)
        obj_cfg = get_object(obj_name)
        part_desc = obj_cfg["parts"][part_name]
        axes[2].set_title(
            f"Affordance: {part_name} ({len(part_3d_points)} 3D pts)\n"
            f"Grasp: {grasp.grasp_type} @ {grasp.confidence:.0%}",
            fontsize=13,
        )
        axes[2].axis('off')

        plt.tight_layout()
        compare_name = f"comparison_{obj_name}_{part_name}.png"
        compare_path = RESULTS_DIR / compare_name
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {compare_path}")

        # Save grasp instruction JSON
        grasp_json = {
            "object_name": obj_name,
            "part_name": part_name,
            "method": method,
            "text_prompt": text_prompt,
            "detection_confidence": seg_result["detection_confidence"],
            "detection_label": seg_result["detection_label"],
            "mask_pixels": seg_result["sam_mask_pixels"],
            "part_3d_points": len(part_3d_points),
            "total_object_points": len(points),
            "grasp": asdict(grasp),
        }
        json_path = RESULTS_DIR / "grasp_poses.json"
        with open(json_path, "w") as f:
            json.dump(grasp_json, f, indent=2)
        print(f"  Saved: {json_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print_header("STAGE 2 COMPLETE")
    obj_cfg = get_object(obj_name)
    part_desc = obj_cfg["parts"][part_name]
    print(f"  Object: {obj_cfg['display_name']}")
    print(f"  Part:   {part_name} ({part_desc})")
    print(f"  Grasp:  {grasp.grasp_type} @ confidence {grasp.confidence:.0%}")
    print(f"  Output: {RESULTS_DIR}/")

    for p in sorted(RESULTS_DIR.iterdir()):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.name:45s} ({size_kb:.1f} KB)")

    # Suggest other parts
    other_parts = [p for p in get_parts(obj_name) if p != part_name]
    if other_parts:
        print(f"\n  Try other parts:")
        for p in other_parts:
            print(f"    python ../affordance-pipeline/visualize_affordance.py "
                  f"--object {obj_name} --part {p}")


if __name__ == "__main__":
    main()
