"""
GraspPlanner — Geometric and neural grasp proposal.

Handles:
  - Geometric grasp heuristics (PCA-based approach directions)
  - GraspNet neural grasp inference
  - Affordance-weighted grasp selection (proximity + score)
  - Full 6-DoF grasp rotation matrix output

The planner is agnostic to the specific affordance detection method —
it takes a point cloud and part indices, and produces a GraspPose.

Usage:
    planner = GraspPlanner()
    grasp = planner.plan("mug", "handle", points, part_indices, metadata)
    # grasp.position, grasp.approach_dir, grasp.grasp_rotation
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

PIPELINE_DIR = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════

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
    grasp_rotation: list = None  # 3x3 rotation matrix


class GraspPlanner:
    """Plans grasps using geometric heuristics or GraspNet inference."""

    # Affordance-weighted selection parameters
    SIGMA = 0.015     # 1.5 cm proximity decay (universal)
    ALPHA = 0.3       # 30% GraspNet score, 70% proximity

    def __init__(self, graspnet_dir: str = None, checkpoint_name: str = "checkpoint-rs.tar"):
        """
        Initialize grasp planner.

        Args:
            graspnet_dir:     Path to graspnet-baseline directory
            checkpoint_name:  Name of GraspNet checkpoint file
        """
        self.graspnet_dir = Path(graspnet_dir) if graspnet_dir else PIPELINE_DIR.parent / "graspnet-baseline"
        self.checkpoint_path = self.graspnet_dir / "checkpoints" / checkpoint_name
        self.input_dir = PIPELINE_DIR / "output"

    # ════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════

    def plan(
        self,
        obj_name: str,
        part_name: str,
        points: np.ndarray,
        part_indices: np.ndarray,
        metadata: dict,
        method: str = "neural",
    ) -> GraspPose:
        """
        Plan a grasp for a specific object part.

        Args:
            obj_name:     Object name
            part_name:    Part name
            points:       Object point cloud (N, 3)
            part_indices: Indices of part points within `points`
            metadata:     Stage 1 metadata
            method:       "neural" (GraspNet) or "geometric"

        Returns:
            GraspPose with position, approach_dir, and grasp_rotation
        """
        if method == "neural":
            return self._plan_neural(obj_name, part_name, points, part_indices, metadata)
        else:
            return self._plan_geometric(obj_name, part_name, points, part_indices)

    # ════════════════════════════════════════════════════════════════
    # GEOMETRIC GRASP PLANNING
    # ════════════════════════════════════════════════════════════════

    def _plan_geometric(self, obj_name, part_name, points, part_indices) -> GraspPose:
        """Propose a grasp using geometric heuristics (PCA + part semantics)."""
        part_points = points[part_indices]
        all_centroid = points.mean(axis=0)
        part_centroid = part_points.mean(axis=0)

        # Default: side approach from body center toward part
        approach_dir = part_centroid - all_centroid
        approach_dir[1] = 0
        norm = np.linalg.norm(approach_dir)
        if norm > 1e-8:
            approach_dir /= norm
        else:
            approach_dir = np.array([1.0, 0.0, 0.0])

        grasp_type = "side"
        confidence = 0.7

        # Part-specific overrides
        if part_name == "rim":
            approach_dir = np.array([0.0, -1.0, 0.0])
            grasp_type = "top_down"
            confidence = 0.8
        elif part_name == "head" and obj_name == "hammer":
            approach_dir = np.array([0.0, -1.0, 0.0])
            grasp_type = "top_down"
            confidence = 0.75
        elif part_name == "chuck":
            _, _, eigenvectors = _compute_pca(points)
            barrel_axis = eigenvectors[0]
            if np.dot(barrel_axis, part_centroid - all_centroid) < 0:
                barrel_axis = -barrel_axis
            approach_dir = barrel_axis
            grasp_type = "axial"
            confidence = 0.7
        elif part_name == "handle":
            grasp_type = "handle"
            confidence = 0.8
        elif part_name == "body":
            _, _, eigenvectors = _compute_pca(points)
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

        # Build rotation matrix: col0=binormal, col1=fingers, col2=approach
        grasp_rot = _build_grasp_rotation(approach_dir)

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

    # ════════════════════════════════════════════════════════════════
    # NEURAL GRASP PLANNING (GraspNet)
    # ════════════════════════════════════════════════════════════════

    def _plan_neural(self, obj_name, part_name, points, part_indices, metadata) -> GraspPose:
        """Run GraspNet inference with affordance-weighted selection."""
        import torch
        from PIL import Image

        if not self.checkpoint_path.exists():
            print(f"  WARNING: GraspNet checkpoint not found at {self.checkpoint_path}")
            print(f"  Falling back to geometric method")
            return self._plan_geometric(obj_name, part_name, points, part_indices)

        # Add GraspNet to path
        sys.path.insert(0, str(self.graspnet_dir / "models"))
        sys.path.insert(0, str(self.graspnet_dir / "dataset"))
        sys.path.insert(0, str(self.graspnet_dir / "utils"))

        # Load data
        depth = np.load(self.input_dir / "depth_raw.npy")
        rgb = np.array(Image.open(self.input_dir / "rgb.png"), dtype=np.float32) / 255.0

        print("  Building workspace-cropped point cloud for GraspNet...")
        end_points, o3d_cloud = self._build_graspnet_cloud(rgb, depth, metadata)

        print("  Loading GraspNet model...")
        from graspnet import GraspNet, pred_decode
        net = GraspNet(
            input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=device)
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
            return self._plan_geometric(obj_name, part_name, points, part_indices)

        # Transform to world frame
        world_positions, world_rotations = self._grasps_to_world(gg, metadata)

        # Affordance-weighted selection
        return self._select_best_grasp(
            gg, world_positions, world_rotations,
            points, part_indices, obj_name, part_name,
        )

    def _select_best_grasp(self, gg, world_positions, world_rotations,
                           points, part_indices, obj_name, part_name) -> GraspPose:
        """Select the best grasp using affordance-weighted scoring."""
        from scipy.spatial import cKDTree

        part_points = points[part_indices]
        part_centroid = part_points.mean(axis=0)
        part_extent = part_points.max(axis=0) - part_points.min(axis=0)
        search_radius = max(part_extent.max() * 0.75, 0.08)

        dists_to_centroid = np.linalg.norm(world_positions - part_centroid, axis=1)
        near_mask = dists_to_centroid < search_radius
        near_indices = np.where(near_mask)[0]

        if len(near_indices) == 0:
            search_radius *= 2
            near_mask = dists_to_centroid < search_radius
            near_indices = np.where(near_mask)[0]

        if len(near_indices) == 0:
            print(f"  WARNING: No GraspNet grasps near {part_name}, falling back to geometric")
            return self._plan_geometric(obj_name, part_name, points, part_indices)

        # Proximity scoring
        part_tree = cKDTree(part_points)
        candidate_positions = world_positions[near_indices]
        min_dists, _ = part_tree.query(candidate_positions)

        proximity_scores = np.exp(-min_dists / self.SIGMA)

        graspnet_scores = gg.scores[near_indices]
        gs_max = graspnet_scores.max()
        graspnet_norm = graspnet_scores / gs_max if gs_max > 0 else graspnet_scores

        combined = self.ALPHA * graspnet_norm + (1 - self.ALPHA) * proximity_scores
        best_local = np.argmax(combined)
        best_idx = near_indices[best_local]

        # Diagnostics
        print(f"  Affordance-weighted selection ({len(near_indices)} candidates):")
        print(f"    Search radius:       {search_radius:.3f}m")
        print(f"    Best GraspNet score: {graspnet_norm[best_local]:.4f}"
              f" (raw {graspnet_scores[best_local]:.4f})")
        print(f"    Nearest part dist:   {min_dists[best_local]*100:.1f}cm")
        print(f"    Proximity score:     {proximity_scores[best_local]:.4f}")
        print(f"    Combined score:      {combined[best_local]:.4f}")

        if len(near_indices) > 1:
            runner_up = np.argsort(combined)[-2]
            print(f"    Runner-up dist:      {min_dists[runner_up]*100:.1f}cm, "
                  f"GraspNet={graspnet_scores[runner_up]:.4f}, "
                  f"combined={combined[runner_up]:.4f}")

        best_pos = world_positions[best_idx]
        best_rot = world_rotations[best_idx]
        approach_dir = best_rot[:, 2]

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

    # ════════════════════════════════════════════════════════════════
    # GRASPNET CLOUD BUILDING
    # ════════════════════════════════════════════════════════════════

    def _build_graspnet_cloud(self, rgb, depth, metadata):
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

        # Workspace crop
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

    def _grasps_to_world(self, gg, metadata):
        """Transform GraspNet grasps from camera (OpenCV) to world frame."""
        opencv_to_habitat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        sensor_R = np.array(metadata["sensor_rotation_matrix"], dtype=np.float32)
        sensor_t = np.array(metadata["sensor_position"], dtype=np.float32)
        R_combined = sensor_R @ opencv_to_habitat

        world_positions = (R_combined @ gg.translations.T).T + sensor_t
        world_rotations = np.array([R_combined @ r for r in gg.rotation_matrices])

        return world_positions, world_rotations


# ════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL UTILITIES
# ════════════════════════════════════════════════════════════════════════

def _compute_pca(points):
    """Compute PCA. Returns centroid, eigenvalues (desc), eigenvectors (rows)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    return centroid, eigenvalues[idx], eigenvectors[:, idx].T


def _build_grasp_rotation(approach_dir):
    """Build a grasp rotation matrix from an approach direction."""
    z_axis = approach_dir.copy()
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 1e-8:
        z_axis /= z_norm

    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_axis, world_up)) > 0.9:
        world_up = np.array([1.0, 0.0, 0.0])

    y_axis = np.cross(z_axis, world_up)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    return np.column_stack([x_axis, y_axis, z_axis])
