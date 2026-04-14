"""
SceneCapture — Habitat-Sim scene setup and sensor capture.

Handles:
  - Simulator configuration (sensors, physics, scene)
  - Agent positioning with clearance checks
  - Object spawning (YCB models)
  - RGB / Depth / Semantic capture
  - Depth → 3D point cloud conversion
  - Output persistence (images, PLY, metadata JSON)

Usage:
    capture = SceneCapture()
    capture.setup()
    agent_pos, obj_info = capture.spawn_object("mug")
    rgb, depth, semantic = capture.capture()
    capture.save(rgb, depth, semantic, obj_info)
    capture.close()
"""

import sys
import json
import numpy as np
from pathlib import Path

import habitat_sim
from habitat_sim.physics import MotionType
import magnum as mn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from objects import get_object


class SceneCapture:
    """Captures RGB, depth, semantic observations and point clouds from Habitat-Sim."""

    # ── Default Configuration ────────────────────────────────────────
    DEFAULT_SENSOR_HEIGHT = 512
    DEFAULT_SENSOR_WIDTH  = 512
    DEFAULT_HFOV_DEG      = 70
    DEFAULT_MAX_DEPTH     = 10.0

    DEFAULT_SCENE_DIR     = "habitat-lab/data/versioned_data/hssd-hab"
    DEFAULT_SCENE_ID      = "102344250"

    def __init__(
        self,
        scene_id: str = None,
        scene_dir: str = None,
        sensor_width: int = None,
        sensor_height: int = None,
        hfov_deg: int = None,
        max_depth: float = None,
        output_dir: str = None,
    ):
        """
        Initialize scene capture configuration.

        Args:
            scene_id:      HSSD scene ID (default: 102344250)
            scene_dir:     Path to scene dataset directory
            sensor_width:  Sensor width in pixels (default: 512)
            sensor_height: Sensor height in pixels (default: 512)
            hfov_deg:      Horizontal field of view in degrees (default: 70)
            max_depth:     Maximum depth in meters (default: 10.0)
            output_dir:    Output directory path (default: affordance-pipeline/output)
        """
        self.scene_id      = scene_id or self.DEFAULT_SCENE_ID
        # Resolve scene_dir relative to the project root so the script can be
        # invoked from any working directory.
        _project_root = Path(__file__).resolve().parent.parent.parent  # …/interactive-robotics
        _default_abs  = str(_project_root / self.DEFAULT_SCENE_DIR)
        self.scene_dir = scene_dir or _default_abs
        self.sensor_width  = sensor_width or self.DEFAULT_SENSOR_WIDTH
        self.sensor_height = sensor_height or self.DEFAULT_SENSOR_HEIGHT
        self.hfov_deg      = hfov_deg or self.DEFAULT_HFOV_DEG
        self.max_depth     = max_depth or self.DEFAULT_MAX_DEPTH

        # Derived camera intrinsics
        hfov_rad = self.hfov_deg * np.pi / 180.0
        self.fx = self.sensor_width / (2.0 * np.tan(hfov_rad / 2.0))
        self.fy = self.fx
        self.cx = self.sensor_width / 2.0
        self.cy = self.sensor_height / 2.0

        # Paths
        pipeline_dir = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else pipeline_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Scene files
        self.scene_file = f"{self.scene_dir}/scenes/{self.scene_id}.scene_instance.json"
        self.scene_dataset = f"{self.scene_dir}/hssd-hab.scene_dataset_config.json"

        # Runtime state
        self.sim = None
        self.agent_pos = None

    # ════════════════════════════════════════════════════════════════
    # SETUP & TEARDOWN
    # ════════════════════════════════════════════════════════════════

    def setup(self):
        """Create and configure the simulator."""
        cfg = self._make_sim_config()
        self.sim = habitat_sim.Simulator(cfg)
        print(f"  Scene loaded: {self.scene_id}")
        print(f"  Sensors: RGB, Depth, Semantic ({self.sensor_width}x{self.sensor_height})")
        print(f"  HFOV: {self.hfov_deg}°, Focal length: {self.fx:.1f}px")

    def close(self):
        """Close the simulator and release resources."""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def _make_sim_config(self):
        """Build habitat-sim configuration with RGB, Depth, and Semantic sensors."""
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_file
        backend_cfg.scene_dataset_config_file = self.scene_dataset
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = 0

        sensor_position = [0.0, 0.88, 0.0]
        sensor_orientation = [np.radians(-15), 0.0, 0.0]

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.sensor_height, self.sensor_width]
        rgb_spec.hfov = mn.Deg(self.hfov_deg)
        rgb_spec.position = sensor_position
        rgb_spec.orientation = sensor_orientation
        rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [self.sensor_height, self.sensor_width]
        depth_spec.hfov = mn.Deg(self.hfov_deg)
        depth_spec.position = sensor_position
        depth_spec.orientation = sensor_orientation
        depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        semantic_spec = habitat_sim.CameraSensorSpec()
        semantic_spec.uuid = "semantic"
        semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_spec.resolution = [self.sensor_height, self.sensor_width]
        semantic_spec.hfov = mn.Deg(self.hfov_deg)
        semantic_spec.position = sensor_position
        semantic_spec.orientation = sensor_orientation
        semantic_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec, semantic_spec]

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    # ════════════════════════════════════════════════════════════════
    # AGENT POSITIONING & OBJECT SPAWNING
    # ════════════════════════════════════════════════════════════════

    def spawn_object(self, obj_name: str):
        """
        Position the agent at a navigable point and spawn a YCB object.

        Args:
            obj_name: Key from objects.json (e.g., "mug", "hammer")

        Returns:
            (agent_pos, obj_info) tuple
        """
        self.agent_pos = self._find_navigable_point()
        print(f"  Agent at: ({self.agent_pos[0]:.2f}, {self.agent_pos[1]:.2f}, "
              f"{self.agent_pos[2]:.2f})")

        agent = self.sim.get_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = self.agent_pos
        agent_state.rotation = np.quaternion(1, 0, 0, 0)
        agent.set_state(agent_state)

        obj_info = self._spawn_single_object(obj_name)

        # Settle physics
        for _ in range(10):
            self.sim.step_physics(1.0 / 60.0)

        return self.agent_pos, obj_info

    def _find_navigable_point(self, max_attempts=200):
        """Find a navigable point with good clearance."""
        sim = self.sim
        if not sim.pathfinder.is_loaded:
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = 0.88
            navmesh_settings.agent_radius = 0.25
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        if not sim.pathfinder.is_loaded:
            return np.array([0.0, 0.1, 0.0])

        TABLE_HEIGHT = 0.8
        best_pt, best_score = None, -1

        for _ in range(max_attempts):
            pt = sim.pathfinder.get_random_navigable_point()
            if np.isnan(pt).any():
                continue

            origin = mn.Vector3(pt[0], TABLE_HEIGHT, pt[2])
            directions = {
                "fwd":   mn.Vector3(0, 0, -1),
                "right": mn.Vector3(1, 0, 0),
                "left":  mn.Vector3(-1, 0, 0),
                "back":  mn.Vector3(0, 0, 1),
            }

            dists = {}
            for name, d in directions.items():
                ray = habitat_sim.geo.Ray(origin, d)
                result = sim.cast_ray(ray)
                dists[name] = result.hits[0].ray_distance if result.has_hits() else 10.0

            score = (min(dists["fwd"], 3.0) * 2 + min(dists["right"], 2.0) +
                     min(dists["left"], 2.0) + min(dists["back"], 1.5))

            meets_min = (dists["fwd"] >= 1.5 and dists["right"] >= 0.6 and
                         dists["left"] >= 0.6 and dists["back"] >= 0.5)

            if meets_min and score > best_score:
                best_score = score
                best_pt = np.array(pt)

        if best_pt is not None:
            return best_pt

        # Fallback
        for _ in range(max_attempts):
            pt = sim.pathfinder.get_random_navigable_point()
            if not np.isnan(pt).any():
                return pt
        return np.array([0.0, 0.1, 0.0])

    def _spawn_single_object(self, obj_name: str):
        """Spawn a single YCB object in front of the agent."""
        obj_cfg = get_object(obj_name)
        _project_root = Path(__file__).resolve().parent.parent.parent
        # Resolve ycb_config relative to project root so the script runs from any cwd
        config_path_raw = obj_cfg["ycb_config"]
        config_path = str(_project_root / config_path_raw) if not Path(config_path_raw).is_absolute() else config_path_raw
        offset = np.array(obj_cfg["offset"])

        obj_templates_mgr = self.sim.get_object_template_manager()
        rigid_obj_mgr = self.sim.get_rigid_object_manager()

        template_ids = obj_templates_mgr.load_configs(config_path)
        if not template_ids:
            raise RuntimeError(f"Could not load template from {config_path}")
        template_handle = obj_templates_mgr.get_template_handles(config_path)[0]

        obj_pos = mn.Vector3(
            self.agent_pos[0] + offset[0],
            self.agent_pos[1] + offset[1],
            self.agent_pos[2] + offset[2],
        )

        # Clearance check
        self._check_and_nudge_position(obj_pos)

        rigid_obj = rigid_obj_mgr.add_object_by_template_handle(template_handle)
        rigid_obj.translation = obj_pos
        rigid_obj.motion_type = MotionType.STATIC
        rigid_obj.semantic_id = 1000

        obj_info = {
            "name": obj_name,
            "rigid_obj_id": rigid_obj.object_id,
            "position": np.array([obj_pos[0], obj_pos[1], obj_pos[2]]),
            "semantic_id": 1000,
        }

        print(f"  Object '{obj_name}' spawned at "
              f"({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})")
        return obj_info

    def _check_and_nudge_position(self, obj_pos, min_clearance=0.15):
        """Check wall clearance and nudge object if too close."""
        origin = mn.Vector3(obj_pos[0], obj_pos[1], obj_pos[2])
        directions = [
            mn.Vector3(1, 0, 0), mn.Vector3(-1, 0, 0),
            mn.Vector3(0, 0, 1), mn.Vector3(0, 0, -1),
        ]
        min_dist = float('inf')
        for d in directions:
            ray = habitat_sim.geo.Ray(origin, d)
            result = self.sim.cast_ray(ray)
            if result.has_hits():
                min_dist = min(min_dist, result.hits[0].ray_distance)

        if min_dist >= min_clearance:
            return

        nudge_dir = np.array([
            self.agent_pos[0] - obj_pos[0], 0, self.agent_pos[2] - obj_pos[2]
        ])
        nudge_norm = np.linalg.norm(nudge_dir)
        if nudge_norm < 0.01:
            return

        nudge_dir /= nudge_norm
        for nudge_dist in [0.1, 0.2, 0.3, 0.5]:
            candidate = mn.Vector3(
                obj_pos[0] + nudge_dir[0] * nudge_dist,
                obj_pos[1],
                obj_pos[2] + nudge_dir[2] * nudge_dist,
            )
            c_origin = mn.Vector3(candidate[0], candidate[1], candidate[2])
            c_min = float('inf')
            for d in directions:
                ray = habitat_sim.geo.Ray(c_origin, d)
                result = self.sim.cast_ray(ray)
                if result.has_hits():
                    c_min = min(c_min, result.hits[0].ray_distance)
            if c_min >= min_clearance:
                obj_pos[0] = candidate[0]
                obj_pos[2] = candidate[2]
                break

    # ════════════════════════════════════════════════════════════════
    # SENSOR CAPTURE
    # ════════════════════════════════════════════════════════════════

    def capture(self):
        """
        Capture RGB, Depth, and Semantic observations.

        Returns:
            (rgb, depth, semantic) tuple of numpy arrays
        """
        obs = self.sim.get_sensor_observations()
        rgb      = obs["rgb"][:, :, :3]
        depth    = obs["depth"]
        semantic = obs["semantic"]

        print(f"  RGB:      {rgb.shape}, dtype={rgb.dtype}")
        print(f"  Depth:    {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]m")
        print(f"  Semantic: {semantic.shape}, unique IDs={len(np.unique(semantic))}")

        return rgb, depth, semantic

    # ════════════════════════════════════════════════════════════════
    # POINT CLOUD
    # ════════════════════════════════════════════════════════════════

    def depth_to_pointcloud(self, depth):
        """Convert depth image to 3D point cloud in camera coordinates.

        Returns:
            (points_camera, valid_mask) tuple
        """
        H, W = depth.shape
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        valid = (depth > 0) & (depth < self.max_depth)
        u_valid, v_valid, d_valid = u[valid], v[valid], depth[valid]

        x =  (u_valid - self.cx) * d_valid / self.fx
        y = -(v_valid - self.cy) * d_valid / self.fy
        z = -d_valid

        return np.stack([x, y, z], axis=-1), valid

    def camera_to_world(self, points_camera):
        """Transform points from camera to world coordinates."""
        agent_state = self.sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states["depth"]

        import quaternion as qt
        R = qt.as_rotation_matrix(sensor_state.rotation)
        t = np.array(sensor_state.position)

        return (R @ points_camera.T).T + t

    def extract_object_pointcloud(self, points_world, rgb, semantic, valid_mask, target_id):
        """Extract points for a single semantic object."""
        semantic_valid = semantic[valid_mask]
        rgb_flat = rgb[valid_mask]

        obj_mask = semantic_valid == target_id
        obj_points = points_world[obj_mask]
        obj_colors = rgb_flat[obj_mask].astype(np.float64) / 255.0

        return obj_points, obj_colors

    # ════════════════════════════════════════════════════════════════
    # SAVE OUTPUTS
    # ════════════════════════════════════════════════════════════════

    def save(self, rgb, depth, semantic, obj_info):
        """Save all outputs: images, point clouds, metadata."""
        self._save_images(rgb, depth, semantic)

        # Point cloud
        points_camera, valid_mask = self.depth_to_pointcloud(depth)
        points_world = self.camera_to_world(points_camera)
        print(f"  Points in world coords: {len(points_world)}")

        rgb_colors = rgb[valid_mask].astype(np.float64) / 255.0
        self._save_pointcloud(points_world, rgb_colors, "scene_full.ply")

        # Object point cloud
        obj_points, obj_colors = self.extract_object_pointcloud(
            points_world, rgb, semantic, valid_mask, obj_info["semantic_id"]
        )
        if len(obj_points) > 0:
            centroid = obj_points.mean(axis=0)
            bbox_size = obj_points.max(axis=0) - obj_points.min(axis=0)
            print(f"  Object '{obj_info['name']}': {len(obj_points)} points")
            print(f"  Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
            print(f"  BBox size: ({bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f})m")
            self._save_pointcloud(
                obj_points, obj_colors,
                f"object_{obj_info['name']}_sem{obj_info['semantic_id']}.ply"
            )

        # Metadata
        self._save_metadata(obj_info, valid_mask)

    def _save_images(self, rgb, depth, semantic):
        """Save RGB, depth, and semantic images."""
        from PIL import Image as PILImage

        # Clean old PLY files
        for old_ply in self.output_dir.glob("object_*.ply"):
            old_ply.unlink()

        PILImage.fromarray(rgb).save(self.output_dir / "rgb.png")

        depth_norm = np.clip(depth / self.max_depth, 0, 1)
        depth_vis = 255 - (depth_norm * 255).astype(np.uint8)
        PILImage.fromarray(depth_vis).save(self.output_dir / "depth.png")

        np.save(self.output_dir / "depth_raw.npy", depth)
        np.save(self.output_dir / "semantic_raw.npy", semantic.astype(np.int32))

        unique_ids = np.unique(semantic)
        rng = np.random.RandomState(42)
        color_map = {sid: rng.randint(0, 255, 3) for sid in unique_ids}
        color_map[0] = np.array([0, 0, 0])
        semantic_colored = np.zeros((*semantic.shape, 3), dtype=np.uint8)
        for sid, color in color_map.items():
            semantic_colored[semantic == sid] = color
        PILImage.fromarray(semantic_colored).save(self.output_dir / "semantic.png")

        print(f"  Saved: rgb.png, depth.png, depth_raw.npy, semantic_raw.npy, semantic.png")

    def _save_pointcloud(self, points, colors, filename):
        """Save a colored point cloud as .ply file."""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None and len(colors) > 0:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            filepath = self.output_dir / filename
            o3d.io.write_point_cloud(str(filepath), pcd)
            print(f"  Saved: {filename} ({len(points)} points)")
        except ImportError:
            print(f"  WARNING: open3d not installed, skipping PLY save")

    def _save_metadata(self, obj_info, valid_mask):
        """Save metadata JSON."""
        agent_state = self.sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states["depth"]
        import quaternion as qt
        sensor_R = qt.as_rotation_matrix(sensor_state.rotation).tolist()
        sensor_t = [float(x) for x in sensor_state.position]

        metadata = {
            "scene_id": self.scene_id,
            "object_name": obj_info["name"],
            "sensor_resolution": [self.sensor_height, self.sensor_width],
            "hfov_deg": self.hfov_deg,
            "focal_length_px": float(self.fx),
            "principal_point": [float(self.cx), float(self.cy)],
            "max_depth_m": self.max_depth,
            "agent_position": [float(x) for x in agent_state.position],
            "sensor_rotation_matrix": sensor_R,
            "sensor_position": sensor_t,
            "total_valid_points": int(valid_mask.sum()),
            "spawned_objects": [{
                "name": obj_info["name"],
                "rigid_obj_id": obj_info["rigid_obj_id"],
                "semantic_id": int(obj_info["semantic_id"]),
                "position": obj_info["position"].tolist(),
            }],
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: metadata.json")

    # ════════════════════════════════════════════════════════════════
    # ACCESSORS
    # ════════════════════════════════════════════════════════════════

    def get_metadata(self) -> dict:
        """Load and return saved metadata.json."""
        meta_path = self.output_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found at {meta_path}")
        with open(meta_path) as f:
            return json.load(f)

    @property
    def intrinsics(self) -> dict:
        """Camera intrinsics dictionary."""
        return {
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "width": self.sensor_width,
            "height": self.sensor_height,
            "hfov_deg": self.hfov_deg,
            "max_depth": self.max_depth,
        }
