"""
Scene Capture — Stage 1 of the Affordance Pipeline
====================================================
Places a single user-chosen object in an HSSD scene and captures:
  - RGB image
  - Depth image (raw metric + colorized)
  - Semantic segmentation
  - Per-object 3D point cloud (.ply)
  - Scene metadata (camera intrinsics, object position, etc.)

Usage:
    cd habitat-lab
    python ../affordance-pipeline/scene_capture.py --object mug
    python ../affordance-pipeline/scene_capture.py --object power_drill
    python ../affordance-pipeline/scene_capture.py --object pitcher
    python ../affordance-pipeline/scene_capture.py --object hammer

Available objects: mug, power_drill, pitcher, hammer
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ── Local imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from objects import OBJECTS, get_object, get_object_names, print_catalog, print_object_parts

# ── Habitat imports ──────────────────────────────────────────────────────
import habitat_sim
from habitat_sim.utils.common import quat_to_magnum
from habitat_sim.physics import MotionType
import magnum as mn

# ── Output directory ─────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

SENSOR_HEIGHT = 512
SENSOR_WIDTH  = 512
HFOV_DEG      = 70
MAX_DEPTH     = 10.0  # meters

# Scene
SCENE_DIR     = "data/scene_datasets/hssd-hab"
SCENE_ID      = "102344250"
SCENE_FILE    = f"{SCENE_DIR}/scenes/{SCENE_ID}.scene_instance.json"
SCENE_DATASET = f"{SCENE_DIR}/hssd-hab.scene_dataset_config.json"

# Camera intrinsics
HFOV_RAD = HFOV_DEG * np.pi / 180.0
FX = SENSOR_WIDTH / (2.0 * np.tan(HFOV_RAD / 2.0))
FY = FX
CX = SENSOR_WIDTH / 2.0
CY = SENSOR_HEIGHT / 2.0


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Capture scene with a single object",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python ../affordance-pipeline/scene_capture.py --object mug",
    )
    parser.add_argument(
        "--object", required=True,
        choices=get_object_names(),
        help="Object to place in the scene",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATOR SETUP
# ═══════════════════════════════════════════════════════════════════════════

def make_sim_config():
    """Build habitat-sim config with RGB, Depth, and Semantic sensors."""

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE_FILE
    backend_cfg.scene_dataset_config_file = SCENE_DATASET
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = 0

    # ── Sensor specs ─────────────────────────────────────────────────
    sensor_position = [0.0, 0.88, 0.0]      # Spot head-camera height
    sensor_orientation = [np.radians(-15), 0.0, 0.0]  # 15° down-tilt

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [SENSOR_HEIGHT, SENSOR_WIDTH]
    rgb_spec.hfov = mn.Deg(HFOV_DEG)
    rgb_spec.position = sensor_position
    rgb_spec.orientation = sensor_orientation
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [SENSOR_HEIGHT, SENSOR_WIDTH]
    depth_spec.hfov = mn.Deg(HFOV_DEG)
    depth_spec.position = sensor_position
    depth_spec.orientation = sensor_orientation
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = [SENSOR_HEIGHT, SENSOR_WIDTH]
    semantic_spec.hfov = mn.Deg(HFOV_DEG)
    semantic_spec.position = sensor_position
    semantic_spec.orientation = sensor_orientation
    semantic_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec, semantic_spec]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def setup_simulator():
    """Create and return a configured simulator."""
    print_header("Setting up Habitat-Sim")
    cfg = make_sim_config()
    sim = habitat_sim.Simulator(cfg)
    print(f"  Scene loaded: {SCENE_ID}")
    print(f"  Sensors: RGB, Depth, Semantic ({SENSOR_WIDTH}x{SENSOR_HEIGHT})")
    print(f"  HFOV: {HFOV_DEG}°, Focal length: {FX:.1f}px")
    return sim


# ═══════════════════════════════════════════════════════════════════════════
# AGENT POSITIONING & OBJECT SPAWNING
# ═══════════════════════════════════════════════════════════════════════════

def check_clearance_at_height(sim, pos, height, min_clearance=0.3):
    """Cast rays in 4 cardinal directions to check for wall clearance."""
    origin = mn.Vector3(pos[0], height, pos[2])
    directions = [
        mn.Vector3(1, 0, 0), mn.Vector3(-1, 0, 0),
        mn.Vector3(0, 0, 1), mn.Vector3(0, 0, -1),
    ]
    min_dist = float('inf')
    for d in directions:
        ray = habitat_sim.geo.Ray(origin, d)
        result = sim.cast_ray(ray)
        if result.has_hits():
            dist = result.hits[0].ray_distance
            min_dist = min(min_dist, dist)
    return min_dist >= min_clearance, min_dist


def find_navigable_point(sim, max_attempts=200):
    """
    Find a navigable point with good clearance at table height
    so the spawned object won't clip into walls or furniture.
    """
    if not sim.pathfinder.is_loaded:
        print("  Navmesh not loaded, recomputing...")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = 0.88
        navmesh_settings.agent_radius = 0.25
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

    if not sim.pathfinder.is_loaded:
        print("  WARNING: Could not load navmesh, using origin")
        return np.array([0.0, 0.1, 0.0])

    TABLE_HEIGHT = 0.8
    MIN_FORWARD_CLEAR = 1.5
    MIN_SIDE_CLEAR = 0.6

    best_pt = None
    best_score = -1

    for attempt in range(max_attempts):
        pt = sim.pathfinder.get_random_navigable_point()
        if np.isnan(pt).any():
            continue

        origin = mn.Vector3(pt[0], TABLE_HEIGHT, pt[2])

        ray_fwd = habitat_sim.geo.Ray(origin, mn.Vector3(0, 0, -1))
        result_fwd = sim.cast_ray(ray_fwd)
        fwd_dist = result_fwd.hits[0].ray_distance if result_fwd.has_hits() else 10.0

        ray_right = habitat_sim.geo.Ray(origin, mn.Vector3(1, 0, 0))
        result_right = sim.cast_ray(ray_right)
        right_dist = result_right.hits[0].ray_distance if result_right.has_hits() else 10.0

        ray_left = habitat_sim.geo.Ray(origin, mn.Vector3(-1, 0, 0))
        result_left = sim.cast_ray(ray_left)
        left_dist = result_left.hits[0].ray_distance if result_left.has_hits() else 10.0

        ray_back = habitat_sim.geo.Ray(origin, mn.Vector3(0, 0, 1))
        result_back = sim.cast_ray(ray_back)
        back_dist = result_back.hits[0].ray_distance if result_back.has_hits() else 10.0

        score = min(fwd_dist, 3.0) * 2 + min(right_dist, 2.0) + min(left_dist, 2.0) + min(back_dist, 1.5)

        meets_min = (fwd_dist >= MIN_FORWARD_CLEAR and
                     right_dist >= MIN_SIDE_CLEAR and
                     left_dist >= MIN_SIDE_CLEAR and
                     back_dist >= 0.5)

        if meets_min and score > best_score:
            best_score = score
            best_pt = np.array(pt)

    if best_pt is not None:
        print(f"  Selected point with clearance score {best_score:.2f}")
        return best_pt

    # Fallback
    for _ in range(max_attempts):
        pt = sim.pathfinder.get_random_navigable_point()
        if not np.isnan(pt).any():
            return pt

    return np.array([0.0, 0.1, 0.0])


def spawn_object(sim, agent_pos, obj_name):
    """
    Spawn a single YCB object in front of the agent.

    Returns dict with object info (rigid_obj_id, name, position, semantic_id).
    """
    print_header(f"Spawning object: {obj_name}")

    obj_cfg = get_object(obj_name)
    config_path = obj_cfg["ycb_config"]
    offset = np.array(obj_cfg["offset"])

    obj_templates_mgr = sim.get_object_template_manager()
    rigid_obj_mgr = sim.get_rigid_object_manager()

    # Load template
    template_ids = obj_templates_mgr.load_configs(config_path)
    if not template_ids:
        raise RuntimeError(f"Could not load template from {config_path}")
    template_handle = obj_templates_mgr.get_template_handles(config_path)[0]

    # Compute world position: agent + offset
    obj_pos = mn.Vector3(
        agent_pos[0] + offset[0],
        agent_pos[1] + offset[1],
        agent_pos[2] + offset[2],
    )

    # Verify clearance
    MIN_CLEARANCE = 0.15
    is_clear, min_dist = check_clearance_at_height(
        sim, [obj_pos[0], obj_pos[1], obj_pos[2]], obj_pos[1], MIN_CLEARANCE
    )

    if not is_clear:
        print(f"  WARNING: Too close to wall (min_dist={min_dist:.3f}m), adjusting...")
        nudge_dir = np.array([agent_pos[0] - obj_pos[0], 0, agent_pos[2] - obj_pos[2]])
        nudge_norm = np.linalg.norm(nudge_dir)
        if nudge_norm > 0.01:
            nudge_dir /= nudge_norm
            for nudge_dist in [0.1, 0.2, 0.3, 0.5]:
                candidate = mn.Vector3(
                    obj_pos[0] + nudge_dir[0] * nudge_dist,
                    obj_pos[1],
                    obj_pos[2] + nudge_dir[2] * nudge_dist,
                )
                is_clear2, min_dist2 = check_clearance_at_height(
                    sim, [candidate[0], candidate[1], candidate[2]],
                    candidate[1], MIN_CLEARANCE
                )
                if is_clear2:
                    print(f"    Nudged {nudge_dist:.1f}m toward agent → OK")
                    obj_pos = candidate
                    break

    # Spawn
    rigid_obj = rigid_obj_mgr.add_object_by_template_handle(template_handle)
    rigid_obj.translation = obj_pos
    rigid_obj.motion_type = MotionType.STATIC

    semantic_id = 1000
    rigid_obj.semantic_id = semantic_id

    obj_info = {
        "name": obj_name,
        "rigid_obj_id": rigid_obj.object_id,
        "position": np.array([obj_pos[0], obj_pos[1], obj_pos[2]]),
        "semantic_id": semantic_id,
    }

    print(f"  Object '{obj_name}' spawned at "
          f"({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})")
    print(f"  Semantic ID: {semantic_id}")

    return obj_info


def position_agent_and_spawn(sim, obj_name):
    """Position agent at a navigable point and spawn the chosen object."""
    print_header("Positioning agent")

    agent_pos = find_navigable_point(sim)
    print(f"  Agent at: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})")

    agent = sim.get_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = agent_pos
    agent_state.rotation = np.quaternion(1, 0, 0, 0)  # face -Z
    agent.set_state(agent_state)

    obj_info = spawn_object(sim, agent_pos, obj_name)

    return agent_pos, obj_info


# ═══════════════════════════════════════════════════════════════════════════
# SENSOR CAPTURE
# ═══════════════════════════════════════════════════════════════════════════

def capture_observations(sim):
    """Capture RGB, Depth, and Semantic observations."""
    print_header("Capturing sensor observations")

    obs = sim.get_sensor_observations()
    rgb      = obs["rgb"][:, :, :3]    # (H, W, 3) uint8
    depth    = obs["depth"]            # (H, W) float32, meters
    semantic = obs["semantic"]         # (H, W) int32

    print(f"  RGB:      {rgb.shape}, dtype={rgb.dtype}")
    print(f"  Depth:    {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]m")
    print(f"  Semantic: {semantic.shape}, unique IDs={len(np.unique(semantic))}")

    return rgb, depth, semantic


# ═══════════════════════════════════════════════════════════════════════════
# DEPTH → 3D POINT CLOUD
# ═══════════════════════════════════════════════════════════════════════════

def depth_to_pointcloud_camera(depth):
    """Convert depth image to 3D point cloud in camera coordinates."""
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    valid = (depth > 0) & (depth < MAX_DEPTH)
    u_valid = u[valid]
    v_valid = v[valid]
    d_valid = depth[valid]

    x =  (u_valid - CX) * d_valid / FX
    y = -(v_valid - CY) * d_valid / FY
    z = -d_valid

    points = np.stack([x, y, z], axis=-1)
    return points, valid


def camera_to_world(points_camera, sim):
    """Transform points from camera to world coordinates."""
    agent_state = sim.get_agent(0).get_state()
    sensor_state = agent_state.sensor_states["depth"]

    import quaternion as qt
    R = qt.as_rotation_matrix(sensor_state.rotation)
    t = np.array(sensor_state.position)

    points_world = (R @ points_camera.T).T + t
    return points_world


# ═══════════════════════════════════════════════════════════════════════════
# PER-OBJECT POINT CLOUD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_object_pointcloud(points_world, rgb, semantic, valid_mask, target_id):
    """Extract points for a single semantic object."""
    semantic_valid = semantic[valid_mask]
    rgb_flat = rgb[valid_mask]

    obj_mask = semantic_valid == target_id
    obj_points = points_world[obj_mask]
    obj_colors = rgb_flat[obj_mask].astype(np.float64) / 255.0

    return obj_points, obj_colors


# ═══════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def save_images(rgb, depth, semantic):
    """Save RGB, depth, and semantic images."""
    from PIL import Image

    Image.fromarray(rgb).save(OUTPUT_DIR / "rgb.png")
    print(f"  Saved: rgb.png")

    # Depth colorized
    depth_norm = np.clip(depth / MAX_DEPTH, 0, 1)
    depth_vis = 255 - (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(OUTPUT_DIR / "depth.png")
    print(f"  Saved: depth.png")

    # Raw depth
    np.save(OUTPUT_DIR / "depth_raw.npy", depth)
    print(f"  Saved: depth_raw.npy")

    # Semantic colorized
    unique_ids = np.unique(semantic)
    rng = np.random.RandomState(42)
    color_map = {sid: rng.randint(0, 255, 3) for sid in unique_ids}
    color_map[0] = np.array([0, 0, 0])
    semantic_colored = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for sid, color in color_map.items():
        semantic_colored[semantic == sid] = color
    Image.fromarray(semantic_colored).save(OUTPUT_DIR / "semantic.png")
    print(f"  Saved: semantic.png")


def save_pointcloud(points, colors, filename):
    """Save a colored point cloud as .ply file."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None and len(colors) > 0:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        filepath = OUTPUT_DIR / filename
        o3d.io.write_point_cloud(str(filepath), pcd)
        print(f"  Saved: {filename} ({len(points)} points)")
    except ImportError:
        print(f"  WARNING: open3d not installed, skipping PLY save")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    obj_name = args.object

    print_header("Scene Capture — Stage 1")
    print(f"  Object:    {obj_name}")
    print(f"  Output:    {OUTPUT_DIR}")

    # Clean old object PLY files to avoid confusion
    for old_ply in OUTPUT_DIR.glob("object_*.ply"):
        old_ply.unlink()
        print(f"  Cleaned: {old_ply.name}")

    obj_cfg = get_object(obj_name)
    print(f"  YCB Model: {obj_cfg['display_name']}")

    # ── Setup ────────────────────────────────────────────────────────
    sim = setup_simulator()

    try:
        # ── Position agent & spawn object ────────────────────────────
        agent_pos, obj_info = position_agent_and_spawn(sim, obj_name)

        # Step physics to settle
        for _ in range(10):
            sim.step_physics(1.0 / 60.0)

        # ── Capture observations ────────────────────────────────────
        rgb, depth, semantic = capture_observations(sim)

        # ── Save images ─────────────────────────────────────────────
        print_header("Saving images")
        save_images(rgb, depth, semantic)

        # ── Depth → point cloud ─────────────────────────────────────
        print_header("Converting depth → 3D point cloud")
        points_camera, valid_mask = depth_to_pointcloud_camera(depth)
        print(f"  Valid depth pixels: {valid_mask.sum()} / {depth.size}")

        points_world = camera_to_world(points_camera, sim)
        print(f"  Points in world coords: {len(points_world)}")

        # ── Save scene point cloud ──────────────────────────────────
        print_header("Saving point clouds")
        rgb_colors = rgb[valid_mask].astype(np.float64) / 255.0
        save_pointcloud(points_world, rgb_colors, "scene_full.ply")

        # ── Extract object point cloud ──────────────────────────────
        print_header("Extracting object point cloud")
        sem_id = obj_info["semantic_id"]
        obj_points, obj_colors = extract_object_pointcloud(
            points_world, rgb, semantic, valid_mask, sem_id
        )

        if len(obj_points) > 0:
            centroid = obj_points.mean(axis=0)
            bbox_size = obj_points.max(axis=0) - obj_points.min(axis=0)
            print(f"  Object '{obj_name}': {len(obj_points)} points")
            print(f"  Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
            print(f"  BBox size: ({bbox_size[0]:.3f}, {bbox_size[1]:.3f}, {bbox_size[2]:.3f})m")

            save_pointcloud(
                obj_points, obj_colors,
                f"object_{obj_name}_sem{sem_id}.ply"
            )
        else:
            print(f"  WARNING: '{obj_name}' not visible (0 pixels)")

        # ── Save metadata ───────────────────────────────────────────
        print_header("Saving metadata")
        agent_state = sim.get_agent(0).get_state()
        sensor_state = agent_state.sensor_states["depth"]
        import quaternion as qt
        sensor_R = qt.as_rotation_matrix(sensor_state.rotation).tolist()
        sensor_t = [float(x) for x in sensor_state.position]

        metadata = {
            "scene_id": SCENE_ID,
            "object_name": obj_name,
            "sensor_resolution": [SENSOR_HEIGHT, SENSOR_WIDTH],
            "hfov_deg": HFOV_DEG,
            "focal_length_px": float(FX),
            "principal_point": [float(CX), float(CY)],
            "max_depth_m": MAX_DEPTH,
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
        with open(OUTPUT_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: metadata.json")

        # ── Done ────────────────────────────────────────────────────
        print_header("STAGE 1 COMPLETE")
        print(f"  Object captured: {obj_cfg['display_name']}")
        print(f"  Object points:   {len(obj_points)}")
        print(f"  Output:          {OUTPUT_DIR}/")

        for p in sorted(OUTPUT_DIR.iterdir()):
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                print(f"    {p.name:40s} ({size_kb:.1f} KB)")

        # Print next step
        print_object_parts(obj_name)
        parts = list(obj_cfg["parts"].keys())
        print(f"\n  Next step:")
        print(f"    python ../affordance-pipeline/visualize_affordance.py "
              f"--object {obj_name} --part {parts[0]}")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
