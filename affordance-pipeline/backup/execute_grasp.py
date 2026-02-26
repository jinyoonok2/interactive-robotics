"""Execute Grasp — Stage 3 of the Affordance Pipeline
====================================================
Loads the grasp pose from Stage 2 (DINO + SAM + GraspNet) and executes it
with a Fetch robot arm in Habitat-Sim.

Uses the high-level FetchRobot wrapper (Manipulator → MobileManipulator
→ FetchRobot) for correct EE offsets, motor configuration, and joint
management, combined with IkHelper for inverse kinematics.

Pipeline:
  Stage 1: scene_capture.py        → RGB, depth, semantic, point cloud
  Stage 2: visualize_affordance.py → DINO + SAM segmentation + GraspNet grasp
  Stage 3: execute_grasp.py        → Robot arm IK → pick execution (THIS FILE)

Usage:
    cd habitat-lab
    python ../affordance-pipeline/execute_grasp.py --object mug --part handle
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace

# ── Local imports ────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
from objects import OBJECTS, get_object, get_object_names

# ── Habitat imports ──────────────────────────────────────────────────────
import habitat_sim
from habitat_sim.physics import MotionType
import magnum as mn

# ── High-level robot wrapper ─────────────────────────────────────────────
from habitat.articulated_agents.robots.fetch_robot import FetchRobot
from habitat.tasks.rearrange.utils import IkHelper
import pybullet as pb
from scipy.spatial.transform import Rotation as spR

# ── Paths ────────────────────────────────────────────────────────────────
INPUT_DIR  = PIPELINE_DIR / "output"
RESULT_DIR = PIPELINE_DIR / "results" / "language"
OUTPUT_DIR = PIPELINE_DIR / "results" / "execution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fetch robot URDFs (relative to habitat-lab working dir)
FETCH_URDF     = "data/robots/hab_fetch/robots/hab_fetch.urdf"
FETCH_ARM_URDF = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"

# ── Sim config (match scene_capture.py) ──────────────────────────────────
SENSOR_HEIGHT = 512
SENSOR_WIDTH  = 512
HFOV_DEG      = 70
SCENE_DIR     = "data/scene_datasets/hssd-hab"
SCENE_ID      = "102344250"
SCENE_FILE    = f"{SCENE_DIR}/scenes/{SCENE_ID}.scene_instance.json"
SCENE_DATASET = f"{SCENE_DIR}/hssd-hab.scene_dataset_config.json"

# ── Animation parameters ────────────────────────────────────────────────
PHYSICS_DT      = 1.0 / 240.0      # 240 Hz physics
CTRL_FREQ        = 120              # control steps per second
RENDER_EVERY     = 8                # render a frame every N physics steps
PRE_GRASP_OFFSET = 0.12            # standoff distance along approach (meters)
LIFT_HEIGHT      = 0.20            # lift after grasp (meters)
SETTLE_STEPS     = 120             # physics steps to let robot settle


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def load_metadata():
    """Load Stage 1 metadata."""
    meta_path = INPUT_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}\n"
            "  Run Stage 1 first: python ../affordance-pipeline/scene_capture.py --object <obj>"
        )
    with open(meta_path) as f:
        return json.load(f)


def load_grasp_pose():
    """Load Stage 2 grasp pose."""
    grasp_path = RESULT_DIR / "grasp_poses.json"
    if not grasp_path.exists():
        raise FileNotFoundError(
            f"grasp_poses.json not found at {grasp_path}\n"
            "  Run Stage 2 first: python ../affordance-pipeline/visualize_affordance.py --object <obj> --part <part>"
        )
    with open(grasp_path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATOR SETUP
# ═══════════════════════════════════════════════════════════════════════════

def make_sim_config():
    """Build habitat-sim config matching scene_capture.py."""

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE_FILE
    backend_cfg.scene_dataset_config_file = SCENE_DATASET
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = 0

    # Sensor offset from agent position (agent pos sets the actual location)
    sensor_position = [0.0, 0.0, 0.0]      # no extra offset — agent pos IS camera pos
    sensor_orientation = [np.radians(-25), 0.0, 0.0]  # slight downward tilt

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [SENSOR_HEIGHT, SENSOR_WIDTH]
    rgb_spec.hfov = mn.Deg(HFOV_DEG)
    rgb_spec.position = sensor_position
    rgb_spec.orientation = sensor_orientation
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


# ═══════════════════════════════════════════════════════════════════════════
# ROBOT SETUP  (using high-level FetchRobot wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def create_fetch_robot(sim, position, rotation=None):
    """
    Create a Fetch robot using the high-level FetchRobot class.

    FetchRobot (MobileManipulator → Manipulator) handles:
      - URDF loading and joint/motor configuration
      - Correct EE offset (0.08m along local X) via ee_transform()
      - arm_joint_pos / gripper_joint_pos properties (with motor sync)
      - Torso lift = 0.15, head locking via update()
      - Motor gain tuning (pos_gain=0.3, vel_gain=0.3, max_impulse=10)

    Args:
        sim: Habitat simulator
        position: mn.Vector3 world position for robot base
        rotation: mn.Quaternion (optional) — defaults to facing -Z

    Returns:
        FetchRobot instance (access underlying sim object via fetch.sim_obj)
    """
    # FetchRobot expects agent_cfg with .articulated_agent_urdf attribute
    agent_cfg = SimpleNamespace(articulated_agent_urdf=FETCH_URDF)

    fetch = FetchRobot(agent_cfg, sim, limit_robo_joints=True, fixed_base=True)
    fetch.reconfigure()  # loads URDF, sets motors, init joints

    # Position the robot
    fetch.sim_obj.translation = position
    if rotation is not None:
        fetch.sim_obj.rotation = rotation

    # Set motion type to DYNAMIC so motors work
    fetch.sim_obj.motion_type = MotionType.DYNAMIC

    # Open gripper (FetchRobot inits with closed gripper by default)
    fetch.gripper_joint_pos = fetch.params.gripper_open_state

    # Apply update to lock head/back joints
    fetch.update()

    print(f"  Fetch robot spawned at "
          f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    print(f"  Arm joints (link IDs):   {fetch.params.arm_joints}")
    print(f"  EE link:                 {fetch.params.ee_links}")
    print(f"  EE offset:               {fetch.params.ee_offset}")
    print(f"  Total joint positions:   {len(fetch.sim_obj.joint_positions)}")

    return fetch


def get_ee_position(fetch):
    """
    Get current end-effector world position using FetchRobot's ee_transform.
    This correctly applies the 0.08m EE offset.
    """
    return fetch.ee_transform().translation


# ═══════════════════════════════════════════════════════════════════════════
# INVERSE KINEMATICS
# ═══════════════════════════════════════════════════════════════════════════

def setup_ik_helper():
    """Initialize PyBullet IK solver for Fetch arm."""
    arm_init = np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005])
    ik = IkHelper(FETCH_ARM_URDF, arm_init)
    print("  IK solver initialized (PyBullet)")
    return ik


def world_to_robot_base(world_pos, fetch):
    """Convert world position to robot base frame for IK."""
    base_T = fetch.base_transformation
    local_pos = base_T.inverted().transform_point(mn.Vector3(*world_pos))
    return np.array([local_pos[0], local_pos[1], local_pos[2]])


def grasp_rotation_to_ee_quat(grasp_rot_world, fetch):
    """
    Convert GraspNet grasp rotation (world frame) to a PyBullet target
    orientation quaternion [x,y,z,w] in the robot's base frame.

    Frame mapping (from URDF analysis):
      GraspNet: col0=binormal, col1=baseline(fingers), col2=approach
      Fetch gripper_link: X=forward(approach), Y=fingers, Z=palm

    So:
      Fetch X  = GraspNet col2 (approach)
      Fetch Y  = GraspNet col1 (fingers)
      Fetch Z  = cross(Fetch_X, Fetch_Y)  (right-handed)
    """
    grasp_rot = np.array(grasp_rot_world)
    approach = grasp_rot[:, 2]   # GraspNet Z → Fetch X (forward)
    fingers  = grasp_rot[:, 1]   # GraspNet Y → Fetch Y (finger opening)
    palm     = np.cross(approach, fingers)  # Fetch Z

    # Build desired EE rotation in world frame
    R_world_ee = np.column_stack([approach, fingers, palm])
    R_world_ee = spR.from_matrix(R_world_ee).as_matrix()  # orthogonalize

    # Transform to robot base frame
    base_T = fetch.base_transformation
    # Extract 3x3 rotation from Magnum Matrix4
    R_base_world = np.array([
        [base_T[0][0], base_T[0][1], base_T[0][2]],
        [base_T[1][0], base_T[1][1], base_T[1][2]],
        [base_T[2][0], base_T[2][1], base_T[2][2]],
    ])
    R_base_ee = R_base_world.T @ R_world_ee

    # To quaternion [x,y,z,w] — same convention as PyBullet
    quat = spR.from_matrix(R_base_ee).as_quat()  # scipy returns [x,y,z,w]
    return quat


def solve_ik(ik_helper, target_world, fetch, target_ori_quat=None,
             max_iters=10, tol=0.005):
    """
    Solve IK with iterative closed-loop correction (position-only).
    """
    original_arm = np.array(fetch.arm_joint_pos)

    target_np = np.array(target_world, dtype=float)
    target_mn = mn.Vector3(*target_np)

    ik_target = target_np.copy()
    best_sol = None
    best_err = float('inf')

    for iteration in range(max_iters):
        target_base = world_to_robot_base(ik_target, fetch)
        seed = original_arm if best_sol is None else best_sol
        ik_helper.set_arm_state(seed)
        ik_sol = np.array(ik_helper.calc_ik(target_base))
        fetch.arm_joint_pos = ik_sol

        actual_ee = get_ee_position(fetch)
        error = target_mn - actual_ee
        err_mag = error.length()

        if err_mag < best_err:
            best_err = err_mag
            best_sol = ik_sol.copy()
        if err_mag < tol:
            break
        ik_target += np.array([error[0], error[1], error[2]])

    fetch.arm_joint_pos = original_arm
    print(f"    IK position: {best_err:.4f}m ({iteration + 1} iters)")
    return best_sol


def adjust_wrist_roll(ik_sol, grasp_rot_world, fetch):
    """
    Adjust the wrist_roll joint (joint 6 of arm, index 6 in ik_sol)
    to make the gripper fingers span ACROSS the graspable feature,
    WITHOUT changing the EE position.

    Strategy: Use the GraspNet binormal (col0 of the rotation matrix)
    as the local surface/handle axis. The fingers should be PERPENDICULAR
    to this axis to grip across it. This works regardless of the robot's
    actual approach direction — it only uses the local geometry.

    The wrist_roll rotates around the gripper's X axis (approach axis).
    Since the EE offset is ALONG X, rotation around X doesn't change
    the EE position — only orientation.

    Args:
        ik_sol: 7-element joint array from position-only IK
        grasp_rot_world: 3x3 GraspNet rotation matrix (world frame)
                         col0=binormal (handle axis), col1=baseline (fingers),
                         col2=approach
        fetch: FetchRobot instance

    Returns:
        Modified 7-element joint array with corrected wrist_roll
    """
    if grasp_rot_world is None:
        return ik_sol

    grasp_rot = np.array(grasp_rot_world)
    # Binormal ≈ local handle/surface axis at the grasp point.
    # For a cylindrical handle, this is ALONG the handle bar.
    handle_axis = grasp_rot[:, 0]
    # Original GraspNet finger direction (for reference / picking best option)
    gn_fingers = grasp_rot[:, 1]

    # Temporarily set arm to get current gripper frame
    original_arm = np.array(fetch.arm_joint_pos)
    fetch.arm_joint_pos = ik_sol

    ee_T = fetch.ee_transform()
    ee_rot = np.array([[ee_T[i][j] for j in range(3)] for i in range(3)])
    curr_x = ee_rot[:, 0]  # approach (wrist roll axis)
    curr_y = ee_rot[:, 1]  # current finger direction
    curr_z = ee_rot[:, 2]  # current palm normal

    fetch.arm_joint_pos = original_arm

    # Project handle axis onto the YZ plane (perpendicular to actual approach)
    handle_proj = handle_axis - np.dot(handle_axis, curr_x) * curr_x
    hp_norm = np.linalg.norm(handle_proj)

    if hp_norm < 1e-6:
        # Handle axis is along the approach — any finger direction works
        print(f"    Wrist roll: skipped (handle axis ∥ approach)")
        return ik_sol

    handle_proj /= hp_norm

    # Desired finger direction: PERPENDICULAR to handle axis in the YZ plane.
    # In the YZ plane, if handle_proj = a·Y + b·Z, the two perpendiculars are:
    #   option1 = -b·Y + a·Z     (rotate +90°)
    #   option2 =  b·Y - a·Z     (rotate -90°)
    # Both are valid (fingers are symmetric). Pick the one closer to
    # GraspNet's intended finger direction for consistency.
    a = np.dot(handle_proj, curr_y)
    b = np.dot(handle_proj, curr_z)

    opt1 = -b * curr_y + a * curr_z
    opt2 =  b * curr_y - a * curr_z

    if abs(np.dot(opt1, gn_fingers)) >= abs(np.dot(opt2, gn_fingers)):
        desired_y = opt1
    else:
        desired_y = opt2

    # Compute wrist_roll angle to align current Y with desired_y
    y_comp = np.dot(desired_y, curr_y)
    z_comp = np.dot(desired_y, curr_z)
    theta = np.arctan2(z_comp, y_comp)

    # Apply to wrist_roll (arm joint index 6, continuous — no limits)
    corrected = ik_sol.copy()
    corrected[6] += theta

    # ---- Verify ----
    fetch.arm_joint_pos = corrected
    new_ee_T = fetch.ee_transform()
    new_rot = np.array([[new_ee_T[i][j] for j in range(3)] for i in range(3)])
    new_y = new_rot[:, 1]  # new finger direction

    # Angle between new fingers and the handle axis (should be ~90° = perpendicular)
    handle_perp_angle = abs(np.degrees(np.arccos(
        np.clip(abs(np.dot(new_y, handle_axis)), -1, 1))))

    # Angle between new fingers and GraspNet fingers (for comparison)
    gn_align_angle = np.degrees(np.arccos(
        np.clip(abs(np.dot(new_y, gn_fingers)), -1, 1)))

    # Position shift (should be ~0)
    new_pos = new_ee_T.translation
    orig_pos = ee_T.translation
    pos_shift = (new_pos - orig_pos).length()

    fetch.arm_joint_pos = original_arm

    print(f"    Wrist roll: Δ={np.degrees(theta):.1f}°, "
          f"fingers⊥handle={handle_perp_angle:.1f}° (target ~90°), "
          f"vs GraspNet={gn_align_angle:.1f}°, "
          f"pos shift={pos_shift*100:.1f}cm")

    return corrected


# ═══════════════════════════════════════════════════════════════════════════
# MOTION EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_joints(start, end, steps):
    """Linear interpolation between joint configurations."""
    alphas = np.linspace(0, 1, steps)
    return [start + a * (end - start) for a in alphas]


def execute_motion(sim, fetch, target_joints, agent, duration_sec=2.0,
                   frames=None, label=""):
    """
    Move robot arm to target joint configuration over `duration_sec`,
    capturing frames for video.

    Uses KINEMATIC interpolation via FetchRobot.arm_joint_pos.
    """
    current_arm = np.array(fetch.arm_joint_pos)

    n_steps = int(duration_sec * CTRL_FREQ)
    waypoints = interpolate_joints(current_arm, target_joints, n_steps)

    ee_start = get_ee_position(fetch)

    for step_i, wp in enumerate(waypoints):
        # Set arm joints via the high-level API (handles motor targets too)
        fetch.arm_joint_pos = wp

        # Step physics
        sim.step_physics(PHYSICS_DT)

        # Capture frame
        if frames is not None and step_i % RENDER_EVERY == 0:
            obs = sim.get_sensor_observations()
            frames.append(obs["rgb"][:, :, :3].copy())

    ee_end = get_ee_position(fetch)
    dist = (ee_end - ee_start).length()
    if label:
        print(f"  {label}: EE moved {dist:.4f}m "
              f"({ee_end[0]:.3f}, {ee_end[1]:.3f}, {ee_end[2]:.3f})")


def execute_gripper(sim, fetch, state, agent, duration_sec=0.5, frames=None):
    """Open or close gripper over duration using FetchRobot gripper API."""
    if state == "open":
        target = np.array(fetch.params.gripper_open_state)
    else:
        target = np.array(fetch.params.gripper_closed_state)

    current_grip = np.array(fetch.gripper_joint_pos)

    n_steps = int(duration_sec * CTRL_FREQ)
    waypoints = interpolate_joints(current_grip, target, n_steps)

    for step_i, wp in enumerate(waypoints):
        fetch.gripper_joint_pos = wp

        sim.step_physics(PHYSICS_DT)

        if frames is not None and step_i % RENDER_EVERY == 0:
            obs = sim.get_sensor_observations()
            frames.append(obs["rgb"][:, :, :3].copy())

    action = "opened" if state == "open" else "closed"
    print(f"  Gripper {action}")


# ═══════════════════════════════════════════════════════════════════════════
# MAGIC GRASP (snap-to constraint)
# ═══════════════════════════════════════════════════════════════════════════

def magic_grasp(sim, fetch, target_obj, grasp_world_pos, grasp_threshold=0.30):
    """
    Snap the target object to the robot's end-effector at the grasp point.
    Uses a rigid constraint (like Habitat's MagicGraspAction).

    Uses FetchRobot's ee_link_id() for correct link identification.

    The constraint attaches:
      - pivot_a: the grasp point expressed in EE-link local frame
      - pivot_b: the grasp point expressed in object local frame
    This ensures the object is held at the handle (or whichever part was targeted),
    not at the object center.

    Returns the constraint ID or None.
    """
    ee_pos = get_ee_position(fetch)
    obj_pos = target_obj.translation

    dist_ee_grasp = (ee_pos - mn.Vector3(*grasp_world_pos)).length()
    dist_ee_obj = (ee_pos - obj_pos).length()

    print(f"  EE → grasp target = {dist_ee_grasp:.3f}m")
    print(f"  EE → object center = {dist_ee_obj:.3f}m")

    if dist_ee_grasp > grasp_threshold:
        print(f"  WARNING: EE is {dist_ee_grasp:.3f}m from grasp target "
              f"(threshold={grasp_threshold}m)")

    # Grasp point in world coordinates
    grasp_pt = mn.Vector3(*grasp_world_pos)

    # pivot_a: grasp point in EE-link local frame
    ee_link_id = fetch.ee_link_id()
    ee_T = fetch.sim_obj.get_link_scene_node(ee_link_id).transformation
    pivot_a = ee_T.inverted().transform_point(grasp_pt)

    # pivot_b: grasp point in object local frame
    obj_T = target_obj.transformation
    pivot_b = obj_T.inverted().transform_point(grasp_pt)

    print(f"  Pivot on EE (local):     ({pivot_a[0]:.4f}, {pivot_a[1]:.4f}, {pivot_a[2]:.4f})")
    print(f"  Pivot on object (local): ({pivot_b[0]:.4f}, {pivot_b[1]:.4f}, {pivot_b[2]:.4f})")

    # Build constraint
    c = habitat_sim.physics.RigidConstraintSettings()
    c.object_id_a = fetch.sim_obj.object_id
    c.link_id_a = ee_link_id
    c.object_id_b = target_obj.object_id
    c.link_id_b = -1  # rigid object base
    c.pivot_a = pivot_a
    c.pivot_b = pivot_b
    c.max_impulse = 1000.0
    c.constraint_type = habitat_sim.physics.RigidConstraintType.Fixed

    # Set frame_a: object rotation in EE-link space (for fixed constraint stability)
    ee_rot = ee_T.rotation()
    obj_rot = obj_T.rotation()
    c.frame_a = ee_rot.inverted().__matmul__(obj_rot)
    c.frame_b = mn.Matrix3.identity_init()

    try:
        constraint_id = sim.create_rigid_constraint(c)
        print(f"  Object attached at grasp point (constraint ID={constraint_id})")
        return constraint_id
    except Exception as e:
        print(f"  WARNING: Failed to create constraint: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# VIDEO OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_video(frames, output_path, fps=15):
    """Save frames as MP4 video."""
    if not frames:
        print("  No frames to save")
        return

    try:
        import imageio
        imageio.mimwrite(str(output_path), frames, fps=fps, quality=8)
        print(f"  Video saved: {output_path.name} ({len(frames)} frames, {fps} fps)")
    except ImportError:
        print("  WARNING: imageio not installed, saving frames as images instead")
        frames_dir = output_path.parent / "frames"
        frames_dir.mkdir(exist_ok=True)
        from PIL import Image
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")
        print(f"  Frames saved to {frames_dir}/ ({len(frames)} images)")


def save_snapshot(sim, output_path, label=""):
    """Save a single RGB snapshot."""
    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3]
    from PIL import Image
    Image.fromarray(rgb).save(str(output_path))
    if label:
        print(f"  Snapshot: {output_path.name} — {label}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN GRASP EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: Execute grasp with Fetch robot arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python ../affordance-pipeline/execute_grasp.py --object mug --part handle\n\n"
            "Prerequisites:\n"
            "  1. Run scene_capture.py (Stage 1) first\n"
            "  2. Run visualize_affordance.py (Stage 2) first"
        ),
    )
    parser.add_argument("--object", required=True, choices=get_object_names(),
                        help="Object to grasp")
    parser.add_argument("--part", required=True,
                        help="Part of the object to grasp")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording")
    return parser.parse_args()


def main():
    args = parse_args()
    obj_name = args.object
    part_name = args.part

    # ── Load Stage 1 & 2 outputs ────────────────────────────────────
    print_header("Execute Grasp — Stage 3")
    print(f"  Object: {obj_name}")
    print(f"  Part:   {part_name}")
    print(f"  Output: {OUTPUT_DIR}")

    metadata = load_metadata()
    grasp_data = load_grasp_pose()

    # Verify consistency
    if grasp_data["object_name"] != obj_name:
        print(f"  WARNING: Grasp was for '{grasp_data['object_name']}', "
              f"but you specified '{obj_name}'")
    if grasp_data["part_name"] != part_name:
        print(f"  WARNING: Grasp was for part '{grasp_data['part_name']}', "
              f"but you specified '{part_name}'")

    grasp = grasp_data["grasp"]
    grasp_pos = np.array(grasp["position"])
    approach_dir = np.array(grasp["approach_dir"])
    approach_dir = approach_dir / np.linalg.norm(approach_dir)  # normalize
    grasp_rot = grasp.get("grasp_rotation", None)  # 3x3 rotation matrix (may be None for old data)

    print(f"\n  Grasp position:   ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
    print(f"  Approach dir:     ({approach_dir[0]:.3f}, {approach_dir[1]:.3f}, {approach_dir[2]:.3f})")
    print(f"  Grasp type:       {grasp['grasp_type']}")
    print(f"  Confidence:       {grasp['confidence']:.1%}")
    print(f"  Orientation data: {'available' if grasp_rot else 'not available (position-only IK)'}")

    # ── Setup simulator ─────────────────────────────────────────────
    print_header("Setting up simulator")
    cfg = make_sim_config()
    sim = habitat_sim.Simulator(cfg)
    print(f"  Scene: {SCENE_ID}")

    try:
        # ── Restore agent position from Stage 1 ─────────────────────
        agent = sim.get_agent(0)
        agent_pos = np.array(metadata["agent_position"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = agent_pos
        agent_state.rotation = np.quaternion(1, 0, 0, 0)
        agent.set_state(agent_state)

        # ── Spawn object at same position as Stage 1 ────────────────
        print_header("Spawning object")
        obj_info = metadata["spawned_objects"][0]
        obj_cfg = get_object(obj_name)
        config_path = obj_cfg["ycb_config"]

        obj_templates_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()

        template_ids = obj_templates_mgr.load_configs(config_path)
        template_handle = obj_templates_mgr.get_template_handles(config_path)[0]

        target_obj = rigid_obj_mgr.add_object_by_template_handle(template_handle)
        obj_world_pos = mn.Vector3(*obj_info["position"])
        target_obj.translation = obj_world_pos
        # Keep STATIC during arm approach so it doesn't get knocked around
        target_obj.motion_type = MotionType.STATIC

        print(f"  Object '{obj_name}' at ({obj_world_pos[0]:.3f}, "
              f"{obj_world_pos[1]:.3f}, {obj_world_pos[2]:.3f})")

        # Diagnostic: show grasp position relative to object
        grasp_vs_obj = grasp_pos - np.array([obj_world_pos[0], obj_world_pos[1], obj_world_pos[2]])
        print(f"  Grasp offset from obj center: "
              f"({grasp_vs_obj[0]:.3f}, {grasp_vs_obj[1]:.3f}, {grasp_vs_obj[2]:.3f})")
        print(f"  Grasp height above obj center: {grasp_vs_obj[1]:.3f}m")

        # ── Spawn Fetch robot (high-level FetchRobot wrapper) ───────
        print_header("Spawning Fetch robot")

        # Position robot so the grasp target is within the arm's workspace.
        # Fetch arm extends ~0.7m forward from base.
        grasp_world = mn.Vector3(*grasp_pos)

        # Robot base: same X as grasp, offset +Z (behind the grasp),
        # at floor level.  Robot faces -Z by default.
        robot_pos = mn.Vector3(
            grasp_world[0],
            agent_pos[1],        # floor level
            grasp_world[2] + 0.6,  # 60cm behind grasp (robot faces -Z)
        )

        fetch = create_fetch_robot(sim, robot_pos)

        # Let physics settle
        for _ in range(SETTLE_STEPS):
            sim.step_physics(PHYSICS_DT)

        # ── Position camera to see the action ───────────────────────
        # Front-side view: in front of the object, looking back at
        # the robot arm reaching toward the mug.
        # Robot faces -Z, so "in front of object" = further -Z.
        cam_pos = np.array([
            obj_world_pos[0] + 0.6,    # to the right side
            obj_world_pos[1] + 0.3,    # slightly above the object (table level + 0.3m)
            obj_world_pos[2] - 0.5,    # in front of the object (-Z)
        ])
        agent_state.position = cam_pos

        # Look at the object (camera will see object in foreground, robot behind)
        look_target = np.array([obj_world_pos[0], obj_world_pos[1], obj_world_pos[2]])
        dx = look_target[0] - cam_pos[0]
        dz = look_target[2] - cam_pos[2]
        yaw = np.arctan2(-dx, -dz)
        half_yaw = yaw / 2.0
        agent_state.rotation = np.quaternion(
            np.cos(half_yaw), 0, np.sin(half_yaw), 0
        )
        agent.set_state(agent_state)
        print(f"  Camera at ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}), "
              f"yaw={np.degrees(yaw):.1f}°")

        # ── Setup IK ────────────────────────────────────────────────
        print_header("Setting up IK solver")
        ik_helper = setup_ik_helper()

        # Check initial EE position (uses FetchRobot.ee_transform with offset)
        ee_init = get_ee_position(fetch)
        print(f"  Initial EE: ({ee_init[0]:.3f}, {ee_init[1]:.3f}, {ee_init[2]:.3f})")
        print(f"  Grasp target: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"  Distance: {(ee_init - mn.Vector3(*grasp_pos)).length():.3f}m")

        # ── Save initial snapshot ───────────────────────────────────
        save_snapshot(sim, OUTPUT_DIR / "00_initial.png", "Initial scene")

        # ── Compute grasp waypoints ─────────────────────────────────
        print_header("Computing grasp trajectory")

        # Pre-grasp: offset along approach direction
        pre_grasp_pos = grasp_pos - approach_dir * PRE_GRASP_OFFSET
        # Grasp position
        final_grasp_pos = grasp_pos
        # Lift position: straight up from grasp
        lift_pos = grasp_pos + np.array([0.0, LIFT_HEIGHT, 0.0])

        print(f"  Pre-grasp:  ({pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, {pre_grasp_pos[2]:.3f})")
        print(f"  Grasp:      ({final_grasp_pos[0]:.3f}, {final_grasp_pos[1]:.3f}, {final_grasp_pos[2]:.3f})")
        print(f"  Lift:       ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")

        # Solve IK for each waypoint (position-only).
        ik_pre    = solve_ik(ik_helper, pre_grasp_pos, fetch)
        ik_grasp  = solve_ik(ik_helper, final_grasp_pos, fetch)
        ik_lift   = solve_ik(ik_helper, lift_pos, fetch)

        # Adjust wrist_roll to align finger direction with GraspNet target.
        # This corrects jaw orientation without changing EE position.
        if grasp_rot is not None:
            print(f"\n  Adjusting wrist roll for finger alignment:")
            ik_pre   = adjust_wrist_roll(ik_pre, grasp_rot, fetch)
            ik_grasp = adjust_wrist_roll(ik_grasp, grasp_rot, fetch)

        print(f"\n  IK solutions computed")

        # Verify IK solutions: temporarily set joints, check actual EE
        for label, ik_sol, target in [
            ("Pre-grasp", ik_pre, pre_grasp_pos),
            ("Grasp", ik_grasp, final_grasp_pos),
            ("Lift", ik_lift, lift_pos),
        ]:
            fetch.arm_joint_pos = ik_sol
            actual_ee = get_ee_position(fetch)
            err = (actual_ee - mn.Vector3(*target)).length()
            print(f"  {label:12s} actual EE error: {err:.4f}m")
        # Restore arm to init
        fetch.arm_joint_pos = fetch.params.arm_init_params

        # ── Verify gripper orientation at grasp ─────────────────────
        # Temporarily set arm to grasp solution to check jaw direction
        fetch.arm_joint_pos = ik_grasp
        ee_T = fetch.ee_transform()
        ee_rot = np.array([[ee_T[i][j] for j in range(3)] for i in range(3)])
        # gripper_link frame: X=forward(approach), Y=fingers, Z=palm
        gripper_x = ee_rot[:, 0]  # approach direction
        gripper_y = ee_rot[:, 1]  # finger opening direction
        gripper_z = ee_rot[:, 2]  # palm normal

        print(f"\n  Gripper orientation at grasp position:")
        print(f"    Approach (X): ({gripper_x[0]:.3f}, {gripper_x[1]:.3f}, {gripper_x[2]:.3f})")
        print(f"    Fingers  (Y): ({gripper_y[0]:.3f}, {gripper_y[1]:.3f}, {gripper_y[2]:.3f})")
        print(f"    Palm     (Z): ({gripper_z[0]:.3f}, {gripper_z[1]:.3f}, {gripper_z[2]:.3f})")

        if grasp_rot is not None:
            target_rot = np.array(grasp_rot)
            target_approach = target_rot[:, 2]   # GraspNet Z
            target_fingers  = target_rot[:, 1]   # GraspNet Y
            # Angle between actual and target approach directions
            cos_approach = np.clip(np.dot(gripper_x, target_approach), -1, 1)
            cos_fingers  = np.clip(np.dot(gripper_y, target_fingers), -1, 1)
            print(f"    Target approach: ({target_approach[0]:.3f}, {target_approach[1]:.3f}, {target_approach[2]:.3f})")
            print(f"    Target fingers:  ({target_fingers[0]:.3f}, {target_fingers[1]:.3f}, {target_fingers[2]:.3f})")
            print(f"    Approach angle:  {np.degrees(np.arccos(cos_approach)):.1f}°")
            print(f"    Fingers angle:   {np.degrees(np.arccos(abs(cos_fingers))):.1f}° "
                  f"(symmetric)")

        fetch.arm_joint_pos = fetch.params.arm_init_params

        # ── Execute grasp sequence ──────────────────────────────────
        frames = [] if not args.no_video else None

        # Capture initial frames
        if frames is not None:
            for _ in range(15):  # ~1 second of initial view
                obs = sim.get_sensor_observations()
                frames.append(obs["rgb"][:, :, :3].copy())

        # Phase 1: Move to pre-grasp
        print_header("Phase 1: Approach (pre-grasp)")
        execute_motion(sim, fetch, ik_pre, agent, duration_sec=2.0,
                       frames=frames, label="Approach")
        save_snapshot(sim, OUTPUT_DIR / "01_pre_grasp.png", "Pre-grasp position")

        # Phase 2: Move to grasp position
        print_header("Phase 2: Final approach")
        execute_motion(sim, fetch, ik_grasp, agent, duration_sec=1.5,
                       frames=frames, label="Final approach")
        save_snapshot(sim, OUTPUT_DIR / "02_at_grasp.png", "At grasp position")

        # Phase 3: Close gripper + magic grasp
        print_header("Phase 3: Grasp")

        # Switch object to DYNAMIC so it can be picked up
        target_obj.motion_type = MotionType.DYNAMIC

        execute_gripper(sim, fetch, "close", agent, duration_sec=0.5,
                        frames=frames)

        # Try magic grasp — attach object at the handle grasp point
        constraint_id = magic_grasp(sim, fetch, target_obj, final_grasp_pos)
        save_snapshot(sim, OUTPUT_DIR / "03_grasped.png", "Object grasped")

        # Phase 4: Lift
        print_header("Phase 4: Lift")
        execute_motion(sim, fetch, ik_lift, agent, duration_sec=2.0,
                       frames=frames, label="Lift")
        save_snapshot(sim, OUTPUT_DIR / "04_lifted.png", "Object lifted")

        # Hold for a moment
        if frames is not None:
            for _ in range(30):
                obs = sim.get_sensor_observations()
                frames.append(obs["rgb"][:, :, :3].copy())

        # ── Save video ──────────────────────────────────────────────
        if frames is not None:
            print_header("Saving video")
            save_video(frames, OUTPUT_DIR / f"grasp_{obj_name}_{part_name}.mp4")

        # ── Save execution log ──────────────────────────────────────
        print_header("Saving execution log")
        ee_final = get_ee_position(fetch)
        obj_final = target_obj.translation

        log = {
            "object_name": obj_name,
            "part_name": part_name,
            "grasp_pose": grasp,
            "robot": {
                "type": "Fetch (FetchRobot wrapper)",
                "base_position": [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])],
                "ee_initial": [float(ee_init[0]), float(ee_init[1]), float(ee_init[2])],
                "ee_final": [float(ee_final[0]), float(ee_final[1]), float(ee_final[2])],
            },
            "waypoints": {
                "pre_grasp": pre_grasp_pos.tolist(),
                "grasp": final_grasp_pos.tolist(),
                "lift": lift_pos.tolist(),
            },
            "object_final_position": [float(obj_final[0]), float(obj_final[1]), float(obj_final[2])],
            "magic_grasp_used": constraint_id is not None,
        }

        log_path = OUTPUT_DIR / "execution_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Saved: execution_log.json")

        # ── Summary ─────────────────────────────────────────────────
        print_header("STAGE 3 COMPLETE")
        print(f"  Object:      {obj_cfg['display_name']}")
        print(f"  Part:        {part_name}")
        print(f"  Grasp type:  {grasp['grasp_type']} ({grasp['confidence']:.0%})")
        print(f"  EE moved:    ({ee_init[0]:.3f},{ee_init[1]:.3f},{ee_init[2]:.3f}) → "
              f"({ee_final[0]:.3f},{ee_final[1]:.3f},{ee_final[2]:.3f})")

        obj_moved = (obj_final - obj_world_pos).length()
        print(f"  Object moved: {obj_moved:.3f}m")

        if obj_final[1] > obj_world_pos[1] + 0.05:
            print(f"  ✓ Object lifted {obj_final[1] - obj_world_pos[1]:.3f}m above start")
        else:
            print(f"  ✗ Object not lifted (may need robot position adjustment)")

        print(f"  Output: {OUTPUT_DIR}/")
        for p in sorted(OUTPUT_DIR.iterdir()):
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                print(f"    {p.name:45s} ({size_kb:.1f} KB)")

    finally:
        sim.close()


if __name__ == "__main__":
    main()
