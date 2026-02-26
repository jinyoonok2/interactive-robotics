"""
RobotExecutor — Fetch robot arm control, IK, and grasp execution.

Handles:
  - FetchRobot wrapper creation and configuration
  - Inverse kinematics (position-only + wrist roll correction)
  - Motion planning and execution (joint interpolation)
  - Magic grasp (rigid constraint attachment)
  - Video capture and snapshot saving

The executor is independent of the affordance detection and grasp
planning stages — it takes a grasp pose and executes it.

Usage:
    executor = RobotExecutor(sim)
    executor.spawn_robot(position)
    executor.execute_grasp(grasp_pos, approach_dir, grasp_rotation, target_obj)
"""

import sys
import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List

import habitat_sim
from habitat_sim.physics import MotionType
import magnum as mn

from habitat.articulated_agents.robots.fetch_robot import FetchRobot
from habitat.tasks.rearrange.utils import IkHelper

PIPELINE_DIR = Path(__file__).resolve().parent.parent

# Fetch robot URDFs
FETCH_URDF     = "data/robots/hab_fetch/robots/hab_fetch.urdf"
FETCH_ARM_URDF = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"


class RobotExecutor:
    """Controls a Fetch robot arm for grasp execution in Habitat-Sim."""

    # Motion parameters
    PHYSICS_DT       = 1.0 / 240.0
    CTRL_FREQ        = 120
    RENDER_EVERY     = 8
    SETTLE_STEPS     = 120

    # Grasp parameters
    PRE_GRASP_OFFSET = 0.12   # standoff distance (meters)
    LIFT_HEIGHT      = 0.20   # lift after grasp (meters)

    def __init__(self, sim: habitat_sim.Simulator):
        """
        Initialize robot executor.

        Args:
            sim: Active Habitat simulator instance
        """
        self.sim = sim
        self.fetch = None
        self.ik_helper = None

        # Output directory
        self.output_dir = PIPELINE_DIR / "results" / "execution"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # ROBOT SETUP
    # ════════════════════════════════════════════════════════════════

    def spawn_robot(self, position: mn.Vector3, rotation: mn.Quaternion = None):
        """
        Spawn a Fetch robot at the given position.

        Args:
            position: World position for the robot base
            rotation: Optional rotation quaternion
        """
        agent_cfg = SimpleNamespace(articulated_agent_urdf=FETCH_URDF)
        self.fetch = FetchRobot(agent_cfg, self.sim, limit_robo_joints=True, fixed_base=True)
        self.fetch.reconfigure()

        self.fetch.sim_obj.translation = position
        if rotation is not None:
            self.fetch.sim_obj.rotation = rotation

        self.fetch.sim_obj.motion_type = MotionType.DYNAMIC
        self.fetch.gripper_joint_pos = self.fetch.params.gripper_open_state
        self.fetch.update()

        # Settle physics
        for _ in range(self.SETTLE_STEPS):
            self.sim.step_physics(self.PHYSICS_DT)

        print(f"  Fetch robot spawned at "
              f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        print(f"  Arm joints:  {self.fetch.params.arm_joints}")
        print(f"  EE link:     {self.fetch.params.ee_links}")
        print(f"  EE offset:   {self.fetch.params.ee_offset}")

    def setup_ik(self):
        """Initialize the PyBullet IK solver."""
        arm_init = np.array([-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005])
        self.ik_helper = IkHelper(FETCH_ARM_URDF, arm_init)
        print("  IK solver initialized (PyBullet)")

    # ════════════════════════════════════════════════════════════════
    # PROPERTIES / ACCESSORS
    # ════════════════════════════════════════════════════════════════

    @property
    def ee_position(self) -> mn.Vector3:
        """Current end-effector world position (with EE offset)."""
        return self.fetch.ee_transform().translation

    @property
    def ee_transform(self):
        """Current end-effector transformation matrix."""
        return self.fetch.ee_transform()

    @property
    def arm_joint_pos(self) -> np.ndarray:
        """Current arm joint positions."""
        return np.array(self.fetch.arm_joint_pos)

    @arm_joint_pos.setter
    def arm_joint_pos(self, joints: np.ndarray):
        """Set arm joint positions."""
        self.fetch.arm_joint_pos = joints

    # ════════════════════════════════════════════════════════════════
    # INVERSE KINEMATICS
    # ════════════════════════════════════════════════════════════════

    def solve_ik(self, target_world: np.ndarray, max_iters: int = 10,
                 tol: float = 0.005) -> np.ndarray:
        """
        Solve IK with iterative closed-loop correction (position-only).

        Args:
            target_world: Target EE position in world coordinates [x, y, z]
            max_iters:    Maximum correction iterations
            tol:          Position error tolerance (meters)

        Returns:
            7-element joint array for the arm
        """
        original_arm = self.arm_joint_pos.copy()

        target_np = np.array(target_world, dtype=float)
        target_mn = mn.Vector3(*target_np)

        ik_target = target_np.copy()
        best_sol = None
        best_err = float('inf')

        for iteration in range(max_iters):
            target_base = self._world_to_robot_base(ik_target)
            seed = original_arm if best_sol is None else best_sol
            self.ik_helper.set_arm_state(seed)
            ik_sol = np.array(self.ik_helper.calc_ik(target_base))
            self.fetch.arm_joint_pos = ik_sol

            actual_ee = self.ee_position
            error = target_mn - actual_ee
            err_mag = error.length()

            if err_mag < best_err:
                best_err = err_mag
                best_sol = ik_sol.copy()
            if err_mag < tol:
                break
            ik_target += np.array([error[0], error[1], error[2]])

        self.fetch.arm_joint_pos = original_arm
        print(f"    IK position: {best_err:.4f}m ({iteration + 1} iters)")
        return best_sol

    def adjust_wrist_roll(self, ik_sol: np.ndarray, grasp_rot_world) -> np.ndarray:
        """
        Adjust the wrist_roll joint to make gripper fingers span ACROSS
        the graspable feature, WITHOUT changing EE position.

        Uses the GraspNet binormal (col0) as the local handle/surface axis.
        Fingers are aligned PERPENDICULAR to this axis.

        This is object-agnostic — it works for any object where the
        rotation matrix encodes local surface geometry.

        Args:
            ik_sol:          7-element joint array from position-only IK
            grasp_rot_world: 3x3 rotation matrix (world frame)
                             col0=binormal, col1=baseline, col2=approach

        Returns:
            Modified 7-element joint array with corrected wrist_roll
        """
        if grasp_rot_world is None:
            return ik_sol

        grasp_rot = np.array(grasp_rot_world)
        handle_axis = grasp_rot[:, 0]   # binormal = handle bar direction
        gn_fingers = grasp_rot[:, 1]    # GraspNet finger direction (reference)

        # Get current gripper frame
        original_arm = self.arm_joint_pos.copy()
        self.fetch.arm_joint_pos = ik_sol

        ee_T = self.fetch.ee_transform()
        ee_rot = np.array([[ee_T[i][j] for j in range(3)] for i in range(3)])
        curr_x = ee_rot[:, 0]  # approach (wrist roll axis)
        curr_y = ee_rot[:, 1]  # current finger direction
        curr_z = ee_rot[:, 2]  # current palm normal

        self.fetch.arm_joint_pos = original_arm

        # Project handle axis onto YZ plane (perpendicular to approach)
        handle_proj = handle_axis - np.dot(handle_axis, curr_x) * curr_x
        hp_norm = np.linalg.norm(handle_proj)

        if hp_norm < 1e-6:
            print(f"    Wrist roll: skipped (handle axis ∥ approach)")
            return ik_sol

        handle_proj /= hp_norm

        # Desired fingers: PERPENDICULAR to handle in YZ plane
        a = np.dot(handle_proj, curr_y)
        b = np.dot(handle_proj, curr_z)

        opt1 = -b * curr_y + a * curr_z    # +90° rotation
        opt2 =  b * curr_y - a * curr_z    # -90° rotation

        # Pick option closer to GraspNet's finger direction
        desired_y = opt1 if abs(np.dot(opt1, gn_fingers)) >= abs(np.dot(opt2, gn_fingers)) else opt2

        # Compute wrist_roll angle
        y_comp = np.dot(desired_y, curr_y)
        z_comp = np.dot(desired_y, curr_z)
        theta = np.arctan2(z_comp, y_comp)

        corrected = ik_sol.copy()
        corrected[6] += theta

        # Verify
        self.fetch.arm_joint_pos = corrected
        new_ee_T = self.fetch.ee_transform()
        new_rot = np.array([[new_ee_T[i][j] for j in range(3)] for i in range(3)])
        new_y = new_rot[:, 1]

        handle_perp_angle = abs(np.degrees(np.arccos(
            np.clip(abs(np.dot(new_y, handle_axis)), -1, 1))))
        gn_align_angle = np.degrees(np.arccos(
            np.clip(abs(np.dot(new_y, gn_fingers)), -1, 1)))
        pos_shift = (new_ee_T.translation - ee_T.translation).length()

        self.fetch.arm_joint_pos = original_arm

        print(f"    Wrist roll: Δ={np.degrees(theta):.1f}°, "
              f"fingers⊥handle={handle_perp_angle:.1f}° (target ~90°), "
              f"vs GraspNet={gn_align_angle:.1f}°, "
              f"pos shift={pos_shift*100:.1f}cm")

        return corrected

    def _world_to_robot_base(self, world_pos):
        """Convert world position to robot base frame for IK."""
        base_T = self.fetch.base_transformation
        local_pos = base_T.inverted().transform_point(mn.Vector3(*world_pos))
        return np.array([local_pos[0], local_pos[1], local_pos[2]])

    # ════════════════════════════════════════════════════════════════
    # MOTION EXECUTION
    # ════════════════════════════════════════════════════════════════

    def move_arm(self, target_joints: np.ndarray, duration_sec: float = 2.0,
                 frames: list = None, label: str = ""):
        """
        Move robot arm to target joint configuration.

        Args:
            target_joints: 7-element target joint array
            duration_sec:  Duration of motion
            frames:        Optional list to append video frames
            label:         Label for logging
        """
        current_arm = self.arm_joint_pos
        n_steps = int(duration_sec * self.CTRL_FREQ)
        waypoints = self._interpolate_joints(current_arm, target_joints, n_steps)

        ee_start = self.ee_position

        for step_i, wp in enumerate(waypoints):
            self.fetch.arm_joint_pos = wp
            self.sim.step_physics(self.PHYSICS_DT)

            if frames is not None and step_i % self.RENDER_EVERY == 0:
                obs = self.sim.get_sensor_observations()
                frames.append(obs["rgb"][:, :, :3].copy())

        ee_end = self.ee_position
        dist = (ee_end - ee_start).length()
        if label:
            print(f"  {label}: EE moved {dist:.4f}m "
                  f"({ee_end[0]:.3f}, {ee_end[1]:.3f}, {ee_end[2]:.3f})")

    def close_gripper(self, duration_sec: float = 0.5, frames: list = None):
        """Close the gripper."""
        self._move_gripper("close", duration_sec, frames)

    def open_gripper(self, duration_sec: float = 0.5, frames: list = None):
        """Open the gripper."""
        self._move_gripper("open", duration_sec, frames)

    def _move_gripper(self, state: str, duration_sec: float, frames: list):
        """Move gripper to open or closed state."""
        target = (np.array(self.fetch.params.gripper_open_state) if state == "open"
                  else np.array(self.fetch.params.gripper_closed_state))

        current = np.array(self.fetch.gripper_joint_pos)
        n_steps = int(duration_sec * self.CTRL_FREQ)
        waypoints = self._interpolate_joints(current, target, n_steps)

        for step_i, wp in enumerate(waypoints):
            self.fetch.gripper_joint_pos = wp
            self.sim.step_physics(self.PHYSICS_DT)

            if frames is not None and step_i % self.RENDER_EVERY == 0:
                obs = self.sim.get_sensor_observations()
                frames.append(obs["rgb"][:, :, :3].copy())

        print(f"  Gripper {'opened' if state == 'open' else 'closed'}")

    @staticmethod
    def _interpolate_joints(start, end, steps):
        """Linear interpolation between joint configurations."""
        alphas = np.linspace(0, 1, steps)
        return [start + a * (end - start) for a in alphas]

    # ════════════════════════════════════════════════════════════════
    # MAGIC GRASP (snap-to constraint)
    # ════════════════════════════════════════════════════════════════

    def attach_object(self, target_obj, grasp_world_pos: np.ndarray,
                      grasp_threshold: float = 0.30) -> Optional[int]:
        """
        Attach target object to EE using a rigid constraint at the grasp point.

        Args:
            target_obj:      Habitat rigid object to attach
            grasp_world_pos: World position of the grasp point
            grasp_threshold: Maximum allowed EE-to-grasp distance

        Returns:
            Constraint ID, or None if failed
        """
        ee_pos = self.ee_position
        grasp_pt = mn.Vector3(*grasp_world_pos)

        dist_ee_grasp = (ee_pos - grasp_pt).length()
        dist_ee_obj = (ee_pos - target_obj.translation).length()

        print(f"  EE → grasp target = {dist_ee_grasp:.3f}m")
        print(f"  EE → object center = {dist_ee_obj:.3f}m")

        if dist_ee_grasp > grasp_threshold:
            print(f"  WARNING: EE is {dist_ee_grasp:.3f}m from grasp target "
                  f"(threshold={grasp_threshold}m)")

        # Compute pivots
        ee_link_id = self.fetch.ee_link_id()
        ee_T = self.fetch.sim_obj.get_link_scene_node(ee_link_id).transformation
        pivot_a = ee_T.inverted().transform_point(grasp_pt)

        obj_T = target_obj.transformation
        pivot_b = obj_T.inverted().transform_point(grasp_pt)

        print(f"  Pivot on EE (local):     ({pivot_a[0]:.4f}, {pivot_a[1]:.4f}, {pivot_a[2]:.4f})")
        print(f"  Pivot on object (local): ({pivot_b[0]:.4f}, {pivot_b[1]:.4f}, {pivot_b[2]:.4f})")

        # Build constraint
        c = habitat_sim.physics.RigidConstraintSettings()
        c.object_id_a = self.fetch.sim_obj.object_id
        c.link_id_a = ee_link_id
        c.object_id_b = target_obj.object_id
        c.link_id_b = -1
        c.pivot_a = pivot_a
        c.pivot_b = pivot_b
        c.max_impulse = 1000.0
        c.constraint_type = habitat_sim.physics.RigidConstraintType.Fixed

        ee_rot = ee_T.rotation()
        obj_rot = obj_T.rotation()
        c.frame_a = ee_rot.inverted().__matmul__(obj_rot)
        c.frame_b = mn.Matrix3.identity_init()

        try:
            constraint_id = self.sim.create_rigid_constraint(c)
            print(f"  Object attached at grasp point (constraint ID={constraint_id})")
            return constraint_id
        except Exception as e:
            print(f"  WARNING: Failed to create constraint: {e}")
            return None

    # ════════════════════════════════════════════════════════════════
    # FULL GRASP EXECUTION
    # ════════════════════════════════════════════════════════════════

    def execute_grasp(
        self,
        grasp_pos: np.ndarray,
        approach_dir: np.ndarray,
        target_obj,
        grasp_rotation=None,
        record_video: bool = True,
    ) -> dict:
        """
        Execute full grasp sequence: approach → grasp → lift.

        Args:
            grasp_pos:      World position of the grasp point
            approach_dir:   Approach direction unit vector
            target_obj:     Habitat rigid object to grasp
            grasp_rotation: Optional 3x3 rotation matrix
            record_video:   Whether to capture video frames

        Returns:
            Dict with execution results (positions, constraint_id, etc.)
        """
        if self.ik_helper is None:
            self.setup_ik()

        frames = [] if record_video else None

        # Compute waypoints
        pre_grasp_pos = grasp_pos - approach_dir * self.PRE_GRASP_OFFSET
        lift_pos = grasp_pos + np.array([0.0, self.LIFT_HEIGHT, 0.0])

        print(f"  Pre-grasp:  ({pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, {pre_grasp_pos[2]:.3f})")
        print(f"  Grasp:      ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"  Lift:       ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")

        # Solve IK
        ik_pre   = self.solve_ik(pre_grasp_pos)
        ik_grasp = self.solve_ik(grasp_pos)
        ik_lift  = self.solve_ik(lift_pos)

        # Wrist roll correction
        if grasp_rotation is not None:
            print(f"\n  Adjusting wrist roll for finger alignment:")
            ik_pre   = self.adjust_wrist_roll(ik_pre, grasp_rotation)
            ik_grasp = self.adjust_wrist_roll(ik_grasp, grasp_rotation)

        print(f"\n  IK solutions computed")

        # Verify
        for label, ik_sol, target in [
            ("Pre-grasp", ik_pre, pre_grasp_pos),
            ("Grasp", ik_grasp, grasp_pos),
            ("Lift", ik_lift, lift_pos),
        ]:
            self.fetch.arm_joint_pos = ik_sol
            actual_ee = self.ee_position
            err = (actual_ee - mn.Vector3(*target)).length()
            print(f"  {label:12s} actual EE error: {err:.4f}m")

        # Print orientation diagnostics
        self._print_orientation_diagnostics(ik_grasp, grasp_rotation)

        # Reset arm
        self.fetch.arm_joint_pos = self.fetch.params.arm_init_params

        # Capture initial frames
        if frames is not None:
            for _ in range(15):
                obs = self.sim.get_sensor_observations()
                frames.append(obs["rgb"][:, :, :3].copy())

        # Phase 1: Approach
        print(f"\n  Phase 1: Approach (pre-grasp)")
        self.move_arm(ik_pre, duration_sec=2.0, frames=frames, label="Approach")
        self.save_snapshot("01_pre_grasp.png", "Pre-grasp position")

        # Phase 2: Final approach
        print(f"\n  Phase 2: Final approach")
        self.move_arm(ik_grasp, duration_sec=1.5, frames=frames, label="Final approach")
        self.save_snapshot("02_at_grasp.png", "At grasp position")

        # Phase 3: Grasp
        print(f"\n  Phase 3: Grasp")
        target_obj.motion_type = MotionType.DYNAMIC
        self.close_gripper(duration_sec=0.5, frames=frames)
        constraint_id = self.attach_object(target_obj, grasp_pos)
        self.save_snapshot("03_grasped.png", "Object grasped")

        # Phase 4: Lift
        print(f"\n  Phase 4: Lift")
        self.move_arm(ik_lift, duration_sec=2.0, frames=frames, label="Lift")
        self.save_snapshot("04_lifted.png", "Object lifted")

        # Hold
        if frames is not None:
            for _ in range(30):
                obs = self.sim.get_sensor_observations()
                frames.append(obs["rgb"][:, :, :3].copy())

        return {
            "frames": frames,
            "constraint_id": constraint_id,
            "ik_solutions": {
                "pre_grasp": ik_pre,
                "grasp": ik_grasp,
                "lift": ik_lift,
            },
            "waypoints": {
                "pre_grasp": pre_grasp_pos.tolist(),
                "grasp": grasp_pos.tolist(),
                "lift": lift_pos.tolist(),
            },
        }

    def _print_orientation_diagnostics(self, ik_grasp, grasp_rotation):
        """Print gripper orientation at grasp position."""
        self.fetch.arm_joint_pos = ik_grasp
        ee_T = self.fetch.ee_transform()
        ee_rot = np.array([[ee_T[i][j] for j in range(3)] for i in range(3)])
        gripper_x = ee_rot[:, 0]
        gripper_y = ee_rot[:, 1]
        gripper_z = ee_rot[:, 2]

        print(f"\n  Gripper orientation at grasp position:")
        print(f"    Approach (X): ({gripper_x[0]:.3f}, {gripper_x[1]:.3f}, {gripper_x[2]:.3f})")
        print(f"    Fingers  (Y): ({gripper_y[0]:.3f}, {gripper_y[1]:.3f}, {gripper_y[2]:.3f})")
        print(f"    Palm     (Z): ({gripper_z[0]:.3f}, {gripper_z[1]:.3f}, {gripper_z[2]:.3f})")

        if grasp_rotation is not None:
            target_rot = np.array(grasp_rotation)
            target_approach = target_rot[:, 2]
            target_fingers = target_rot[:, 1]
            cos_approach = np.clip(np.dot(gripper_x, target_approach), -1, 1)
            cos_fingers = np.clip(np.dot(gripper_y, target_fingers), -1, 1)
            print(f"    Target approach: ({target_approach[0]:.3f}, {target_approach[1]:.3f}, {target_approach[2]:.3f})")
            print(f"    Target fingers:  ({target_fingers[0]:.3f}, {target_fingers[1]:.3f}, {target_fingers[2]:.3f})")
            print(f"    Approach angle:  {np.degrees(np.arccos(cos_approach)):.1f}°")
            print(f"    Fingers angle:   {np.degrees(np.arccos(abs(cos_fingers))):.1f}° (symmetric)")

    # ════════════════════════════════════════════════════════════════
    # VIDEO & SNAPSHOTS
    # ════════════════════════════════════════════════════════════════

    def save_snapshot(self, filename: str, label: str = ""):
        """Save a single RGB snapshot."""
        obs = self.sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]
        from PIL import Image
        filepath = self.output_dir / filename
        Image.fromarray(rgb).save(str(filepath))
        if label:
            print(f"  Snapshot: {filename} — {label}")

    def save_video(self, frames: list, filename: str, fps: int = 15):
        """Save frames as MP4 video."""
        if not frames:
            print("  No frames to save")
            return

        filepath = self.output_dir / filename
        try:
            import imageio
            imageio.mimwrite(str(filepath), frames, fps=fps, quality=8)
            print(f"  Video saved: {filename} ({len(frames)} frames, {fps} fps)")
        except ImportError:
            print("  WARNING: imageio not installed, saving as images")
            frames_dir = filepath.parent / "frames"
            frames_dir.mkdir(exist_ok=True)
            from PIL import Image
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")

    def save_execution_log(self, obj_name, part_name, grasp, result,
                           robot_pos, ee_init, target_obj, obj_world_pos):
        """Save execution log as JSON."""
        ee_final = self.ee_position
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
            "waypoints": result["waypoints"],
            "object_final_position": [float(obj_final[0]), float(obj_final[1]), float(obj_final[2])],
            "magic_grasp_used": result["constraint_id"] is not None,
        }

        log_path = self.output_dir / "execution_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  Saved: execution_log.json")

    # ════════════════════════════════════════════════════════════════
    # CAMERA POSITIONING
    # ════════════════════════════════════════════════════════════════

    def position_camera(self, agent, obj_world_pos):
        """
        Position the camera for a good view of the grasp action.

        Args:
            agent:         Habitat agent
            obj_world_pos: Object world position (mn.Vector3)
        """
        cam_pos = np.array([
            obj_world_pos[0] + 0.6,
            obj_world_pos[1] + 0.3,
            obj_world_pos[2] - 0.5,
        ])

        agent_state = habitat_sim.AgentState()
        agent_state.position = cam_pos

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
