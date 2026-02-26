"""
AffordancePipeline — Orchestrator for the full 3-stage pipeline.

Ties together:
  1. SceneCapture:       Setup scene, spawn object, capture sensors
  2. AffordanceDetector: DINO + SAM language-guided segmentation
  3. GraspPlanner:       Geometric / GraspNet grasp proposals
  4. RobotExecutor:      Fetch arm IK + motion + magic grasp

Can run all stages end-to-end, or individual stages.

Usage:
    # Full pipeline
    pipeline = AffordancePipeline()
    pipeline.run("mug", "handle")

    # Individual stages
    pipeline.run_capture("mug")
    pipeline.run_affordance("mug", "handle")
    pipeline.run_grasp("mug", "handle")
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import asdict

import habitat_sim
from habitat_sim.physics import MotionType
import magnum as mn

from core.scene_capture import SceneCapture
from core.affordance_detector import AffordanceDetector
from core.grasp_planner import GraspPlanner
from core.robot_executor import RobotExecutor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from objects import get_object, get_object_names, get_parts


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


PIPELINE_DIR = Path(__file__).resolve().parent


class AffordancePipeline:
    """Orchestrates the 3-stage affordance-based grasp pipeline."""

    def __init__(
        self,
        scene_id: str = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        """
        Initialize the pipeline.

        Args:
            scene_id:       HSSD scene ID
            box_threshold:  Grounding DINO detection threshold
            text_threshold: Grounding DINO text matching threshold
        """
        self.capture = SceneCapture(scene_id=scene_id)
        self.detector = AffordanceDetector(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        self.planner = GraspPlanner()

        # Paths for inter-stage data
        self.input_dir = PIPELINE_DIR / "output"
        self.result_dir = PIPELINE_DIR / "results" / "language"
        self.output_dir = PIPELINE_DIR / "results" / "execution"

    # ════════════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ════════════════════════════════════════════════════════════════

    def run(self, obj_name: str, part_name: str, prompt: str = None,
            record_video: bool = True):
        """
        Run the complete 3-stage pipeline.

        Args:
            obj_name:     Object name (e.g., "mug")
            part_name:    Part name (e.g., "handle")
            prompt:       Custom text prompt for affordance detection
            record_video: Whether to capture execution video
        """
        print_header(f"Full Pipeline: {obj_name}/{part_name}")

        self.run_capture(obj_name)
        self.run_affordance(obj_name, part_name, prompt=prompt)
        self.run_grasp(obj_name, part_name, record_video=record_video)

    # ════════════════════════════════════════════════════════════════
    # STAGE 1: SCENE CAPTURE
    # ════════════════════════════════════════════════════════════════

    def run_capture(self, obj_name: str):
        """
        Stage 1: Capture scene with object.

        Args:
            obj_name: Object to place in scene
        """
        print_header("Stage 1: Scene Capture")
        print(f"  Object: {obj_name}")

        try:
            self.capture.setup()
            agent_pos, obj_info = self.capture.spawn_object(obj_name)
            rgb, depth, semantic = self.capture.capture()
            self.capture.save(rgb, depth, semantic, obj_info)
            print_header("Stage 1 Complete")
        finally:
            self.capture.close()

    # ════════════════════════════════════════════════════════════════
    # STAGE 2: AFFORDANCE DETECTION + GRASP PLANNING
    # ════════════════════════════════════════════════════════════════

    def run_affordance(self, obj_name: str, part_name: str, prompt: str = None):
        """
        Stage 2: Detect affordance and plan grasp.

        Args:
            obj_name:  Object name
            part_name: Part to detect
            prompt:    Custom text prompt
        """
        print_header("Stage 2: Affordance Detection + Grasp Planning")
        print(f"  Object: {obj_name}")
        print(f"  Part:   {part_name}")

        # Load Stage 1 data
        metadata = self.capture.get_metadata()
        from PIL import Image as PILImage
        import open3d as o3d

        rgb = np.array(PILImage.open(self.input_dir / "rgb.png"))
        depth = np.load(self.input_dir / "depth_raw.npy")

        # Detect affordance
        print_header(f"Detecting: '{part_name}'")
        seg_result = self.detector.detect(
            obj_name, part_name, rgb, depth, metadata, prompt=prompt,
        )
        if seg_result is None:
            raise RuntimeError(f"Affordance detection failed for '{part_name}'")

        # Load object point cloud for grasp planning
        ply_files = list(self.input_dir.glob(f"object_{obj_name}_sem*.ply"))
        if ply_files:
            pcd = o3d.io.read_point_cloud(str(ply_files[0]))
            points = np.asarray(pcd.points)
        else:
            points = seg_result["world_points"]

        # Plan grasp
        print_header(f"Planning grasp for '{part_name}'")
        part_3d_points = seg_result["world_points"]
        part_indices = np.arange(len(part_3d_points))

        grasp = self.planner.plan(
            obj_name, part_name, part_3d_points, part_indices, metadata,
        )

        print(f"  Type:       {grasp.grasp_type}")
        print(f"  Position:   ({grasp.position[0]:.4f}, {grasp.position[1]:.4f}, {grasp.position[2]:.4f})")
        print(f"  Approach:   ({grasp.approach_dir[0]:.3f}, {grasp.approach_dir[1]:.3f}, {grasp.approach_dir[2]:.3f})")
        print(f"  Confidence: {grasp.confidence:.0%}")

        # Visualize
        print_header("Rendering visualization")
        img_annotated = self.detector.visualize(
            rgb, seg_result, obj_name, part_name,
            grasp=grasp, metadata=metadata,
        )

        # Save
        text_prompt = prompt or f"{part_name} of the {obj_name}"
        self.detector.save_results(
            rgb, img_annotated, seg_result, grasp,
            obj_name, part_name, metadata, text_prompt,
        )

        print_header("Stage 2 Complete")

    # ════════════════════════════════════════════════════════════════
    # STAGE 3: ROBOT EXECUTION
    # ════════════════════════════════════════════════════════════════

    def run_grasp(self, obj_name: str, part_name: str, record_video: bool = True):
        """
        Stage 3: Execute grasp with Fetch robot.

        Args:
            obj_name:     Object name
            part_name:    Part name
            record_video: Whether to capture video
        """
        print_header("Stage 3: Robot Execution")
        print(f"  Object: {obj_name}")
        print(f"  Part:   {part_name}")

        # Load Stage 1 & 2 data
        metadata = self.capture.get_metadata()
        grasp_path = self.result_dir / "grasp_poses.json"
        with open(grasp_path) as f:
            grasp_data = json.load(f)

        grasp = grasp_data["grasp"]
        grasp_pos = np.array(grasp["position"])
        approach_dir = np.array(grasp["approach_dir"])
        approach_dir /= np.linalg.norm(approach_dir)
        grasp_rot = grasp.get("grasp_rotation", None)

        print(f"  Grasp position: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"  Approach dir:   ({approach_dir[0]:.3f}, {approach_dir[1]:.3f}, {approach_dir[2]:.3f})")
        print(f"  Orientation:    {'available' if grasp_rot else 'position-only'}")

        # Setup simulator
        print_header("Setting up simulator")
        sim_cfg = self._make_exec_sim_config()
        sim = habitat_sim.Simulator(sim_cfg)

        try:
            # Restore agent
            agent = sim.get_agent(0)
            agent_pos = np.array(metadata["agent_position"])
            agent_state = habitat_sim.AgentState()
            agent_state.position = agent_pos
            agent_state.rotation = np.quaternion(1, 0, 0, 0)
            agent.set_state(agent_state)

            # Spawn object
            print_header("Spawning object")
            obj_info = metadata["spawned_objects"][0]
            obj_cfg = get_object(obj_name)
            target_obj = self._spawn_exec_object(sim, obj_name, obj_info)
            obj_world_pos = mn.Vector3(*obj_info["position"])

            # Spawn robot
            print_header("Spawning Fetch robot")
            executor = RobotExecutor(sim)
            robot_pos = mn.Vector3(
                grasp_pos[0],
                agent_pos[1],
                grasp_pos[2] + 0.6,
            )
            executor.spawn_robot(robot_pos)
            executor.setup_ik()

            ee_init = executor.ee_position

            # Position camera
            executor.position_camera(agent, obj_world_pos)
            executor.save_snapshot("00_initial.png", "Initial scene")

            # Execute grasp
            print_header("Executing grasp")
            result = executor.execute_grasp(
                grasp_pos, approach_dir, target_obj,
                grasp_rotation=grasp_rot,
                record_video=record_video,
            )

            # Save video
            if result["frames"]:
                print_header("Saving video")
                executor.save_video(result["frames"], f"grasp_{obj_name}_{part_name}.mp4")

            # Save log
            print_header("Saving execution log")
            executor.save_execution_log(
                obj_name, part_name, grasp, result,
                robot_pos, ee_init, target_obj, obj_world_pos,
            )

            # Summary
            print_header("STAGE 3 COMPLETE")
            ee_final = executor.ee_position
            obj_final = target_obj.translation
            obj_moved = (obj_final - obj_world_pos).length()

            print(f"  Object:      {obj_cfg['display_name']}")
            print(f"  Part:        {part_name}")
            print(f"  Grasp type:  {grasp['grasp_type']} ({grasp['confidence']:.0%})")
            print(f"  Object moved: {obj_moved:.3f}m")

            if obj_final[1] > obj_world_pos[1] + 0.05:
                print(f"  ✓ Object lifted {obj_final[1] - obj_world_pos[1]:.3f}m above start")
            else:
                print(f"  ✗ Object not lifted")

            print(f"  Output: {executor.output_dir}/")
            for p in sorted(executor.output_dir.iterdir()):
                if p.is_file():
                    size_kb = p.stat().st_size / 1024
                    print(f"    {p.name:45s} ({size_kb:.1f} KB)")

        finally:
            sim.close()

    # ════════════════════════════════════════════════════════════════
    # STAGE 3 HELPERS
    # ════════════════════════════════════════════════════════════════

    def _make_exec_sim_config(self):
        """Build sim config for execution (RGB sensor only)."""
        scene_dir = self.capture.scene_dir
        scene_id = self.capture.scene_id

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = f"{scene_dir}/scenes/{scene_id}.scene_instance.json"
        backend_cfg.scene_dataset_config_file = f"{scene_dir}/hssd-hab.scene_dataset_config.json"
        backend_cfg.enable_physics = True
        backend_cfg.gpu_device_id = 0

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [self.capture.sensor_height, self.capture.sensor_width]
        rgb_spec.hfov = mn.Deg(self.capture.hfov_deg)
        rgb_spec.position = [0.0, 0.0, 0.0]
        rgb_spec.orientation = [np.radians(-25), 0.0, 0.0]
        rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec]

        return habitat_sim.Configuration(backend_cfg, [agent_cfg])

    def _spawn_exec_object(self, sim, obj_name, obj_info):
        """Spawn object for execution stage."""
        obj_cfg = get_object(obj_name)
        config_path = obj_cfg["ycb_config"]

        obj_templates_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()

        obj_templates_mgr.load_configs(config_path)
        template_handle = obj_templates_mgr.get_template_handles(config_path)[0]

        target_obj = rigid_obj_mgr.add_object_by_template_handle(template_handle)
        obj_world_pos = mn.Vector3(*obj_info["position"])
        target_obj.translation = obj_world_pos
        target_obj.motion_type = MotionType.STATIC

        print(f"  Object '{obj_name}' at ({obj_world_pos[0]:.3f}, "
              f"{obj_world_pos[1]:.3f}, {obj_world_pos[2]:.3f})")
        return target_obj
