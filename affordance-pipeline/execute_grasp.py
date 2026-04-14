"""
Execute Grasp — Stage 3 of the Affordance Pipeline
====================================================
Entry point for Stage 3. Uses the RobotExecutor class.

Usage:
    cd habitat-lab
    python ../affordance-pipeline/execute_grasp.py --object mug --part handle
    python ../affordance-pipeline/execute_grasp.py --object hammer --part handle --method clipseg
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import habitat_sim
from habitat_sim.physics import MotionType
import magnum as mn

from core.robot_executor import RobotExecutor
from objects import get_object_names, get_object


PIPELINE_DIR = Path(__file__).resolve().parent
INPUT_DIR    = PIPELINE_DIR / "output"

# Sim config
SENSOR_HEIGHT = 512
SENSOR_WIDTH  = 512
HFOV_DEG      = 70
SCENE_DIR     = "habitat-lab/data/versioned_data/hssd-hab"
SCENE_ID      = "102344250"
SCENE_FILE    = f"{SCENE_DIR}/scenes/{SCENE_ID}.scene_instance.json"
SCENE_DATASET = f"{SCENE_DIR}/hssd-hab.scene_dataset_config.json"


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Stage 3: Execute grasp with Fetch robot arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python ../affordance-pipeline/execute_grasp.py --object mug --part handle"
        ),
    )
    parser.add_argument("--object", required=True, choices=get_object_names(),
                        help="Object to grasp")
    parser.add_argument("--part", required=True,
                        help="Part of the object to grasp")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording")
    parser.add_argument("--method", choices=["clipseg", "uad"], default="clipseg",
                        help="Which affordance method's grasp to execute (default: clipseg)")
    return parser.parse_args()


def make_sim_config():
    """Build habitat-sim config for execution."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE_FILE
    backend_cfg.scene_dataset_config_file = SCENE_DATASET
    backend_cfg.enable_physics = True
    backend_cfg.gpu_device_id = 0

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [SENSOR_HEIGHT, SENSOR_WIDTH]
    rgb_spec.hfov = mn.Deg(HFOV_DEG)
    rgb_spec.position = [0.0, 0.0, 0.0]
    rgb_spec.orientation = [np.radians(-25), 0.0, 0.0]
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def main():
    args = parse_args()
    obj_name = args.object
    part_name = args.part

    # Per-object output directories
    method = args.method
    RESULT_DIR = PIPELINE_DIR / "results" / obj_name / method
    OUTPUT_DIR = PIPELINE_DIR / "results" / obj_name / "execution"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 & 2 data
    print_header("Execute Grasp — Stage 3")
    print(f"  Object: {obj_name}")
    print(f"  Part:   {part_name}")
    print(f"  Method: {method}")
    print(f"  Output: {OUTPUT_DIR}")

    meta_path = INPUT_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"  ERROR: metadata.json not found. Run Stage 1 first.")
        sys.exit(1)
    with open(meta_path) as f:
        metadata = json.load(f)

    grasp_path = RESULT_DIR / "grasp_poses.json"
    if not grasp_path.exists():
        print(f"  ERROR: grasp_poses.json not found. Run Stage 2 first.")
        sys.exit(1)
    with open(grasp_path) as f:
        grasp_data = json.load(f)

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
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    grasp_rot = grasp.get("grasp_rotation", None)

    print(f"\n  Grasp position:   ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
    print(f"  Approach dir:     ({approach_dir[0]:.3f}, {approach_dir[1]:.3f}, {approach_dir[2]:.3f})")
    print(f"  Grasp type:       {grasp['grasp_type']}")
    print(f"  Confidence:       {grasp['confidence']:.1%}")
    print(f"  Orientation data: {'available' if grasp_rot else 'not available (position-only IK)'}")

    # Setup simulator
    print_header("Setting up simulator")
    cfg = make_sim_config()
    sim = habitat_sim.Simulator(cfg)
    print(f"  Scene: {SCENE_ID}")

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

        grasp_vs_obj = grasp_pos - np.array([obj_world_pos[0], obj_world_pos[1], obj_world_pos[2]])
        print(f"  Grasp offset from obj center: "
              f"({grasp_vs_obj[0]:.3f}, {grasp_vs_obj[1]:.3f}, {grasp_vs_obj[2]:.3f})")

        # Spawn robot
        print_header("Spawning Fetch robot")
        executor = RobotExecutor(sim)
        executor.set_output_dir(obj_name)
        robot_pos = mn.Vector3(
            grasp_pos[0],
            agent_pos[1],
            grasp_pos[2] + 0.6,
        )
        executor.spawn_robot(robot_pos)
        executor.setup_ik()

        ee_init = executor.ee_position
        print(f"  Initial EE: ({ee_init[0]:.3f}, {ee_init[1]:.3f}, {ee_init[2]:.3f})")
        print(f"  Grasp target: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        print(f"  Distance: {(ee_init - mn.Vector3(*grasp_pos)).length():.3f}m")

        # Position camera
        executor.position_camera(agent, obj_world_pos)
        executor.save_snapshot("00_initial.png", "Initial scene")

        # Execute grasp
        print_header("Computing grasp trajectory")
        result = executor.execute_grasp(
            grasp_pos, approach_dir, target_obj,
            grasp_rotation=grasp_rot,
            record_video=not args.no_video,
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
        print(f"  EE moved:    ({ee_init[0]:.3f},{ee_init[1]:.3f},{ee_init[2]:.3f}) → "
              f"({ee_final[0]:.3f},{ee_final[1]:.3f},{ee_final[2]:.3f})")
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
