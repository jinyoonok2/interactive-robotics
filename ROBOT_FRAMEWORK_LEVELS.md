# Habitat-Lab Robot Framework Levels

Quick reference for the abstraction layers available for robot control in Habitat-Lab.

---

## Level 0 — Raw Simulator API

**What:** Load URDF manually, set `joint_positions` by raw indices, no motor management.

```python
robot = ao_mgr.add_articulated_object_from_urdf("fetch.urdf", ...)
jpos = robot.joint_positions
jpos[15] = -0.45  # hardcoded index
robot.joint_positions = jpos
```

**Pros:** Full control, no dependencies.  
**Cons:** Must hardcode every joint index, no EE offset, no motor sync, fragile.

---

## Level 1 — `Manipulator`

**File:** `habitat-lab/habitat/articulated_agents/manipulator.py`

**What:** Base class that wraps an articulated object with named joint access, EE transform with offset, gripper control, and motor management.

**Key properties/methods:**
| API | Description |
|-----|-------------|
| `arm_joint_pos` | Read/write arm joints (auto-syncs motor targets in DYNAMIC mode) |
| `gripper_joint_pos` | Read/write gripper joints |
| `ee_transform()` | End-effector `Matrix4` with `ee_offset` applied (e.g., +0.08m along local X for Fetch) |
| `ee_link_id` | Link ID of the EE (e.g., 22 for Fetch) |
| `close_gripper() / open_gripper()` | Interpolates gripper toward closed/open state |
| `params` | `MobileManipulatorParams` — all constants in one place |

---

## Level 2 — `MobileManipulator`

**File:** `habitat-lab/habitat/articulated_agents/mobile_manipulator.py`

**What:** Extends `Manipulator` + `ArticulatedAgentBase`. Adds base locomotion, camera transforms, and URDF loading via `agent_cfg`.

**Constructor:** `MobileManipulator(params, agent_cfg, sim, limit_robo_joints, fixed_base)`  
- `agent_cfg` must have `.articulated_agent_urdf` attribute (path to URDF file)

**Key additions:**
| API | Description |
|-----|-------------|
| `base_transformation` | Robot base frame (with -90° X correction for Fetch) |
| `base_pos / base_rot` | Get/set base position and rotation |
| `reconfigure()` | Loads URDF, configures motors, sets initial joint positions |

---

## Level 3 — (No distinct level)

---

## Level 4 — `FetchRobot` ⭐ (We use this)

**File:** `habitat-lab/habitat/articulated_agents/robots/fetch_robot.py`

**What:** Concrete `MobileManipulator` with all Fetch-specific constants pre-defined. Calls `_get_fetch_params()` to populate everything.

**Pre-defined constants (no hardcoding needed):**
```
arm_joints:        [15, 16, 17, 18, 19, 20, 21]
gripper_joints:    [23, 24]
ee_links:          [22]
ee_offset:         Vector3(0.08, 0, 0)    # fingertip offset
arm_init_params:   [-0.45, -1.08, 0.1, 0.935, -0.001, 1.573, 0.005]
gripper_open:      [0.04, 0.04]
gripper_closed:    [0.0, 0.0]
arm_mtr_pos_gain:  0.3
arm_mtr_vel_gain:  0.3
arm_mtr_max_impulse: 10.0
```

**`update()` method auto-locks:**
- Head pan → 0, Head tilt → π/2
- Back (torso_lift) → 0.15

**Usage (our current approach):**
```python
from habitat.articulated_agents.robots.fetch_robot import FetchRobot
from types import SimpleNamespace

agent_cfg = SimpleNamespace(articulated_agent_urdf=FETCH_URDF)
fetch = FetchRobot(agent_cfg, sim, limit_robo_joints=True, fixed_base=True)
fetch.reconfigure()
fetch.update()

# Read/write arm
fetch.arm_joint_pos = new_joints

# EE position (with 0.08m offset)
ee_pos = fetch.ee_transform().translation

# Gripper
fetch.gripper_joint_pos = fetch.params.gripper_open_state
```

---

## Level 5 — `ArticulatedAgentManager`

**File:** `habitat-lab/habitat/articulated_agents/articulated_agent_manager.py`

**What:** Bundles `FetchRobot` + `IkHelper` + `RearrangeGraspManager` into one manager. Used by the full Rearrange task system.

**What it adds:**
- Automatic IK helper creation from agent config
- `RearrangeGraspManager` for snap/desnap grasping
- Multi-agent support

**When to use:** Only if you're running inside the full `RearrangeSim` environment with habitat config files. Overkill for standalone scripts.

---

## Level 6 — `ArmEEAction` (RL Action Wrapper)

**File:** `habitat-lab/habitat/tasks/rearrange/actions/actions.py`

**What:** RL-style action space that converts EE position deltas → IK → joint targets. Used by RL training and HITL apps.

**Flow:** `action_delta` → `ee_target` (base frame) → `IkHelper.calc_ik()` → `arm_motor_pos`

**When to use:** Only for RL training or HITL interactive apps. Requires full `RearrangeSim` + task config + episode datasets.

---

## IkHelper (Standalone IK Solver)

**File:** `habitat-lab/habitat/tasks/rearrange/utils.py`

**What:** PyBullet DIRECT-mode IK solver. Works independently of the framework level.

**Key details:**
- Uses a **separate arm-only URDF** (`hab_fetch_arm.urdf`, 8 links)
- `pb_link_idx = 7` → targets `gripper_link` (not fingertip)
- `calc_ik(target_pos)` — target must be in **robot base frame**
- Returns 7-element joint array matching `FetchRobot.arm_joint_pos`

**Usage:**
```python
from habitat.tasks.rearrange.utils import IkHelper

ik_helper = IkHelper(robot_urdf, arm_start=0, arm_len=7, open_gripper_val=0.04)

# Convert world → base frame
base_T = fetch.base_transformation
local_target = base_T.inverted().transform_point(world_target)

# Solve
joints = ik_helper.calc_ik(local_target)
fetch.arm_joint_pos = joints
```

---

## Summary Table

| Level | Class | What You Get | When to Use |
|-------|-------|-------------|-------------|
| 0 | Raw sim API | Nothing — manual everything | Never (fragile) |
| 1 | `Manipulator` | Joint props, EE transform, motors | Building custom robots |
| 2 | `MobileManipulator` | + Base locomotion, cameras, URDF loading | Building custom mobile manipulators |
| **4** | **`FetchRobot`** ⭐ | **+ All Fetch constants, head/back locking** | **Standalone scripts (our choice)** |
| 5 | `ArticulatedAgentManager` | + IkHelper + GraspManager bundled | Full Rearrange task system |
| 6 | `ArmEEAction` | + RL action space | RL training / HITL apps |

**Our stack:** `FetchRobot` (Level 4) + standalone `IkHelper` + manual `RigidConstraintSettings` for grasping.

---

## Available Robots

| Robot | URDF | Status |
|-------|------|--------|
| `hab_fetch` | `data/robots/hab_fetch/robots/hab_fetch.urdf` | ✅ In use |
| `fetch_no_base` | Same arm, no wheels | Available |
| `hab_spot_arm` | Boston Dynamics Spot | Available (needs full IK rewrite) |
| `franka_panda` | 7-DoF arm | URDF not downloaded |
