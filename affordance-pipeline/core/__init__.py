"""
Core modules for the Affordance Pipeline.

This package contains the modularized components:
  - SceneCapture:       Habitat-Sim scene setup and sensor capture
  - AffordanceDetector: DINO + SAM language-guided part segmentation
  - GraspPlanner:       Geometric + GraspNet grasp proposals
  - RobotExecutor:      Fetch robot IK, motion planning, and grasp execution
"""

from .scene_capture import SceneCapture
from .affordance_detector import AffordanceDetector
from .grasp_planner import GraspPlanner
from .robot_executor import RobotExecutor
