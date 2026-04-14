"""
Core modules for the Affordance Pipeline.

This package contains the modularized components:
  - SceneCapture:       Habitat-Sim scene setup and sensor capture
  - AffordanceDetector: UAD unsupervised affordance detection
  - CLIPSegDetector:    CLIPSeg text-guided segmentation
  - GraspPlanner:       Geometric + GraspNet grasp proposals
  - RobotExecutor:      Fetch robot IK, motion planning, and grasp execution
"""

from .scene_capture import SceneCapture
from .affordance_detector import AffordanceDetector
from .clipseg_detector import CLIPSegDetector
from .grasp_planner import GraspPlanner
from .robot_executor import RobotExecutor
