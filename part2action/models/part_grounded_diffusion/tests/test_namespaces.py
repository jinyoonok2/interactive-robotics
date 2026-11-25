"""Naming must preserve old class identity and keep the new family independent."""
import json
import subprocess
import sys
import unittest

from models.part2action_model import Part2ActionModel
from models.visual_pose_policy_v1 import VisualPosePolicyV1
from models.grounded_visual_pose_policy_v1 import GroundedVisualPosePolicyV1
from models.pose_conditioning_v1 import TargetPoseConditionerV1
from models.part_grounded_diffusion import (
    PartGroundedDiffusionPolicy, PartGroundedDiffusionWithPose,
    PartGroundedDiffusionPosePrototype)
from models.part_grounded_diffusion.gt_pose import TargetPoseConditioner
from models.pose_token_diffusion import (
    ARCHITECTURE_ID, PoseTokenDiffusionConfig, PoseTokenDiffusionPolicy)


class ModelNamespaceTests(unittest.TestCase):
    def test_existing_names_are_exact_aliases_not_wrapped_or_subclassed(self):
        self.assertIs(PartGroundedDiffusionPolicy, Part2ActionModel)
        self.assertIs(PartGroundedDiffusionWithPose, VisualPosePolicyV1)
        self.assertIs(PartGroundedDiffusionPosePrototype, GroundedVisualPosePolicyV1)
        self.assertIs(TargetPoseConditioner, TargetPoseConditionerV1)

    def test_new_family_has_its_own_name_and_configuration_identity(self):
        self.assertEqual(ARCHITECTURE_ID, 'pose_token_diffusion_v1')
        self.assertEqual(PoseTokenDiffusionPolicy.__module__, 'models.pose_token_diffusion.policy')
        self.assertFalse(issubclass(PoseTokenDiffusionPolicy, Part2ActionModel))
        with self.assertRaises(ValueError):
            PoseTokenDiffusionConfig.from_mapping({'architecture': 'pose_first_v1'})
        with self.assertRaises(ValueError):
            PoseTokenDiffusionConfig.from_mapping({'architecture': 'grounded_visual_pose_v1'})

    def test_package_names_and_preservation_check_do_not_eagerly_load_models(self):
        code = '''import json, sys
from models.part_grounded_diffusion.verify_sources import verify_legacy_sources
from models.pose_token_diffusion import ARCHITECTURE_ID, PoseTokenDiffusionConfig
assert 'torch' not in sys.modules
assert 'models.backbone' not in sys.modules
assert 'models.part2action_model' not in sys.modules
assert PoseTokenDiffusionConfig().architecture == ARCHITECTURE_ID
print(json.dumps(verify_legacy_sources()))
'''
        completed = subprocess.run([sys.executable, '-c', code], check=True,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(completed.stdout)['preserved_sources'], 30)


if __name__ == '__main__':
    unittest.main()
