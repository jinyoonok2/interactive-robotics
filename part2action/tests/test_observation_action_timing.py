"""Check that the audit distinguishes pre-action from post-action transitions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/audit_observation_action_timing.py'
spec = importlib.util.spec_from_file_location('timing_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TimingAuditTests(unittest.TestCase):
    def test_transition_distinguishes_post_and_pre_action(self):
        for motion_frame, expected_lag in [(16, 0), (17, 1)]:
            with self.subTest(motion_frame=motion_frame):
                actions = np.zeros((20, 7))
                actions[16:, 0] = .7
                poses = np.zeros((20, 7))
                poses[:, 3] = 1
                poses[motion_frame:, 4] = .03
                result = audit.inspect_transition(actions, poses, np.zeros(20))
                self.assertEqual(result['motion_minus_command_frames'], expected_lag)
                self.assertEqual(result['preceding_observation_index'], 15)
                self.assertIs(result['preceding_observation_sampled_stride2'], False)

    def test_no_approach_and_length_mismatch_are_not_positive_evidence(self):
        actions = np.zeros((3, 7))
        poses = np.zeros((3, 7))
        result = audit.inspect_transition(actions, poses, np.zeros(3))
        self.assertIsNone(result['first_approach_command'])
        self.assertIsNone(result['motion_minus_command_frames'])
        with self.assertRaisesRegex(ValueError, 'lengths differ'):
            audit.inspect_transition(actions, poses[:2], np.zeros(3))


if __name__ == '__main__':
    unittest.main()
