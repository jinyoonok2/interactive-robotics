import numpy as np

from data.geometry import bound_gripper_command


def test_bound_gripper_preserves_arm_pose_and_raw_predictions():
    for raw, expected in [(-.14, 0.), (.062, .04), (.02, .02), (0., 0.), (.04, .04)]:
        action = np.array([.2, .1, .3, -2., .5, 1., raw], dtype=np.float32)
        original = action.copy()
        bounded = bound_gripper_command(action)
        assert np.array_equal(action, original)
        assert np.array_equal(bounded[:6], action[:6])
        assert np.isclose(bounded[6], expected)


def test_invalid_action_is_rejected_before_controller():
    for action in [np.zeros(6), np.array([0.] * 6 + [np.nan]), np.array([np.inf] + [0.] * 6)]:
        try:
            bound_gripper_command(action)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid action must not reach controller")
