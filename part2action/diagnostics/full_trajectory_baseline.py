"""Eligibility for a successful reproducible baseline, independent of HDF5 fidelity."""
import numpy as np
from scipy.spatial.transform import Rotation


REPEAT_LIMITS = {'tcp_position_m': .001, 'tcp_rotation_deg': 1., 'finger_m': .001,
                 'arm_joint_rad': .001, 'object_position_m': .001, 'object_rotation_deg': 1.,
                 'object_joint_rad_or_m': .001}


def compare_repeats(first, second):
    for key in ['actions', 'applied_actions', 'replay_tcp', 'replay_joints',
                'replay_fingers', 'object_xyz', 'object_xyzw', 'object_joints']:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid or misaligned repeat arrays: ' + key)
    if not (np.array_equal(first['actions'], second['actions']) and
            np.array_equal(first['actions'], first['applied_actions']) and
            np.array_equal(second['actions'], second['applied_actions'])):
        raise ValueError('Repeats did not execute the same unchanged source actions')
    def position(a, b): return float(np.linalg.norm(a-b, axis=-1).max())
    def angle(a, b): return float(np.rad2deg((Rotation.from_quat(a).inv()*Rotation.from_quat(b)).magnitude()).max())
    joints_a, joints_b = first['object_joints'], second['object_joints']
    metrics = {
        'tcp_position_m': position(first['replay_tcp'][:, 4:], second['replay_tcp'][:, 4:]),
        'tcp_rotation_deg': angle(first['replay_tcp'][:, :4], second['replay_tcp'][:, :4]),
        'finger_m': float(np.max(np.abs(first['replay_fingers']-second['replay_fingers']))),
        'arm_joint_rad': float(np.max(np.abs(first['replay_joints']-second['replay_joints']))),
        'object_position_m': position(first['object_xyz'], second['object_xyz']),
        'object_rotation_deg': angle(first['object_xyzw'], second['object_xyzw']),
        'object_joint_rad_or_m': float(np.max(np.abs(joints_a[..., 0]-joints_b[..., 0]))) if joints_a.size else 0.,
    }
    return {'reproducible': all(metrics[k] <= limit for k, limit in REPEAT_LIMITS.items()),
            'metrics': metrics, 'limits': REPEAT_LIMITS,
            'compared_against': 'Another fresh execution with the same source, seed and controller; not the HDF5 path'}


def eligibility(summaries, comparison):
    complete = len(summaries) == 2 and all(
        row.get('status') == 'completed' and row.get('all_actions_identical_and_in_order') and
        row.get('source_action_count') == row.get('executed_action_count') for row in summaries)
    # Native episodes terminate on first success. Touch can be transient; do not
    # add a final-frame retention requirement after replaying the whole file.
    successful = complete and all(row.get('native_task_success_any') for row in summaries)
    return {'complete_replays': complete, 'both_original_tasks_successful': bool(successful),
            'eligible': bool(successful and comparison['reproducible']),
            'hdf5_tracking_error_is_gate': False}
