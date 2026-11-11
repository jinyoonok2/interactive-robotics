import numpy as np
from diagnostics.full_trajectory_baseline import compare_repeats, eligibility


def trial():
    tcp=np.tile([0.,0.,0.,1.,.1,.2,.3],(3,1))
    actions=np.zeros((3,7))
    return {'actions':actions,'applied_actions':actions.copy(),'replay_tcp':tcp,
            'replay_joints':np.zeros((3,7)),'replay_fingers':np.zeros(3),
            'object_xyz':np.zeros((3,3)),'object_xyzw':np.tile([0.,0.,0.,1.],(3,1)),
            'object_joints':np.empty((3,0))}


def test_baseline_ignores_original_hdf5_path_error_but_requires_successful_repeats():
    comparison=compare_repeats(trial(),trial())
    summary={'status':'completed','all_actions_identical_and_in_order':True,
             'source_action_count':3,'executed_action_count':3,'native_task_success_any':True,
             'native_task_success_final':False,
             'tcp_position_rmse_mm':9999.}
    assert eligibility([summary,summary],comparison)['eligible']
    failed={**summary,'native_task_success_any':False}
    assert not eligibility([summary,failed],comparison)['eligible']


def test_reproducibility_compares_robot_and_object_motion_and_checks_source_actions():
    first,second=trial(),trial();second['object_xyz'][1,0]=.002
    assert not compare_repeats(first,second)['reproducible']
    second=trial();second['applied_actions'][0,0]=1
    try:compare_repeats(first,second)
    except ValueError:pass
    else:raise AssertionError('Changed commands must not qualify as a baseline repeat')
