import numpy as np
from diagnostics.replay_validation import gain_hypothesis, repeat_metrics


def test_quaternion_sign_is_not_a_pose_change_and_units_stay_separate():
    a=np.tile([0.,0.,0.,1.,0.,0.,0.],(3,1));b=a.copy();b[:,:4]*=-1
    result=repeat_metrics(a,b)
    assert result['xyz_max_m']==0 and result['orientation_max_deg']==0
    b[2,4]=.001
    result=repeat_metrics(a,b)
    assert result['xyz_max_m']==.001 and result['first_xyz_divergence_frame']==2


def test_controller_hypotheses_preserve_commands_and_separate_changes():
    a=np.zeros((45,7));a[:,6]=.04;a[5:,0]=.3;a[25:,:3]=[.35,0,0];a[35:,6]=0
    original=a.copy();w={'start':5,'close':35,'end':45}
    old=gain_hypothesis(a,w,'legacy');g=gain_hypothesis(a,w,'closure_001');n=gain_hypothesis(a,w,'closure_001_pregrasp015')
    assert old[-1][1]==.1 and all(x[1]==.01 for x in g)
    assert all(x[0]==y[0] and x[2]==y[2] for x,y in zip(old,g))
    assert n[5][2]==.15 and g[5][2]==.2
    assert n[26]==g[26]
    np.testing.assert_array_equal(a,original)
