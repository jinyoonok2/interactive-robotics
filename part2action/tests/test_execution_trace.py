from types import SimpleNamespace

import numpy as np

from diagnostics.execution_trace import ExecutionTrace, can_expand, replay_agreement


def test_recorder_preserves_motor_calls_and_physics_cadence():
    class Pose:
        def to_list(self): return np.array([0., 0., 0., 1., 0., 0., 0.])
    class Client:
        def __init__(self): self.motors = []; self.steps = 0; self.filters = []
        def getBasePositionAndOrientation(self, uid): return [0.,0.,0.], [0.,0.,0.,1.]
        def getBaseVelocity(self, uid): return [0.,0.,0.], [0.,0.,0.]
        def getNumJoints(self, uid): return 2 if uid == 1 else 0
        def getJointStates(self, uid, indices): return [(0.,0.,None,0.) for _ in indices]
        def getJointState(self, uid, index): return 0.,0.
        def setJointMotorControl2(self, *args, **kwargs): self.motors.append((args,kwargs)); return 27
        def stepSimulation(self): self.steps += 1; return 13
        def setCollisionFilterPair(self, *args): self.filters.append(args); return 19
    class Robot:
        panda = 1
        def get_tcp_pose(self): return Pose()
        def get_gripper_state(self): return .04
        def apply_action(self, action, gain=.01, gain_gripper=.001, gain_near_target=.01):
            return p.setJointMotorControl2(1, 0, 2, float(action[0]), positionGain=gain, force=2400.)
    p=Client(); env=SimpleNamespace(robot=Robot(),obj=SimpleNamespace(uid=2),world=SimpleNamespace(p=p))
    trace=ExecutionTrace(env)
    try:
        with trace.scope('approach', {'gain':.005}):
            assert env.robot.apply_action([.2,0,0,0,0,0,.04], gain=.005)==27
            for _ in range(12): assert p.stepSimulation()==13
            trace.finish_action()
        assert len(trace.events)==1 and trace.events[0]['physics_steps']==12
        assert trace.events[0]['kwargs']=={'gain':.005,'gain_gripper':.001,'gain_near_target':.01}
        assert trace.events[0]['motors'][0]['kwargs']['positionGain']==.005
        assert p.motors==[((1,0,2,.2),{'positionGain':.005,'force':2400.})]
        assert trace.phase_events[0]['end_physics_step']==12
        assert p.setCollisionFilterPair(1,1,9,10,1)==19
        p.stepSimulation()
        assert p.setCollisionFilterPair(1,1,9,10,0)==19
        assert [e['kind'] for e in trace.events]==['action','client_call','physics','client_call']
        assert trace.events[2]['physics_steps']==1
        assert p.filters==[(1,1,9,10,1),(1,1,9,10,0)]
    finally: trace.close()
    env.robot.apply_action([0]*7)
    assert len(trace.events)==4 and len(p.motors)==2


def test_agreement_handles_quaternion_sign_and_detects_actual_motion():
    a=np.tile([0.,0.,0.,1.,.2,.3,.4],(3,1)); b=a.copy(); b[:,:4]*=-1
    assert replay_agreement(a,b)['max_tcp_rotation_deg']==0
    b[1,4]+=.002
    assert abs(replay_agreement(a,b)['max_tcp_position_m']-.002)<1e-10


def test_state_agreement_rejects_object_drift_even_when_robot_matches():
    import copy
    from diagnostics.execution_trace import state_trace_agreement
    state={'tcp_xyzw_xyz':[0,0,0,1,0,0,0], 'finger_m':.04,'robot_q':[0.]*12,
           'object_xyz':[0.,0.,.2],'object_xyzw':[0,0,0,1]}
    a=[{'kind':'action','after':state}];b=copy.deepcopy(a)
    b[0]['after']['object_xyz'][0]=.002
    result=state_trace_agreement(a,b)
    assert result['max_tcp_position_m']==0 and result['max_object_position_m']==.002


def test_expansion_requires_category_coverage_replay_and_both_labels():
    rows=[{'case':{'object':name},'generated_stable_success':True,'eligible':True,
           'variation_successes':2,'variation_failures':1} for name in ['bottle','mug','pliers','scissors']]
    assert can_expand(rows,['bottle','mug','pliers','scissors'])['passed']
    rows[-1]['eligible']=False
    assert not can_expand(rows,['bottle','mug','pliers','scissors'])['passed']
    rows[-1]['eligible']=True
    for r in rows:r['variation_failures']=0
    assert not can_expand(rows,['bottle','mug','pliers','scissors'])['passed']


def test_joint_conditions_have_real_multivariable_changes():
    from generate_logged_references import pilot_conditions
    rows=pilot_conditions()
    assert len(rows)==18 and rows[0]['name']=='identity'
    joint=[r for r in rows if r['name'].startswith('joint_')]
    assert len(joint)==2
    assert all(np.any(r['translation_m']) and np.any(r['rotation_deg']) and r['close_offset_steps'] for r in joint)


def test_source_inventory_excludes_unmapped_and_unsupported_training_entries():
    import json
    from pathlib import Path
    import tempfile
    import h5py
    from generate_logged_references import enumerate_sources
    with tempfile.TemporaryDirectory() as name:
        root=Path(name);(root/'demos').mkdir()
        episodes=[{'ep_id':'1','obj_id':'a','chain_params':[{'skill_name':'grasp_obj','params':{'part_grasp':'handle'}}],
                   'skill_instructions':['Grasp the mug at its handle']},
                  {'ep_id':'2','obj_id':'b','chain_params':[{'skill_name':'touch_obj','params':{}}],
                   'skill_instructions':['Touch the mug']}]
        (root/'episodes_meta_train.json').write_text(json.dumps({'mug':{'train':{'1':episodes},'test1':{'1':[]}}}))
        with h5py.File(root/'demos/mug.hdf5','w') as h:
            for index,ep in enumerate(['1','2','3',None]):
                d=h.create_group(f'data/demo_{index}')
                d.create_dataset('actions',data=np.array([[0,0,0,0,0,0,.04]]))
                if ep is not None:d.create_dataset('metadata/ep_id',data=ep)
        cases,skipped=enumerate_sources(root,['mug'])
        assert len(cases)==1 and cases[0]['episode_id']=='1' and cases[0]['part']=='handle'
        assert len(skipped)==3 and all('reason' in row for row in skipped)
