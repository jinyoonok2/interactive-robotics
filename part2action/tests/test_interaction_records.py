import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diagnostics.interaction_tolerances import conditions, perturb_actions
from diagnostics.interaction_records import decision_frame, candidate_vector, outcome_components, depth_points, motion_delta
from data.interaction_outcomes import InteractionOutcomeDataset, instance_split
from models.interaction_success import InteractionSuccessHead


def test_observation_cutoff_precedes_every_intervention_and_candidate_units_are_explicit():
    actions=np.zeros((35,7));actions[:,6]=.04;actions[25:,6]=0
    window={'start':5,'close':25,'end':35};candidates=conditions(.03,30,6)
    frame=decision_frame(actions,window,candidates)
    assert 0<frame<25
    for c in candidates:
        np.testing.assert_array_equal(perturb_actions(actions,window,c)[:frame],actions[:frame])
    np.testing.assert_allclose(candidate_vector(candidates[-1])[-1],.3)
    np.testing.assert_allclose(np.linalg.norm(candidate_vector(candidates[7])[3:6]),np.pi/6)


def test_nontrivial_euler_commands_are_exactly_preserved_before_rotation_intervention():
    actions=np.zeros((45,7));actions[:,6]=.04;actions[35:,6]=0
    actions[:20,3:6]=[3.0068489786699906,-1.2990458863926615,-1.0387557055231569]
    actions[20:,3:6]=[2.91,-1.2,-.98]
    window={'start':5,'close':35,'end':45};candidates=conditions(.01,10,2)
    frame=decision_frame(actions,window,candidates)
    assert 0<frame<35
    for candidate in candidates:
        changed=perturb_actions(actions,window,candidate)
        np.testing.assert_array_equal(changed[:frame],actions[:frame])
    rotated=next(c for c in candidates if c['name']=='rotation_deg_x_minus')
    changed=perturb_actions(actions,window,rotated)
    assert np.any(changed[frame:,3:6]!=actions[frame:,3:6])


def test_outcome_components_preserve_threshold_sensitive_failures_and_achieved_motion():
    row={'lift_m':.02934,'hold':{'bilateral_fraction':1.,'target_contact_fraction':1.},'hold_relative_position_drift_m':.006}
    result=outcome_components(row)
    assert result['failure_causes']==['lift'] and result['margins']['lift']<0
    reference=np.tile([0.,0.,0.,1.,0.,0.,0.],(3,1));achieved=reference.copy()
    achieved[:,4]=-.02;achieved[:,:4]=Rotation.from_euler('z',10,degrees=True).as_quat()
    delta=motion_delta(achieved,reference,1)
    np.testing.assert_allclose(delta['world_translation_m'],[-.02,0,0])
    np.testing.assert_allclose(delta['reference_tcp_local_rotation_vector_rad'],[0,0,np.pi/18])


def test_scene_point_sampling_is_finite_reproducible_and_preserves_rng():
    depth=np.ones((4,4),dtype=np.float32);intrinsic=np.array([[2.,0,2],[0,2,2],[0,0,1]])
    np.random.seed(73);before=np.random.get_state()
    a,w=depth_points(depth,intrinsic,np.eye(4),8)
    b,_=depth_points(depth,intrinsic,np.eye(4),8)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(np.random.get_state()[1],before[1])
    np.testing.assert_allclose(w,a*np.array([1,-1,-1]))


def test_instance_splits_group_all_sources_and_refuse_one_instance():
    rows=[{'instance_group':instance,'source_group':f'demo_{demo}'} for instance in ['mug/1','mug/2','bottle/3'] for demo in range(2)]
    training,held=instance_split(rows)
    assert not {r['instance_group'] for r in training}&{r['instance_group'] for r in held}
    try:instance_split(rows[:2])
    except ValueError:pass
    else:raise AssertionError('Single-instance split accepted')


def test_dataset_separates_post_action_outcomes_from_predictor_inputs():
    with tempfile.TemporaryDirectory() as name:
        root=Path(name)
        tcp=np.tile([0.,0.,0.,1.,.1,.2,.3],(2,1)).astype(np.float32)
        np.savez(root/'observation.npz',rgb=np.zeros((2,3,4,4),dtype=np.uint8),pcd_camera_m=np.zeros((2,8,3),dtype=np.float32),tcp_pose_xyzw_xyz=tcp,finger_position_m=np.full((2,1),.04,dtype=np.float32))
        row={'stable_target_grasp_lift':False,'primitive_context':[0]*8,'candidate_parameters':[0]*7,
             'lift_m':.02,'hold':{'bilateral_fraction':1.,'target_contact_fraction':1.},'hold_relative_position_drift_m':.002,
             'outcome_components':{'passes':{'lift':False,'bilateral':True,'target_contact':True,'drift':True}},
             'achieved_tcp_at_nominal_close':[999]*7,'native_grasp_predicate_equivalent':True}
        (root/'result.json').write_text(json.dumps(row))
        (root/'records.json').write_text(json.dumps({'source_split':'training only','records':[{'result_file':'result.json','observation_file':'observation.npz','skill_instruction':'Grasp the mug at its handle','instance_group':'mug/1','source_group':'mug/demo_1'}]}))
        sample=InteractionOutcomeDataset(root/'records.json')[0]
        assert set(sample['inputs'])=={'rgb','pcd_camera_m','robot_state','primitive_context','candidate','skill_instruction'}
        assert sample['labels']['success'].item()==0
        np.testing.assert_allclose(sample['inputs']['robot_state'][:3],[.1,.2,.3])


def test_success_head_scores_candidates_without_upstream_gradients_or_order_dependence():
    torch.manual_seed(42)
    head=InteractionSuccessHead(12,6,hidden_dim=16)
    observation=torch.randn(2,12,requires_grad=True);skill=torch.randn(2,6,requires_grad=True)
    state=torch.randn(2,8);primitive=torch.randn(2,8);candidates=torch.randn(2,3,7,requires_grad=True)*.01
    output=head(observation,skill,state,primitive,candidates)
    assert output['success_logits'].shape==(2,3) and output['criterion_logits'].shape==(2,3,4)
    permuted=head(observation,skill,state,primitive,candidates[:,[2,0,1]])
    torch.testing.assert_close(output['success_logits'][:,[2,0,1]],permuted['success_logits'])
    loss=output['success_logits'].square().mean()+output['criterion_logits'].square().mean();loss.backward()
    assert observation.grad is None and skill.grad is None
    assert head.score[-1].weight.grad.abs().sum()>0
