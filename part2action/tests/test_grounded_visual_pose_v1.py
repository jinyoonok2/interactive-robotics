"""Information boundaries, checkpoint reuse and actual observation/gradient paths."""
from unittest.mock import patch
import inspect
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F
import models.part2action_model as original
from models.grounded_pose_head_v1 import rotation_from_six
from models.grounded_visual_pose_policy_v1 import GroundedVisualPosePolicyV1
from train_grounded_visual_pose_v1 import forward_kwargs, compute_losses, load_initialization
from test_visual_pose_v1 import Vision, Text, inputs


def policy(cls=GroundedVisualPosePolicyV1, **overrides):
    with patch.object(original, 'FrozenDINOv2', Vision), patch.object(original, 'FrozenT5', Text):
        options=dict(heads=['heatmap', 'approach', 'action'], hidden_dim=16,
            use_pcd=True, use_gripper_state=True, use_contact_xyz=True,
            use_contact_action_token=True, use_unified_contact_conditioning=True,
            contact_conditioning_enabled=False, contact_conditioning_mode='normalized_global',
            action_head_type='diffusion', diffusion_steps=4,
            diffusion_denoiser_type='transformer', diffusion_transformer_layers=1,
            diffusion_transformer_heads=4)
        options.update(overrides)
        return cls(**options)


def sample(model, rgb, obs, target):
    torch.manual_seed(101)
    return model(rgb, ['grasp the handle'], target_action=target, **obs)


def test_rotation_is_so3_even_for_degenerate_predictions():
    six = torch.cat([torch.zeros(1, 6), torch.tensor([[1., 1., 0., 0., 0., 0.]]), torch.randn(5, 6)])
    rotation = rotation_from_six(six)
    torch.testing.assert_close(rotation.transpose(-1,-2) @ rotation,
                               torch.eye(3).expand(7, -1, -1), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(torch.det(rotation), torch.ones(7))


def test_zero_conditioner_reuses_baseline_actions_and_removes_unused_heads():
    torch.manual_seed(22); baseline = policy(original.Part2ActionModel).eval()
    torch.manual_seed(22); model = policy().eval()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'baseline.pt'
        torch.save({'model':baseline.state_dict()}, path)
        receipt = load_initialization(model, path)
        assert any(k.startswith('contact_attention.token.') for k in receipt['omitted_unused_tensors'])
    rgb, obs, target = torch.rand(1,3,28,28), inputs(), torch.rand(1,8,7)
    a, b = sample(baseline,rgb,obs,target), sample(model,rgb,obs,target)
    torch.testing.assert_close(a['action_noise_pred'], b['action_noise_pred'], rtol=0, atol=0)
    assert not hasattr(model, 'heatmap_head') and not hasattr(model, 'approach_head')
    assert not hasattr(model.contact_attention, 'token')
    assert b['pose_features'].shape == (1,10)


def test_inference_has_no_pose_or_contact_label_inputs():
    names = inspect.signature(GroundedVisualPosePolicyV1.forward).parameters
    for forbidden in ('goal_pose','goal_valid','grasp_pose_target','contact_xyz_target','gt_part_mask','agentview_part_pcd'):
        assert forbidden not in names
    batch = {**inputs(), 'goal_pose':'poison', 'grasp_pose_target':'poison',
             'point_part_labels':'poison', 'contact_xyz':'poison'}
    assert set(forward_kwargs(batch, False)) == {'agentview_pcd','tcp_pose','gripper_state'}


def test_pose_and_action_use_rgb_and_observed_scene_geometry():
    torch.manual_seed(24); model = policy().eval()
    with torch.no_grad():
        model.pose_conditioner.projection[-1].weight.normal_(std=.05)
    rgb, obs, target = torch.rand(1,3,28,28), inputs(), torch.rand(1,8,7)
    a = sample(model,rgb,obs,target)
    changed_rgb = rgb.clone(); changed_rgb[:,0] += .7
    b = sample(model,changed_rgb,obs,target)
    assert not torch.allclose(a['local_visual_features'],b['local_visual_features'])
    assert not torch.allclose(a['predicted_pose_rotation_world'],b['predicted_pose_rotation_world'])
    assert not torch.allclose(a['action_noise_pred'],b['action_noise_pred'])
    shifted = dict(obs,agentview_pcd=obs['agentview_pcd']+.05)
    c = sample(model,rgb,shifted,target)
    assert not torch.allclose(a['predicted_pose_position_world'],c['predicted_pose_position_world'])
    assert not torch.allclose(a['action_noise_pred'],c['action_noise_pred'])


def test_action_gradients_reach_pose_estimator_and_grounding_without_detach():
    torch.manual_seed(25); model = policy().eval()
    with torch.no_grad():
        model.pose_conditioner.projection[-1].weight.normal_(std=.05)
    out = sample(model,torch.rand(1,3,28,28),inputs(),torch.rand(1,8,7))
    F.mse_loss(out['action_noise_pred'],out['action_noise']).backward()
    for parameter in (model.pose_head.rotation.weight, model.pose_head.translation.weight,
                      model.pose_head.local_visual.in_proj_weight,
                      model.contact_attention.point_proj.weight, model.fusion.visual_proj.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_invalid_pose_labels_are_ignored_while_action_training_continues():
    torch.manual_seed(26); model = policy().eval()
    target = torch.rand(1,8,7)
    out = sample(model,torch.rand(1,3,28,28),inputs(),target)
    batch = dict(action_chunk=target, grasp_pose_target=torch.full((1,7),float('nan')),
                 grasp_pose_valid=torch.zeros(1))
    loss, parts = compute_losses(out,batch,{'action_weight':1.})
    assert torch.isfinite(loss) and parts['pose_position'] == 0 and parts['pose_rotation'] == 0
    loss.backward()
    assert model.action_head is not None and model.pose_head.applicability.weight.grad.abs().sum() > 0


def test_supervised_pose_trains_new_head_before_zero_projection_opens():
    torch.manual_seed(27); model=policy().eval()
    obs=inputs(); target=torch.rand(1,8,7)
    out=sample(model,torch.rand(1,3,28,28),obs,target)
    label=obs['tcp_pose'].clone(); label[:,4:7] += .05
    batch=dict(action_chunk=target,grasp_pose_target=label,grasp_pose_valid=torch.ones(1))
    loss,_=compute_losses(out,batch,{'action_weight':1.})
    loss.backward()
    assert model.pose_head.translation.weight.grad.abs().sum() > 0
    assert model.pose_head.rotation.weight.grad.abs().sum() > 0
    assert model.pose_conditioner.projection[-1].weight.grad.abs().sum() > 0


def test_pose_labels_stop_at_closure_without_removing_later_action_samples():
    from data.grounded_grasp_pose_v1 import GroundedGraspPoseDatasetV1
    from data.oracle_grasp_pose_v1 import OracleGraspPoseDatasetV1
    dataset=object.__new__(GroundedGraspPoseDatasetV1)
    dataset._samples=[None]
    dataset.target_for=lambda spec: {'valid':True,'closure_action_index':3}
    for timestep,expected in [(0,1.),(2,1.),(3,0.),(20,0.)]:
        parent=dict(meta={'t':timestep},goal_pose=torch.tensor([0.,0.,0.,1.,.1,.2,.3]),
                    goal_valid=torch.ones(()),action_chunk=torch.ones(8,7))
        with patch.object(OracleGraspPoseDatasetV1,'__getitem__',return_value=parent):
            sample=dataset[0]
        assert float(sample['grasp_pose_valid'])==expected
        assert sample['action_chunk'].shape==(8,7)
        assert 'goal_pose' not in sample and 'goal_valid' not in sample


def test_two_frame_hierarchy_plan_cache_and_bfloat16_backward():
    torch.manual_seed(29)
    model=policy(use_hierarchy=True,use_part_gate=True,n_obs_steps=2,
                 temporal_encoder_type='transformer',temporal_heads=4).eval()
    obs=inputs();rgb=torch.rand(1,2,3,28,28);target=torch.rand(1,8,7)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        first=model(rgb,['grasp handle then lift'],target_action=target,**obs)
        cached=model(rgb,None,target_action=target,
            plan_slots_override=first['plan_slots'],
            slot_valid_logits_override=first['slot_valid_logits'],
            current_phase=first['predicted_phase'],force_current_phase=True,**obs)
        loss=F.mse_loss(cached['action_noise_pred'].float(),cached['action_noise'].float())
    assert cached['action_noise_pred'].shape==(1,8,7)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
