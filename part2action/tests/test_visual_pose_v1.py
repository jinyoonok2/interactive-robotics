"""Regression and information-boundary checks for the ORIGINAL visual policy."""
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from scipy.spatial.transform import Rotation

import models.part2action_model as original
from models.visual_pose_policy_v1 import VisualPosePolicyV1
from models.pose_conditioning_v1 import TargetPoseConditionerV1, relative_pose_features
from data.oracle_grasp_pose_v1 import first_grasp_target


class Vision(nn.Module):
    embed_dim, grid = 16, 2
    def __init__(self, **kwargs):
        super().__init__(); self.model = nn.Identity()
    def forward(self, rgb):
        value = rgb.mean(dim=(-1, -2))
        value = torch.cat([value] * 6, -1)[:, :16]
        return value[:, None].expand(-1, 4, -1)


class Text(nn.Module):
    embed_dim = 16
    def __init__(self, **kwargs):
        super().__init__(); self.model = nn.Identity(); self.device_str = 'cpu'
    def forward(self, texts):
        return torch.ones(len(texts), 2, 16), torch.ones(len(texts), 2, dtype=torch.bool)


def policy(cls):
    with patch.object(original, 'FrozenDINOv2', Vision), patch.object(original, 'FrozenT5', Text):
        return cls(heads=['action'], hidden_dim=16, use_pcd=True, use_gripper_state=True,
                   action_head_type='diffusion', diffusion_steps=4,
                   diffusion_denoiser_type='transformer', diffusion_transformer_layers=1,
                   diffusion_transformer_heads=4)


def inputs():
    tcp = torch.tensor([[0., 0., 0., 1., .1, .2, .3]])
    return dict(tcp_pose=tcp, gripper_state=torch.tensor([[.04]]), agentview_pcd=torch.randn(1, 16, 3))


def test_relative_pose_frame_invariance_and_invalid_data_boundary():
    tcp = np.r_[Rotation.from_euler('xyz', [.1, .2, .3]).as_quat(), [.1, .2, .3]]
    goal = np.r_[Rotation.from_euler('xyz', [.2, .5, .1]).as_quat(), [.3, .4, .2]]
    a = relative_pose_features(torch.tensor(tcp[None]), torch.tensor(goal[None]), torch.ones(1))
    transform = Rotation.from_euler('xyz', [.3, -.2, .4])
    def move(p):
        return np.r_[(transform * Rotation.from_quat(p[:4])).as_quat(), transform.apply(p[4:]) + 2.]
    b = relative_pose_features(torch.tensor(move(tcp)[None]), torch.tensor(move(goal)[None]), torch.ones(1))
    torch.testing.assert_close(a, b, atol=5e-6, rtol=1e-5)
    invalid = relative_pose_features(torch.full((1, 7), float('nan')), torch.full((1, 7), float('nan')), torch.zeros(1))
    assert torch.count_nonzero(invalid) == 0


def test_gt_label_is_measured_preclose_not_command_or_postclose():
    actions = np.zeros((6, 7)); actions[:, 6] = [.04, .04, .04, 0., 0., 0.]
    actions[:, :3] = 999
    poses = np.tile([0., 0., 0., 1., 0., 0., 0.], (6, 1)); poses[:, 4] = np.arange(6)
    result = first_grasp_target(actions, poses, ['Grasp handle']*5 + ['Move up'])
    assert result['valid'] and result['closure_action_index'] == 3
    np.testing.assert_array_equal(result['pose'], poses[2])
    assert not first_grasp_target(actions, poses, ['Touch handle']*6)['valid']


def test_disabled_matches_original_checkpoint_and_noise_exactly():
    torch.manual_seed(42); base = policy(original.Part2ActionModel).eval()
    torch.manual_seed(42); augmented = policy(VisualPosePolicyV1).eval()
    for name, value in base.state_dict().items():
        assert torch.equal(value, augmented.state_dict()[name]), name
    rgb, obs, target = torch.rand(1, 3, 28, 28), inputs(), torch.rand(1, 8, 7)
    torch.manual_seed(99); a = base(rgb, ['grasp'], target_action=target, **obs)
    torch.manual_seed(99); b = augmented(rgb, ['grasp'], target_action=target, **obs)
    assert torch.equal(a['action_noise_pred'], b['action_noise_pred'])
    assert torch.count_nonzero(b['pose_conditioning_features']) == 0


def test_same_action_decoder_preserves_vision_and_learns_pose_conditioning():
    torch.manual_seed(43); model = policy(VisualPosePolicyV1).eval()
    model.pose_conditioning_enabled = True
    obs = inputs(); goal = obs['tcp_pose'].clone(); goal[:, 4] += .1
    rgb = torch.rand(1, 3, 28, 28, requires_grad=True)
    target = torch.rand(1, 8, 7)
    torch.manual_seed(12)
    out = model(rgb, ['grasp'], target_action=target, goal_pose=goal, goal_valid=torch.ones(1), **obs)
    loss = (out['action_noise_pred'] - out['action_noise']).square().mean(); loss.backward()
    assert model.pose_conditioner.projection[-1].weight.grad.abs().sum() > 0
    # The existing DINO encoder intentionally runs without gradients. Its
    # learned fusion projection must still be trained and consume visual input.
    assert model.fusion.visual_proj.weight.grad.abs().sum() > 0
    assert out['action_noise_pred'].shape == (1, 8, 7)
    assert not hasattr(model, 'pose_actor')
    with torch.no_grad():
        model.pose_conditioner.projection[-1].weight.add_(.01 * torch.randn_like(model.pose_conditioner.projection[-1].weight))
    torch.manual_seed(12)
    a = model(rgb.detach(), ['grasp'], target_action=target, goal_pose=goal, goal_valid=torch.ones(1), **obs)
    shifted = goal.clone(); shifted[:, 4] += .05
    torch.manual_seed(12)
    b = model(rgb.detach(), ['grasp'], target_action=target, goal_pose=shifted, goal_valid=torch.ones(1), **obs)
    assert not torch.allclose(a['action_noise_pred'], b['action_noise_pred'])
    changed_rgb = rgb.detach().clone(); changed_rgb[:, 0] += .5
    torch.manual_seed(12)
    c = model(changed_rgb, ['grasp'], target_action=target, goal_pose=goal, goal_valid=torch.ones(1), **obs)
    assert not torch.allclose(a['fused'], c['fused'])
    assert not torch.allclose(a['action_noise_pred'], c['action_noise_pred'])


def test_zero_projection_cannot_replace_actions_with_goal():
    module = TargetPoseConditionerV1(16)
    features = torch.randn(1, 4, 16)
    obs = inputs(); goal = obs['tcp_pose'] + torch.tensor([[0., 0., 0., 0., 5., 5., 5.]])
    conditioned, _ = module(features, obs['tcp_pose'], goal, torch.ones(1))
    assert torch.equal(conditioned, features)
    try:
        module(features, obs['tcp_pose'], None, None)
    except ValueError:
        pass
    else:
        raise AssertionError('Enabled oracle input cannot silently fall back when missing')


def test_reporting_keeps_missing_trials_unknown_and_uses_same_pairs():
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from run_visual_oracle_pose_v1 import report, dump, read
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        plan = {'trials':[{'seed':42,'needs_grasp_pose':True}], 'protocol':'test', 'limits':[]}
        dump(root/'evaluations/control/000.json', {'status':'completed','trial_index':0,
            'needs_grasp_pose':True,'success':False,'grasp':{'grasp_acquired':False,'initial_phase_last_grasp':False}})
        report(SimpleNamespace(out_dir=root),plan)
        result = read(root/'summary.json')
        assert result['groups']['control']['completed'] == 1
        assert result['groups']['oracle_pose']['completed'] == 0
        assert result['paired_completed'] == 0
        assert result['primary_paired_comparison']['oracle']['paired_task_trials'] == 0
        assert result['unknown'][0]['status'] == 'missing'
