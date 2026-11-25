"""Actual geometry/grounding/pose/diffusion blocks; only frozen backbones are tiny."""
from dataclasses import replace
import unittest
import torch
from torch import nn
from torch.nn import functional as F

from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.config import PoseTokenDiffusionConfig
from models.pose_token_diffusion.contracts import Observation, TargetPose
from models.pose_token_diffusion.policy import build_policy
from models.pose_token_diffusion.blocks.geometry import camera_grid, sample_image_features, relative_pose, rotation_matrix


class Vision(nn.Module):
    embed_dim = 16
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 16)
        self.calls = 0
    def forward(self, rgb):
        self.calls += 1
        values = F.adaptive_avg_pool2d(rgb, (2, 2)).flatten(2).transpose(1, 2)
        return self.projection(values)


class Language(nn.Module):
    embed_dim = 16
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(31, 16)
    def forward(self, texts):
        indices = torch.tensor([[sum(map(ord, text)) % 31, len(text) % 31] for text in texts], device=self.embedding.weight.device)
        return self.embedding(indices), torch.ones(indices.shape, device=indices.device, dtype=torch.bool)


def fixture(source='provided', progress=False):
    torch.manual_seed(77)
    config = PoseTokenDiffusionConfig(hidden_dim=16, action_horizon=3, img_size=28,
        attention_heads=4, decoder_layers=1, hierarchy_layers=1, diffusion_steps=4,
        visual_token_grid=2, scene_token_budget=4, local_token_budget=2,
        pose_source=source, progress_enabled=progress)
    model = build_policy(config, backbones=FrozenBackbones(Vision(), Language()))
    points = torch.tensor([[[-.2, -.1, 1.], [-.1, .15, 1.], [0., 0., .9], [.1, -.2, 1.], [.2, .2, 1.]]]).repeat(2, 1, 1)
    tcp = torch.tensor([[0., 0., 0., 1., -.5, .05, .4]]).repeat(2, 1)
    observation = Observation(torch.rand(2, 2, 3, 28, 28), points,
        ['grasp handle then lift', 'grasp neck then lift'], tcp, torch.full((2, 1), .04))
    goal = TargetPose(tcp[:, 4:7]+.02, torch.eye(3).expand(2, -1, -1), torch.ones(2, dtype=torch.bool))
    return model, observation, goal


class ActualModelTests(unittest.TestCase):
    def test_camera_projection_matches_pixel_centers_and_rejects_bad_depth(self):
        grid, valid = camera_grid(torch.tensor([[[0., 0., 1.], [0., 0., 0.], [99., 0., 1.]]]), 300, 300)
        torch.testing.assert_close(grid[0, 0], torch.full((2,), 1/300), atol=1e-6, rtol=1e-5)
        self.assertEqual(valid.tolist(), [[True, False, False]])
        yy, xx = torch.meshgrid(torch.arange(300), torch.arange(300), indexing='ij')
        image = torch.stack([xx, yy]).float()[None]
        values, _ = sample_image_features(image, torch.tensor([[[0., 0., 1.]]]), 300, 300)
        torch.testing.assert_close(values, torch.full((1, 1, 2), 150.))

    def test_relative_geometry_is_frame_correct_invalid_safe_and_differentiable(self):
        tcp = torch.tensor([[0., 0., 2**-.5, 2**-.5, 0., 0., 0.]])
        position = torch.tensor([[0., .1, 0.]], requires_grad=True)
        target = TargetPose(position, torch.eye(3)[None], torch.ones(1, dtype=torch.bool))
        error = relative_pose(tcp, target)
        torch.testing.assert_close(error[0, :3], torch.tensor([.1, 0., 0.]), atol=1e-6, rtol=1e-5)
        error.square().sum().backward()
        self.assertGreater(position.grad.abs().sum().item(), 0)
        invalid = TargetPose(torch.full((1, 3), float('nan')), torch.full((1, 3, 3), float('nan')), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(relative_pose(tcp, invalid).count_nonzero(), 0)
        rotations = rotation_matrix(torch.cat([torch.zeros(1, 6), torch.randn(4, 6)]))
        torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations, torch.eye(3).expand(5, -1, -1), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(torch.det(rotations), torch.ones(5), atol=2e-6, rtol=1e-5)

    def test_real_diffusion_backpropagates_and_cross_attention_is_present(self):
        model, observation, goal = fixture()
        out = model(observation, provided_pose=goal, target_actions=torch.rand(2, 3, 7))
        F.mse_loss(out.actions['action_noise_pred'].float(), out.actions['action_noise']).backward()
        for parameter in (model.blocks['encoder'].scene_projection[0].weight,
                          model.blocks['pose_tokenizer'].translation[0].weight,
                          model.blocks['action_decoder'].denoiser.layers.layers[0].multihead_attn.in_proj_weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertFalse(any(p.grad is not None for p in model.blocks['encoder'].backbones.parameters()))
        self.assertEqual(out.encoded.visual_tokens.shape, (2, 4, 16))
        self.assertEqual(out.grounding.tokens.shape, (2, 3, 16))
        self.assertEqual(out.pose_memory.tokens.shape, (2, 2, 16))

    def test_sampling_reuses_features_and_retains_learned_plan_cache(self):
        model, observation, goal = fixture()
        model.eval()
        first = model(observation, provided_pose=goal)
        self.assertEqual(first.actions['action_chunk'].shape, (2, 3, 7))
        self.assertTrue(torch.isfinite(first.actions['action_chunk']).all())
        self.assertEqual(model.blocks['encoder'].backbones.visual.calls, 1)
        hierarchy = first.encoded.hierarchy
        cached = replace(observation, instructions=None, plan_slots=hierarchy['plan_slots'],
            slot_valid_logits=hierarchy['slot_valid_logits'], current_phase=hierarchy['predicted_phase'], force_current_phase=True)
        second = model(cached, provided_pose=goal)
        torch.testing.assert_close(second.encoded.hierarchy['plan_slots'], hierarchy['plan_slots'])
        self.assertEqual(model.blocks['encoder'].backbones.visual.calls, 2)

    def test_predicted_pose_uses_rgb_and_scene_and_action_loss_reaches_pose_head(self):
        model, observation, _ = fixture('predicted')
        target = torch.rand(2, 3, 7)
        torch.manual_seed(2)
        first = model(observation, target_actions=target)
        F.mse_loss(first.actions['action_noise_pred'].float(), first.actions['action_noise']).backward()
        for parameter in (model.blocks['pose_head'].position.weight,
                          model.blocks['pose_head'].rotation.weight,
                          model.blocks['grounding'].key.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        torch.manual_seed(2)
        changed = model(replace(observation, rgb=observation.rgb+.3), target_actions=target)
        self.assertFalse(torch.allclose(first.predicted_pose.position_world, changed.predicted_pose.position_world))
        self.assertFalse(torch.allclose(first.actions['action_noise_pred'], changed.actions['action_noise_pred']))
        torch.manual_seed(2)
        shifted = model(replace(observation, scene_points_camera=observation.scene_points_camera+torch.tensor([.03, 0., 0.])), target_actions=target)
        self.assertFalse(torch.allclose(first.predicted_pose.position_world, shifted.predicted_pose.position_world))

    def test_provided_goal_changes_actions_disabled_pose_keeps_vision_and_amp_is_finite(self):
        model, observation, goal = fixture()
        model.eval()
        target = torch.rand(2, 3, 7)
        torch.manual_seed(3)
        a = model(observation, provided_pose=goal, target_actions=target)
        torch.manual_seed(3)
        b = model(observation, provided_pose=replace(goal, position_world=goal.position_world+.1), target_actions=target)
        self.assertFalse(torch.allclose(a.actions['action_noise_pred'], b.actions['action_noise_pred']))
        disabled, observation, _ = fixture('disabled', progress=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = disabled(observation, target_actions=target)
            loss = F.mse_loss(out.actions['action_noise_pred'].float(), out.actions['action_noise'])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(out.pose_memory.tokens.count_nonzero(), 0)
        self.assertGreater(disabled.blocks['encoder'].visual_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(out.progress_logits.shape, (2,))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in disabled.parameters()))


if __name__ == '__main__':
    unittest.main()
