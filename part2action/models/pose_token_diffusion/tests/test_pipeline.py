"""Real new blocks through losses, checkpoint transfer and native rollout API."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import models.part2action_model as legacy
from models.pose_token_diffusion.checkpoint import (
    checkpoint_state, initialize_from_legacy, load_checkpoint, save_checkpoint)
from models.pose_token_diffusion.dataset import observation_from_batch, provided_pose_from_batch
from models.pose_token_diffusion.evaluation import build_population, summarize
from models.pose_token_diffusion.losses import compute_losses
from models.pose_token_diffusion.runtime import PoseTokenRolloutPolicy
from models.pose_token_diffusion.policy import build_policy
from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.training import train_step
from models.pose_token_diffusion.tests.test_model import fixture, Vision as TestVision, Language as TestLanguage
from models.pose_token_diffusion.blocks.geometry import quaternion_matrix


def batch_for(observation):
    batch_size = observation.rgb.shape[0]
    pose = observation.tcp_pose.clone()
    pose[:, 4:7] += .02
    return dict(rgb=observation.rgb, agentview_pcd=observation.scene_points_camera,
        task_instruction=observation.instructions, tcp_pose=observation.tcp_pose,
        gripper_state=observation.gripper_opening, action_chunk=torch.rand(batch_size, 3, 7),
        goal_pose=pose, goal_valid=torch.ones(batch_size), pose_supervision_valid=torch.ones(batch_size),
        point_part_labels=torch.tensor([[1., 1., 0., 0., 0.]]).repeat(batch_size, 1),
        point_part_valid=torch.ones(batch_size), part_mask=torch.ones(batch_size, 28, 28),
        skill_plan=[['grasp handle', 'lift'] for _ in range(batch_size)],
        skill_valid_mask=torch.tensor([[1., 1., 0., 0.]]).repeat(batch_size, 1),
        phase_index=torch.ones(batch_size, dtype=torch.long), phase_termination=torch.zeros(batch_size))


class PipelineTests(unittest.TestCase):
    def test_all_losses_update_real_blocks_without_label_inputs(self):
        model, observation, _ = fixture()
        batch = batch_for(observation)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        before = model.blocks['action_decoder'].denoiser.layers.layers[0].multihead_attn.in_proj_weight.detach().clone()
        metrics, gradients = train_step(model, batch, optimizer, {'action_weight': 1.}, collect_gradients=True)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertGreater(gradients['blocks.pose_head.position.weight'], 0)
        self.assertFalse(torch.equal(before, model.blocks['action_decoder'].denoiser.layers.layers[0].multihead_attn.in_proj_weight))
        self.assertEqual(set(vars(observation_from_batch(batch)))-{'rgb','scene_points_camera','instructions','tcp_pose','gripper_opening'},
                         {'plan_slots','slot_valid_logits','current_phase','force_current_phase'})
        bad = dict(batch, goal_pose=torch.full_like(batch['goal_pose'], float('nan')),
                   goal_valid=torch.zeros(2), pose_supervision_valid=torch.zeros(2))
        out = model(observation_from_batch(bad), provided_pose=provided_pose_from_batch(bad), target_actions=bad['action_chunk'])
        loss, parts = compute_losses(out, bad, model, {'action_weight': 1.})
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts['pose_position'], 0)
        self.assertEqual(parts['pose_rotation'], 0)

    def test_real_checkpoint_excludes_frozen_backbones_and_roundtrips(self):
        model, observation, goal = fixture()
        model.eval()
        torch.manual_seed(5)
        original = model(observation, provided_pose=goal).actions['action_chunk']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'policy.pt'
            save_checkpoint(model, path)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['checkpoint_schema'], 2)
            self.assertFalse(any(key.startswith('blocks.encoder.backbones.') for key in payload['model']))
            restored, _, _ = fixture()
            load_checkpoint(restored, path)
            restored.eval()
            torch.manual_seed(5)
            torch.testing.assert_close(restored(observation, provided_pose=goal).actions['action_chunk'], original, rtol=0, atol=0)
            predicted, _, _ = fixture('predicted')
            with self.assertRaises(ValueError):
                load_checkpoint(predicted, path)
            load_checkpoint(predicted, path, allow_pose_source_change=True)

    def test_legacy_transfer_is_explicit_preserves_cross_attention_and_old_bytes(self):
        # Construct an actual legacy policy with tiny frozen encoders.
        class Vision(torch.nn.Module):
            embed_dim, grid = 16, 2
            def __init__(self, **kwargs):
                super().__init__()
                self.model = torch.nn.Identity()
        class Text(torch.nn.Module):
            embed_dim = 16
            def __init__(self, **kwargs):
                super().__init__()
                self.model = torch.nn.Identity()
                self.device_str = 'cpu'
        with patch.object(legacy, 'FrozenDINOv2', Vision), patch.object(legacy, 'FrozenT5', Text):
            old = legacy.Part2ActionModel(heads=['action'], hidden_dim=256, action_chunk=3,
                use_pcd=True, use_gripper_state=True, use_part_gate=True, use_hierarchy=True,
                hierarchy_layers=1, n_obs_steps=2, temporal_encoder_type='transformer',
                temporal_layers=1, temporal_heads=4, action_head_type='diffusion', diffusion_steps=4,
                diffusion_schedule='cosine',
                diffusion_action_scale=[.5, .5, .5, 3.141592653589793, 3.141592653589793, 3.141592653589793, .02],
                diffusion_action_offset=[0., 0., 0., 0., 0., 0., .02],
                diffusion_denoiser_type='transformer', diffusion_transformer_layers=1, diffusion_transformer_heads=4)
        small, _, _ = fixture()
        # Legacy DiffusionActionHead uses a 256-wide denoiser even when other
        # fixture widths are smaller. Match the actual production decoder width.
        model = build_policy(replace(small.config, hidden_dim=256),
                             backbones=FrozenBackbones(TestVision(), TestLanguage()))
        cross = model.blocks['action_decoder'].denoiser.layers.layers[0].multihead_attn.in_proj_weight.detach().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'legacy.pt'
            torch.save({'model': old.state_dict()}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = initialize_from_legacy(model, path)
            self.assertGreater(receipt['loaded_tensors'], 20)
            torch.testing.assert_close(model.blocks['encoder'].pcd_encoder.point_mlp[0].weight, old.pcd_encoder.point_mlp[0].weight)
            torch.testing.assert_close(model.blocks['action_decoder'].denoiser.layers.layers[0].linear1.weight,
                                       old.action_head.net.encoder.layers[0].linear1.weight)
            torch.testing.assert_close(model.blocks['action_decoder'].denoiser.layers.layers[0].multihead_attn.in_proj_weight, cross)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_native_rollout_adapter_outputs_actions_and_plan_cache(self):
        model, observation, goal = fixture()
        model.eval()
        adapter = PoseTokenRolloutPolicy(model, provided_pose=goal)
        first = adapter(observation.rgb, observation.instructions, agentview_pcd=observation.scene_points_camera,
                        tcp_pose=observation.tcp_pose, gripper_state=observation.gripper_opening)
        self.assertEqual(first['action_chunk'].shape, (2, 3, 7))
        second = adapter(observation.rgb, None, agentview_pcd=observation.scene_points_camera,
            tcp_pose=observation.tcp_pose, gripper_state=observation.gripper_opening,
            plan_slots_override=first['plan_slots'], slot_valid_logits_override=first['slot_valid_logits'],
            current_phase=first['predicted_phase'], force_current_phase=True)
        torch.testing.assert_close(second['predicted_phase'], first['predicted_phase'])
        predicted, _, _ = fixture('predicted')
        with self.assertRaises(ValueError):
            PoseTokenRolloutPolicy(predicted, provided_pose=goal)

    def test_matched_pose_modes_start_from_identical_weights(self):
        provided, _, _ = fixture('provided')
        disabled, _, _ = fixture('disabled')
        for key, value in checkpoint_state(provided).items():
            torch.testing.assert_close(checkpoint_state(disabled)[key], value, rtol=0, atol=0)

    def test_eval_reference_indices_preserved_and_unknowns_not_policy_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {'scenarios': [{'obj_class': 'bottle'}], 'trials': [
                {'scenario_index': 0, 'seed': 42, 'needs_grasp_pose': True},
                {'scenario_index': 0, 'seed': 43, 'needs_grasp_pose': True}]}
            (root/'plan.json').write_text(json.dumps(plan))
            _, trials = build_population(root, [43], root)
            self.assertEqual(trials[0][0], 1)
            with self.assertRaises(ValueError):
                build_population(root, [42, 42], root)
            with self.assertRaises(ValueError):
                build_population(root, [99], root)
        summary = summarize([{'status': 'blocked_missing_oracle'},
            {'status': 'completed', 'needs_grasp_pose': True, 'success': True,
             'grasp': {'grasp_acquired': True, 'initial_phase_last_grasp': True}}])
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['unknown'], 1)
        self.assertEqual(summary['task_successes'], 1)


if __name__ == '__main__':
    unittest.main()
