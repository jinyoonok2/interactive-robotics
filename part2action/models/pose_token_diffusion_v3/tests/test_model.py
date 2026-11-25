from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.tests.test_model import Vision, Language
from models.pose_token_diffusion_v2.tests.test_model import fixture as v2_fixture
from models.pose_token_diffusion_v2.checkpoint import save_checkpoint as save_v2
from models.pose_token_diffusion_v3.config import PoseTokenDiffusionV3Config
from models.pose_token_diffusion_v3.skills import SkillSpec, annotated_skill
from models.pose_token_diffusion_v3.policy import build_policy
from models.pose_token_diffusion_v3.dataset import observation_from_batch
from models.pose_token_diffusion_v3.training import train_step, validate
from models.pose_token_diffusion_v3.checkpoint import checkpoint_state, save_checkpoint, load_checkpoint
from models.pose_token_diffusion_v3.planner import PlannerDecision, JsonSkillPlanner, QwenVLMPlanner
from models.pose_token_diffusion_v3.runtime import HierarchicalRolloutPolicy
from models.pose_token_diffusion_v3.losses import compute_losses
from models.pose_token_diffusion_v3.evaluation import SuppliedPlanDiagnostic
from models.pose_token_diffusion_v3.planner import PlannerRequest


def fixture(source='predicted'):
    torch.manual_seed(83)
    config = PoseTokenDiffusionV3Config(hidden_dim=16, action_horizon=3, img_size=28,
        attention_heads=4, decoder_layers=1, hierarchy_layers=1, diffusion_steps=4,
        visual_token_grid=2, scene_token_budget=4, local_token_budget=2, pose_source=source)
    model = build_policy(config, backbones=FrozenBackbones(Vision(), Language()))
    tcp = torch.tensor([[0., 0., 0., 1., -.5, .05, .4]]).repeat(2, 1)
    batch = {'rgb': torch.rand(2, 2, 3, 28, 28),
        'agentview_pcd': torch.tensor([[[-.2, -.1, 1.], [-.1, .15, 1.], [0., 0., .9], [.1, -.2, 1.], [.2, .2, 1.]]]).repeat(2, 1, 1),
        'tcp_pose': tcp, 'skill_start_tcp': tcp.clone(), 'gripper_state': torch.full((2, 1), .04),
        'skill_spec': [annotated_skill('Grasp the mug at its handle'), annotated_skill('Touch the bottle at its cap')],
        'action_chunk': torch.rand(2, 3, 7), 'action_valid': torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        'goal_poses': tcp[:, None].repeat(1, 4, 1), 'goal_opening': torch.zeros(2, 4),
        'goal_supervision_valid': torch.zeros(2, 4, 3, dtype=torch.bool),
        'phase_index': torch.zeros(2, dtype=torch.long),
        'point_part_labels': torch.tensor([[1., 1., 0., 0., 0.]]).repeat(2, 1),
        'point_part_valid': torch.ones(2, dtype=torch.bool), 'part_mask': torch.ones(2, 28, 28),
        'meta': [{'file': 'mug.hdf5', 'demo': 'demo_1', 't': 0},
                 {'file': 'bottle.hdf5', 'demo': 'demo_2', 't': 0}]}
    batch['goal_poses'][:, 0, 4:7] += .02
    batch['goal_supervision_valid'][:, 0] = True
    batch['goal_supervision_valid'][1, 0, 1] = False
    return model, batch


def progress_batch(batch):
    keep = {'rgb', 'agentview_pcd', 'tcp_pose', 'skill_start_tcp', 'gripper_state', 'skill_spec', 'meta'}
    result = {k: v for k, v in batch.items() if k in keep}
    result['progress_target'] = torch.tensor([1., 0.])
    result['meta'] = [{'skill': 'grasp_obj', 'source': 'annotated_transition', 'target': 1},
                      {'skill': 'touch_obj', 'source': 'complete_chain_endpoint_proxy', 'target': 0}]
    return result


class ModelTests(unittest.TestCase):
    def test_action_gradients_reach_predicted_goals_grounding_and_context(self):
        model, batch = fixture()
        output = model(observation_from_batch(batch), target_actions=batch['action_chunk'])
        F.mse_loss(output.actions['action_noise_pred'], output.actions['action_noise']).backward()
        for name in ('position', 'rotation', 'opening', 'applicability'):
            self.assertGreater(float(getattr(model.blocks['goal_head'], name).weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.blocks['grounding'].key.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.blocks['encoder'].skill_embedding.weight.grad.abs().sum()), 0)
        self.assertFalse(any(p.grad is not None for p in model.blocks['encoder'].backbones.parameters()))
        self.assertEqual(output.predicted_goal.position_world.shape, (2, 1, 3))
        self.assertEqual(output.pose_memory.tokens.shape, (2, 3, 16))
        self.assertFalse(hasattr(model.blocks['encoder'], 'skill_planner'))

    def test_joint_update_includes_independent_progress_queries(self):
        model, batch = fixture()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        metrics, gradients = train_step(model, batch, optimizer,
            progress_batch=progress_batch(batch), collect_gradients=True, amp=True)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertGreater(gradients['blocks.progress_monitor.net.0.weight'], 0)
        self.assertFalse(any(p.grad is not None for p in model.blocks['encoder'].backbones.parameters()))

    def test_future_targets_full_task_and_annotation_phase_do_not_enter_observations(self):
        model, batch = fixture()
        model.eval()
        altered = dict(batch, goal_poses=torch.randn_like(batch['goal_poses']),
                       phase_index=torch.ones(2, dtype=torch.long), task_instruction=['other task']*2)
        a, b = observation_from_batch(batch), observation_from_batch(altered)
        self.assertNotIn('phase_index', vars(a))
        self.assertNotIn('goal_poses', vars(a))
        torch.manual_seed(6)
        first = model(a, target_actions=batch['action_chunk'])
        torch.manual_seed(6)
        second = model(b, target_actions=batch['action_chunk'])
        torch.testing.assert_close(first.actions['action_noise_pred'], second.actions['action_noise_pred'], rtol=0, atol=0)
        with self.assertRaises(TypeError):
            model(a, provided_goals=batch['goal_poses'])

    def test_skill_change_affects_conditioning_without_changing_the_scene(self):
        model, batch = fixture()
        model.eval()
        observation = observation_from_batch(batch)
        a = model.encode_and_ground(observation)[0]
        skills = tuple(annotated_skill('Touch the mug at its handle') for _ in range(2))
        b = model.encode_and_ground(replace(observation, skills=skills))[0]
        self.assertFalse(torch.equal(a.language_tokens, b.language_tokens))
        self.assertFalse(torch.equal(a.visual_tokens, b.visual_tokens))

    def test_disabled_pose_keeps_visual_and_skill_paths(self):
        model, batch = fixture('disabled')
        output = model(observation_from_batch(batch), target_actions=batch['action_chunk'])
        self.assertEqual(int(output.pose_memory.tokens.count_nonzero()), 0)
        self.assertFalse(output.pose_memory.valid_mask.any())
        self.assertGreater(int(output.encoded.visual_tokens.count_nonzero()), 0)
        self.assertGreater(int(output.encoded.language_tokens.count_nonzero()), 0)

    def test_padded_commands_do_not_contribute_action_loss(self):
        model, batch = fixture()
        output = model(observation_from_batch(batch), target_actions=batch['action_chunk'])
        _, first = compute_losses(output, batch, model)
        prediction = output.actions['action_noise_pred'].clone()
        prediction[~batch['action_valid']] += 10000.
        modified = replace(output, actions={**output.actions, 'action_noise_pred': prediction})
        _, second = compute_losses(modified, batch, model)
        torch.testing.assert_close(first['action'], second['action'], rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_v2_rejection_before_mutation(self):
        model, batch = fixture()
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'v3.pt'
            save_checkpoint(model, path)
            payload = torch.load(path, weights_only=True)
            self.assertEqual(payload['architecture'], 'pose_token_diffusion_v3')
            self.assertFalse(any('backbones' in k for k in payload['model']))
            restored, _ = fixture()
            load_checkpoint(restored, path)
            restored.eval()
            torch.manual_seed(7)
            a = model(observation_from_batch(batch)).actions['action_chunk']
            torch.manual_seed(7)
            b = restored(observation_from_batch(batch)).actions['action_chunk']
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, path)
            previous, _ = v2_fixture()
            old = Path(directory)/'v2.pt'
            save_v2(previous, old)
            before = {k: v.clone() for k, v in checkpoint_state(model).items()}
            with self.assertRaises(ValueError):
                load_checkpoint(model, old)
            self.assertTrue(all(torch.equal(v, checkpoint_state(model)[k]) for k, v in before.items()))

    def test_validation_is_readable_and_preserves_state_and_rng(self):
        model, batch = fixture()
        model.train()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report'
            result = validate(model, [batch], [progress_batch(batch)], None, path,
                              epoch=0, device=torch.device('cpu'))
            self.assertEqual(result['overall']['samples'], 2)
            self.assertEqual(result['by_skill']['touch_obj']['goal_rotation_degrees']['count'], 0)
            self.assertTrue((path/'samples.csv').is_file())
            self.assertTrue((path/'progress.csv').is_file())
            self.assertTrue((path/'README.md').is_file())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))


class SemanticTests(unittest.TestCase):
    def test_annotation_compiler_and_structured_roundtrip(self):
        text = 'Reorient the back of the mug to face downwards'
        spec = annotated_skill(text, previous=['Grasp the mug at its handle'], step_id='2')
        self.assertEqual((spec.object_ref, spec.target_part, spec.interaction), ('mug', 'back', 'maintain_grasp'))
        self.assertEqual(SkillSpec.from_json(json.dumps(spec.to_dict())), spec)
        move = annotated_skill('Move upwards', previous=['Touch the mug at its body'])
        self.assertEqual(move.direction, 'up')
        self.assertEqual(move.interaction, 'unspecified')
        self.assertIsNone(move.distance_m)

    def test_invalid_or_privileged_planner_fields_are_rejected(self):
        spec = annotated_skill('Grasp the mug at its handle').to_dict()
        for values in ({**spec, 'pose': [1, 2, 3]}, {**spec, 'kind': 'touch_obj'},
                       {**spec, 'interaction': 'open'}, {**spec, 'distance_m': float('nan')}):
            with self.assertRaises(ValueError):
                SkillSpec.from_mapping(values)
        with self.assertRaises(ValueError):
            PlannerDecision.from_json('{"status":"done","skill":' + json.dumps(spec) + '}')
        with self.assertRaises(ValueError):
            QwenVLMPlanner(revision='main')

    def test_configs_do_not_convert_v2_or_enable_gt_pose(self):
        with self.assertRaises(ValueError):
            PoseTokenDiffusionV3Config.from_mapping({'architecture': 'pose_token_diffusion_v2'})
        with self.assertRaises(ValueError):
            PoseTokenDiffusionV3Config(pose_source='provided')


class RuntimeTests(unittest.TestCase):
    def test_supplied_plan_is_explicitly_diagnostic_and_does_not_advance_on_periodic_review(self):
        planner = SuppliedPlanDiagnostic(['Grasp the mug at its handle', 'Move upwards'])
        request = PlannerRequest('Lift mug', None, (), .04, None, .99, (), 'periodic_review')
        self.assertEqual(planner.plan(request).skill.kind, 'grasp_obj')
        self.assertEqual(planner.plan(replace(request, reason='monitor_requested_review')).skill.kind, 'move_gripper')
        self.assertEqual(planner.source, 'supplied_plan_diagnostic')

    def test_geometric_refresh_preserves_anchor_and_completion_only_requests_review(self):
        model, batch = fixture()
        requests = []
        first = annotated_skill('Move upwards', previous=['Grasp the mug at its handle'], step_id='lift')
        def generate(request):
            requests.append(request)
            return json.dumps({'status': 'execute', 'skill': first.to_dict()})
        runtime = HierarchicalRolloutPolicy(model, JsonSkillPlanner(generate), review_every_chunks=1)
        args = {key: batch[source][:1].clone() for key, source in
                [('rgb', 'rgb'), ('scene_points_camera', 'agentview_pcd'), ('tcp_pose', 'tcp_pose'), ('gripper_opening', 'gripper_state')]}
        a = runtime.act(**args, task='Lift the mug')
        anchor = runtime.anchor.clone()
        args['tcp_pose'][:, 6] += .05
        b = runtime.act(**args)
        torch.testing.assert_close(anchor, runtime.anchor)
        self.assertEqual(a['current_skill'], b['current_skill'])
        self.assertEqual(len(requests), 2)
        # Zero-initialized displacement gives the same anchored move target
        # even though the observation encoder and goal predictor run again.
        torch.testing.assert_close(a['predicted_goal_position_world'], b['predicted_goal_position_world'])
        self.assertEqual(model.blocks['encoder'].backbones.visual.calls, 2)
        self.assertEqual(requests[1].reason, 'periodic_review')
        self.assertFalse(b['physical_success_verified'])

    def test_new_step_latches_new_anchor_and_done_is_not_physical_success(self):
        model, batch = fixture()
        decisions = iter([PlannerDecision('execute', annotated_skill('Grasp the mug at its handle', step_id='a')),
                          PlannerDecision('execute', annotated_skill('Move upwards', step_id='b')),
                          PlannerDecision('done')])
        class Planner:
            source = 'test_planner'
            def plan(self, request):
                return next(decisions)
        runtime = HierarchicalRolloutPolicy(model, Planner(), review_every_chunks=1)
        args = {key: batch[source][:1].clone() for key, source in
                [('rgb', 'rgb'), ('scene_points_camera', 'agentview_pcd'), ('tcp_pose', 'tcp_pose'), ('gripper_opening', 'gripper_state')]}
        runtime.act(**args, task='Lift the mug')
        args['tcp_pose'][:, 6] += .05
        runtime.act(**args)
        torch.testing.assert_close(runtime.anchor, args['tcp_pose'])
        stopped = runtime.act(**args)
        self.assertEqual(stopped['status'], 'planner_done')
        self.assertIsNone(stopped['action_chunk'])
        self.assertFalse(stopped['physical_success_verified'])
        with self.assertRaises(RuntimeError):
            runtime.act(**args)


if __name__ == '__main__':
    unittest.main()
