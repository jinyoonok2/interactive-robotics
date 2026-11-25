from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from diagnostics.v3_grasp_probe import (GoalIntervention, choose_reference,
    inference_with_fixed_noise, matching_states, relative_grasp_goal, retention_score,
    world_grasp_goal)
from models.pose_token_diffusion_v3.dataset import observation_from_batch
from models.pose_token_diffusion_v3.tests.test_model import fixture


def singleton():
    model, batch = fixture()
    batch = {key: value[:1] if isinstance(value, (torch.Tensor, list)) else value for key, value in batch.items()}
    model.requires_grad_(False).eval()
    return model, observation_from_batch(batch)


def state(tcp_z=0., object_z=0.):
    return {'tcp_xyzw_xyz': [0., 0., 0., 1., 0., 0., tcp_z],
            'object_xyz': [0., 0., object_z], 'object_xyzw': [0., 0., 0., 1.]}


class GraspProbeTests(unittest.TestCase):
    def test_identity_hook_preserves_predictions_and_rng(self):
        model, obs = singleton()
        rng = torch.get_rng_state().clone()
        first = inference_with_fixed_noise(model, obs, 42, 0)
        with GoalIntervention(model) as hook:
            second = inference_with_fixed_noise(model, obs, 42, 0)
            self.assertFalse(hook.last['privileged_target'])
        torch.testing.assert_close(first.actions['action_chunk'], second.actions['action_chunk'], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(len(model.blocks['goal_head']._forward_hooks), 0)

    def test_only_target_geometry_changes_while_visual_context_and_weights_remain(self):
        model, obs = singleton()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        original = inference_with_fixed_noise(model, obs, 42, 1)
        position = original.predicted_goal.position_world[0, 0].tolist()
        position[0] += .05
        reference = {'position_world': position,
                     'rotation_world': Rotation.from_euler('z', .5).as_matrix().tolist(), 'opening_m': .01}
        with GoalIntervention(model, lambda: reference) as hook:
            altered = inference_with_fixed_noise(model, obs, 42, 1)
            self.assertTrue(hook.last['privileged_target'])
            np.testing.assert_allclose(hook.last['raw_prediction']['position_world'], original.predicted_goal.position_world[0, 0].tolist())
        torch.testing.assert_close(altered.encoded.visual_tokens, original.encoded.visual_tokens, rtol=0, atol=0)
        torch.testing.assert_close(altered.grounding.center_world, original.grounding.center_world, rtol=0, atol=0)
        torch.testing.assert_close(altered.predicted_goal.skill_logits, original.predicted_goal.skill_logits, rtol=0, atol=0)
        torch.testing.assert_close(altered.predicted_goal.applicability_logits, original.predicted_goal.applicability_logits, rtol=0, atol=0)
        self.assertFalse(torch.equal(original.pose_memory.tokens, altered.pose_memory.tokens))
        self.assertFalse(torch.equal(original.actions['action_chunk'], altered.actions['action_chunk']))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_hook_is_removed_even_if_provider_fails(self):
        model, obs = singleton()
        def bad():
            raise ValueError('invalid reference')
        with self.assertRaises(ValueError):
            with GoalIntervention(model, bad):
                inference_with_fixed_noise(model, obs, 42, 0)
        self.assertEqual(len(model.blocks['goal_head']._forward_hooks), 0)

    def test_forced_geometry_gate_preserves_geometry_opening_confidence_and_features(self):
        model, obs = singleton()
        original = inference_with_fixed_noise(model, obs, 42, 0)
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        with GoalIntervention(model, force_geometry_confidence=True) as hook:
            changed = inference_with_fixed_noise(model, obs, 42, 0)
            self.assertTrue(hook.last['force_geometry_confidence'])
            self.assertFalse(hook.last['privileged_target'])
        for name in ('position_world', 'rotation_world', 'opening', 'component_valid', 'skill_logits'):
            torch.testing.assert_close(getattr(changed.predicted_goal, name), getattr(original.predicted_goal, name), rtol=0, atol=0)
        torch.testing.assert_close(changed.predicted_goal.applicability_logits[..., 2], original.predicted_goal.applicability_logits[..., 2], rtol=0, atol=0)
        self.assertTrue(torch.equal(changed.predicted_goal.applicability_logits[..., :2].sigmoid(), torch.ones(1, 1, 2)))
        torch.testing.assert_close(changed.encoded.visual_tokens, original.encoded.visual_tokens, rtol=0, atol=0)
        self.assertFalse(torch.equal(changed.pose_memory.tokens, original.pose_memory.tokens))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_object_relative_grasp_transforms_consistently(self):
        initial = state(tcp_z=.1, object_z=.05)
        initial['object_xyzw'] = Rotation.from_euler('z', np.pi/2).as_quat().tolist()
        initial['tcp_xyzw_xyz'][4] = .02
        reference = relative_grasp_goal(initial, .003)
        same = world_grasp_goal(reference, [*initial['object_xyzw'], *initial['object_xyz']])
        np.testing.assert_allclose(same['position_world'], initial['tcp_xyzw_xyz'][4:], atol=1e-12)
        moved = world_grasp_goal(reference, [*initial['object_xyzw'], 1., 0., .05])
        np.testing.assert_allclose(moved['position_world'], [1.02, 0., .1], atol=1e-12)
        self.assertEqual(moved['opening_m'], .003)

    def test_native_contact_alone_does_not_validate_reference(self):
        before, after = state(), state(tcp_z=.05)
        hold = [{'state': after, 'force_qualified_fingers': [9]} for _ in range(10)]
        outcome = retention_score(before, after, hold, native_grasp=True, commanded_opening=0.)
        self.assertFalse(outcome['retained_grasp'])
        self.assertFalse(outcome['passes']['object_lift'])
        self.assertIsNone(choose_reference({'reference_replay': {'status': 'completed', 'retention': outcome}})[0])

    def test_retention_needs_lift_contact_and_closed_command(self):
        before, after = state(), state(tcp_z=.05, object_z=.05)
        hold = [{'state': after, 'force_qualified_fingers': [9, 10]} for _ in range(10)]
        self.assertTrue(retention_score(before, after, hold, native_grasp=True, commanded_opening=0.)['retained_grasp'])
        self.assertFalse(retention_score(before, after, hold, native_grasp=True, commanded_opening=.04)['retained_grasp'])
        self.assertFalse(retention_score(before, after, hold, native_grasp=False, commanded_opening=0.)['retained_grasp'])
        no_contact = [{'state': after, 'force_qualified_fingers': []} for _ in range(10)]
        self.assertFalse(retention_score(before, after, no_contact, native_grasp=True, commanded_opening=0.)['retained_grasp'])

    def test_reference_choice_is_predeclared_and_requires_valid_goal(self):
        row = {'status': 'completed', 'retention': {'retained_grasp': True}, 'reference_goal': {'pose': 'present'}}
        name, _ = choose_reference({'reference_native': row, 'reference_replay': row})
        self.assertEqual(name, 'reference_replay')
        with self.assertRaises(ValueError):
            choose_reference({'reference_replay': {**row, 'reference_goal': None}})

    def test_reset_mismatch_is_not_a_valid_paired_comparison(self):
        a, b = state(), state()
        self.assertTrue(matching_states(a, b))
        b['object_xyz'][0] += .01
        self.assertFalse(matching_states(a, b))


if __name__ == '__main__':
    unittest.main()
