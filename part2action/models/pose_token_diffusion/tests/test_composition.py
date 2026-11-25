"""Use tiny injected blocks; never load pretrained encoders or the simulator."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.checkpoint import load_checkpoint, save_checkpoint
from models.pose_token_diffusion.config import PoseTokenDiffusionConfig, load_config
from models.pose_token_diffusion.contracts import (
    EncodedObservation, GroundedRegion, Observation, PoseMemory, TargetPose)
from models.pose_token_diffusion.policy import PoseTokenDiffusionBlocks, PoseTokenDiffusionPolicy, build_policy
from models.part_grounded_diffusion.verify_sources import verify_legacy_sources

PACKAGE = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    """Synthetic data adapter. Its identity camera transform is test-only."""
    def __init__(self, width):
        super().__init__()
        self.visual = nn.Linear(3, width)
        self.points = nn.Linear(3, width)
        self.robot = nn.Linear(8, width)
        self.language = nn.Parameter(torch.ones(1, 1, width))

    def forward(self, observation):
        rgb = observation.rgb
        if rgb.ndim == 4:
            rgb = rgb[:, None]
        batch_size = rgb.shape[0]
        return EncodedObservation(
            self.visual(rgb.mean((-1, -2))), self.points(observation.scene_points_camera),
            observation.scene_points_camera, self.language.expand(batch_size, -1, -1),
            self.robot(torch.cat([observation.tcp_pose, observation.gripper_opening], -1))[:, None],
            observation.tcp_pose)


class TinyGrounding(nn.Module):
    def forward(self, encoded):
        return GroundedRegion(encoded.scene_tokens.mean(1, keepdim=True), encoded.scene_xyz_world.mean(1))


class TinyPoseHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.position = nn.Linear(width*2, 3)

    def forward(self, encoded, region):
        offset = self.position(torch.cat([region.tokens[:, 0], encoded.visual_tokens.mean(1)], -1))
        batch_size = offset.shape[0]
        return TargetPose(region.center_world+offset,
            torch.eye(3, device=offset.device).expand(batch_size, -1, -1),
            torch.ones(batch_size, device=offset.device, dtype=torch.bool))


class TinyTokenizer(nn.Module):
    """Simple routing fixture; full TCP rotation handling is a future real block."""
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(9, width)
        self.width = width

    def forward(self, encoded, goal):
        batch_size = encoded.visual_tokens.shape[0]
        if goal is None:
            return PoseMemory(encoded.visual_tokens.new_zeros(batch_size, 1, self.width),
                torch.zeros(batch_size, 1, dtype=torch.bool), encoded.visual_tokens.new_zeros(batch_size, 9))
        relative = torch.cat([goal.position_world-encoded.tcp_pose[:, 4:7],
                              goal.rotation_world[:, :, :2].reshape(batch_size, 6)], -1)
        return PoseMemory(self.projection(relative)[:, None], goal.valid[:, None], relative)


class TinyAttentionDecoder(nn.Module):
    """Attention fixture only; it is not a diffusion implementation."""
    def __init__(self, config):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, config.action_horizon, config.hidden_dim))
        self.attention = nn.MultiheadAttention(config.hidden_dim, 2, batch_first=True)
        self.output = nn.Linear(config.hidden_dim, config.action_dim)
        self.last_encoded = self.last_memory = self.last_target = None

    def forward(self, encoded, memory, *, region, target_actions=None):
        self.last_encoded, self.last_memory, self.last_target = encoded, memory, target_actions
        context = torch.cat([encoded.visual_tokens, encoded.scene_tokens, encoded.language_tokens,
                             encoded.robot_tokens, memory.tokens], 1)
        batch_size = context.shape[0]
        mask = torch.zeros(batch_size, context.shape[1], dtype=torch.bool)
        mask[:, -memory.tokens.shape[1]:] = ~memory.valid_mask
        attended, _ = self.attention(self.queries.expand(batch_size, -1, -1), context, context,
                                     key_padding_mask=mask, need_weights=False)
        return {'action_chunk': self.output(attended)}


class TinyProgress(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output = nn.Linear(width*2+9, 1)

    def forward(self, encoded, memory):
        return self.output(torch.cat([encoded.visual_tokens.mean(1), encoded.robot_tokens.mean(1),
                                      memory.relative_error], -1)).squeeze(-1)


def fixture(source='provided', progress=False):
    torch.manual_seed(14)
    config = PoseTokenDiffusionConfig(hidden_dim=8, action_horizon=3, pose_source=source, progress_enabled=progress)
    blocks = PoseTokenDiffusionBlocks(TinyEncoder(8), TinyGrounding(), TinyTokenizer(8),
        TinyAttentionDecoder(config), pose_head=TinyPoseHead(8), progress_head=TinyProgress(8))
    model = build_policy(config, blocks=blocks)
    observation = Observation(torch.randn(2, 2, 3, 8, 8), torch.randn(2, 5, 3),
        ['grasp handle', 'grasp neck'], torch.tensor([[0., 0., 0., 1., .1, .2, .3]]).repeat(2, 1),
        torch.full((2, 1), .04))
    goal = TargetPose(torch.tensor([[.2, .3, .4], [.4, .3, .2]]), torch.eye(3).expand(2, -1, -1),
                      torch.ones(2, dtype=torch.bool))
    return model, observation, goal


class CompositionTests(unittest.TestCase):
    def test_config_is_explicit_and_separate_from_legacy_recipes(self):
        config = load_config(PACKAGE/'configs/model.yaml')
        self.assertEqual(config.architecture, 'pose_token_diffusion_v1')
        self.assertEqual(config.language_backbone, 'google/flan-t5-base')
        with self.assertRaises(ValueError):
            PoseTokenDiffusionConfig.from_mapping({'model': {}, 'train': {}})
        with self.assertRaises(ValueError):
            PoseTokenDiffusionConfig.from_mapping({**config.to_dict(), 'use_contact_residual': True})
        with self.assertRaises(ValueError):
            replace(config, hidden_dim=True)

    def test_composition_keeps_contexts_separate_and_targets_decoder_only(self):
        model, observation, goal = fixture()
        target = torch.randn(2, 3, 7)
        out = model(observation, provided_pose=goal, target_actions=target)
        self.assertEqual(PoseTokenDiffusionPolicy.__bases__, (nn.Module,))
        self.assertEqual(out.actions['action_chunk'].shape, (2, 3, 7))
        self.assertIs(out.target_pose, goal)
        decoder = model.blocks['action_decoder']
        self.assertIs(decoder.last_memory, out.pose_memory)
        self.assertIs(decoder.last_target, target)
        self.assertEqual(decoder.last_encoded.visual_tokens.shape, (2, 2, 8))
        self.assertEqual(decoder.last_encoded.scene_tokens.shape, (2, 5, 8))
        self.assertIsNone(out.progress_logits)

    def test_predicted_route_has_action_gradients_into_pose_and_vision(self):
        model, observation, _ = fixture('predicted', progress=True)
        out = model(observation)
        out.actions['action_chunk'].square().mean().backward()
        for parameter in (model.blocks['encoder'].visual.weight, model.blocks['encoder'].points.weight,
                          model.blocks['pose_head'].position.weight, model.blocks['pose_tokenizer'].projection.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(out.progress_logits.shape, (2,))
        with self.assertRaises(ValueError):
            model(observation, provided_pose=out.target_pose)

    def test_pose_modes_fail_explicitly_and_disabled_preserves_visual_context(self):
        model, observation, goal = fixture()
        with self.assertRaises(ValueError):
            model(observation)
        model, observation, goal = fixture('disabled')
        out = model(observation)
        self.assertFalse(out.pose_memory.valid_mask.any())
        self.assertEqual(out.pose_memory.tokens.count_nonzero(), 0)
        out.actions['action_chunk'].square().mean().backward()
        self.assertGreater(model.blocks['encoder'].visual.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model(observation, provided_pose=goal)
        blocks = PoseTokenDiffusionBlocks(model.blocks['encoder'], model.blocks['grounding'],
            model.blocks['pose_tokenizer'], model.blocks['action_decoder'])
        with self.assertRaises(ValueError):
            build_policy(replace(model.config, pose_source='predicted'), blocks=blocks)
        with self.assertRaises(ValueError):
            build_policy(replace(model.config, progress_enabled=True), blocks=blocks)

    def test_observation_contract_rejects_label_fields_and_mismatched_geometry(self):
        model, observation, goal = fixture()
        with self.assertRaises(TypeError):
            Observation(**vars(observation), grasp_pose_target=goal)
        with self.assertRaises(ValueError):
            model(replace(observation, tcp_pose=torch.zeros(2, 6)), provided_pose=goal)
        encoded = model.blocks['encoder'](observation)
        with self.assertRaises(ValueError):
            replace(encoded, scene_xyz_world=torch.zeros(2, 4, 3)).validate(2, 8)
        with self.assertRaises(ValueError):
            model(observation, provided_pose=goal, target_actions=torch.zeros(2, 8, 7))

    def test_frozen_backbone_adapter_retains_history_and_eval_mode(self):
        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(3, 8)
                self.calls = 0
            def forward(self, rgb):
                self.calls += 1
                return self.projection(rgb.mean((-1, -2)))[:, None]
        class Language(nn.Module):
            def forward(self, texts):
                return torch.ones(len(texts), 2, 8), torch.ones(len(texts), 2, dtype=torch.bool)
        _, observation, _ = fixture()
        pair = FrozenBackbones(Vision(), Language())
        pair.train()
        out = pair(observation.rgb, observation.instructions)
        self.assertEqual(out['visual_history'].shape, (2, 2, 1, 8))
        self.assertEqual(pair.visual.calls, 1)
        self.assertFalse(any(module.training for module in pair.modules()))
        self.assertFalse(any(parameter.requires_grad for parameter in pair.parameters()))

    def test_checkpoint_roundtrip_cannot_overwrite_or_accept_legacy_identity(self):
        model, observation, goal = fixture()
        expected = model(observation, provided_pose=goal).actions['action_chunk'].detach()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'new.pt'
            save_checkpoint(model, path)
            original_bytes = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, path)
            self.assertEqual(original_bytes, path.read_bytes())
            restored, _, _ = fixture()
            load_checkpoint(restored, path)
            torch.testing.assert_close(restored(observation, provided_pose=goal).actions['action_chunk'], expected)
            foreign = Path(folder)/'old.pt'
            torch.save({'architecture': 'grounded_visual_pose_v1', 'model': model.state_dict()}, foreign)
            before = {key: value.clone() for key, value in restored.state_dict().items()}
            with self.assertRaises(ValueError):
                load_checkpoint(restored, foreign)
            for key, value in before.items():
                torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
            with self.assertRaises(TypeError):
                save_checkpoint(nn.Linear(2, 2), Path(folder)/'wrong.pt')

    def test_incompatible_checkpoint_is_rejected_before_any_weight_changes(self):
        model, _, _ = fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'new.pt'
            save_checkpoint(model, path)
            payload = torch.load(path, weights_only=True)
            keys = list(payload['model'])
            payload['model'][keys[0]] = payload['model'][keys[0]]+100
            payload['model'][keys[-1]] = torch.zeros(99)
            bad = Path(folder)/'bad.pt'
            torch.save(payload, bad)
            before = {key: value.clone() for key, value in model.state_dict().items()}
            with self.assertRaises(ValueError):
                load_checkpoint(model, bad)
            for key, value in before.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_legacy_integrity_guard_detects_missing_or_modified_sources(self):
        self.assertGreater(verify_legacy_sources()['preserved_sources'], 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root/'old.py'
            original.write_text('original bytes\n')
            manifest = root/'snapshot.json'
            manifest.write_text(json.dumps({'scope': 'test fixture', 'source_sha256': {
                'old.py': hashlib.sha256(original.read_bytes()).hexdigest()}}))
            verify_legacy_sources(root=root, manifest_path=manifest)
            original.write_text('modified bytes\n')
            with self.assertRaises(RuntimeError):
                verify_legacy_sources(root=root, manifest_path=manifest)
            original.unlink()
            with self.assertRaises(RuntimeError):
                verify_legacy_sources(root=root, manifest_path=manifest)


if __name__ == '__main__':
    unittest.main()
