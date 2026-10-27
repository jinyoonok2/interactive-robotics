"""Early contact must affect temporal features, without privileged inputs."""
from pathlib import Path
from unittest.mock import patch

import unittest
import torch
from torch import nn
import yaml

import models.part2action_model as architecture
from models.temporal import TemporalPatchTransformer
from scripts.validate_configs import validate


class Vision(nn.Module):
    embed_dim, grid = 16, 2

    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, rgb):
        return rgb.mean((1, 2, 3))[:, None, None].expand(-1, 4, 16)


class Text(nn.Module):
    embed_dim = 16

    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, texts):
        return torch.ones(len(texts), 2, 16), torch.ones(len(texts), 2, dtype=torch.bool)


def build():
    with patch.object(architecture, 'FrozenDINOv2', Vision), patch.object(architecture, 'FrozenT5', Text):
        return architecture.Part2ActionModel(
            heads=['action'], hidden_dim=16, use_pcd=True, use_contact_xyz=True,
            use_contact_action_token=True, use_unified_contact_conditioning=True,
            contact_conditioning_stage='early_temporal', use_gripper_state=True,
            n_obs_steps=2, temporal_encoder_type='transformer',
            use_part_gate=True, use_hierarchy=True,
            action_head_type='diffusion', diffusion_steps=4).eval()


def test_predicted_contact_enters_temporal_before_fusion_and_receives_gradients():
    torch.manual_seed(7)
    model = build()
    assert not hasattr(model, 'contact_geometry_conditioner')
    inputs = dict(rgb=torch.randn(1, 2, 3, 28, 28), instructions=['grasp'],
                  tcp_pose=torch.tensor([[0., 0., 0., 1., .1, .2, .3]]),
                  agentview_pcd=torch.randn(1, 16, 3), gripper_state=torch.tensor([[.04]]),
                  target_action=torch.randn(1, 8, 7))
    events = []
    handles = [getattr(model, name).register_forward_hook(
        lambda module, args, output, name=name: events.append(name))
        for name in ['contact_attention', 'temporal', 'fusion', 'action_head']]
    torch.manual_seed(42)
    original = model(**inputs, contact_xyz_target=torch.ones(1, 3) * 100)
    assert events == ['contact_attention', 'temporal', 'fusion', 'action_head']
    for handle in handles:
        handle.remove()
    torch.manual_seed(42)
    changed_gt = model(**inputs, contact_xyz_target=torch.ones(1, 3) * -100)
    assert torch.equal(original['action_noise_pred'], changed_gt['action_noise_pred'])

    predict = model.contact_attention.forward

    def shifted(*args, **kwargs):
        xyz, token, logits = predict(*args, **kwargs)
        return xyz + torch.tensor([[.3, -.2, .1]]), token, logits

    with patch.object(model.contact_attention, 'forward', shifted):
        torch.manual_seed(42)
        changed = model(**inputs)
    assert not torch.allclose(original['fused'], changed['fused'])
    assert not torch.allclose(original['action_noise_pred'], changed['action_noise_pred'])
    loss = (original['action_noise_pred'] - original['action_noise']).square().mean()
    loss.backward()
    for module in [model.early_contact_encoder, model.temporal]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    # Action gradients cannot move the contact head's target; auxiliary loss trains it.
    assert all(p.grad is None for p in model.contact_attention.parameters())
    model.zero_grad(set_to_none=True)
    auxiliary = model(**inputs)['contact_xyz'].square().mean()
    auxiliary.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.contact_attention.parameters())

    model.contact_conditioning_enabled = False
    torch.manual_seed(42)
    control = model(**inputs)
    with patch.object(model.contact_attention, 'forward', shifted):
        torch.manual_seed(42)
        shifted_control = model(**inputs)
    assert torch.count_nonzero(control['contact_conditioning_features']) == 0
    assert torch.equal(control['action_noise_pred'], shifted_control['action_noise_pred'])
    with torch.no_grad():
        inference = model(**{k: v for k, v in inputs.items() if k != 'target_action'})
    assert inference['action_chunk'].shape == (1, 8, 7)
    assert torch.isfinite(inference['action_chunk']).all()


def test_temporal_context_attends_before_readout():
    encoder = TemporalPatchTransformer(16, max_steps=2).eval()
    history = torch.randn(2, 2, 4, 16)
    a = encoder(history, context_token=torch.zeros(2, 16))
    b = encoder(history, context_token=torch.randn(2, 16))
    assert a.shape == b.shape == (2, 4, 16)
    assert not torch.allclose(a, b)
    with unittest.TestCase().assertRaises(ValueError):
        encoder(history, context_token=torch.zeros(2, 8))


def test_early_configs_match_and_propagate_to_training_factory():
    from scripts.train import build_model
    root = Path(__file__).resolve().parents[1] / 'configs/architecture_update'
    configs = []
    for variant in ['control', 'contact']:
        path = root / f'hierarchical_world_geometry_30epoch_diffusion_early_{variant}.yaml'
        assert validate(path, check_files=True) == []
        cfg = yaml.safe_load(path.read_text())
        cfg['model']['text_device'] = 'cpu'
        with patch.object(architecture, 'FrozenDINOv2', Vision), patch.object(architecture, 'FrozenT5', Text):
            model = build_model(cfg)
        assert model.contact_conditioning_stage == 'early_temporal'
        cfg.pop('name')
        cfg.pop('output_dir')
        cfg['model'].pop('contact_conditioning_enabled')
        configs.append(cfg)
    assert configs[0] == configs[1]
