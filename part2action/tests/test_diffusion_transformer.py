"""Transformer diffusion shares DDPM semantics and existing conditioning inputs."""
from pathlib import Path
from unittest.mock import patch
import unittest

import torch
import yaml

from models.heads import DiffusionActionHead
from scripts.validate_configs import validate


def test_transformer_training_sampling_checkpoint_and_action_interaction():
    torch.manual_seed(12)
    head = DiffusionActionHead(16, hidden=16, num_steps=5, schedule='cosine',
        denoiser_type='transformer', transformer_layers=2, transformer_heads=4,
        action_scale=[.5] * 3 + [3.14] * 3 + [.02],
        action_offset=[0.] * 6 + [.02], clip_sample=2.)
    features = torch.randn(2, 4, 16, requires_grad=True)
    target = torch.randn(2, 8, 7)
    out = head(features, target)
    assert out['action_noise_pred'].shape == target.shape
    loss = (out['action_noise_pred'] - out['action_noise']).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and features.grad.abs().sum() > 0
    assert head.net.input.weight.grad.abs().sum() > 0
    assert head.net.position.grad.abs().sum() > 0
    assert head.cond[0].weight.grad.abs().sum() > 0
    head.eval()
    noisy, time = torch.randn(2, 8, 7), torch.tensor([1, 2])
    a = head._denoise(features, noisy, time)
    other = noisy.clone()
    other[:, 0] += 1
    b = head._denoise(features, other, time)
    assert not torch.allclose(a[:, 1:], b[:, 1:])  # Action steps interact.
    # Retain pooled inputs for this experiment, rather than adding token memory.
    torch.testing.assert_close(head._denoise(features.flip(1), noisy, time), a)
    restored = DiffusionActionHead(16, hidden=16, num_steps=5, schedule='cosine',
        denoiser_type='transformer', transformer_layers=2, transformer_heads=4,
        clip_sample=2.)
    restored.load_state_dict(head.state_dict(), strict=True)
    restored.eval()
    torch.manual_seed(42)
    sample = head(features)['action_chunk']
    torch.manual_seed(42)
    copy = restored(features)['action_chunk']
    assert torch.equal(sample, copy)
    assert sample.shape == (2, 8, 7) and torch.isfinite(sample).all()
    assert head.normalize(sample).abs().max() <= 2.00001
    clean = head.normalize(target)

    def oracle(fused, action, t):
        bar = head.alpha_bars[t].reshape(-1, 1, 1)
        return (action - bar.sqrt() * clean) / (1 - bar).sqrt()

    head.clip_sample = None
    with patch.object(head, '_denoise', oracle):
        recovered = head.sample(features.detach())
    torch.testing.assert_close(recovered, target, atol=1e-4, rtol=1e-4)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        amp = head(features, target)
    assert torch.isfinite(amp['action_noise_pred']).all()


def test_transformer_configuration_changes_only_decoder():
    base = Path(__file__).resolve().parents[1] / 'configs/architecture_update'
    path = base / 'hierarchical_world_geometry_30epoch_diffusion_transformer_control.yaml'
    assert validate(path, check_files=True) == []
    old, new = [yaml.safe_load(p.read_text()) for p in [
        base / 'hierarchical_world_geometry_30epoch_unified_diffusion_control.yaml', path]]
    assert new['model']['use_part_gate']
    assert not new['model']['use_point_part_head']
    assert not new['model']['contact_conditioning_enabled']
    assert not new['train'].get('init_checkpoint')
    for cfg in [old, new]:
        cfg.pop('name'); cfg.pop('output_dir')
    for key in ['diffusion_denoiser_type', 'diffusion_transformer_layers',
                'diffusion_transformer_heads', 'diffusion_transformer_dropout']:
        new['model'].pop(key)
    assert old == new
    with unittest.TestCase().assertRaises(ValueError):
        DiffusionActionHead(16, hidden=16, denoiser_type='transformer', transformer_heads=3)


def test_model_factories_pass_transformer_options():
    import models.part2action_model as architecture
    from scripts.train import build_model
    # Reuse lightweight perception stubs; no model downloads or GPUs.
    import runpy
    stubs = runpy.run_path(str(Path(__file__).with_name('test_early_contact.py')))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'configs/architecture_update/hierarchical_world_geometry_30epoch_diffusion_transformer_control.yaml').read_text())
    cfg['model']['text_device'] = 'cpu'
    with patch.object(architecture, 'FrozenDINOv2', stubs['Vision']), patch.object(architecture, 'FrozenT5', stubs['Text']):
        torch.manual_seed(42)
        model = build_model(cfg).eval()
        cfg['model']['diffusion_denoiser_type'] = 'mlp'
        torch.manual_seed(42)
        reference = build_model(cfg).eval()
    shared = {key: value for key, value in model.state_dict().items()
              if not key.startswith('action_head.net.')}
    assert all(torch.equal(value, reference.state_dict()[key]) for key, value in shared.items())
    assert model.action_head.denoiser_type == 'transformer'
    assert model.use_part_gate and not model.use_point_part_head
    inputs = dict(rgb=torch.randn(1, 2, 3, 28, 28), instructions=['grasp'],
        agentview_pcd=torch.randn(1, 16, 3),
        tcp_pose=torch.tensor([[0., 0., 0., 1., .1, .2, .3]]),
        gripper_state=torch.tensor([[.04]]), target_action=torch.randn(1, 8, 7))
    output = model(**inputs)
    assert output['action_noise_pred'].shape == (1, 8, 7)
    assert torch.count_nonzero(output['contact_conditioning_features']) == 0
    assert output['gate_logits'].shape == (1, 4)


def test_ten_epoch_next_action_pair_is_matched_from_scratch():
    base = Path(__file__).resolve().parents[1] / 'configs/architecture_update'
    configs = []
    for variant, kind in [('baseline', 'mlp'), ('transformer', 'transformer')]:
        path = base / f'hierarchical_world_geometry_10epoch_diffusion_next_action_{variant}.yaml'
        assert validate(path, check_files=True) == []
        cfg = yaml.safe_load(path.read_text())
        assert cfg['model']['action_head_type'] == 'diffusion'
        assert cfg['model'].pop('diffusion_denoiser_type') == kind
        assert cfg['train']['epochs'] == 10
        assert cfg['data']['action_target_offset'] == cfg['data']['sample_stride'] == 1
        assert cfg['data']['n_obs_steps'] == 2
        assert not cfg['model']['contact_conditioning_enabled']
        assert not cfg['train'].get('init_checkpoint')
        assert not cfg['train'].get('resume')
        cfg.pop('name'); cfg.pop('output_dir')
        configs.append(cfg)
    assert configs[0] == configs[1]
