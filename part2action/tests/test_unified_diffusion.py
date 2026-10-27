"""Integration checks for normalized diffusion with predicted contact conditioning."""
from unittest.mock import patch

import torch
from torch import nn

from models.heads import DiffusionActionHead
import models.part2action_model as architecture


def test_diffusion_normalization_reverse_process_and_gradients():
    scale = [.5, .5, .5, 3.14, 3.14, 3.14, .02]
    offset = [0.] * 6 + [.02]
    head = DiffusionActionHead(16, hidden=16, num_steps=10, schedule="cosine",
                               action_scale=scale, action_offset=offset)
    features = torch.randn(2, 4, 16, requires_grad=True)
    target = torch.randn(2, 8, 7) * .1
    target[..., 6] = .04
    assert torch.allclose(head.denormalize(head.normalize(target)), target, atol=1e-6)
    assert head.alpha_bars[-1] < 1e-4
    restored = DiffusionActionHead(16, hidden=16, num_steps=10)
    restored.load_state_dict(head.state_dict())
    assert torch.equal(restored.action_scale, head.action_scale)
    assert torch.equal(restored.action_offset, head.action_offset)
    assert torch.equal(restored.alpha_bars, head.alpha_bars)
    out = head(features, target)
    loss = (out["action_noise_pred"] - out["action_noise"]).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and features.grad.abs().sum() > 0
    normalized = head.normalize(target)

    def oracle(fused, noisy, time):
        cumulative = head.alpha_bars[time].reshape(-1, 1, 1)
        return (noisy - cumulative.sqrt() * normalized) / (1 - cumulative).sqrt()

    with patch.object(head, "_denoise", oracle):
        recovered = head.sample(features.detach())
    assert torch.allclose(recovered, target, atol=1e-5)
    head.clip_sample = 2.
    sampled = head.sample(features.detach())
    assert torch.isfinite(sampled).all()
    assert head.normalize(sampled).abs().max() <= 2.00001


def test_unified_diffusion_train_and_inference_use_predicted_geometry():
    class Vision(nn.Module):
        embed_dim, grid = 16, 2

        def __init__(self, **kwargs):
            super().__init__()
            self.model = nn.Identity()

        def forward(self, rgb):
            return torch.ones(rgb.shape[0], 4, 16)

    class Text(nn.Module):
        embed_dim = 16

        def __init__(self, **kwargs):
            super().__init__()
            self.model, self.device_str = nn.Identity(), "cpu"

        def forward(self, texts):
            return torch.ones(len(texts), 2, 16), torch.ones(len(texts), 2, dtype=torch.bool)

    with patch.object(architecture, "FrozenDINOv2", Vision), patch.object(architecture, "FrozenT5", Text):
        model = architecture.Part2ActionModel(
            heads=["action"], hidden_dim=16, use_pcd=True, use_contact_xyz=True,
            use_contact_action_token=True, use_unified_contact_conditioning=True,
            contact_conditioning_mode="normalized_global",
            contact_conditioning_initial_weight=.01,
            action_head_type="diffusion", diffusion_steps=10, diffusion_schedule="cosine",
            diffusion_action_scale=[.5] * 3 + [3.14] * 3 + [.02],
            diffusion_action_offset=[0.] * 6 + [.02])
    model.eval()  # Disable dropout while comparing fixed-seed noise predictions.
    inputs = dict(tcp_pose=torch.tensor([[0., 0., 0., 1., .1, .2, .3]]),
                  agentview_pcd=torch.randn(1, 16, 3))
    rgb, target = torch.zeros(1, 3, 28, 28), torch.randn(1, 8, 7)
    torch.manual_seed(42)
    a = model(rgb, ["touch"], target_action=target,
              contact_xyz_target=torch.full((1, 3), 100.), **inputs)
    torch.manual_seed(42)
    b = model(rgb, ["touch"], target_action=target,
              contact_xyz_target=torch.full((1, 3), -100.), **inputs)
    assert torch.equal(a["action_noise_pred"], b["action_noise_pred"])
    loss = (a["action_noise_pred"] - a["action_noise"]).square().mean()
    loss.backward()
    assert model.contact_geometry_conditioner.projection[0].weight.grad.abs().sum() > 0
    assert model.contact_geometry_conditioner.contact_weight.grad.abs() > 0
    assert "near_contact" not in a and not hasattr(model, "near_contact_head")
    with torch.no_grad():
        out = model(rgb, ["touch"], **inputs)
    assert out["action_chunk"].shape == (1, 8, 7)
    assert torch.isfinite(out["action_chunk"]).all()
    assert torch.allclose(out["contact_conditioning_features"],
                          (out["contact_xyz"] - inputs["tcp_pose"][:, 4:7]) / .1)
    model.contact_conditioning_enabled = False
    with torch.no_grad():
        out = model(rgb, ["touch"], **inputs)
    assert torch.count_nonzero(out["contact_conditioning_features"]) == 0


def test_diffusion_configs_are_matched_and_reject_invalid_scales():
    from pathlib import Path
    import copy
    import yaml
    from scripts.validate_configs import validate

    root = Path(__file__).resolve().parents[1] / "configs" / "architecture_update"
    control_path = root / "hierarchical_world_geometry_30epoch_unified_diffusion_control.yaml"
    contact_path = root / "hierarchical_world_geometry_30epoch_unified_diffusion_contact.yaml"
    assert validate(control_path, check_files=True) == []
    assert validate(contact_path, check_files=True) == []
    control, contact = (yaml.safe_load(p.read_text()) for p in [control_path, contact_path])
    for cfg in [control, contact]:
        cfg.pop("name")
        cfg.pop("output_dir")
        cfg["model"].pop("contact_conditioning_enabled")
    assert control == contact
    bad = copy.deepcopy(control["model"]["diffusion_action_scale"])
    bad[-1] = 0
    try:
        DiffusionActionHead(16, action_scale=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero normalization scale must be rejected")
