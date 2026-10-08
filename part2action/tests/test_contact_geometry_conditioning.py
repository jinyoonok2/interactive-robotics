import torch
from models.heads import ContactGeometryConditioner, ActionChunkHead


def test_enabled_changes_actions_control_ignores_geometry_and_detaches():
    torch.manual_seed(4)
    module = ContactGeometryConditioner(16)
    decoder = ActionChunkHead(16, 8, 7)
    features = torch.randn(2, 4, 16, requires_grad=True)
    tcp = torch.tensor([[0., 0., 0., 1., .1, .2, .3]]).repeat(2, 1)
    contact = torch.tensor([[.2, .2, .3], [.4, .2, .3]], requires_grad=True)
    conditioned, relative = module(features, contact, tcp)
    assert torch.allclose(relative[:, 0], torch.tensor([1., 3.]))
    shifted, _ = module(features, contact + .05, tcp)
    assert not torch.allclose(decoder(conditioned), decoder(shifted))
    control, zeros = module(features, contact, tcp, enabled=False)
    other, _ = module(features, contact + 10, tcp, enabled=False)
    assert torch.equal(control, other) and torch.count_nonzero(zeros) == 0
    decoder(conditioned).sum().backward()
    assert contact.grad is None and features.grad is not None
    assert module.projection[0].weight.grad is not None


def test_invalid_geometry_is_zero_and_common_translation_is_invariant():
    module = ContactGeometryConditioner(16)
    features = torch.zeros(1, 16)
    tcp = torch.tensor([[0., 0., 0., 1., .1, .2, .3]])
    contact = torch.tensor([[.4, .2, .3]])
    a, _ = module(features, contact, tcp)
    shifted_tcp = tcp.clone(); shifted_tcp[:, 4:7] += 2
    b, _ = module(features, contact + 2, shifted_tcp)
    assert torch.allclose(a, b, atol=1e-5)
    invalid, relative = module(features, torch.full_like(contact, float('nan')), tcp)
    assert torch.isfinite(invalid).all() and torch.count_nonzero(relative) == 0


def test_model_training_and_rollout_use_predictions_not_labels():
    from unittest.mock import patch
    from torch import nn
    import models.part2action_model as architecture
    class Vision(nn.Module):
        embed_dim = 16
        grid = 2
        def __init__(self, **kwargs):
            super().__init__(); self.model = nn.Identity()
        def forward(self, rgb):
            return torch.ones(rgb.shape[0], 4, 16)
    class Text(nn.Module):
        embed_dim = 16
        def __init__(self, **kwargs):
            super().__init__(); self.model = nn.Identity(); self.device_str = 'cpu'
        def forward(self, texts):
            return torch.ones(len(texts), 2, 16), torch.ones(len(texts), 2, dtype=torch.bool)
    with patch.object(architecture, 'FrozenDINOv2', Vision), patch.object(architecture, 'FrozenT5', Text):
        model = architecture.Part2ActionModel(heads=['action'], hidden_dim=16,
            use_pcd=True, use_contact_xyz=True, use_contact_action_token=True,
            use_unified_contact_conditioning=True)
    rgb = torch.zeros(1, 3, 28, 28)
    tcp = torch.tensor([[0., 0., 0., 1., .1, .2, .3]])
    cloud = torch.randn(1, 16, 3)
    model.train()
    a = model(rgb, ['touch'], tcp_pose=tcp, agentview_pcd=cloud,
              contact_xyz_target=torch.full((1, 3), 100.))
    b = model(rgb, ['touch'], tcp_pose=tcp, agentview_pcd=cloud,
              contact_xyz_target=torch.full((1, 3), -100.))
    assert torch.equal(a['contact_conditioning_features'], b['contact_conditioning_features'])
    assert a['action_chunk'].shape == (1, 8, 7)
    assert 'near_contact' not in a and not hasattr(model, 'near_contact_head')
    model.eval()
    out = model(rgb, ['touch'], tcp_pose=tcp, agentview_pcd=cloud)
    assert torch.isfinite(out['action_chunk']).all()
    assert torch.allclose(out['contact_conditioning_features'],
                          (out['contact_xyz'].detach()-tcp[:, 4:7])/.1)
