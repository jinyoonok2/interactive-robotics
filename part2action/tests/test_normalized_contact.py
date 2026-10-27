import torch

from models.heads import ContactGeometryConditioner


def test_control_checkpoint_bias_and_zero_weight_preserve_original_features():
    torch.manual_seed(42)
    old = ContactGeometryConditioner(16)
    new = ContactGeometryConditioner(16, mode="normalized_global")
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert missing == ["contact_weight"] and not unexpected
    features = torch.randn(2,4,16)
    contact = torch.randn(2,3,requires_grad=True)
    tcp = torch.randn(2,7)
    expected = old(features,contact,tcp,enabled=False)[0]
    actual = new(features,contact,tcp,enabled=False)[0]
    assert torch.equal(actual, expected)
    new.contact_weight.data.zero_()
    assert torch.equal(new(features,contact,tcp)[0], expected)


def test_normalized_branch_is_small_trainable_and_detaches_contact():
    torch.manual_seed(42)
    module = ContactGeometryConditioner(16, mode="normalized_global", initial_weight=.01)
    features = torch.randn(2,4,16,requires_grad=True)
    contact = torch.randn(2,3,requires_grad=True)
    tcp = torch.randn(2,7)
    conditioned, _ = module(features,contact,tcp)
    control, _ = module(features,contact,tcp,enabled=False)
    signal = conditioned-control
    assert signal.square().mean().sqrt() <= .01001
    (conditioned*torch.randn_like(conditioned)).sum().backward()
    assert module.contact_weight.grad is not None and module.contact_weight.grad.abs()>0
    assert contact.grad is None and features.grad is not None
    assert module.projection[0].weight.grad is not None
    fresh = ContactGeometryConditioner(16, mode="normalized_global")
    fresh.load_state_dict(module.state_dict())
    torch.testing.assert_close(fresh(features,contact,tcp)[0],conditioned)
