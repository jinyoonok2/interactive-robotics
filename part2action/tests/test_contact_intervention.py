import types

import torch

from diagnostics.contact_intervention import install_contact_intervention, rotation_matrix
from models.heads import ActionChunkHead, ContactGeometryConditioner


def make_model():
    torch.manual_seed(3)
    return types.SimpleNamespace(use_unified_contact_conditioning=True,
                                 contact_conditioning_enabled=True, action_head_type="mlp",
                                 split_action_heads=False,
                                 contact_geometry_conditioner=ContactGeometryConditioner(16),
                                 action_head=ActionChunkHead(16, 2))


def test_normal_is_exact_and_probes_do_not_modify_features():
    model = make_model()
    f, c, tcp = torch.randn(1, 4, 16), torch.randn(1, 3), torch.randn(1, 7)
    saved = f.clone()
    expected = model.contact_geometry_conditioner(f, c, tcp)
    handle = install_contact_intervention(model, 1.0)
    actual = model.contact_geometry_conditioner(f, c, tcp)
    assert torch.equal(expected[0], actual[0]) and torch.equal(f, saved)
    diagnostic = model.contact_intervention_diagnostic
    torch.testing.assert_close(torch.tensor(diagnostic["candidates"]["1.0"]["action0"]),
                               model.action_head(actual[0])[0, 0])
    assert diagnostic["candidates"]["0.0"]["contact_signal_rms"] == 0
    handle.remove()


def test_zero_and_reduced_scale_input_keep_bias_and_weights():
    for strength in [0., .25]:
        model = make_model()
        f, c, tcp = torch.randn(1, 4, 16), torch.randn(1, 3), torch.randn(1, 7)
        original = model.contact_geometry_conditioner(f, c, tcp)
        weights = {k: v.clone() for k, v in model.contact_geometry_conditioner.state_dict().items()}
        expected = f + model.contact_geometry_conditioner.projection(original[1]*strength)[:, None, :]
        handle = install_contact_intervention(model, strength)
        actual = model.contact_geometry_conditioner(f, c, tcp)
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], original[1]*strength)
        for k, v in weights.items():
            assert torch.equal(v, model.contact_geometry_conditioner.state_dict()[k])
        handle.remove()


def test_rotation_comparison_handles_euler_wrap():
    import numpy as np
    assert np.allclose(rotation_matrix([0, 0, 0]), rotation_matrix([0, 0, 2*np.pi]))
