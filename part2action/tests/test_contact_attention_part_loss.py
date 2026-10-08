import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import contact_attention_part_loss


def test_contact_attention_prefers_flagged_part_points():
    scene_xyz = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    contact_xyz = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    part_labels = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    sample_valid = torch.ones(1)
    on_part = torch.tensor([[8.0, 7.0, -4.0]])
    off_part = torch.tensor([[-4.0, -4.0, 8.0]])
    weights = {"contact_attn_sigma": 0.03}
    on_loss, on_mass = contact_attention_part_loss(
        on_part, scene_xyz, contact_xyz, part_labels, sample_valid, weights
    )
    off_loss, off_mass = contact_attention_part_loss(
        off_part, scene_xyz, contact_xyz, part_labels, sample_valid, weights
    )
    assert on_loss < off_loss
    assert on_mass > off_mass
    assert on_mass > 0.8


def test_contact_attention_skips_frames_without_part_points():
    scene_xyz = torch.zeros(1, 3, 3)
    contact_xyz = torch.zeros(1, 3)
    part_labels = torch.zeros(1, 3)
    sample_valid = torch.ones(1)
    loss, attn = contact_attention_part_loss(
        torch.zeros(1, 3),
        scene_xyz,
        contact_xyz,
        part_labels,
        sample_valid,
        {"contact_attn_sigma": 0.03},
    )
    assert float(loss) == 0.0
    assert float(attn) == 0.0
