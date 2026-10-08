"""Tests for TCP XYZ delta action conversion helpers."""
from __future__ import annotations

import torch

from data.geometry import (
    absolute_actions_to_tcp_deltas,
    tcp_deltas_to_absolute_actions,
)


def test_absolute_delta_roundtrip():
    tcp = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.5, -0.2, 0.3]])
    absolute = torch.tensor(
        [
            [
                [0.52, -0.18, 0.35, 0.1, 0.0, -0.2, 0.04],
                [0.55, -0.15, 0.33, 0.1, 0.0, -0.2, 0.00],
            ]
        ]
    )
    deltas = absolute_actions_to_tcp_deltas(absolute, tcp)
    assert torch.allclose(deltas[..., :3], absolute[..., :3] - tcp[:, None, 4:7])
    assert torch.allclose(deltas[..., 3:], absolute[..., 3:])
    recovered = tcp_deltas_to_absolute_actions(deltas, tcp)
    assert torch.allclose(recovered, absolute, atol=1e-6)
