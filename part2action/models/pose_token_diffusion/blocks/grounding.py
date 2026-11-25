"""Task/state-conditioned soft region grounding, shared by pose and action."""
import torch
from torch import nn
from ..contracts import GroundedRegion


class TaskRegionGrounding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.budget = config.local_token_budget
        self.query = nn.Linear(config.hidden_dim*2, config.hidden_dim)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.score = nn.Sequential(nn.GELU(), nn.Linear(config.hidden_dim, 1))

    def forward(self, encoded):
        query = self.query(torch.cat([encoded.language_tokens[:, 0], encoded.robot_tokens[:, 0]], -1))
        logits = self.score(self.key(encoded.scene_tokens)+query[:, None]).squeeze(-1).float()
        if encoded.scene_valid_mask is not None:
            logits = logits.masked_fill(~encoded.scene_valid_mask, -1e9)
        weights = logits.softmax(-1)
        center = (weights[..., None]*encoded.scene_xyz_world.float()).sum(1)
        weighted = (weights[..., None]*encoded.scene_tokens).sum(1, keepdim=True)
        count = min(self.budget, logits.shape[1])
        indices = logits.topk(count, dim=1).indices
        local = encoded.scene_tokens.gather(1, indices[..., None].expand(-1, -1, encoded.scene_tokens.shape[-1]))
        # Soft weighting keeps actor gradients connected to region scores.
        selected_weights = weights.gather(1, indices)
        selected_weights = selected_weights/selected_weights.sum(-1, keepdim=True).clamp(min=1e-8)
        local = local*selected_weights[..., None]*count
        with torch.autocast(device_type=center.device.type, enabled=False):
            delta = encoded.scene_xyz_world.float()-center[:, None]
            covariance = torch.einsum('bn,bni,bnj->bij', weights, delta, delta)
        return GroundedRegion(torch.cat([weighted, local], 1), center, logits, weights, covariance)
