"""Masked current-skill action/pose/grounding losses; progress has its own queries."""
import torch
from torch.nn import functional as F

from models.pose_token_diffusion.blocks.geometry import quaternion_matrix


def average(value, mask):
    return (value * mask).sum() / mask.sum().clamp(min=1)


def active_targets(batch):
    phases = batch['phase_index'].long()
    index = torch.arange(len(phases), device=phases.device)
    return (batch['goal_poses'][index, phases], batch['goal_opening'][index, phases],
            batch['goal_supervision_valid'][index, phases].bool())


def compute_losses(output, batch, model, weights=None):
    weights = weights or {}
    prediction, opening, mask = active_targets(batch)
    goal = output.predicted_goal
    noise_error = (output.actions['action_noise_pred'].float() - output.actions['action_noise'].float()).square().mean(-1)
    parts = {'action': average(noise_error, batch['action_valid'].bool())}
    delta = (goal.position_world[:, 0] - prediction[:, 4:7]) / model.config.translation_scale_m
    parts['goal_position'] = average(F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction='none').mean(-1), mask[:, 0])
    rotation = quaternion_matrix(prediction[:, :4])
    parts['goal_rotation'] = average((goal.rotation_world[:, 0] - rotation).square().sum((-1, -2))/6, mask[:, 1])
    parts['goal_opening'] = average(F.smooth_l1_loss(goal.opening[:, 0]/.04, opening/.04, reduction='none'), mask[:, 2])
    parts['goal_applicability'] = F.binary_cross_entropy_with_logits(goal.applicability_logits[:, 0].float(), mask.float())
    if 'point_part_labels' in batch:
        labels = batch['point_part_labels'].float() * output.encoded.scene_valid_mask
        counts = labels.sum(-1)
        eligible = batch['point_part_valid'].bool() & (counts > 0)
        parts['grounding_region'] = average(-(labels/counts.clamp(min=1)[:, None] *
            output.grounding.logits.log_softmax(-1)).sum(-1), eligible)
        gate = output.encoded.hierarchy['gate_logits']
        if gate is not None:
            side = int(gate.shape[1] ** .5)
            target = F.interpolate(batch['part_mask'][:, None].float(), (side, side), mode='nearest').flatten(1)
            parts['gate'] = average(F.binary_cross_entropy_with_logits(gate.float(), target,
                reduction='none', pos_weight=gate.new_tensor(10.).float()).mean(-1), eligible)
    unknown = set(weights) - {name + '_weight' for name in parts} - {'progress_weight'}
    if unknown:
        raise ValueError('Unknown/inapplicable v3 losses: ' + str(sorted(unknown)))
    total = sum(value * float(weights.get(key + '_weight', 1.)) for key, value in parts.items())
    return total, {**{key: value.detach() for key, value in parts.items()}, 'total': total.detach()}
