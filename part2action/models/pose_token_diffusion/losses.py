"""Action, pose, region and skill supervision; no labels enter encoder inputs."""
import torch
from torch.nn import functional as F
from .dataset import target_pose_from_tensor


def compute_losses(output, batch, model, weights):
    parts = {}
    parts['action'] = F.mse_loss(output.actions['action_noise_pred'].float(), output.actions['action_noise'].float())
    valid = batch['pose_supervision_valid'].bool()
    target = target_pose_from_tensor(batch['goal_pose'], valid)
    predicted = output.predicted_pose
    denominator = valid.sum().clamp(min=1)
    if predicted is not None:
        displacement = (predicted.position_world-target.position_world)/model.config.translation_scale_m
        position = F.smooth_l1_loss(displacement, torch.zeros_like(displacement), reduction='none').mean(-1)
        rotation = (predicted.rotation_world-target.rotation_world).square().sum((-1, -2))/6
        parts['pose_position'] = (position*valid).sum()/denominator
        parts['pose_rotation'] = (rotation*valid).sum()/denominator
        parts['pose_applicability'] = F.binary_cross_entropy_with_logits(predicted.applicability_logits.float(), valid.float())
    if 'point_part_labels' in batch:
        labels = batch['point_part_labels'].float()
        counts = labels.sum(-1)
        eligible = batch['point_part_valid'].bool() & (counts > 0)
        if output.encoded.scene_valid_mask is not None:
            labels = labels*output.encoded.scene_valid_mask
            counts = labels.sum(-1)
            eligible = eligible & (counts > 0)
        distribution = labels/counts.clamp(min=1)[:, None]
        per_sample = -(distribution*output.grounding.logits.float().log_softmax(-1)).sum(-1)
        parts['grounding_region'] = (per_sample*eligible).sum()/eligible.sum().clamp(min=1)
    hierarchy = output.encoded.hierarchy or {}
    if hierarchy:
        slot_targets = model.blocks['encoder'].encode_skill_targets(batch['skill_plan'])
        slot_valid = batch['skill_valid_mask'].bool()
        slot_error = 1-F.cosine_similarity(hierarchy['plan_slots'].float(), slot_targets.float(), dim=-1)
        parts['skill_embedding'] = (slot_error*slot_valid).sum()/slot_valid.sum().clamp(min=1)
        parts['slot_validity'] = F.binary_cross_entropy_with_logits(hierarchy['slot_valid_logits'].float(), slot_valid.float())
        parts['phase'] = F.cross_entropy(hierarchy['phase_logits'].float(), batch['phase_index'].long())
        termination = output.progress_logits if output.progress_logits is not None else hierarchy['termination_logits']
        parts['termination'] = F.binary_cross_entropy_with_logits(termination.float(), batch['phase_termination'].float(),
            pos_weight=termination.new_tensor(weights.get('termination_pos_weight', 10.)).float())
        if 'gate_logits' in hierarchy:
            logits = hierarchy['gate_logits'].float()
            side = int(logits.shape[1]**.5)
            labels = F.interpolate(batch['part_mask'][:, None].float(), (side, side), mode='nearest').flatten(1)
            parts['gate'] = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(10.))
    allowed = {'action', 'pose_position', 'pose_rotation', 'pose_applicability', 'grounding_region',
               'skill_embedding', 'slot_validity', 'phase', 'termination', 'gate'}
    unknown = set(weights)-{name+'_weight' for name in allowed}-{'termination_pos_weight'}
    if unknown:
        raise ValueError('Unknown or inapplicable loss weights: '+', '.join(sorted(unknown)))
    total = sum(value*float(weights.get(name+'_weight', 1.)) for name, value in parts.items())
    return total, {**{key: value.detach() for key, value in parts.items()}, 'total': total.detach()}
