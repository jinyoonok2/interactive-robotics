"""Per-skill goal, action and hierarchy objectives; no labels enter observations."""
import torch
from torch.nn import functional as F

from .dataset import provided_goals_from_batch


def average(values, mask):
    return (values*mask).sum()/mask.sum().clamp(min=1)


def compute_losses(output,batch,model,weights=None):
    weights = weights or {}
    parts = {'action':F.mse_loss(output.actions['action_noise_pred'].float(),output.actions['action_noise'].float())}
    target = provided_goals_from_batch(batch)
    predicted = output.predicted_goals
    mask = batch['goal_supervision_valid'].bool()
    phases = batch['phase_index'].long()
    index = torch.arange(phases.shape[0],device=phases.device)
    position_error = (predicted.position_world-target.position_world)/model.config.translation_scale_m
    parts['goal_position'] = average(F.smooth_l1_loss(position_error,torch.zeros_like(position_error),reduction='none').mean(-1),mask[...,0])
    parts['goal_rotation'] = average((predicted.rotation_world-target.rotation_world).square().sum((-1,-2))/6,mask[...,1])
    parts['goal_opening'] = average(F.smooth_l1_loss(predicted.opening/.04,target.opening/.04,reduction='none'),mask[...,2])
    parts['goal_applicability'] = F.binary_cross_entropy_with_logits(
        predicted.applicability_logits[index,phases].float(),mask[index,phases].float())
    slot_valid = batch['skill_valid_mask'].bool()
    parts['goal_skill'] = average(F.cross_entropy(predicted.skill_logits.float().transpose(1,2),
        batch['goal_skill_kind'].long(),reduction='none'),slot_valid)
    if 'point_part_labels' in batch:
        labels = batch['point_part_labels'].float()
        if output.encoded.scene_valid_mask is not None:
            labels = labels*output.encoded.scene_valid_mask
        counts = labels.sum(-1)
        eligible = batch['point_part_valid'].bool() & (counts>0)
        logits = torch.stack([r.logits for r in output.grounding_bank],1)[index,phases]
        parts['grounding_region'] = average(-(labels/counts.clamp(min=1)[:,None]*logits.log_softmax(-1)).sum(-1),eligible)
    hierarchy = output.encoded.hierarchy
    slot_targets = model.blocks['encoder'].encode_skill_targets(batch['skill_plan'])
    parts['skill_embedding'] = average(1-F.cosine_similarity(hierarchy['plan_slots'].float(),slot_targets.float(),dim=-1),slot_valid)
    parts['slot_validity'] = F.binary_cross_entropy_with_logits(hierarchy['slot_valid_logits'].float(),slot_valid.float())
    parts['phase'] = F.cross_entropy(hierarchy['phase_logits'].float(),phases)
    parts['termination'] = F.binary_cross_entropy_with_logits(hierarchy['termination_logits'].float(),
        batch['phase_termination'].float(),pos_weight=hierarchy['termination_logits'].new_tensor(weights.get('termination_pos_weight',10.)).float())
    if 'gate_logits' in hierarchy:
        logits = hierarchy['gate_logits'].float()
        side = int(logits.shape[1]**.5)
        labels = F.interpolate(batch['part_mask'][:,None].float(),(side,side),mode='nearest').flatten(1)
        parts['gate'] = F.binary_cross_entropy_with_logits(logits,labels,pos_weight=logits.new_tensor(10.))
    unknown = set(weights)-{key+'_weight' for key in parts}-{'termination_pos_weight'}
    if unknown:
        raise ValueError('Unknown v2 loss weights: '+str(sorted(unknown)))
    total = sum(value*float(weights.get(key+'_weight',1.)) for key,value in parts.items())
    return total,{**{key:value.detach() for key,value in parts.items()},'total':total.detach()}
