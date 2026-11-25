"""Readable component diagnostics from one held-out forward pass.

Annotated phase selects a bank slot for diagnostic scoring only. The policy
forward pass always uses its learned phase and predicted goals. Also report
selected-goal errors to expose phase-selection errors in the integrated path.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .dataset import provided_goals_from_batch
from .targets import SKILLS


def component_rows(output, batch):
    target = provided_goals_from_batch(batch)
    goal = output.predicted_goals
    hierarchy = output.encoded.hierarchy
    phases = batch['phase_index'].long()
    predicted_phase = hierarchy['predicted_phase'].long()
    index = torch.arange(len(phases),device=phases.device)
    mask = batch['goal_supervision_valid'][index,phases].bool()
    kinds = batch['goal_skill_kind'][index,phases].long()
    truth = target.position_world[index,phases].float()
    predicted = goal.position_world[index,phases].float()
    selected = goal.position_world[index,predicted_phase].float()
    relative_rotation = goal.rotation_world[index,phases].float().transpose(-1,-2) @ target.rotation_world[index,phases].float()
    angle = torch.rad2deg(torch.acos(((relative_rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)))
    applicability = goal.applicability_logits[index,phases].float().sigmoid()
    weights = torch.stack([r.weights for r in output.grounding_bank],1)[index,phases].float()
    selected_weights = torch.stack([r.weights for r in output.grounding_bank],1)[index,predicted_phase].float()
    labels = batch.get('point_part_labels',torch.zeros_like(weights)).bool()
    if output.encoded.scene_valid_mask is not None:
        labels = labels & output.encoded.scene_valid_mask.bool()
    grounding_valid = batch.get('point_part_valid',torch.zeros(len(phases),device=phases.device)).bool() & labels.any(-1)
    loss = (output.actions['action_noise_pred'].float()-output.actions['action_noise'].float()).square().mean((1,2))
    rows = []
    for i, meta in enumerate(batch['meta']):
        current, chosen = int(phases[i]),int(predicted_phase[i])
        row = {'object':Path(meta['file']).stem,'demo':meta['demo'],'t':int(meta['t']),
            'goal_label_quality':meta.get('goal_label_quality','unspecified'),
            'goal_label_reason':meta.get('goal_label_reason',''),
            'goal_target_t':meta.get('goal_target_t'),
            'touch_proxy_distance_m':meta.get('touch_proxy_distance_m'),
            'skill':SKILLS[int(kinds[i])],'reference_phase':current,'predicted_phase':chosen,
            'phase_correct':int(current==chosen),
            'predicted_skill':SKILLS[int(goal.skill_logits[i,current].argmax())],
            'skill_correct':int(goal.skill_logits[i,current].argmax()==kinds[i]),
            'termination_probability':float(hierarchy['termination_logits'][i].float().sigmoid()),
            'termination_target':int(batch['phase_termination'][i]>.5),
            'termination_valid':True,
            'action_noise_mse':float(loss[i]),
            'position_valid':int(mask[i,0]),'rotation_valid':int(mask[i,1]),'opening_valid':int(mask[i,2]),
            'goal_position_cm':float((predicted[i]-truth[i]).norm()*100) if mask[i,0] else None,
            'selected_goal_position_cm':float((selected[i]-truth[i]).norm()*100) if mask[i,0] else None,
            'goal_rotation_degrees':float(angle[i]) if mask[i,1] else None,
            'goal_opening_mm':float((goal.opening[i,current]-target.opening[i,current]).abs()*1000) if mask[i,2] else None,
            'grounding_top1_hit':int(labels[i,weights[i].argmax()]) if grounding_valid[i] else None,
            'selected_grounding_top1_hit':int(labels[i,selected_weights[i].argmax()]) if grounding_valid[i] else None,
            'grounding_target_mass':float(weights[i,labels[i]].sum()) if grounding_valid[i] else None}
        for j, name in enumerate(('position','rotation','opening')):
            row[name+'_applicability_probability'] = float(applicability[i,j])
        for j, axis in enumerate('xyz'):
            row['predicted_goal_'+axis+'_m'] = float(predicted[i,j])
            row['reference_goal_'+axis+'_m'] = float(truth[i,j]) if mask[i,0] else None
        rows.append(row)
    return rows


def distribution(values):
    values = np.asarray([x for x in values if x is not None],dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite validation metric')
    return {'count':len(values),'mean':float(values.mean()) if len(values) else None,
        'median':float(np.median(values)) if len(values) else None,
        'p90':float(np.quantile(values,.9)) if len(values) else None}


def classification(predicted, targets):
    predicted, targets = np.asarray(predicted,dtype=bool),np.asarray(targets,dtype=bool)
    tp,fp,fn,tn = [int(x.sum()) for x in (predicted&targets,predicted&~targets,~predicted&targets,~predicted&~targets)]
    return {'count':len(targets),'positive_targets':tp+fn,'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def summarize(rows):
    names = ('goal_position_cm','selected_goal_position_cm','goal_rotation_degrees','goal_opening_mm',
        'grounding_top1_hit','selected_grounding_top1_hit','grounding_target_mass','phase_correct','skill_correct','action_noise_mse')
    result = {'samples':len(rows),**{name:distribution([r[name] for r in rows]) for name in names}}
    terminal_rows = [r for r in rows if r.get('termination_valid',True)]
    result['termination'] = classification([r['termination_probability']>=.5 for r in terminal_rows],
                                          [r['termination_target'] for r in terminal_rows])
    result['applicability'] = {name:classification([r[name+'_applicability_probability']>=.5 for r in rows],
        [r[name+'_valid'] for r in rows]) for name in ('position','rotation','opening')}
    return result


def write_report(directory, rows, *, epoch, losses, scope):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    by_skill, by_object = defaultdict(list),defaultdict(list)
    for row in rows:
        by_skill[row['skill']].append(row)
        by_object[row['object']].append(row)
    result = {'epoch':epoch,'scope':scope,'losses':losses,'overall':summarize(rows),
        'by_skill':{k:summarize(v) for k,v in sorted(by_skill.items())},
        'by_object':{k:summarize(v) for k,v in sorted(by_object.items())},
        'definitions':{'goal_errors':'Bank slot matched to annotated phase for scoring only; reference is demonstrated TCP.',
            'selected_goal_error':'Actual learned-phase goal compared with the current annotated target.',
            'grounding':'Top-weight point membership and probability mass on annotated target-part points; valid labels only.',
            'touch':'Fingertip-distance proxy labels; not verified physical contact.',
            'action_noise_mse':'Diffusion noise-prediction loss, not a task-success rate.',
            'coverage':'Null metrics mean no eligible labels, not zero error. Samples are correlated trajectory frames.'}}
    (directory/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (directory/'samples.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    columns = ['skill','samples','position labels','position mean cm','position p90 cm','rotation mean deg',
               'grounding top1 %','phase accuracy %','action noise MSE']
    table = [columns]
    for skill,item in [('ALL',result['overall']),*result['by_skill'].items()]:
        def fmt(metric, scale=1):
            value = item[metric]['mean']
            return 'N/A' if value is None else f'{value*scale:.3f}'
        pos = item['goal_position_cm']
        table.append([skill,str(item['samples']),str(pos['count']),fmt('goal_position_cm'),
            'N/A' if pos['p90'] is None else f"{pos['p90']:.3f}",fmt('goal_rotation_degrees'),
            fmt('grounding_top1_hit',100),fmt('phase_correct',100),fmt('action_noise_mse')])
    with (directory/'summary.csv').open('w',newline='') as handle:
        csv.writer(handle).writerows(table)
    text = [f'# Validation epoch {epoch}', '',scope,'',
        '| '+' | '.join(table[0])+' |','| '+' | '.join(['---']*len(columns))+' |']
    text += ['| '+' | '.join(row)+' |' for row in table[1:]]
    text += ['', 'See metrics.json for applicability/termination precision and recall, coverage, and per-object results.',
        'These are held-out demonstration diagnostics, not simulator task-success results. Touch is a geometry proxy.',
        'The annotated phase is used only to score bank slots; policy inputs and goal selection remain predicted.','']
    (directory/'README.md').write_text('\n'.join(text))
    return result
