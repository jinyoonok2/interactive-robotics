"""Human-readable skill-conditioned diagnostics; these are not task success rates."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import torch

from models.pose_token_diffusion.blocks.geometry import quaternion_matrix
from models.pose_token_diffusion_v2.metrics import classification
from .losses import active_targets


def component_rows(output, batch):
    target, opening, valid = active_targets(batch)
    goal = output.predicted_goal
    distance = (goal.position_world[:, 0] - target[:, 4:7]).norm(dim=-1)*100
    rotation = goal.rotation_world[:, 0].float().transpose(-1, -2) @ quaternion_matrix(target[:, :4])
    angles = torch.rad2deg(torch.acos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))
    noise = (output.actions['action_noise_pred'].float()-output.actions['action_noise'].float()).square().mean(-1)
    rows = []
    for i, meta in enumerate(batch['meta']):
        count = int(batch['action_valid'][i].sum())
        row = {'object': Path(meta['file']).stem, 'demo': meta['demo'], 't': int(meta['t']),
            'skill': batch['skill_spec'][i].kind, 'conditioning': 'annotated_current_skill',
            'goal_quality': meta.get('goal_quality', ''), 'valid_action_commands': count,
            'goal_position_cm': float(distance[i]) if valid[i, 0] else None,
            'goal_rotation_degrees': float(angles[i]) if valid[i, 1] else None,
            'opening_error_mm': float((goal.opening[i, 0]-opening[i]).abs()*1000) if valid[i, 2] else None,
            'action_noise_mse': float(noise[i][batch['action_valid'][i]].mean()) if count else None,
            'grounding_top1_hit': None, 'grounding_part_mass': None}
        if 'point_part_labels' in batch and bool(batch['point_part_valid'][i]):
            labels = batch['point_part_labels'][i].float()*output.encoded.scene_valid_mask[i]
            if labels.sum() > 0:
                weights = output.grounding.weights[i]
                row['grounding_top1_hit'] = float(labels[weights.argmax()])
                row['grounding_part_mass'] = float((weights*labels).sum())
        rows.append(row)
    return rows


def summarize(rows):
    result = {'samples': len(rows)}
    for key in ('goal_position_cm', 'goal_rotation_degrees', 'opening_error_mm',
                'action_noise_mse', 'grounding_top1_hit', 'grounding_part_mass'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = {'count': len(values), 'mean': sum(values)/len(values) if values else None}
    return result


def write_report(directory, rows, *, epoch, losses, progress_rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    by_skill, by_object = defaultdict(list), defaultdict(list)
    for row in rows:
        by_skill[row['skill']].append(row)
        by_object[row['object']].append(row)
    def score_progress(subset):
        result = classification([r['probability'] >= .5 for r in subset], [r['target'] for r in subset])
        result['false_positive_rate'] = result['fp']/(result['fp']+result['tn']) if result['fp']+result['tn'] else None
        return result
    progress = {'overall': score_progress(progress_rows),
        'by_skill': {k: score_progress([r for r in progress_rows if r['skill'] == k])
                     for k in sorted({r['skill'] for r in progress_rows})},
        'by_source': {k: score_progress([r for r in progress_rows if r['source'] == k])
                      for k in sorted({r['source'] for r in progress_rows})}}
    result = {'epoch': epoch, 'losses': losses, 'overall': summarize(rows),
        'by_skill': {k: summarize(v) for k, v in sorted(by_skill.items())},
        'by_object': {k: summarize(v) for k, v in sorted(by_object.items())}, 'progress': progress,
        'scope': 'Whole-demo held-out skill-conditioned diagnostics. Supplied annotated current skills; predicted geometric goals. Not VLM planning or simulator task success.',
        'limits': 'Touch geometry and complete-chain endpoints are proxies. Frames are correlated. Null means no eligible label. Validation coverage is recorded explicitly.'}
    (directory/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, values in [('samples.csv', rows), ('progress.csv', progress_rows)]:
        if values:
            with (directory/name).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
                writer.writeheader()
                writer.writerows(values)
    text = [f'# V3 validation epoch {epoch}', '', result['scope'], '',
        '| Skill | Frames | Position cm | Rotation degrees | Opening mm | Action noise MSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind, item in [('ALL', result['overall']), *result['by_skill'].items()]:
        fmt = lambda k: 'N/A' if item[k]['mean'] is None else f"{item[k]['mean']:.4f}"
        text.append('| ' + ' | '.join([kind, str(item['samples']), fmt('goal_position_cm'),
            fmt('goal_rotation_degrees'), fmt('opening_error_mm'), fmt('action_noise_mse')]) + ' |')
    text += ['', 'Progress readiness uses annotation-transition/endpoint proxies. See progress.csv and metrics.json.',
             result['limits'], '']
    (directory/'README.md').write_text('\n'.join(text))
    return result
