"""Opt-in real-HDF5 integration check with tiny frozen test backbones.

Exercises production-width v2 trainable blocks, not pretrained DINO/T5 or a
simulator. This is an implementation check, never a policy performance result.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import torch
import yaml

from data.partinstruct_loader import collate_part2action
from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.tests.test_model import Vision, Language
from models.pose_token_diffusion_v2.config import load_config
from models.pose_token_diffusion_v2.dataset import build_dataset, observation_from_batch
from models.pose_token_diffusion_v2.policy import build_policy
from models.pose_token_diffusion_v2.training import train_step

PACKAGE = Path(__file__).resolve().parents[1]


def check(max_demos_per_file=32):
    recipe = yaml.safe_load((PACKAGE/'configs/train.yaml').read_text())
    data = {**recipe['data'], 'max_demos_per_file': max_demos_per_file,
            'sample_stride': 1, 'split': 'all'}
    dataset = build_dataset(data, recipe['goal_labels'])
    phases, valid, reasons, qualities = Counter(), Counter(), Counter(), Counter()
    touch_distances, examples, samples = [], [], []
    seen, selected = set(), set()
    for index, spec in enumerate(dataset._samples):
        key = spec.file_path, spec.demo_key
        if key in seen:
            continue
        seen.add(key)
        goals = dataset.targets_for(spec)
        for phase_index, phase in enumerate(goals):
            kind = phase['kind']
            phases[kind] += 1
            valid[kind] += int(phase['component_valid'].any())
            reasons[phase['reason'] or 'valid'] += 1
            qualities[phase['quality']] += 1
            distance = phase['touch_proxy_to_part_distance_m']
            if distance is not None:
                touch_distances.append(float(distance))
            # One real observation per skill kind, including later phases.
            if kind not in selected and phase['start'] < spec.n_steps-data['action_chunk']:
                sample_index = index+phase['start']  # stride=1, contiguous demo
                sample = dataset[sample_index]
                assert sample['meta']['demo'] == spec.demo_key
                assert int(sample['phase_index']) == phase_index
                samples.append(sample)
                selected.add(kind)
                examples.append({'object':Path(spec.file_path).stem,'demo':spec.demo_key,
                    'skill':kind,'instruction':phase['instruction'],'phase_start':phase['start'],
                    'target_t':phase['target_t'],'quality':phase['quality'],
                    'components':phase['component_valid'].tolist(),
                    'touch_proxy_to_part_distance_m':distance})
    if not {'grasp_obj','touch_obj','move_gripper','rotate_obj'}.issubset(selected):
        raise AssertionError('Real integration sample did not cover contact and later motion skills')
    # Independent construction must keep whole demonstrations disjoint.
    train = build_dataset({**data,'split':'train'}, recipe['goal_labels'])
    validation = build_dataset({**data,'split':'validation'}, recipe['goal_labels'])
    train_demos = set(map(tuple,train.split_info['demonstrations']))
    val_demos = set(map(tuple,validation.split_info['demonstrations']))
    assert train_demos and val_demos and train_demos.isdisjoint(val_demos)
    assert train_demos|val_demos == seen
    train.close(); validation.close(); dataset.close()

    torch.manual_seed(42)
    config = replace(load_config(PACKAGE/'configs/model.yaml'),text_device='cpu')
    model = build_policy(config,backbones=FrozenBackbones(Vision(),Language()))
    batch = collate_part2action(samples)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    metrics, gradients = train_step(model,batch,optimizer,recipe['losses'],collect_gradients=True)
    assert gradients['blocks.goal_head.position.weight'] > 0
    model.eval()
    with torch.inference_mode():
        output = model(observation_from_batch(batch))
    actions = output.actions['action_chunk']
    assert actions.shape == (len(samples),config.action_horizon,7)
    assert torch.isfinite(actions).all()
    return {'status':'passed','scope':'Real HDF5 labels/observations; tiny frozen test backbones; production-width v2 blocks. No simulator or success-rate measurement.',
        'demonstrations':len(seen),'max_demos_per_file':max_demos_per_file,
        'phase_counts':dict(phases),'valid_goal_counts':dict(valid),'label_reasons':dict(reasons),
        'label_qualities':dict(qualities),'touch_proxy_distance_range_m':
            [min(touch_distances),max(touch_distances)] if touch_distances else None,
        'train_demonstrations':len(train_demos),'validation_demonstrations':len(val_demos),
        'demonstration_overlap':0,'examples':examples,'losses':metrics,
        'nonzero_goal_gradient_parameters':sum(v>0 for k,v in gradients.items() if 'goal_head' in k),
        'sampled_action_shape':list(actions.shape),'diffusion_steps':config.diffusion_steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-demos-per-file',type=int,default=32)
    args = parser.parse_args()
    result = check(args.max_demos_per_file)
    with args.out.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
