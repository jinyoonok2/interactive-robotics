"""Opt-in real HDF5 integration using tiny frozen encoders, not a success evaluation."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import torch
import yaml

from models.pose_token_diffusion.backbones import FrozenBackbones
from models.pose_token_diffusion.tests.test_model import Vision, Language
from ..config import load_config
from ..dataset import build_dataset, collate, observation_from_batch, ProgressDataset
from ..skills import annotated_skill
from models.pose_token_diffusion_v2.targets import segments
from ..policy import build_policy
from ..training import train_step, validate

PACKAGE = Path(__file__).resolve().parents[1]


def check(manifest, out_dir):
    recipe = yaml.safe_load((PACKAGE/'configs/pilot.yaml').read_text())
    data = {**recipe['data'], 'demo_manifest': str(manifest)}
    train = build_dataset(data, recipe['goal_labels'])
    validation = build_dataset({**data, 'split': 'validation'}, recipe['goal_labels'])
    train_keys = set(map(tuple, train.split_info['demonstrations']))
    val_keys = set(map(tuple, validation.split_info['demonstrations']))
    assert train_keys and val_keys and train_keys.isdisjoint(val_keys)
    seen, counts, examples, selected = set(), Counter(), [], set()
    for i, spec in enumerate(train._samples):
        key = spec.file_path, spec.demo_key
        if key in seen:
            continue
        seen.add(key)
        # Audit every selected semantic annotation without loading every full
        # touch cloud; geometric extraction is exercised on the selected batch.
        demo = train._open(spec.file_path)['data'][spec.demo_key]
        phases = segments(demo['skill_instructions'][:])
        for number, phase in enumerate(phases):
            annotated_skill(phase['instruction'], previous=[p['instruction'] for p in phases[:number]])
            counts[phase['kind']] += 1
            if phase['kind'] not in selected:
                item = train[i+phase['start']]
                assert item['skill_spec'].kind == phase['kind']
                examples.append(item)
                selected.add(phase['kind'])
    assert len(selected) == 5
    progress = ProgressDataset(manifest, split='train', n_obs_steps=2, **recipe['progress'])
    progress_ids = []
    for label in (0, 1):
        for kind in sorted(selected):
            progress_ids.append(next(i for i, r in enumerate(progress._samples) if r['skill'] == kind and r['target'] == label))
    auxiliary = collate([progress[i] for i in progress_ids])
    batch = collate(examples)
    config = replace(load_config(PACKAGE/'configs/model.yaml'), text_device='cpu', diffusion_steps=4)
    torch.manual_seed(42)
    model = build_policy(config, backbones=FrozenBackbones(Vision(), Language()))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses, gradients = train_step(model, batch, optimizer, recipe['losses'], progress_batch=auxiliary, collect_gradients=True)
    assert gradients['blocks.goal_head.position.weight'] > 0
    assert gradients['blocks.progress_monitor.net.0.weight'] > 0
    model.eval()
    with torch.inference_mode():
        action = model(observation_from_batch(batch)).actions['action_chunk']
    assert torch.isfinite(action).all()
    report = validate(model, [batch], [auxiliary], recipe['losses'], out_dir/'component_check', epoch=0, device=torch.device('cpu'))
    result = {'status': 'passed', 'scope': 'Real HDF5 annotation/interface audit and one CPU update with tiny frozen test encoders. Production-width trainable blocks, four diffusion steps for sampling. Not pretrained-backbone or simulator performance.',
        'train_demos': len(train_keys), 'validation_demos': len(val_keys), 'demo_overlap': 0,
        'train_frames': len(train), 'validation_frames': len(validation), 'phase_counts': dict(counts),
        'sampled_action_shape': list(action.shape), 'losses': losses,
        'nonzero_goal_gradient_parameters': sum(v > 0 for k, v in gradients.items() if k.startswith('blocks.goal_head.')),
        'progress_queries': len(progress_ids), 'report': 'component_check/README.md'}
    for dataset in (train, validation, progress):
        dataset.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    result = check(args.manifest, args.out_dir)
    (args.out_dir/'check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
