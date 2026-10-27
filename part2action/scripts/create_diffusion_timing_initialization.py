"""Create shared random policy weights for a from-scratch timing comparison."""
import argparse
import copy
from pathlib import Path

import torch

from _common import ROOT, load_yaml, resolve_paths, set_seed
from train import build_model, policy_state_digest, trainable_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = resolve_paths(load_yaml(args.config), ROOT)
    if cfg['model'].get('action_head_type') != 'diffusion' or cfg['model'].get('contact_conditioning_enabled'):
        raise ValueError('Expected contact-disabled diffusion configuration')
    seed = int(cfg['train']['seed'])
    set_seed(seed)
    torch.set_num_threads(2)
    cpu_config = copy.deepcopy(cfg)
    cpu_config['model']['text_device'] = 'cpu'
    cpu_config['train']['device'] = 'cpu'
    model = build_model(cpu_config)
    state = trainable_state_dict(model)
    fingerprint = policy_state_digest(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': state, 'cfg': cfg, 'epoch': -1, 'training_steps': 0,
                'initialization_kind': 'fresh_random_policy', 'seed': seed,
                'policy_state_sha256': fingerprint, 'frozen_backbones_excluded': True}, args.output)
    print('Created untrained policy initialization:', args.output, flush=True)
    print('Policy state SHA256:', fingerprint, flush=True)


if __name__ == '__main__':
    main()
