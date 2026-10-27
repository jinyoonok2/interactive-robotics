"""Evaluate released PartInstruct weights with its native runner and GT parts.

Policy architecture comes from the checkpoint, rather than our training configs.
Native sampling and oracle skill progression are retained; this is not the
fixed 149-scenario Part2Action protocol or the VLM/SAM pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['DP-S', 'DP3-S'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--framework-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--load-only', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    # Official trusted checkpoint uses dill and includes its training config.
    payload = torch.load(str(args.checkpoint), map_location='cpu', pickle_module=dill)
    saved = payload['cfg']
    base = args.framework_root / 'PartInstruct/baselines/evaluation/config'
    cfg = OmegaConf.load(base / f'{args.model}_evaluator.yaml')
    cfg.pop('defaults', None)
    cfg.pop('logging', None)
    cfg.pop('hydra', None)
    cfg.env = OmegaConf.load(base / 'env/env_config.yaml')
    from compat import resolve_legacy_targets
    cfg.policy = OmegaConf.create(resolve_legacy_targets(OmegaConf.to_container(saved.policy, resolve=True)))
    if args.model == 'DP-S':
        channels = payload['state_dicts']['model']['obs_encoder.key_model_map.agentview_rgb.conv1.weight'].shape[1]
        cfg.policy.obs_encoder.image_encoder.input_channels = int(channels)
        assert channels == 5, 'Public release encoder expects RGB plus repeated mask'
    cfg.shape_meta = OmegaConf.create(OmegaConf.to_container(saved.shape_meta, resolve=True))
    if args.model == 'DP-S' and 'agentview_mask' in cfg.shape_meta.obs:
        cfg.shape_meta.obs.agentview_part_mask = cfg.shape_meta.obs.pop('agentview_mask')
    cfg.n_obs_steps = int(saved.policy.n_obs_steps)
    cfg.n_action_steps = int(saved.policy.n_action_steps)
    cfg.horizon = int(saved.policy.horizon)
    cfg.training.use_ema = bool(saved.training.use_ema)
    cfg.data_root = str(args.data_root)
    cfg.env.data_root = str(args.data_root)
    cfg.env.device = 'cuda:0'
    cfg.meta_path = 'episodes_meta_test.json'
    cfg.env.meta_path = 'episodes_meta_test.json'
    cfg.output_dir = str(args.out_dir)
    cfg.ckpt_path = str(args.checkpoint)
    cfg.rollout_mode = 'specific_ckpt'
    cfg.split = 'test1'
    cfg.epoch = 0  # Output label only; actual training metadata stays in checkpoint.
    cfg.obj_classes = ['scissors'] if args.smoke else None
    cfg.task.env_runner._target_ = ('PartInstruct.baselines.evaluation.env_runner.'
        + ('dp_env_runner.DPEnvRunner' if args.model == 'DP-S' else 'dp3_env_runner.DP3EnvRunner'))
    cfg.task.env_runner.bullet_env = 'PartInstruct.PartGym.env.bullet_env'
    cfg.task.env_runner.n_envs = 1 if args.smoke else 16
    cfg.task.env_runner.n_vis = cfg.task.env_runner.n_envs
    cfg.task.env_runner.debug_output = str(args.out_dir / 'debug')
    if args.smoke:
        # One complete native scissors trial before the all-category pass.
        cfg.task_type = '1'
    OmegaConf.save(cfg, args.out_dir / 'evaluation_config.yaml', resolve=True)
    info = {'model': args.model, 'checkpoint': str(args.checkpoint),
            'checkpoint_policy_config': OmegaConf.to_container(saved.policy, resolve=True),
            'compatibility': 'Legacy class paths mapped; Mask-name alias and DP convolution input dimension recovered from weights; strict parameter loading.',
            'state_dict_keys': list(payload['state_dicts']),
            'use_ema': cfg.training.use_ema, 'split': 'test1',
            'protocol': 'native task-group sampling, GT parts and oracle skill progression',
            'scope': 'one scissors task-group smoke' if args.smoke else 'all test1 object categories; 16 sampled rollouts per category',
            'limits': ['Not the fixed-scenario 447-rollout Part2Action protocol.',
                       'Not an exact reproduction of every paper evaluation setting.',
                       'Uses privileged target-part observations and oracle skill progression.']}
    (args.out_dir / 'provenance.json').write_text(json.dumps(info, indent=2)+'\n')
    policy = hydra.utils.instantiate(cfg.policy)
    key = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][key], strict=True)
    del payload
    policy.eval()
    if args.load_only:
        print('STRICT_CHECKPOINT_LOAD_OK', args.model, flush=True)
        return
    policy.cuda()
    # Import only after config preparation. Use the native evaluator's sampling
    # and runner, without loading optimizer state or overriding the policy.
    from PartInstruct.baselines.evaluation.evaluator import Evaluator
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.cfg = cfg
    evaluator.model = policy
    evaluator.ema_model = policy if cfg.training.use_ema else None
    evaluator.epoch = 0
    evaluator.dataset_meta_path = str(args.data_root / 'episodes_meta_test.json')
    evaluator.dataset_meta = json.loads(Path(evaluator.dataset_meta_path).read_text())
    evaluator.n_envs = cfg.task.env_runner.n_envs
    evaluator.specified_obj_classes = cfg.obj_classes
    evaluator.specified_splits = cfg.split
    evaluator.specified_task_type = cfg.task_type
    evaluator.roll_out(cfg)
    rows=[]
    for path in sorted(args.out_dir.glob('media/**/rollout_logs.json')):
        data=json.loads(path.read_text())
        successes=[v for k,v in data.items() if '/success_' in k]
        rows.append({'path':str(path.relative_to(args.out_dir)), 'trials':len(successes),
                     'successes':sum(bool(v) for v in successes)})
    summary={'model':args.model,'groups':rows,'trials':sum(r['trials'] for r in rows),
             'successes':sum(r['successes'] for r in rows),'protocol':info['protocol']}
    assert summary['trials'] > 0, 'No evaluated trials found'
    (args.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
