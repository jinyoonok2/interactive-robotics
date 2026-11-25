"""Observation-only or explicitly GT-conditioned native PartGym evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import PoseTokenDiffusionConfig
from .policy import build_policy
from .checkpoint import load_checkpoint
from .dataset import target_pose_from_tensor
from .runtime import PoseTokenRolloutPolicy
from diagnostics.evaluation_protocol import execution_settings, DEFAULT_PROTOCOL_PATH, align_native_environment

REPO = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def build_population(data_root, seeds, oracle_run=None):
    from diagnostics.evaluation_scenarios import load_scenarios, rollout_plan
    if oracle_run is not None:
        plan = read(Path(oracle_run)/'plan.json')
        scenarios = plan['scenarios']
        if not set(seeds) <= {row['seed'] for row in plan['trials']}:
            raise ValueError('Requested seeds are not in the supplied oracle reference plan')
        rollout_plan(scenarios, seeds)  # validate unique, legal seeds
        trials = [(i, trial) for i, trial in enumerate(plan['trials']) if trial['seed'] in seeds]
    else:
        scenarios = load_scenarios(Path(data_root)/'episodes_meta_test.json', 'test1',
                                  ['bottle', 'mug', 'pliers', 'scissors'])
        rollout_plan(scenarios, seeds)
        trials = [(i, row) for i, row in enumerate(
            {'scenario_index': index, 'seed': seed,
             'needs_grasp_pose': bool(scenario['episode']['chain_params'] and
                 scenario['episode']['chain_params'][0]['skill_name'] == 'grasp_obj')}
            for index, scenario in enumerate(scenarios) for seed in seeds)]
    return scenarios, trials


def summarize(records):
    complete = [row for row in records if row['status'] == 'completed']
    grasp = [row for row in complete if row['needs_grasp_pose']]
    return {'planned': len(records), 'completed': len(complete), 'unknown': len(records)-len(complete),
        'task_successes': sum(bool(row['success']) for row in complete), 'grasp_trials': len(grasp),
        'grasp_acquired': sum(bool(row['grasp']['grasp_acquired']) for row in grasp),
        'initial_phase_end_grasp': sum(bool(row['grasp']['initial_phase_last_grasp']) for row in grasp),
        'limits': ['Missing references and runtime errors are unknown, not physical policy failures.',
                   'Native grasp predicate permits either-finger contact; sustained retention is not established.',
                   'Combine shards and compare identical completed scenario/seed pairs before reporting improvements.']}


def main(*, architecture='pose_token_diffusion_v1'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/p/part2action/data/PartInstruct'))
    parser.add_argument('--oracle-run-dir', type=Path, help='Exact-reset GT reference catalog; required for provided mode')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, help='Explicit debugging subset only')
    parser.add_argument('--evaluation-config',type=Path,default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument('--max-steps',type=int,default=None)
    parser.add_argument('--execute-steps',type=int,default=None)
    args = parser.parse_args()
    settings = execution_settings('test1',config_path=args.evaluation_config,
        max_steps=args.max_steps,execute_steps=args.execute_steps)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards or (args.limit is not None and args.limit < 1):
        parser.error('Invalid shard or debugging limit')
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation for native visual evaluation')
    # Simulation-specific imports stay lazy; training and CLI help do not need OmegaConf.
    from omegaconf import OmegaConf
    import rollout_partgym as rollout
    from diagnostics.execution_trace import physical_state
    from generate_logged_references import native_classes
    payload = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    if payload.get('architecture') != architecture:
        raise ValueError('Checkpoint architecture differs from the selected evaluation entry point')
    if architecture == 'pose_token_diffusion_v2':
        from models.pose_token_diffusion_v2.config import PoseTokenDiffusionV2Config as Config
        from models.pose_token_diffusion_v2.policy import build_policy as builder
        from models.pose_token_diffusion_v2.checkpoint import load_checkpoint as restore
        from models.pose_token_diffusion_v2.runtime import PoseTokenV2RolloutPolicy as Adapter
    else:
        Config,builder,restore,Adapter = PoseTokenDiffusionConfig,build_policy,load_checkpoint,PoseTokenRolloutPolicy
    config = Config.from_mapping(payload['config'])
    if config.pose_source == 'provided' and args.oracle_run_dir is None:
        raise ValueError('Provided-pose evaluation requires an exact-reset oracle reference catalog')
    model = builder(config)
    restore(model, args.ckpt)
    model.cuda().eval()
    env_class = rollout._load_partgym_env(REPO/'PartInstruct')
    _, planner_class = native_classes()
    sim = OmegaConf.load(REPO/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    native = OmegaConf.load(REPO/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml')
    align_native_environment(sim,native)
    sim.data_root, sim.device = str(args.data_root), 'cpu'
    full = OmegaConf.load(REPO/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    rollout.configure_partgym_agentview_pcd(sim, full)
    scenarios, trials = build_population(args.data_root, args.seeds, args.oracle_run_dir)
    if args.limit is not None:
        trials = trials[:args.limit]
    trials = [(index, trial) for index, trial in trials if index % args.shards == args.shard_index]
    if args.out_dir.exists():
        raise FileExistsError('Choose a fresh per-shard evaluation output directory')
    args.out_dir.mkdir(parents=True)
    digest = hashlib.sha256(args.ckpt.read_bytes()).hexdigest()
    dump(args.out_dir/'population.json', {'trials': trials, 'config': config.to_dict(),
         'checkpoint_sha256': digest, 'debug_subset': args.limit is not None,
         'execution_settings':settings})
    records = []
    for index, trial in trials:
        scenario = scenarios[trial['scenario_index']]
        ref_path = args.oracle_run_dir/'oracle_references'/f'{index:03d}.json' if args.oracle_run_dir else None
        reference = read(ref_path) if ref_path and ref_path.is_file() else None
        available = bool(reference and reference['status'] == 'completed')
        path = args.out_dir/f'{index:03d}.json'
        if config.pose_source == 'provided' and trial['needs_grasp_pose'] and not available:
            row = {'status': 'blocked_missing_oracle', 'trial_index': index, **trial}
            dump(path, row)
            records.append(row)
            continue
        tracker = {'phase': 0, 'grasp_acquired': False, 'initial_phase_last_grasp': False, 'grasp_steps': []}
        part = scenario['episode']['chain_params'][0]['params'].get('part_grasp', '')

        class MonitoredEnvironment(env_class):
            def reset(self, *values, **kwargs):
                obs = super().reset(*values, **kwargs)
                if available:
                    state = physical_state(self)
                    for name in ('tcp_xyzw_xyz', 'robot_q', 'object_xyz', 'object_xyzw'):
                        np.testing.assert_allclose(state[name], reference['initial_state'][name], atol=1e-6, rtol=0)
                return obs
            def step(self, action, *values, **kwargs):
                result = super().step(action, *values, **kwargs)
                if trial['needs_grasp_pose'] and tracker['phase'] == 0:
                    before = self.world.sim_time
                    grasp = bool(planner_class.predicate_grasping(self, part_name=part)) and float(action[6]) <= .02
                    if self.world.sim_time != before:
                        raise ValueError('Read-only grasp scoring advanced physics')
                    tracker['grasp_acquired'] |= grasp
                    tracker['initial_phase_last_grasp'] = grasp
                    tracker['grasp_steps'].append(grasp)
                return result

        goal = None
        if config.pose_source == 'provided':
            pose = reference['goal_pose'] if available else [0., 0., 0., 1., 0., 0., 0.]
            goal = target_pose_from_tensor(torch.tensor([pose], device='cuda'), torch.tensor([available], device='cuda'))
        adapter = Adapter(model) if architecture == 'pose_token_diffusion_v2' else Adapter(model,provided_pose=goal)

        class MonitoredPolicy(torch.nn.Module):
            use_action_deltas = False
            def forward(self, rgb, instructions, **kwargs):
                out = adapter(rgb, instructions, **kwargs)
                tracker['phase'] = int(out['predicted_phase'][0])
                return out

        try:
            result = rollout.rollout_one(env_cls=MonitoredEnvironment, model=MonitoredPolicy(),
                model_cfg={'data': {'n_obs_steps': config.n_obs_steps}}, partgym_cfg=sim,
                config_path=Path(__file__).with_name('configs')/'model.yaml',
                obj_class=scenario['obj_class'], task_type=scenario['task_type'], split='test1',
                trial_index=index, episode_seed=trial['seed'], device=torch.device('cuda'),
                max_steps=settings['max_steps'], execute_steps=settings['execute_steps'], record=False, gif_fps=10., gif_width=400,
                gif_every_n=2, diagnostics=False, predicate_diagnostics=False, contact_diagnostics=False,
                diagnostic_every=20, termination_mode='legacy', min_phase_cycles=2,
                out_dir=args.out_dir, selected_episode=scenario['episode'])
            row = {'status': 'completed', 'trial_index': index, **trial, 'grasp': tracker,
                   'checkpoint_sha256': digest, 'pose_source': config.pose_source, **result}
        except Exception as error:
            row = {'status': 'runtime_error', 'trial_index': index, **trial, 'error': repr(error)}
            records.append(row)
            dump(path, row)
            dump(args.out_dir/'summary.json', summarize(records))
            raise
        records.append(row)
        dump(path, row)
    dump(args.out_dir/'summary.json', summarize(records))


if __name__ == '__main__':
    main()
