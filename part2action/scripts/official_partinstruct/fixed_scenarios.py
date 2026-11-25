"""Scenario/budget-matched released-policy references, with privileged inputs.

Retains native observation preprocessing and sequential-skill success. This is
not an equal-information or common-success-definition architecture comparison.
"""
import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from diagnostics.evaluation_scenarios import load_scenarios, pin_scenario, rollout_plan, success_statistics
from diagnostics.evaluation_protocol import execution_settings


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def validate_manifest(path, metadata, indices=None, seeds=None):
    manifest = json.loads(path.read_text())
    if manifest['split'] != 'test1':
        raise ValueError('Expected the existing test1 scenario manifest')
    canonical = load_scenarios(metadata, 'test1', ['bottle', 'mug', 'pliers', 'scissors'])
    if manifest['scenarios'] != canonical or len(canonical) != 149:
        raise ValueError('Manifest differs from the canonical 149-scenario selection')
    if manifest['evaluation_seeds'] != [42, 43, 44]:
        raise ValueError('Expected the original evaluation seeds 42, 43, 44')
    chosen = list(range(149)) if indices is None else indices
    if not chosen or len(set(chosen)) != len(chosen) or any(i < 0 or i >= 149 for i in chosen):
        raise ValueError('Scenario indices must be unique and within the source manifest')
    trial_seeds = manifest['evaluation_seeds'] if seeds is None else seeds
    if any(s not in manifest['evaluation_seeds'] for s in trial_seeds):
        raise ValueError('Preflight seeds must be drawn from the original manifest')
    cases = [canonical[i] for i in chosen]
    return manifest, rollout_plan(cases, trial_seeds), chosen, trial_seeds


def action_prefix(predicted, execute_steps, remaining_steps):
    actions = np.asarray(predicted, dtype=float)
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7 or not np.isfinite(actions).all():
        raise ValueError('Expected finite single-policy actions shaped (1, T, 7)')
    if execute_steps < 1 or remaining_steps < 1 or actions.shape[1] < execute_steps:
        raise ValueError('Invalid execution budget or action chunk')
    executed = actions[0, :min(execute_steps, remaining_steps)].copy()
    executed[:, 6] = np.clip(executed[:, 6], 0., .04)
    return executed


def last_info(info):
    # Native MultiStepWrapper returns arrays of the last n_obs_steps of info.
    result = {}
    for key in ['Success', 'Completion Rate', 'Steps', 'ep_id', 'Object id']:
        value = info.get(key)
        if value is not None:
            result[key] = np.asarray(value).reshape(-1)[-1].item()
    return result


def batch_observations(observations, observation_space):
    """Match native AsyncVectorEnv shared-memory shape/dtype conversion.

    BulletEnv tokens carry a singleton tokenizer batch axis and part masks
    omit the declared channel axis. The native vector buffer flattens and
    reshapes both to their declared spaces; direct stepping must do the same.
    """
    batch={}
    for key,space in observation_space.spaces.items():
        value=np.asarray(observations[key],dtype=space.dtype)
        if value.size != int(np.prod(space.shape)):
            raise ValueError(f'Observation {key} has shape {value.shape}, expected space {space.shape}')
        batch[key]=value.reshape((1,)+tuple(space.shape))
    return batch


def write_summary(out, rows, provenance, expected):
    groups = {}
    for row in rows:
        group = groups.setdefault(row['obj_class'], {'trials':0, 'successes':0})
        group['trials'] += 1; group['successes'] += int(row['success'])
    summary = {
        'status':'completed' if len(rows) == expected else 'in_progress',
        'model':provenance['model'], 'protocol':provenance['protocol'],
        'trials':len(rows), 'planned_trials':expected,
        'successes':sum(r['success'] for r in rows),
        'success_rate':sum(r['success'] for r in rows)/len(rows) if rows else 0.,
        'success_definition':provenance['success_definition'],
        'execution_settings':provenance.get('execution_settings'),
        'per_object':groups, **success_statistics(rows), 'limits':provenance['limits'],
    }
    atomic_json(out/'summary.json', summary)
    with (out/'per_trial.csv').open('w', newline='') as f:
        fields = ['obj_class','task_type','requested_episode_id','seed','success','completion_rate','num_actions','native_steps','policy_forward_mean_ms']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    # Use the shared table renderer for copyable presentation images.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'docs/slide_history/scripts'))
    from export_slide_tables import render
    media = out/'media'/provenance['media_date']; media.mkdir(parents=True, exist_ok=True)
    table = [['Privileged released-policy reference', 'Completed', 'Task successes'],
             [provenance['model'], f'{len(rows)} / {expected}', str(summary['successes'])]]
    table += [[name, str(group['trials']), str(group['successes'])] for name, group in sorted(groups.items())]
    render(table, [.56,.22,.22], media/'results_table.png')
    image = f'media/{provenance["media_date"]}/results_table.png'
    (out/'index.html').write_text(
        f'<!doctype html><title>Fixed-scenario released-policy reference</title>'
        f'<h1>{provenance["model"]}: privileged reference</h1><img width="1000" src="{image}">'
        f'<p>Use this image: <a href="{image}">results_table.png</a></p>'
        '<p>Scenario and budget matching does not equalize information access or success definitions.</p>'
        '<p><a href="per_trial.csv">Per-trial CSV</a> | <a href="summary.json">Summary</a> | '
        '<a href="provenance.json">Protocol and limitations</a></p>')
    return summary


def rollout(policy, cfg, preprocess, env_cls, scenario, trial_index, seed, args, directory):
    import torch
    from PartInstruct.baselines.utils.multistep_wrapper import MultiStepWrapper
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    raw = env_cls(config=cfg.env, shape_meta=cfg.shape_meta, gui=False,
                  record=False, evaluation=True, skill_mode=True,
                  obj_class=scenario['obj_class'], split='test1', task_type=scenario['task_type'], track_samples=False)
    env = MultiStepWrapper(raw, n_obs_steps=cfg.n_obs_steps, n_action_steps=args.execute_steps,
                           max_episode_steps=args.max_steps)
    try:
        pin_scenario(raw, scenario['episode'], seed)
        obs = env.reset()
        if str(raw.ep_id) != str(scenario['episode']['ep_id']):
            raise RuntimeError('Reset selected a different scenario')
        xyz, quat = raw.world.p.getBasePositionAndOrientation(raw.obj.uid)
        atomic_json(directory/'initial_state.json', {
            'object_xyz':list(xyz), 'object_quat_xyzw':list(quat),
            'robot_joints':np.asarray(raw.robot.get_joint_states()).tolist(),
            'tcp_pose':raw.robot.get_tcp_pose().to_list().tolist(),
            'object_id':str(raw.obj_id), 'episode_id':str(raw.ep_id), 'seed':seed,
        })
        policy.reset(); actions = []; raw_actions = []; times = []; phases = []; info = {}; done = False
        while len(actions) < args.max_steps and not done:
            batch = batch_observations(obs, env.observation_space)
            with torch.no_grad():
                model_obs = preprocess.get_obs(batch, None, policy.device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                predicted = policy.predict_action(model_obs)['action'].detach().cpu().numpy()
                torch.cuda.synchronize()
                times.append((time.perf_counter()-started)*1000.)
            prefix = action_prefix(predicted, args.execute_steps, args.max_steps-len(actions))
            raw_actions.extend(predicted[0, :len(prefix)].astype(float).tolist())
            phases.append(int(raw.cur_skill_idx))
            before = len(env.reward)
            obs, _, done, info = env.step(prefix)
            count = len(env.reward)-before
            actions.extend(prefix[:count].tolist())
        final = last_info(info)
        if str(final.get('ep_id')) != str(scenario['episode']['ep_id']):
            raise RuntimeError('Final episode ID does not match requested scenario')
        row = {
            'obj_class':scenario['obj_class'], 'task_type':scenario['task_type'],
            'requested_episode_id':str(raw.ep_id), 'trial_index':trial_index, 'seed':seed,
            'success':bool(final['Success']),  # Never treat wrapper time limit as success.
            'completion_rate':float(final['Completion Rate']), 'num_actions':len(actions),
            'native_steps':int(final['Steps']), 'policy_forward_mean_ms':float(np.mean(times)),
            'policy_forward_latency_ms':times, 'oracle_phase_trace':phases,
            'actions':actions, 'raw_actions':raw_actions, 'final_info':final,
        }
        atomic_json(directory/'result.json', row)
        return row
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['DP-S','DP3-S'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--framework-root', type=Path, required=True)
    parser.add_argument('--scenarios-manifest', type=Path, required=True)
    parser.add_argument('--scenario-indices', nargs='+', type=int)
    parser.add_argument('--eval-seeds', nargs='+', type=int)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--execute-steps', type=int, default=None)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    settings = execution_settings('test1',max_steps=args.max_steps,execute_steps=args.execute_steps)
    args.max_steps,args.execute_steps = settings['max_steps'],settings['execute_steps']
    if args.max_steps < 1 or args.execute_steps < 1:
        parser.error('Budgets must be positive')
    manifest, plan, indices, seeds = validate_manifest(args.scenarios_manifest, args.data_root/'episodes_meta_test.json', args.scenario_indices, args.eval_seeds)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    selected = copy.deepcopy(manifest)
    selected.update({'source_manifest':str(args.scenarios_manifest),
                     'source_manifest_sha256':hashlib.sha256(args.scenarios_manifest.read_bytes()).hexdigest(),
                     'source_scenario_indices':indices, 'evaluation_seeds':seeds,
                     'trials_per_scenario':len(seeds), 'expected_rollouts_per_model':len(plan),
                     'scenarios':[manifest['scenarios'][i] for i in indices]})
    atomic_json(args.out_dir/'evaluation_scenarios.json', selected)
    if args.prepare_only:
        print(json.dumps({'planned_trials':len(plan), 'indices':indices, 'seeds':seeds})); return
    import hydra
    import torch
    from omegaconf import OmegaConf
    from evaluate import load_released_policy
    cfg, policy, provenance = load_released_policy(SimpleNamespace(**vars(args), smoke=False))
    # Retain released camera/robot/physical thresholds and observation shapes.
    original_env = OmegaConf.to_container(cfg.env, resolve=True)
    cfg.env.data_root = str(args.data_root); cfg.env.device = 'cuda:0'
    cfg.env.shape_meta = cfg.shape_meta
    cfg.task.env_runner.env_config = cfg.env
    cfg.task.env_runner.max_steps = args.max_steps
    cfg.task.env_runner.n_action_steps = args.execute_steps
    provenance.update({
        'media_date':datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d'),
        'protocol':f"fixed_149_{settings['max_steps']}_{settings['execute_steps']}_privileged_native_skill_success",
        'scope':f'{len(plan)} explicit trials', 'evaluation_seeds':seeds,
        'source_scenario_manifest':str(args.scenarios_manifest),
        'max_steps':args.max_steps, 'execute_steps':args.execute_steps,
        'execution_settings':settings,
        'success_definition':'Native sequential skill completion (skill_mode=True), not the Part2Action full-task predicate scorer',
        'env_config_changes':{k:{'native':v,'fixed_scenario':OmegaConf.to_container(cfg.env,resolve=True).get(k)} for k,v in original_env.items() if k not in cfg.env or v != OmegaConf.to_container(cfg.env,resolve=True).get(k)},
        'language_encoder_eval_mode':True,
        'limits':[
            'Action budgets, execution cadence and physical thresholds use the released native evaluator settings; input privilege and success definitions still differ from Part2Action.',
            'Released policies receive GT target-part observations and oracle skill progression; our standard policies predict grounding and phase progression.',
            'Native sequential-skill success differs from our full-task predicate scoring; do not interpret score differences as architecture gains.',
            'Checkpoint modalities, native observation preprocessing, denoising settings and all learned weights are retained; language preprocessing is in eval mode.',
            'Identical seed numbers do not guarantee identical stochastic observation streams across different input modalities.',
        ],
    })
    policy.cuda()
    cls = hydra.utils.get_class(cfg.task.env_runner._target_)
    preprocess = cls.__new__(cls)
    preprocess.lang_encoder = hydra.utils.instantiate(cfg.task.env_runner.lang_encoder).eval()
    preprocess.past_action = False
    from PartInstruct.PartGym.env.bullet_env import BulletEnv
    OmegaConf.save(cfg, args.out_dir/'evaluation_config.yaml', resolve=True)
    provenance['gpu'] = torch.cuda.get_device_name(0)
    atomic_json(args.out_dir/'provenance.json', provenance)
    rows = []; write_summary(args.out_dir, rows, provenance, len(plan))
    try:
        for index, (scenario, trial_index, seed) in enumerate(plan):
            directory = args.out_dir/'trials'/f'{index:03d}_{scenario["obj_class"]}_{scenario["episode"]["ep_id"]}_seed{seed}'
            directory.mkdir(parents=True)
            row = rollout(policy, cfg, preprocess, BulletEnv, scenario, trial_index, seed, args, directory)
            rows.append(row); summary = write_summary(args.out_dir, rows, provenance, len(plan))
            print(f'[fixed] {index+1}/{len(plan)} {args.model} {row["obj_class"]} ep={row["requested_episode_id"]} seed={seed} success={row["success"]}', flush=True)
    except Exception as exc:
        summary = write_summary(args.out_dir, rows, provenance, len(plan))
        summary.update({'status':'failed','error':repr(exc)})
        atomic_json(args.out_dir/'summary.json', summary)
        raise
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
