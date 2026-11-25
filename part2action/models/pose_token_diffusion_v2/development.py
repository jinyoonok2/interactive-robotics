"""Matched, video-recorded development rollouts of the two v2 pilot epochs.

Uses scenarios belonging to held-out training-release demonstrations. The lift
group keeps its original full instruction/chain and scores the grasp/lift prefix
separately. This is not the reserved Test1 benchmark or an expert-phase rollout.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PART = ROOT/'part2action'
LAUNCHER = PART/'slurm/evaluate_pose_token_v2_development.sbatch'
ARMS = ('epoch_001', 'epoch_002')
FAMILIES = ('grasp', 'touch', 'grasp_lift_sequence')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scenario_family(episode):
    chain = episode['chain_params']
    kinds = [entry['skill_name'] for entry in chain]
    if kinds == ['grasp_obj']:
        return 'grasp'
    if kinds == ['touch_obj']:
        return 'touch'
    if (len(chain) >= 2 and kinds[:2] == ['grasp_obj', 'move_gripper']
            and chain[1]['params'].get('dir_move') == 'top'
            and chain[1]['params'].get('grasping') is True):
        return 'grasp_lift_sequence'
    return None


def select_scenarios(candidates, objects, *, seed=42):
    """One per object/family, before observing either checkpoint's outcomes."""
    selected = []
    for obj in objects:
        for family in FAMILIES:
            pool = [row for row in candidates if row['obj_class'] == obj
                    and scenario_family(row['episode']) == family]
            if not pool:
                raise ValueError(f'No eligible held-out scenario for {obj}/{family}')
            def rank(row):
                key = f"{seed}/{obj}/{family}/{row['episode']['ep_id']}/{row['source_demo']}"
                return hashlib.sha256(key.encode()).hexdigest()
            selected.append({**copy.deepcopy(min(pool, key=rank)), 'family': family})
    return selected


def prepare(args):
    import h5py
    import torch
    from diagnostics.evaluation_protocol import DEFAULT_PROTOCOL_PATH, execution_settings
    from .training import source_paths
    from register_result import register

    training = args.training_study.resolve(strict=True)
    completion = read(training/'training/completion.json')
    if completion.get('status') != 'completed' or completion.get('epochs') != 2 or completion.get('smoke'):
        raise ValueError('Expected the completed two-epoch pilot')
    initialization = read(training/'training/initialization.json')
    if initialization.get('checkpoint_loaded') is not False:
        raise ValueError('Expected fresh-policy initialization')
    original_plan = read(training/'plan.json')
    for name, expected in original_plan['source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Pilot implementation has changed: '+name)
    manifest_path = training/'demonstrations.json'
    manifest = read(manifest_path)
    data_root = Path(manifest['demonstrations'][0]['file']).parents[1]
    metadata_path = data_root/'episodes_meta_train.json'
    metadata = read(metadata_path)
    candidates, train_episodes, exclusions = [], set(), []
    for item in manifest['demonstrations']:
        with h5py.File(item['file'], 'r') as handle:
            value = handle['data'][item['demo']]['metadata/ep_id'][()]
        episode_id = value.decode() if isinstance(value, bytes) else str(value)
        if item['split'] == 'train':
            train_episodes.add((item['object'], episode_id))
            continue
        matches = [(task, episode) for task, episodes in metadata[item['object']]['train'].items()
                   for episode in episodes if str(episode['ep_id']) == episode_id]
        if len(matches) != 1:
            raise ValueError('Ambiguous or missing training metadata: '+episode_id)
        task, episode = matches[0]
        # Some released demonstrations end early: do not pretend a two-phase
        # recording is a complete three/four-phase source task.
        if item['skills'] != [entry['skill_name'] for entry in episode['chain_params']]:
            exclusions.append({'object': item['object'], 'demo': item['demo'],
                               'reason': 'Recorded skill chain differs from complete metadata chain'})
            continue
        candidates.append({'obj_class': item['object'], 'task_type': task,
            'episode': episode, 'source_demo': item['demo'], 'source_hdf5': item['file']})
    candidates = [row for row in candidates
                  if (row['obj_class'], str(row['episode']['ep_id'])) not in train_episodes]
    objects = ('bottle', 'mug', 'pliers', 'scissors')
    scenarios = select_scenarios(candidates, objects, seed=args.seed)
    if args.reference_development:
        reference = read(args.reference_development/'plan.json')
        if scenarios != reference['scenarios'] or args.seed != reference['seed']:
            raise ValueError('Corrected pilot must reuse exactly the original development scenarios and seed')
    checkpoints = {}
    configs = []
    for arm in ARMS:
        path = training/'training'/f'{arm}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('architecture') != 'pose_token_diffusion_v2' or payload['config']['pose_source'] != 'predicted':
            raise ValueError('Expected a predicted-goal v2 checkpoint')
        configs.append(payload['config'])
        checkpoints[arm] = {'path': str(path), 'sha256': sha(path)}
        del payload
    if configs[0] != configs[1]:
        raise ValueError('Checkpoint configurations differ')
    paths = source_paths()+[Path(__file__), LAUNCHER,
        PART/'scripts/run_pose_token_v2_development.py', PART/'scripts/rollout_partgym.py',
        PART/'diagnostics/evaluation_scenarios.py', PART/'diagnostics/execution_trace.py',
        PART/'diagnostics/phase_controller.py', PART/'diagnostics/partgym_reach.py',
        PART/'diagnostics/partgym_skill_oracle.py', PART/'scripts/_common.py',
        ROOT/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml']
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym').rglob('*.py'))
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym/config').glob('*.yaml'))
    sources = {str(p.resolve()): sha(p) for p in sorted(set(paths))}
    settings = execution_settings('test1')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Development simulator diagnostics on held-out training-release scenarios; not Test1.',
        'training_study': str(training), 'data_root': str(data_root),
        'reference_development':str(args.reference_development) if args.reference_development else None,
        'initialization': initialization, 'checkpoints': checkpoints,
        'scenarios': scenarios, 'seed': args.seed,
        'trials': [{'scenario_index': i, 'seed': args.seed} for i in range(len(scenarios))],
        'execution_settings': settings, 'environment_split': 'train',
        'model_inputs': ['RGB history', 'observed scene XYZ', 'full task instruction', 'measured TCP/opening', 'cached learned plan/phase'],
        'progression': 'Existing learned termination at 0.5; minimum two prediction cycles. No oracle advancement.',
        'scoring': {'task_success': 'Native info.Success for the entire original task.',
            'grasp_acquired': 'Named-part native either-finger predicate, with commanded finger opening <= 0.02 m.',
            'grasp_lift_prefix': 'Earlier named-part grasp, current grasp/closure, and object within 5 cm of initial object XYZ + native translate_distance along +Z.',
            'touch': 'Native task success on touch-only tasks; no additional state-mutating touch predicate.',
            'limits': 'Grasp predicate is not sustained retention. Lift-containing tasks keep all later skills/instructions. Native reset is not replay of recorded HDF5 states.'},
        'selection_exclusions': exclusions,
        'source_sha256': sources,
        'metadata_sha256': {str(p): sha(p) for p in (metadata_path, manifest_path, training/'plan.json', DEFAULT_PROTOCOL_PATH)},
        'data_files': original_plan['data_files']}
    if args.reference_development:
        paths=[args.reference_development/'plan.json']
        paths += [p for arm in ARMS for p in (args.reference_development/arm).glob('[0-9][0-9][0-9].json')]
        plan['metadata_sha256'].update({str(p):sha(p) for p in paths})
    write(args.out_dir/'plan.json', plan)
    snapshot = args.out_dir/'source_snapshot'; snapshot.mkdir()
    for index, name in enumerate(sorted(sources)):
        shutil.copy2(name, snapshot/f'{index:03d}_{Path(name).name}')
    register('diagnostics', args.out_dir)
    write_report(args.out_dir, plan, {})
    print(json.dumps({'study': str(args.out_dir), 'trials_per_checkpoint': len(scenarios),
                      'groups': dict(Counter(row['family'] for row in scenarios))}), flush=True)


def verify(plan):
    for group in ('source_sha256', 'metadata_sha256'):
        for path, expected in plan[group].items():
            if sha(path) != expected:
                raise ValueError('Pinned input changed: '+path)
    for checkpoint in plan['checkpoints'].values():
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Checkpoint changed')
    for name, expected in plan['data_files'].items():
        stat = Path(name).stat()
        if stat.st_size != expected['size'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError('Training data changed')


def submit(args, plan):
    receipt = args.out_dir/'submissions.json'
    if receipt.exists():
        raise FileExistsError('This study already has job submissions')
    common = ['sbatch', '--parsable', '--no-requeue', '--cpus-per-task=4', '--mem=32G',
        '--output='+str(args.out_dir/'%x_%A_%a.out'), '--error='+str(args.out_dir/'%x_%A_%a.err')]
    exports = '--export=ALL,P2A_V2_DEV_STUDY='+str(args.out_dir)
    jobs = []
    command = common+['--partition=gpu', '--gres=gpu:1', '--array=0-1%2', '--time=02:00:00',
        '--constraint=a40|a100_40gb|a100_80gb|h100_94gb|a4000|a4500|rtx_4000_ada',
        '--job-name=p2a-v2-dev', exports+',MODE=evaluate', str(LAUNCHER)]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
    jobs.append({'mode': 'evaluate', 'job_id': job, 'command': command})
    write(receipt, jobs)
    command = common+['--partition=cpu', '--time=00:10:00', '--dependency=afterany:'+job,
        '--kill-on-invalid-dep=yes', '--job-name=p2a-v2-dev-report', exports+',MODE=report', str(LAUNCHER)]
    report_job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(';')[0]
    jobs.append({'mode': 'report', 'job_id': report_job, 'command': command})
    write(receipt, jobs)
    print(json.dumps(jobs, indent=2), flush=True)


def native_grasp(env, part):
    """Same native contact test on already-refreshed part points; no world step."""
    points = env.parser.part_pcds[part]
    return any(env.world.check_obj_gripper_collision(env.robot.panda, finger, points, 0.003, 20.0)
               for finger in (env.robot.LEFTFINGERIDX, env.robot.RIGHTFINGERIDX))


def evaluate(args, plan):
    import gc
    import numpy as np
    import torch
    import imageio_ffmpeg
    from omegaconf import OmegaConf
    import rollout_partgym as rollout
    from diagnostics.evaluation_protocol import align_native_environment
    from diagnostics.execution_trace import physical_state
    from .config import PoseTokenDiffusionV2Config
    from .policy import build_policy
    from .checkpoint import load_checkpoint
    from .runtime import PoseTokenV2RolloutPolicy
    from .targets import SKILLS

    if not 0 <= args.index < len(ARMS) or not torch.cuda.is_available():
        raise ValueError('Use a valid checkpoint index and Slurm GPU allocation')
    arm = ARMS[args.index]
    folder = args.out_dir/arm
    folder.mkdir(exist_ok=False)
    checkpoint = plan['checkpoints'][arm]
    payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    config = PoseTokenDiffusionV2Config.from_mapping(payload['config'])
    del payload
    # Deterministic construction, then rollout_one reseeds before each reset.
    rollout.set_seed(plan['seed'])
    model = build_policy(config)
    load_checkpoint(model, checkpoint['path'])
    model.cuda().eval()
    env_class = rollout._load_partgym_env(ROOT/'PartInstruct')
    sim = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    native = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml')
    align_native_environment(sim, native)
    sim.data_root, sim.device, sim.meta_path = plan['data_root'], 'cpu', 'episodes_meta_train.json'
    full = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    rollout.configure_partgym_agentview_pcd(sim, full)
    settings = plan['execution_settings']
    records = []
    for index, trial in enumerate(plan['trials']):
        scenario = plan['scenarios'][trial['scenario_index']]
        tracker = {'initial_state': None, 'final_state': None, 'actions': [], 'goals': [],
                   'grasp_acquired': False, 'grasp_lift_prefix': False}
        first = scenario['episode']['chain_params'][0]
        part = first['params'].get('part_grasp')

        class MonitoredEnvironment(env_class):
            def reset(self, *values, **kwargs):
                obs = super().reset(*values, **kwargs)
                tracker['initial_state'] = physical_state(self)
                return obs

            def step(self, action, *values, **kwargs):
                result = super().step(action, *values, **kwargs)
                before_time = self.world.sim_time
                xyz = np.asarray(self.obj.get_pose().translation, dtype=float)
                grasp = bool(part and native_grasp(self, part) and float(action[6]) <= .02)
                previous_grasp = tracker['grasp_acquired']
                tracker['grasp_acquired'] |= grasp
                target = np.asarray(tracker['initial_state']['object_xyz'])+np.array([0., 0., float(self.config.translate_distance)])
                lifted = bool(scenario['family'] == 'grasp_lift_sequence' and previous_grasp
                              and grasp and np.linalg.norm(xyz-target) < .05)
                tracker['grasp_lift_prefix'] |= lifted
                state = physical_state(self)
                tracker['final_state'] = state
                tracker['actions'].append({'action_index': len(tracker['actions']),
                    'named_part_grasp': grasp if part else None,
                    'grasp_lift_prefix': lifted if scenario['family'] == 'grasp_lift_sequence' else None,
                    'object_xyz': xyz.tolist(), 'tcp_xyzw_xyz': state['tcp_xyzw_xyz'],
                    'measured_finger_m': state['finger_m'], 'native_task_success': bool(result[3].get('Success', False))})
                if self.world.sim_time != before_time:
                    raise RuntimeError('Diagnostic scoring advanced simulation')
                return result

        adapter = PoseTokenV2RolloutPolicy(model)

        class MonitoredPolicy(torch.nn.Module):
            use_action_deltas = False

            def forward(self, rgb, instructions, **kwargs):
                out = adapter(rgb, instructions, **kwargs)
                selected = adapter.last_output.selected_goal
                phase = int(out['predicted_phase'][0])
                tracker['goals'].append({'prediction_index': len(tracker['goals']),
                    'phase': phase, 'goal_xyz_world': selected.position_world[0, 0].detach().float().cpu().tolist(),
                    'goal_rotation_world': selected.rotation_world[0, 0].detach().float().cpu().tolist(),
                    'goal_opening_m': float(selected.opening[0, 0]),
                    'component_valid': selected.component_valid[0, 0].cpu().tolist(),
                    'predicted_skill': SKILLS[int(selected.skill_logits[0, 0].argmax())],
                    'termination_probability': float(out['termination_logits'][0].sigmoid())})
                return out

        print(json.dumps({'stage': 'rollout_start', 'checkpoint': arm, 'trial': index,
                          'object': scenario['obj_class'], 'family': scenario['family']}), flush=True)
        identity = {'trial_index': index, **trial, 'obj_class': scenario['obj_class'],
                    'family': scenario['family'], 'checkpoint_sha256': checkpoint['sha256']}
        try:
            result = rollout.rollout_one(env_cls=MonitoredEnvironment, model=MonitoredPolicy(),
                model_cfg={'data': {'n_obs_steps': config.n_obs_steps}}, partgym_cfg=sim,
                config_path=PART/'models/pose_token_diffusion_v2/configs/model.yaml',
                obj_class=scenario['obj_class'], task_type=scenario['task_type'], split='train',
                trial_index=index, episode_seed=trial['seed'], device=torch.device('cuda'),
                max_steps=settings['max_steps'], execute_steps=settings['execute_steps'],
                record=True, gif_fps=10., gif_width=400, gif_every_n=2,
                diagnostics=True, predicate_diagnostics=False, contact_diagnostics=False,
                diagnostic_every=20, termination_mode='legacy', min_phase_cycles=2,
                out_dir=folder, selected_episode=scenario['episode'])
            if not result['rollout_video'] or not Path(result['rollout_video']).is_file():
                raise RuntimeError('Expected recorded rollout video')
            video = Path(result['rollout_video'])
            browser_video = video.with_name('rollout.browser.mp4')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                '-i', str(video), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(browser_video)], check=True)
            browser_video.replace(video)
            row = {**result, **identity, 'status': 'completed', 'monitor': tracker}
            row['phase_advances'] = sum(bool(event['advanced']) for event in result['phase_events'])
        except Exception as error:
            row = {**identity, 'status': 'runtime_error', 'error': repr(error), 'monitor': tracker}
            write(folder/f'{index:03d}.json', row)
            raise
        records.append(row)
        write(folder/f'{index:03d}.json', row)
        print(json.dumps({'stage': 'rollout_end', 'checkpoint': arm, 'trial': index,
                          'success': row['success'], 'phase_advances': row['phase_advances'],
                          'grasp_acquired': tracker['grasp_acquired'],
                          'grasp_lift_prefix': tracker['grasp_lift_prefix']}), flush=True)
        del adapter
        gc.collect()
    write(folder/'completion.json', {'status': 'completed', 'trials': len(records),
        'checkpoint_sha256': checkpoint['sha256'], 'execution_settings': settings})


def paired_summary(plan, records):
    paired, mismatches = [], []
    for index, trial in enumerate(plan['trials']):
        rows = [records.get(arm, {}).get(index, {}) for arm in ARMS]
        for arm, row in zip(ARMS, rows):
            if not row:
                continue
            if (row.get('trial_index') != index or any(row.get(k) != v for k, v in trial.items())
                    or row.get('checkpoint_sha256') != plan['checkpoints'][arm]['sha256']):
                raise ValueError('Result identity differs from the planned trial')
            if row['status'] == 'completed':
                settings = plan['execution_settings']
                if (any(row.get(k) != settings[k] for k in ('max_steps', 'execute_steps'))
                        or not 0 < row['num_actions'] <= settings['max_steps']):
                    raise ValueError('Result differs from the execution protocol')
        if all(row.get('status') == 'completed' for row in rows):
            a, b = [row['monitor']['initial_state'] for row in rows]
            same = all(len(a[key]) == len(b[key]) and max(abs(x-y) for x, y in zip(a[key], b[key])) <= 1e-6
                       for key in ('tcp_xyzw_xyz', 'robot_q', 'object_xyz', 'object_xyzw'))
            if same:
                paired.append(index)
            else:
                mismatches.append(index)
    groups = {}
    for family in FAMILIES:
        indices = [i for i in paired if plan['scenarios'][plan['trials'][i]['scenario_index']]['family'] == family]
        groups[family] = {'paired_trials': len(indices), 'checkpoints': {}}
        for arm in ARMS:
            rows = [records[arm][i] for i in indices]
            groups[family]['checkpoints'][arm] = {
                'task_successes': sum(bool(row['success']) for row in rows),
                'grasp_acquired': sum(bool(row['monitor']['grasp_acquired']) for row in rows) if family != 'touch' else None,
                'grasp_lift_prefix': sum(bool(row['monitor']['grasp_lift_prefix']) for row in rows) if family == 'grasp_lift_sequence' else None,
                'trials_with_phase_advance': sum(row['phase_advances'] > 0 for row in rows)}
    return {'planned_per_checkpoint': len(plan['trials']), 'paired_trial_indices': paired,
        'reset_mismatch_indices': mismatches, 'by_family': groups,
        'coverage': {arm: dict(Counter(records.get(arm, {}).get(i, {}).get('status', 'not_recorded')
                                      for i in range(len(plan['trials'])))) for arm in ARMS}}


def write_report(directory, plan, records):
    summary = paired_summary(plan, records)
    summary['status'] = 'complete' if len(summary['paired_trial_indices']) == len(plan['trials']) else 'incomplete'
    summary['scope'] = plan['scope']
    write(directory/'summary.json', summary)
    lines = ['# V2 checkpoint simulator development evaluation', '', plan['scope'], '',
        'Epoch 1 and epoch 2 of the same fresh-policy run. Frozen pretrained DINOv2/Flan-T5; predicted pose goals.', '',
        f"Planned: {len(plan['trials'])} scenarios per checkpoint. Matched completed resets: {len(summary['paired_trial_indices'])}.",
        '250 executed actions maximum; eight actions per prediction; learned phase advancement. Videos and per-action/goal/phase logs are retained.', '',
        'The lift group keeps the original full lift-containing task, including later rotation. Its grasp/lift prefix is scored separately; full-task success requires the entire instruction.', '',
        '| Group | Paired trials | Epoch 1 task success | Epoch 2 task success |',
        '| --- | ---: | ---: | ---: |']
    for family, group in summary['by_family'].items():
        n = group['paired_trials']
        counts = [group['checkpoints'][arm]['task_successes'] for arm in ARMS]
        lines.append(f'| {family} | {n} | {counts[0]}/{n} | {counts[1]}/{n} |')
    lines += ['', '[Comparison and videos](index.html) · [Full counts and coverage](summary.json) · [Pinned plan](plan.json)', '',
        'This small development set and single seed do not establish benchmark improvement. Missing/error/reset-mismatched trials are coverage gaps. Grasp contact does not prove sustained retention.']
    if plan.get('reference_development') and summary['status']=='complete':
        reference=Path(plan['reference_development'])
        before_after={}
        lines += ['', '## Before / after completion correction', '',
            'Same epoch number, source demonstrations, action update budget, scenarios and reset seed. Completion-only training adds encoder/MLP computation.', '',
            '| Checkpoint | Group | Matched resets | Previous task successes | Corrected task successes | Previous phase advances (trials) | Corrected phase advances (trials) |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            for family in FAMILIES:
                pairs=[]
                for i in summary['paired_trial_indices']:
                    if plan['scenarios'][plan['trials'][i]['scenario_index']]['family']!=family:continue
                    old=read(reference/arm/f'{i:03d}.json');new=records[arm][i]
                    a,b=old['monitor']['initial_state'],new['monitor']['initial_state']
                    same=all(max(abs(x-y) for x,y in zip(a[k],b[k]))<=1e-6 for k in
                             ('tcp_xyzw_xyz','robot_q','object_xyz','object_xyzw'))
                    if old['status']=='completed' and same:pairs.append((old,new))
                item={'matched_resets':len(pairs),'previous_task_successes':sum(bool(a['success']) for a,b in pairs),
                    'corrected_task_successes':sum(bool(b['success']) for a,b in pairs),
                    'previous_trials_with_advance':sum(a['phase_advances']>0 for a,b in pairs),
                    'corrected_trials_with_advance':sum(b['phase_advances']>0 for a,b in pairs)}
                before_after[arm+'/'+family]=item
                lines.append('| '+' | '.join(map(str,[arm,family,*item.values()]))+' |')
        summary['before_after']=before_after
        write(directory/'summary.json',summary)
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    panels = []
    for i, trial in enumerate(plan['trials']):
        scenario = plan['scenarios'][trial['scenario_index']]
        panels.append('<h2>'+html.escape(f"{i:02d} {scenario['obj_class']} / {scenario['family']}")+'</h2><p>'+html.escape(scenario['episode']['task_instruction'])+'</p><div class="pair">')
        for arm in ARMS:
            row = records.get(arm, {}).get(i, {})
            panels.append('<section><h3>'+arm+'</h3>')
            if row.get('status') == 'completed':
                path = Path(row['rollout_video']).relative_to(directory)
                label = f"Task success: {row['success']}; phase advances: {row['phase_advances']}; grasp: {row['monitor']['grasp_acquired']}; grasp/lift: {row['monitor']['grasp_lift_prefix']}"
                panels.append('<p>'+html.escape(label)+'</p><video controls preload="none" src="'+html.escape(str(path))+'"></video>')
            else:
                panels.append('<p>'+html.escape(row.get('status', 'pending'))+'</p>')
            panels.append('</section>')
        panels.append('</div>')
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V2 development comparison</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}.pair{display:flex;gap:24px}.pair section{width:50%}video{width:100%}</style><h1>V2 epoch 1 / epoch 2</h1><p>'+html.escape(plan['scope'])+'</p>'+''.join(panels))
    return summary


def report(args, plan):
    records = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for path in (args.out_dir/arm).glob('[0-9][0-9][0-9].json'):
            row = read(path)
            if row['trial_index'] in records[arm]:
                raise ValueError('Duplicate trial')
            records[arm][row['trial_index']] = row
    summary = write_report(args.out_dir, plan, records)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('prepare', 'submit', 'evaluate', 'report'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--training-study', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--reference-development', type=Path)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.mode == 'prepare':
        if args.training_study is None:
            parser.error('--training-study is required for preparation')
        prepare(args)
        return
    plan = read(args.out_dir/'plan.json')
    verify(plan)
    {'submit': submit, 'evaluate': evaluate, 'report': report}[args.mode](args, plan)


if __name__ == '__main__':
    main()
