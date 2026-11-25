"""Frozen v3 single-scene grasp debugging; privileged references are evaluation only."""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal

from diagnostics.run_health import RunHealth, atomic_json
from diagnostics.storage import configure_cache_environment, register_result_dir, result_dir

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    import h5py
    import numpy as np
    import torch
    from models.pose_token_diffusion_v3.training import source_paths
    from diagnostics.v3_grasp_probe import REFERENCE_ORDER, RETENTION_LIMITS
    previous = read(args.development_plan)
    case = previous['scenarios'][args.scenario_index]
    if case['family'] != 'grasp' or len(case['episode']['skill_instructions']) != 1:
        raise ValueError('Choose one isolated grasp development scenario')
    checkpoint = Path(previous['checkpoint']).resolve(strict=True)
    if sha(checkpoint) != previous['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the completed development evaluation')
    training_receipt = Path(previous['training_study'])/'train/source_hashes.json'
    for name, wanted in read(training_receipt).items():
        if sha(name) != wanted:
            raise ValueError('Frozen training implementation changed: '+name)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload['architecture'] != 'pose_token_diffusion_v3' or payload['training']['epoch'] != 10 or payload['config']['pose_source'] != 'predicted':
        raise ValueError('Expected the trained epoch-10 predicted-pose v3 checkpoint')
    if args.out_dir.exists():
        raise FileExistsError('Choose a fresh diagnostic directory')
    with h5py.File(case['source_hdf5'], 'r') as handle:
        demo = handle['data'][case['source_demo']]
        episode = demo['metadata/ep_id'][()]
        episode = episode.decode() if isinstance(episode, bytes) else str(episode)
        if episode != str(case['episode']['ep_id']):
            raise ValueError('Reference demo does not match the selected episode')
        arrays = {'actions': demo['actions'][:], 'tcp': demo['obs/tcp_pose'][:],
                  'gripper': demo['obs/gripper_state'][:], 'joints': demo['obs/joint_states'][:]}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite reference demonstration')
    reused = {}
    if args.reference_study:
        previous_reference_plan = read(args.reference_study/'plan.json')
        if (previous_reference_plan['checkpoint_sha256'] != sha(checkpoint)
                or previous_reference_plan['scenario'] != case
                or previous_reference_plan['seed'] != previous['trials'][args.scenario_index]['seed']
                or previous_reference_plan['execution_settings'] != previous['execution_settings']
                or previous_reference_plan['controller'] != {'gain': .01, 'gain_gripper': .01, 'gain_near_target': .01}):
            raise ValueError('Reused reference comes from a different diagnostic setting')
        for name in REFERENCE_ORDER:
            path = args.reference_study/name/'result.json'
            if path.exists():
                row = read(path)
                if row.get('status') == 'completed' and row.get('retention', {}).get('retained_grasp') and row.get('reference_goal'):
                    reused[name] = {'source': str(path.resolve()), 'sha256': sha(path), 'row': row}
        if not reused:
            raise ValueError('No verified reference is available for reuse')
    args.out_dir.mkdir(parents=True)
    register_result_dir(args.out_dir)
    np.savez_compressed(args.out_dir/'reference_demo.npz', **arrays)
    for name, item in reused.items():
        (args.out_dir/name).mkdir()
        atomic_json(args.out_dir/name/'result.json', item['row'])
    vgn = Path(previous['data_root'])/'assets/models/vgn_conv.pth'
    if not vgn.is_file():
        raise FileNotFoundError('Native expert weights must already exist: '+str(vgn))
    paths = source_paths()+[Path(__file__), ROOT/'part2action/diagnostics/v3_grasp_probe.py',
        ROOT/'part2action/slurm/evaluate_v3_grasp_probe.sbatch',
        ROOT/'part2action/scripts/rollout_partgym.py',
        ROOT/'part2action/scripts/run_pose_token_v3_phase_diagnostic.py',
        ROOT/'part2action/diagnostics/execution_trace.py',
        ROOT/'part2action/diagnostics/physics_collision.py',
        ROOT/'part2action/diagnostics/partgym_skill_oracle.py',
        ROOT/'part2action/diagnostics/evaluation_scenarios.py',
        ROOT/'part2action/diagnostics/evaluation_protocol.py',
        Path(args.development_plan).resolve(), args.out_dir/'reference_demo.npz', vgn]
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym').rglob('*.py'))
    paths += list((ROOT/'PartInstruct/PartInstruct/PartGym').rglob('*.yaml'))
    paths += [args.out_dir/name/'result.json' for name in reused]
    paths += [Path(item['source']) for item in reused.values()]
    plan = {'schema': 1, 'scope': 'One fixed held-out development scene; inference diagnostics, not training or a benchmark.',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint), 'checkpoint_epoch': 10,
        'training_source_receipt_sha256': sha(training_receipt),
        'data_root': previous['data_root'], 'scenario': case,
        'seed': previous['trials'][args.scenario_index]['seed'],
        'historical_initial_state': read(Path(args.development_plan).parent/'learned_progression'/f'{args.scenario_index:03d}'/'result.json')['initial_state'],
        'execution_settings': previous['execution_settings'],
        'controller': {'gain': .01, 'gain_gripper': .01, 'gain_near_target': .01},
        'reference_order': list(REFERENCE_ORDER), 'vgn_checkpoint': str(vgn),
        'force_geometry_confidence': args.force_geometry_confidence,
        'reused_references': {name: {key: item[key] for key in ('source', 'sha256')} for name, item in reused.items()},
        'reference_study': str(args.reference_study.resolve()) if args.reference_study else None,
        'planned_conditions': [*(list(reused) if reused else REFERENCE_ORDER), 'predicted', 'verified'],
        'retention': {'lift_m': .05, 'lift_steps': 20, 'hold_steps': 10, 'limits': RETENTION_LIMITS},
        'source_sha256': {str(path.resolve()): sha(path) for path in sorted(set(paths))},
        'limits': ['Reference replay uses released commands, not undocumented historical motor gains.',
                   'The native expert uses privileged planning; its env control calls are normalized to the same evaluation gains.',
                   'Verified targets are object-relative measured poses from a reference that passes the added lift/hold check.',
                   'The verified arm reads simulator object pose only to construct pose tokens; RGB/scene/robot/skill encoders remain unchanged.',
                   ('Position/orientation applicability logits are forced to 20 (unit sigmoid multipliers); opening confidence, masks and skill logits remain unchanged.'
                    if args.force_geometry_confidence else 'Pose confidence/validity/skill logits remain learned and unchanged.')+' This intervention can introduce distribution shift.',
                   'Matched per-prediction diffusion noise does not keep later observations identical.',
                   'One scene and seed cannot establish architecture superiority.']}
    atomic_json(args.out_dir/'plan.json', plan)
    snapshot = args.out_dir/'source_snapshot'
    snapshot.mkdir()
    for index, path in enumerate(sorted(set(paths))):
        shutil.copy2(path, snapshot/f'{index:04d}_{path.name}')
    (args.out_dir/'README.md').write_text('# Frozen v3 grasp probe\n\nPrepared; jobs have not yet been submitted.\n\n'
        'Two reference checks (released demonstration replay and native VGN/BulletPlanner) '
        'and the ordinary predicted-pose policy can run concurrently. The verified-pose '
        'policy waits for references and reports blocked if neither passes grasp/lift validation. '
        'All learned policy weights and visual pathways remain unchanged.\n')


def verify(plan):
    if sha(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('Trained checkpoint changed')
    for name, wanted in plan['source_sha256'].items():
        if sha(name) != wanted:
            raise ValueError('Diagnostic source changed: '+name)


def make_env(plan):
    import open3d as o3d
    import rollout_partgym as helpers
    from omegaconf import OmegaConf
    from diagnostics.evaluation_protocol import align_native_environment
    from diagnostics.evaluation_scenarios import pin_scenario
    cls = helpers._load_partgym_env(ROOT/'PartInstruct')
    sim = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    native = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/baselines/evaluation/config/env/env_config.yaml')
    align_native_environment(sim, native)
    sim.data_root, sim.device, sim.meta_path = plan['data_root'], 'cpu', 'episodes_meta_train.json'
    full = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config.yaml')
    helpers.configure_partgym_agentview_pcd(sim, full)
    helpers.set_seed(plan['seed'])
    o3d.utility.random.seed(plan['seed'])
    case = plan['scenario']
    env = cls(config=sim, gui=False, record=True, evaluation=True, skill_mode=False,
              obj_class=case['obj_class'], split='train', task_type=case['task_type'], track_samples=False)
    try:
        pin_scenario(env, case['episode'], plan['seed'])
        observation = env.reset()
        if str(env.ep_id) != str(case['episode']['ep_id']):
            raise ValueError('Pinned scene mismatch')
        return env, observation
    except BaseException:
        env.close()
        raise


class ProbeLimit(BaseException):
    """Stop reference search cleanly without being swallowed by planner handlers."""


class Recorder:
    def __init__(self, env, plan, health):
        import numpy as np
        from diagnostics.execution_trace import physical_state
        from diagnostics.physics_collision import PhysicsCollisionLogger
        self.env, self.plan, self.health = env, plan, health
        self.physics = PhysicsCollisionLogger(env)
        self.original_take, self.original_step = env._take_action, env.step
        self.rows, self.goals, self.phase = [], [], 'base'
        self.native_success_observed = False
        self.goal = None
        self.initial = physical_state(env)
        self.initial_observation_hashes = {}
        env._take_action = self.take
        env.step = self.step

    def step(self, *args, **kwargs):
        result = self.original_step(*args, **kwargs)
        self.native_success_observed |= bool(result[3].get('Success', False))
        if self.rows:
            self.rows[-1]['native_success_after_step'] = bool(result[3].get('Success', False))
        return result

    def take(self, command, *args, **kwargs):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from data.geometry import bound_gripper_command
        from diagnostics.execution_trace import physical_state, plain
        if self.phase == 'base' and len(self.rows) >= self.plan['execution_settings']['max_steps']:
            raise ProbeLimit('Reference exhausted the executed-command budget')
        command = bound_gripper_command(command)
        self.physics.action_index, self.physics.command = len(self.rows), command.tolist()
        self.health.context.update(action_index=len(self.rows), phase=self.phase)
        self.original_take(command, **self.plan['controller'])
        state = physical_state(self.env)
        contacts = self.env.world.p.getContactPoints(bodyA=self.env.robot.panda, bodyB=self.env.obj.uid)
        fingers = [self.env.robot.LEFTFINGERIDX, self.env.robot.RIGHTFINGERIDX]
        qualified = sorted({int(c[3]) for c in contacts if c[3] in fingers and c[9] >= .1})
        measured = state['tcp_xyzw_xyz']
        row = {'action_index': len(self.rows), 'phase': self.phase, 'command': command.tolist(), 'state': state,
               'force_qualified_fingers': qualified, 'contacts': [
                   {'robot_link': int(c[3]), 'object_link': int(c[4]), 'normal_force_n': float(c[9]),
                    'distance_m': float(c[8]), 'position_on_object': list(c[6])} for c in contacts],
               'controller_applied': self.plan['controller'],
               'controller_requested': {'args': plain(args), 'kwargs': plain(kwargs)},
               'command_position_tracking_error_m': float(np.linalg.norm(np.asarray(measured[4:])-command[:3])),
               'command_orientation_tracking_error_deg': float(np.rad2deg((Rotation.from_quat(measured[:4]).inv()*Rotation.from_euler('xyz', command[3:6])).magnitude()))}
        if self.goal:
            row['goal'] = self.goal
            goal = self.goal['conditioned_goal']
            row['target_position_error_m'] = float(np.linalg.norm(np.asarray(measured[4:])-goal['position_world']))
            row['target_orientation_error_deg'] = float(np.rad2deg((Rotation.from_quat(measured[:4]).inv()*Rotation.from_matrix(goal['rotation_world'])).magnitude()))
        self.rows.append(row)
        if len(self.rows) == 1 or len(self.rows) % 25 == 0:
            self.health.event('executing', action_index=len(self.rows)-1, phase=self.phase)

    def close(self):
        self.env._take_action = self.original_take
        self.env.step = self.original_step
        self.physics.close()


def end_and_lift(env, recorder, plan):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from diagnostics.execution_trace import physical_state
    from diagnostics.partgym_skill_oracle import PartGymSkillOracle
    from diagnostics.v3_grasp_probe import relative_grasp_goal, retention_score
    from run_pose_token_v3_phase_diagnostic import preserved_completion_query
    before = physical_state(env)
    count = len(recorder.physics.rows)
    decision = preserved_completion_query(env, PartGymSkillOracle(), 0)
    del recorder.physics.rows[count:]
    if not decision.supported:
        raise ValueError('Unsupported grasp predicate: '+decision.reason)
    opening = float(recorder.rows[-1]['command'][6]) if recorder.rows else .04
    candidate = relative_grasp_goal(before, opening)
    recorder.phase, recorder.goal = 'added_lift_hold', None
    origin = np.asarray(before['tcp_xyzw_xyz'][4:])
    angles = Rotation.from_quat(before['tcp_xyzw_xyz'][:4]).as_euler('xyz')
    settings = plan['retention']
    for step in range(settings['lift_steps']+settings['hold_steps']):
        xyz = origin+[0., 0., settings['lift_m']*min(1., (step+1)/settings['lift_steps'])]
        env._take_action(np.r_[xyz, angles, opening], **plan['controller'])
        env._get_observation()  # Added continuation video only; not a policy input.
    after = physical_state(env)
    score = retention_score(before, after, recorder.rows[-settings['hold_steps']:],
        native_grasp=decision.complete, commanded_opening=opening, limits=settings['limits'])
    return {'before_lift': before, 'after_lift': after, 'native_grasp_before_lift': decision.complete,
            'native_predicate_reason': decision.reason, 'retention': score,
            'candidate_reference_goal': candidate,
            'reference_goal': candidate if score['retained_grasp'] else None}


def finish(folder, env, recorder, row):
    # Native save_renders deletes its destination directory: isolate video files.
    env.dump_buffers()
    env.save_renders(str(folder/'video/rollout.mp4'), video_only=True)
    with gzip.open(folder/'commands.json.gz', 'wt') as handle:
        json.dump(recorder.rows, handle, allow_nan=False)
    with gzip.open(folder/'physics.json.gz', 'wt') as handle:
        json.dump(recorder.physics.rows, handle, allow_nan=False)
    atomic_json(folder/'goals.json', recorder.goals)
    atomic_json(folder/'result.json', {**row, 'status': 'completed', 'initial_state': recorder.initial,
        'initial_observation_hashes': recorder.initial_observation_hashes,
        'base_commands': sum(r['phase'] == 'base' for r in recorder.rows),
        'additional_commands': sum(r['phase'] != 'base' for r in recorder.rows),
        'video': str(folder/'video/rollout.mp4'), 'command_trace': str(folder/'commands.json.gz')})


def observation_hashes(observation):
    import numpy as np
    return {key: hashlib.sha256(np.asarray(observation[key]).tobytes()).hexdigest()
            for key in ('agentview_rgb', 'agentview_pcd')}


def reference(args, plan, health):
    import numpy as np
    from diagnostics.v3_grasp_probe import REFERENCE_ORDER, matching_states
    name = REFERENCE_ORDER[args.index]
    folder = args.out_dir/name
    folder.mkdir(exist_ok=False)
    env, observation = make_env(plan)
    planner = None
    if args.index == 1:
        # BulletPlanner deep-copies its parser/world at construction. Build it
        # before attaching recorder callbacks that reference open health logs.
        try:
            from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner
            planner = BulletPlanner(env, planner_model_path=Path(plan['vgn_checkpoint']), generation_mode=True)
            planner.grasp_planner.net.eval()
        except BaseException:
            env.close()
            raise
    recorder = Recorder(env, plan, health)
    recorder.initial_observation_hashes = observation_hashes(observation)
    row = {'arm': name, 'native_success_observed': False, 'stop_reason': 'reference_finished',
           'matches_historical_initial_state': matching_states(recorder.initial, plan['historical_initial_state'])}
    old_alarm = signal.getsignal(signal.SIGALRM)
    def timed_out(*unused):
        raise ProbeLimit('Reference search exceeded fifteen minutes')
    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(900)
    try:
        try:
            if args.index == 0:
                with np.load(args.out_dir/'reference_demo.npz') as data:
                    actions = data['actions'].copy()
                    recorded_tcp = data['tcp'].copy()
                for command in actions:
                    _, _, _, info = env.step(command)
                    row['native_success_observed'] |= bool(info.get('Success', False))
                measured = np.asarray([r['state']['tcp_xyzw_xyz'][4:] for r in recorder.rows])
                errors = np.linalg.norm(measured-recorded_tcp[:len(measured), 4:], axis=1)
                row['replay_tracking'] = {'mean_position_error_m': float(errors.mean()),
                                          'maximum_position_error_m': float(errors.max()),
                                          'note': 'Released HDF5 does not store per-step historical gains; current evaluation gains are used.'}
            else:
                params = plan['scenario']['episode']['chain_params'][0]['params']
                row['planner_return'] = bool(planner.grasp_obj(plan['scenario']['obj_class'], **params))
        except ProbeLimit as limit:
            row['stop_reason'] = str(limit)
        finally:
            signal.alarm(0)
        row['native_success_observed'] = recorder.native_success_observed
        if recorder.rows:
            row.update(end_and_lift(env, recorder, plan))
        else:
            row.update(retention={'retained_grasp': False, 'reason': 'No reference actions'}, reference_goal=None)
        finish(folder, env, recorder, row)
    finally:
        signal.signal(signal.SIGALRM, old_alarm)
        recorder.close()
        env.close()


def policy(args, plan, health):
    import numpy as np
    import torch
    from collections import deque
    import rollout_partgym as helpers
    from diagnostics.evaluation_protocol import execution_prefix
    from diagnostics.v3_grasp_probe import (POLICY_ARMS, REFERENCE_ORDER, GoalIntervention,
        choose_reference, matching_states, inference_with_fixed_noise, policy_digest, world_grasp_goal)
    from models.pose_token_diffusion_v3.checkpoint import load_checkpoint
    from models.pose_token_diffusion_v3.config import PoseTokenDiffusionV3Config
    from models.pose_token_diffusion_v3.contracts import SkillObservation
    from models.pose_token_diffusion_v3.policy import build_policy
    from models.pose_token_diffusion_v3.skills import annotated_skill
    name = POLICY_ARMS[args.index]
    folder = args.out_dir/name
    folder.mkdir(exist_ok=False)
    selected_name, selected = None, None
    if name == 'verified':
        references = {arm: read(args.out_dir/arm/'result.json') for arm in REFERENCE_ORDER
                      if (args.out_dir/arm/'result.json').exists()}
        selected_name, selected = choose_reference(references)
        if selected is None:
            atomic_json(folder/'result.json', {'status': 'blocked', 'arm': name,
                'reason': 'Neither reference passed grasp/lift validation. No GT-target policy result exists.',
                'reference_statuses': {k: r.get('retention') for k, r in references.items()}})
            return
    if not torch.cuda.is_available():
        raise RuntimeError('Frozen policy evaluation requires a Slurm GPU allocation')
    payload = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    config = PoseTokenDiffusionV3Config.from_mapping(payload['config'])
    del payload
    helpers.set_seed(plan['seed'])
    health.event('loading_frozen_policy', arm=name)
    model = build_policy(config)
    load_checkpoint(model, plan['checkpoint'])
    model.requires_grad_(False).cuda().eval()
    before_digest = policy_digest(model)
    env, observation = make_env(plan)
    recorder = Recorder(env, plan, health)
    recorder.initial_observation_hashes = observation_hashes(observation)
    row = {'arm': name, 'checkpoint_sha256': plan['checkpoint_sha256'], 'pose_source': name,
           'force_geometry_confidence': plan.get('force_geometry_confidence', False),
           'reference_source': selected_name, 'native_success_observed': False,
           'gpu': torch.cuda.get_device_name(), 'stop_reason': 'command_budget',
           'matches_historical_initial_state': matching_states(recorder.initial, plan['historical_initial_state'])}
    provider = None
    if selected is not None:
        if not matching_states(recorder.initial, selected['initial_state']):
            recorder.close(); env.close()
            raise ValueError('Verified reference was obtained from a different physical reset')
        row['reference_result_sha256'] = sha(args.out_dir/selected_name/'result.json')
        def provider():
            xyz, quat = env.world.p.getBasePositionAndOrientation(env.obj.uid)
            return world_grasp_goal(selected['reference_goal'], [*quat, *xyz])
    device = torch.device('cuda')
    history = deque([np.asarray(observation['agentview_rgb'], dtype=np.uint8)], maxlen=config.n_obs_steps)
    anchor = helpers._model_observations(observation, device)['tcp_pose'].detach().clone()
    skill = annotated_skill(plan['scenario']['episode']['skill_instructions'][0])
    try:
        with GoalIntervention(model, provider, force_geometry_confidence=plan.get('force_geometry_confidence', False)) as intervention:
            prediction = 0
            while len(recorder.rows) < plan['execution_settings']['max_steps']:
                inputs = helpers._model_observations(observation, device)
                obs = SkillObservation(helpers._make_rgb_tensor(history, config.n_obs_steps, device),
                    inputs['agentview_pcd'][..., :3], inputs['tcp_pose'], inputs['gripper_state'].reshape(1, 1), (skill,), anchor)
                health.context.update(prediction=prediction)
                output = inference_with_fixed_noise(model, obs, plan['seed'], prediction)
                recorder.goal = copy.deepcopy(intervention.last)
                recorder.goals.append({'prediction': prediction, **copy.deepcopy(intervention.last)})
                chunk = output.actions['action_chunk'][0].detach().float().cpu().numpy()
                done = False
                for command in execution_prefix(chunk, plan['execution_settings']['execute_steps'], plan['execution_settings']['max_steps']-len(recorder.rows)):
                    observation, _, done, info = env.step(command)
                    row['native_success_observed'] |= bool(info.get('Success', False))
                    history.append(np.asarray(observation['agentview_rgb'], dtype=np.uint8))
                    if done:
                        row['stop_reason'] = 'native_environment_done'
                        break
                prediction += 1
                if done:
                    break
        after_digest = policy_digest(model)
        if after_digest != before_digest or sha(plan['checkpoint']) != plan['checkpoint_sha256']:
            raise ValueError('Frozen policy weights or checkpoint changed')
        row.update(policy_weights_unchanged=True, policy_state_sha256=before_digest,
                   rgb_scene_skill_robot_inputs_retained=True, **end_and_lift(env, recorder, plan))
        finish(folder, env, recorder, row)
    finally:
        recorder.close()
        env.close()


def report(args, plan):
    from diagnostics.v3_grasp_probe import REFERENCE_ORDER, POLICY_ARMS, matching_states
    records = {name: read(args.out_dir/name/'result.json') for name in (*REFERENCE_ORDER, *POLICY_ARMS)
               if (args.out_dir/name/'result.json').exists()}
    expected = plan.get('planned_conditions', [*REFERENCE_ORDER, *POLICY_ARMS])
    complete = all(name in records for name in expected)
    comparison = {'status': 'complete' if complete else 'incomplete', 'scope': plan['scope'],
        'checkpoint_epoch': 10, 'conditions': {name: {key: row.get(key) for key in
            ('status', 'native_success_observed', 'retention', 'reference_source', 'reason', 'policy_weights_unchanged')}
            for name, row in records.items()}, 'limits': plan['limits']}
    a, b = records.get('predicted', {}), records.get('verified', {})
    comparison['paired_policy_comparison'] = 'available' if a.get('status') == b.get('status') == 'completed' else 'unavailable'
    if a.get('status') == b.get('status') == 'completed':
        comparison['matching_initial_states'] = matching_states(a['initial_state'], b['initial_state'])
        comparison['matching_initial_observation_hashes'] = a['initial_observation_hashes'] == b['initial_observation_hashes']
        comparison['matching_policy_weights'] = a['policy_state_sha256'] == b['policy_state_sha256']
    comparison['interpretation'] = ('A blocked verified condition means the setup has no validated reference yet; it is not a model failure. '
        'A verified-pose gain would support investigating grounding/pose accuracy. Failure with a verified target leaves decoder, controller, confidence, approach and closure unresolved; injection can shift the input distribution.')
    if plan.get('reference_study'):
        previous = Path(plan['reference_study'])
        factorial = {}
        factorial_rows = []
        for source, gate in ((previous, 'learned'), (args.out_dir, 'forced_full')):
            for arm in POLICY_ARMS:
                path = source/arm/'result.json'
                if path.exists():
                    row = read(path)
                    if row.get('status') == 'completed':
                        factorial_rows.append(row)
                    factorial[gate+'_'+arm] = {'status': row['status'],
                        'native_success': row.get('native_success_observed'),
                        'retained_grasp': row.get('retention', {}).get('retained_grasp'),
                        'source': str(path), 'source_sha256': sha(path)}
        comparison['geometry_gate_conditions'] = factorial
        if len(factorial_rows) == 4:
            first = factorial_rows[0]
            comparison['gate_comparison_controls'] = {
                'same_checkpoint': all(row['checkpoint_sha256'] == plan['checkpoint_sha256'] for row in factorial_rows),
                'same_policy_state': all(row['policy_state_sha256'] == first['policy_state_sha256'] for row in factorial_rows),
                'same_initial_physical_state': all(matching_states(row['initial_state'], first['initial_state']) for row in factorial_rows),
                'same_initial_observation_hashes': all(row['initial_observation_hashes'] == first['initial_observation_hashes'] for row in factorial_rows),
                'same_gpu_type': all(row['gpu'] == first['gpu'] for row in factorial_rows)}
    atomic_json(args.out_dir/'summary.json', comparison)
    lines = ['# Frozen v3 grasp diagnostic', '', plan['scope'], '',
        '| Condition | Status | Native success observed | Retained grasp after added lift |',
        '| --- | --- | --- | --- |']
    for name in expected:
        row = records.get(name, {})
        lines.append(f"| {name} | {row.get('status', 'missing')} | {row.get('native_success_observed', 'N/A')} | {row.get('retention', {}).get('retained_grasp', 'N/A')} |")
    lines += ['', comparison['interpretation'], '', 'Weights remain frozen; no new training. Both policy arms retain visual/scene features.', '']+plan['limits']
    if comparison.get('geometry_gate_conditions'):
        lines += ['', '## Geometry and gating diagnostic', '',
                  '| Gate | Pose source | Status | Native success | Retained grasp |',
                  '| --- | --- | --- | --- | --- |']
        for key, row in comparison['geometry_gate_conditions'].items():
            gate, arm = key.rsplit('_', 1)
            lines.append(f"| {gate} | {arm} | {row['status']} | {row['native_success']} | {row['retained_grasp']} |")
    (args.out_dir/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'reference', 'policy', 'report'], required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--development-plan', type=Path)
    parser.add_argument('--scenario-index', type=int, default=3)
    parser.add_argument('--index', type=int, choices=[0, 1], default=0)
    parser.add_argument('--force-geometry-confidence', action='store_true',
                        help='Preparation only: force position/orientation token multipliers to one')
    parser.add_argument('--reference-study', type=Path,
                        help='Preparation only: reuse validated references from the exact prior scene/protocol')
    args = parser.parse_args()
    configure_cache_environment()
    args.out_dir = result_dir(args.out_dir)
    if args.mode == 'prepare':
        prepare(args)
        return
    plan = read(args.out_dir/'plan.json')
    if args.mode == 'report':
        report(args, plan)
        return
    health_dir = args.out_dir/'health'/f'{args.mode}_{args.index}_{os.environ.get("SLURM_JOB_ID", os.getpid())}'
    with RunHealth(health_dir) as health:
        try:
            verify(plan)
            health.event('starting_diagnostic', mode=args.mode, index=args.index)
            {'reference': reference, 'policy': policy}[args.mode](args, plan, health)
        except BaseException as error:
            from diagnostics.v3_grasp_probe import REFERENCE_ORDER, POLICY_ARMS
            name = (REFERENCE_ORDER if args.mode == 'reference' else POLICY_ARMS)[args.index]
            atomic_json(args.out_dir/name/'result.json', {'status': 'runtime_error', 'arm': name,
                'error': repr(error), 'health_directory': str(health_dir)})
            raise


if __name__ == '__main__':
    main()
