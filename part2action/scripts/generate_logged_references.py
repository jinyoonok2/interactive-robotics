"""Fresh native expert grasps with exact controller logs and gated perturbations.

Separate training-source dataset; never edits released HDF5s. This first adapter
supports initially-open, first-skill grasps, not arbitrary later skills or full
task regeneration. A failed generation/replay is not a negative variation label.
"""
from __future__ import annotations

import argparse
import copy
import functools
import gzip
import gc
import hashlib
import inspect
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from diagnostics.execution_trace import ExecutionTrace, can_expand, physical_state, plain, state_trace_agreement, trace_arrays
from diagnostics.interaction_records import candidate_vector, decision_frame, observe, outcome_components
from diagnostics.interaction_tolerances import conditions, grasp_window, perturb_actions
from diagnostics.physics_collision import PhysicsCollisionLogger
from diagnostics.evaluation_scenarios import pin_scenario
from diagnostics.lift_continuation import lift_settings
from run_interaction_tolerance_pilot import contact_summary, dump

ROOT = Path(__file__).resolve().parents[2]
REPLAY_LIMITS = {'max_tcp_position_m': 1e-6, 'max_tcp_rotation_deg': .001,
                 'max_finger_difference_m': 1e-6, 'initial_state_max_difference': 1e-9,
                 'max_robot_joint_difference_rad': 1e-6,
                 'max_object_position_m': 1e-6, 'max_object_rotation_deg': .001}


def enumerate_sources(data, objects):
    """Every installed training demo is inventoried, including unsupported starts."""
    metadata = json.loads((data/'episodes_meta_train.json').read_text())
    cases, skipped = [], []
    for obj in objects:
        lookup = {str(e['ep_id']): (task, e) for task, rows in metadata[obj]['train'].items() for e in rows}
        with h5py.File(data/'demos'/f'{obj}.hdf5', 'r') as handle:
            for key in sorted(handle['data'], key=lambda v: int(v.removeprefix('demo_'))):
                demo = handle['data'][key]
                if 'metadata/ep_id' not in demo or 'actions' not in demo or not len(demo['actions']):
                    skipped.append({'object': obj, 'demo': key, 'reason': 'Missing metadata or action sequence'})
                    continue
                ep = demo['metadata/ep_id'][()]
                ep = ep.decode() if isinstance(ep, bytes) else str(ep)
                base = {'object': obj, 'demo': key, 'episode_id': ep}
                if ep not in lookup:
                    skipped.append({**base, 'reason': 'No matching training episode metadata'})
                    continue
                task, episode = lookup[ep]
                if not episode['chain_params'] or episode['chain_params'][0]['skill_name'] != 'grasp_obj':
                    skipped.append({**base, 'reason': 'First skill is not grasp_obj; adapter not implemented'})
                    continue
                if float(demo['actions'][0, 6]) <= .02:
                    skipped.append({**base, 'reason': 'Source starts with closed gripper; adapter not implemented'})
                    continue
                part = episode['chain_params'][0]['params'].get('part_grasp', '')
                cases.append({**base, 'object_id': str(episode['obj_id']), 'task_type': task,
                              'part': part, 'instruction': episode['skill_instructions'][0],
                              'episode': episode})
    return cases, skipped


def pilot_selection(cases, per_object, seed):
    """Select before generating outcomes; diversify part/instance groups."""
    rng = random.Random(seed)
    selected = []
    for obj in sorted({c['object'] for c in cases}):
        groups = {}
        for case in cases:
            if case['object'] == obj:
                groups.setdefault((case['part'], case['object_id']), []).append(case)
        keys = sorted(groups)
        rng.shuffle(keys)
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while keys and len(chosen) < per_object:
            remaining = []
            for key in keys:
                if len(chosen) >= per_object:
                    break
                chosen.append(groups[key].pop())
                if groups[key]:
                    remaining.append(key)
            keys = remaining
        selected.extend(chosen)
    return selected


def native_classes():
    sys.path.insert(0, str(ROOT/'PartInstruct'))
    from PartInstruct.PartGym.env.bullet_env import BulletEnv
    from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner

    class ExpertStateEnv(BulletEnv):
        def _get_observation(self):
            # The native expert uses geometry and low-dimensional state, not
            # RGB or token embeddings. Avoid unused render cost in its timeouts.
            return {'tcp_pose': self.robot.get_tcp_pose().to_list(),
                    'joint_states': np.asarray(self.robot.get_joint_states()),
                    'gripper_state': np.array([self.robot.get_gripper_state()])}

    return ExpertStateEnv, BulletPlanner


def make_env(data, case, seed, native_task_context=False):
    # Native BulletPlanner deep-copies parsers containing BulletClients. Collect
    # closed prior-scene cycles BEFORE Bullet reuses a connection identifier.
    gc.collect()
    import open3d as o3d
    import torch
    from omegaconf import OmegaConf
    cls, _ = native_classes()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    o3d.utility.random.seed(seed)
    config = OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    config.data_root = str(data)
    config.meta_path = 'episodes_meta_train.json'
    config.device = 'cpu'
    config.shape_meta.obs = {k: v for k, v in config.shape_meta.obs.items()
                             if k in {'tcp_pose', 'joint_states', 'gripper_state'}}
    env = cls(config=config, gui=False, evaluation=True, skill_mode=not native_task_context,
              obj_class=case['object'], split='train', task_type=case['task_type'], record=False)
    try:
        episode = copy.deepcopy(case['episode'])
        if not native_task_context:
            episode['chain_params'] = episode['chain_params'][:1]
            episode['skill_instructions'] = episode['skill_instructions'][:1]
        pin_scenario(env, episode, seed)
        env.reset()
        if str(env.ep_id) != case['episode_id'] or str(env.obj_id) != case['object_id']:
            raise RuntimeError('Fresh scene selected the wrong source episode')
        points = env.parser.part_pcds[case['part'] or env.parser.obj_name]
        xyz, quat = env.world.p.getBasePositionAndOrientation(env.obj.uid)
        env.tree = cKDTree(Rotation.from_quat(quat).inv().apply(points - xyz))
        env.initial_z = float(xyz[2])
        env.reference_part = case['part'] or env.parser.obj_name
        return env
    except BaseException:
        env.close()
        raise


def native_grasp(env):
    from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner
    return bool(BulletPlanner.predicate_grasping(env, part_name=env.reference_part))


def lift_and_score(env, trace, physics, continuation='legacy'):
    """Fixed 5 cm / 1 s relative lift, 0.5 s hold; same earlier pilot criteria."""
    before_lift_native = native_grasp(env)
    last_action = next(e for e in reversed(trace.events) if e['kind'] == 'action')
    settings = lift_settings(continuation, last_action)
    pose = env.robot.get_tcp_pose()
    origin, angles = pose.translation.copy(), pose.rotation.as_euler('xyz')
    with trace.scope('additional_settle', settings):
        for _ in range(settings['settle_steps']):
            physics.action_index = sum(e['kind'] == 'action' for e in trace.events)
            physics.command = last_action['action']
            env.robot.apply_action(last_action['action'], **last_action['kwargs'])
            env.world.step()
            trace.finish_action()
    with trace.scope('additional_lift_hold', settings):
        for t in range(settings['lift_steps']+settings['hold_steps']):
            action = np.r_[origin + [0, 0, .05*min(1., (t+1)/settings['lift_steps'])], angles, 0.]
            physics.action_index = sum(e['kind'] == 'action' for e in trace.events)
            physics.command = action.tolist()
            env.robot.apply_action(action, gain=.01, gain_gripper=settings['finger_gain'], gain_near_target=.01)
            env.world.step()
            trace.finish_action()
    hold = physics.rows[-72:]
    if len(hold) != 72:
        raise RuntimeError('Incomplete hold instrumentation')
    contact = contact_summary(env, hold)
    relative = np.array([Rotation.from_quat(r['tcp_quat_xyzw']).inv().apply(
        np.asarray(r['object_xyz']) - r['tcp_xyz']) for r in hold])
    xyz, _ = env.world.p.getBasePositionAndOrientation(env.obj.uid)
    row = {'native_grasp_predicate_equivalent': before_lift_native,
           'continuation': settings,
           'lift_m': float(xyz[2] - env.initial_z), 'hold': contact,
           'hold_relative_position_drift_m': float(np.linalg.norm(relative-relative[0], axis=1).max())}
    row['outcome_components'] = outcome_components(row)
    row['stable_target_grasp_lift'] = all(row['outcome_components']['passes'].values())
    return row


def save_trial(directory, trace, physics, row, observations=None):
    directory.mkdir(parents=True, exist_ok=False)
    actions, tcp, fingers = trace_arrays(trace.events)
    dump(directory/'result.json', plain(row))
    np.savez_compressed(directory/'trajectory.npz', actions=actions, tcp=tcp, fingers=fingers)
    with gzip.open(directory/'execution.json.gz', 'wt') as handle:
        json.dump({'initial': trace.initial, 'events': trace.events,
                   'phase_events': trace.phase_events, 'physics_steps': trace.physics_steps}, handle)
    with gzip.open(directory/'physics.json.gz', 'wt') as handle:
        json.dump(physics.rows, handle)
    if observations is not None:
        np.savez_compressed(directory/'observations.npz', **observations)


def generate(data, case, seed, directory, native_task_context=False):
    _, planner_cls = native_classes()
    env = make_env(data, case, seed, native_task_context)
    trace = physics = None
    try:
        planner = planner_cls(env, planner_model_path=data/'assets/models/vgn_conv.pth', generation_mode=True)
        planner.grasp_planner.net.eval()
        from omegaconf import OmegaConf
        context = {
            'config': OmegaConf.to_container(env.config, resolve=True),
            'seed': seed, 'python_numpy_torch_open3d_seeded': True,
            'physics_engine': plain(env.world.p.getPhysicsEngineParameters()),
            'physics_dt_s': env.world.dt, 'control_hz': env.world.control_hz,
            'physics_steps_per_control': env.world.n_steps_per_control,
            'robot_home_joints': env.robot.HOME_JOINT_POS,
            'native_apply_action_argument_wiring_preserved': True,
            'native_full_task_termination_context': native_task_context,
            'versions': {name: importlib.metadata.version(name) for name in ['torch', 'pybullet', 'open3d', 'numpy', 'scipy']},
            'dynamics': {name: [plain(env.world.p.getDynamicsInfo(body, link))
                                for link in range(-1, env.world.p.getNumJoints(body))]
                         for name, body in [('robot', env.robot.panda), ('object', env.obj.uid),
                                            ('table', env.table.uid), ('floor', env.floor.uid)]},
            'asset_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                            sorted((data/'assets/partnet-grasping'/case['object']/case['object_id']).rglob('*'))
                            if path.is_file()},
            'scope': 'Fresh initial grasp only; native expert + additional fixed lift test, not full-task expert regeneration',
        }
        trace = ExecutionTrace(env)
        physics = PhysicsCollisionLogger(env)
        native_step = env.step

        def step(*args, **kwargs):
            physics.action_index = sum(e['kind'] == 'action' for e in trace.events)
            physics.command = plain(args[0] if args else kwargs['action'])
            result = native_step(*args, **kwargs)
            trace.finish_action()
            return result
        env.step = step
        for name in ['grasp_obj', 'plan_grasps', 'execute_grasp', 'target_grasp', 'pose_tcp', 'switch_gripper']:
            original = getattr(planner, name)
            def wrap(method, phase):
                @functools.wraps(method)
                def call(*args, **kwargs):
                    # Numeric settings are logged; full motor calls carry the
                    # actual targets. Avoid serializing whole scene objects.
                    settings = {k: plain(v) for k, v in kwargs.items()
                                if isinstance(v, (str, bool, int, float, list, tuple))}
                    with trace.scope(phase, settings) as phase_row:
                        result = method(*args, **kwargs)
                        phase_row['returned_bool'] = bool(result[0] if isinstance(result, tuple) else result) if isinstance(result, (bool, tuple)) else None
                        return result
                return call
            setattr(planner, name, wrap(original, name))
        started = time.monotonic()
        ret = planner.grasp_obj(case['object'], **case['episode']['chain_params'][0]['params'])
        grasp_events = len(trace.events)
        grasp_physics = trace.physics_steps
        # Some planner attempts can end before a close command. Preserve those
        # attempts as generation failures, never perturbation negatives.
        grasp_actions, _, _ = trace_arrays(trace.events)
        row = {'planner_reported_success': bool(ret), 'source_seed': seed,
               'native_full_task_termination_context': native_task_context,
               'grasp_event_count': grasp_events, 'grasp_physics_steps': grasp_physics,
               'generation_wall_s': time.monotonic()-started}
        if len(grasp_actions) and np.any(grasp_actions[:, 6] <= .02):
            row.update(lift_and_score(env, trace, physics))
        else:
            row.update(stable_target_grasp_lift=False, generation_failure='No executed gripper-close event')
        save_trial(directory, trace, physics, row)
        dump(directory/'generation_context.json', context)
        return row, copy.deepcopy(trace.events), trace.initial
    finally:
        if physics is not None:
            physics.close()
        if trace is not None:
            trace.close()
        env.close()


def replay(data, case, seed, reference_events, grasp_event_count, condition, window, directory,
           capture_frame=None, reference_initial=None, continuation='legacy', native_task_context=False):
    env = make_env(data, case, seed, native_task_context)
    trace = ExecutionTrace(env)
    physics = PhysicsCollisionLogger(env)
    prefix = reference_events[:grasp_event_count]
    original, _, _ = trace_arrays(prefix)
    actions = perturb_actions(original, window, condition)
    observations = []
    action_index = 0
    initial_difference = max(float(np.max(np.abs(np.asarray(trace.initial[k])-reference_initial[k])))
                             for k in ['robot_q', 'robot_qd', 'object_xyz', 'object_xyzw',
                                       'object_linear_velocity', 'object_angular_velocity']) if reference_initial else None
    try:
        for event in prefix:
            if event['kind'] == 'client_call':
                trace.phase = event['phase']
                getattr(env.world.p, event['function'])(*event['args'], **event['kwargs'])
            if event['kind'] == 'action':
                if capture_frame is not None and action_index in {max(0, capture_frame-1), capture_frame}:
                    observations.append(observe(env, env.config))
                trace.phase = event['phase']
                physics.action_index = action_index
                physics.command = actions[action_index].tolist()
                env.robot.apply_action(actions[action_index], **event['kwargs'])
                action_index += 1
            for _ in range(event['physics_steps']):
                env.world.p.stepSimulation()
                env.world.sim_time += env.world.dt
            trace.finish_action()
        prefix_agreement = state_trace_agreement(prefix, trace.events) if not np.any(candidate_vector(condition)) else None
        row = lift_and_score(env, trace, physics, continuation)
        row.update(condition=condition, candidate_parameters=candidate_vector(condition).tolist(),
                   initial_state_max_difference=initial_difference, grasp_prefix_agreement=prefix_agreement)
        if condition['name'] == 'identity':
            row['agreement_to_generated'] = state_trace_agreement(reference_events, trace.events)
        packed = None
        if capture_frame is not None:
            if len(observations) == 1 and capture_frame == 0:
                observations *= 2
            if len(observations) != 2:
                raise RuntimeError('Missing decision observations')
            packed = {key: np.stack([o[key] for o in observations]) for key in observations[0]}
            row['decision_frame'] = capture_frame
            row['primitive_context'] = np.r_[original[window['close']], (window['close']-capture_frame)/20.].tolist()
        save_trial(directory, trace, physics, row, packed)
        return row, trace_arrays(trace.events), packed
    finally:
        physics.close()
        trace.close()
        env.close()


def pilot_conditions():
    """Fixed before outcomes: small spatial/timing changes and selected joint tests."""
    rows = conditions(.01, 10., 2)
    base = rows[0]
    for sign in [-1, 1]:
        rows.append({**base, 'name': f'joint_x_y_close_{sign}',
                     'translation_m': [sign*.02, 0., 0.], 'rotation_deg': [0., sign*20., 0.],
                     'close_offset_steps': sign*4})
    # Predeclared larger orientation probes supply potential failures without
    # selecting per-case magnitudes after inspecting their success labels.
    rows.append({**base, 'name': 'rotation_x_plus_60', 'rotation_deg': [60., 0., 0.]})
    return rows


def run_case(plan, case, index, out):
    data = Path(plan['data_root'])
    seed = plan['seed'] + index
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'case.json', case)
    row = {'case': case, 'source_index': index, 'seed': seed, 'eligible': False,
           'generated_stable_success': False, 'variation_successes': 0, 'variation_failures': 0}
    try:
        task_context=bool(plan.get('native_task_context',False))
        generation, events, initial = generate(data, case, seed, out/'generated',task_context)
        row['generation'] = generation
        row['generated_stable_success'] = generation['stable_target_grasp_lift']
        if not row['generated_stable_success']:
            row['status'] = 'generation_failed_physical_criteria'
            return row
        prefix = events[:generation['grasp_event_count']]
        actions, _, _ = trace_arrays(prefix)
        window = grasp_window(actions, [case['instruction']]*len(actions))
        proposed = plan['conditions']
        valid, unsupported = [], []
        for condition in proposed:
            onset = window['close'] + condition['close_offset_steps']
            if not 0 < onset < len(actions):
                unsupported.append(condition['name'])
            else:
                valid.append(condition)
        row['unsupported_timing_conditions'] = unsupported
        frame = decision_frame(actions, window, valid)
        references = []
        _, ref_tcp, ref_fingers = trace_arrays(events)
        reference_observation = None
        for repeat in range(plan['repeats']):
            result, arrays, observation = replay(data, case, seed, events, generation['grasp_event_count'],
                valid[0], window, out/f'identity_{repeat}', frame if repeat else None, initial,native_task_context=task_context)
            agreement = result['agreement_to_generated']
            agreement['initial_state_max_difference'] = result['initial_state_max_difference']
            ok = all(agreement[k] <= v for k, v in REPLAY_LIMITS.items()) and result['stable_target_grasp_lift']
            references.append({'passed': ok, **agreement})
            if observation is not None:
                reference_observation = observation
        row['reference_checks'] = references
        row['eligible'] = all(r['passed'] for r in references)
        if not row['eligible']:
            row['status'] = 'replay_validation_failed'
            return row
        if reference_observation is None:
            raise RuntimeError('Need a capture-enabled identity reference')
        records = []
        records.append({'result_file': 'identity_1/result.json', 'observation_file': 'identity_1/observations.npz',
                        'instance_group': f'{case["object"]}/{case["object_id"]}',
                        'source_group': f'{case["object"]}/{case["demo"]}', 'skill_instruction': case['instruction']})
        for condition in valid[1:]:
            # One fresh execution per condition. Identity repeats validate
            # reconstruction; exact repeats are not independent training data.
            name = condition['name']
            result, _, observation = replay(data, case, seed, events, generation['grasp_event_count'],
                condition, window, out/name, frame, initial,native_task_context=task_context)
            for key in ['rgb', 'depth_m', 'pcd_camera_m', 'tcp_pose_xyzw_xyz', 'finger_position_m']:
                if not np.array_equal(observation[key], reference_observation[key]):
                    raise RuntimeError('Pre-intervention observations changed across candidates: '+key)
            success = result['stable_target_grasp_lift']
            row['variation_successes' if success else 'variation_failures'] += 1
            records.append({'result_file': f'{name}/result.json', 'observation_file': f'{name}/observations.npz',
                            'instance_group': f'{case["object"]}/{case["object_id"]}',
                            'source_group': f'{case["object"]}/{case["demo"]}', 'skill_instruction': case['instruction']})
        dump(out/'records.json', {'source_split': 'training only', 'validation_status': 'passed',
                                  'reference_checks': references, 'records': records})
        row['status'] = 'completed'
        return row
    except BaseException as exc:
        row.update(status='runtime_error', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        dump(out/'summary.json', row)


def summarize(directory, plan):
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob('cases/*/summary.json'))]
    expected = len(plan['pilot_indices']) if plan['stage'] == 'pilot' else len(plan['cases'])
    errors = sum(r.get('status') == 'runtime_error' for r in rows)
    summary = {'status': 'completed' if len(rows) == expected and not errors else 'incomplete' if errors else 'in_progress',
               'stage': plan['stage'], 'cases': len(rows), 'planned_cases': expected,
               'runtime_error_cases': errors,
               'eligible_cases': sum(r['eligible'] for r in rows),
               'variation_successes': sum(r['variation_successes'] for r in rows),
               'variation_failures': sum(r['variation_failures'] for r in rows),
               'expansion_gate': can_expand(rows, plan['objects']),
               'limits': plan['limits']}
    dump(directory/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--require-eligible', action='store_true', help='Smoke gate: exit nonzero for an invalid reference')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed after submission: '+name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(args.out_dir, plan)
        return
    if args.case_index is None or not 0 <= args.case_index < len(plan['cases']):
        parser.error('--case-index must identify one planned source')
    case = plan['cases'][args.case_index]
    row = run_case(plan, case, args.case_index, args.out_dir/'cases'/f'{args.case_index:05d}')
    if args.require_eligible and not row['eligible']:
        raise RuntimeError('Fresh expert smoke did not pass generation/replay validation: '+row['status'])


if __name__ == '__main__':
    main()
