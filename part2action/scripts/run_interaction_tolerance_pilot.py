"""CPU-only, replay-gated grasp perturbations; no learned policy or encoder loads.

Initial pilot, not a benchmark or a training-ready observation dataset. Native
controller gains/frame cadence are not saved in HDF5. The explicit replay recipe
below is a hypothesis tested against recorded post-action TCP/finger states.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import random
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from diagnostics.interaction_tolerances import (
    conditions, controller_gains, grasp_window, perturb_actions, replay_fidelity_passes, tracking_metrics,
)
from diagnostics.physics_collision import PhysicsCollisionLogger

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {'tcp_rmse_m': .01, 'tcp_p95_m': .02,
              'orientation_p95_deg': 10., 'finger_rmse_m': .005,
              'closure_tcp_error_m': .01}


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def entries(value):
    if isinstance(value, dict):
        if 'ep_id' in value and 'chain_params' in value:
            yield value
        else:
            for child in value.values(): yield from entries(child)
    elif isinstance(value, list):
        for child in value: yield from entries(child)


def dump(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n'); temp.replace(path)


def select_cases(data, objects, per_object, seed):
    """Training-only, round-robin over part/instance groups, before any replay."""
    metadata = {str(x['ep_id']): x for x in entries(json.loads((data/'episodes_meta_train.json').read_text()))}
    rng = random.Random(seed); selected = []; skipped = defaultdict(int)
    for obj in objects:
        groups = defaultdict(list)
        with h5py.File(data/'demos'/f'{obj}.hdf5', 'r') as h:
            for key in sorted(h['data'], key=lambda x: int(x.removeprefix('demo_'))):
                d = h['data'][key]
                if 'metadata/ep_id' not in d:
                    skipped['missing episode metadata'] += 1
                    continue
                ep = decode(d['metadata/ep_id'][()]); info = metadata.get(ep)
                if not info or not info['chain_params'] or info['chain_params'][0]['skill_name'] != 'grasp_obj':
                    continue
                try:
                    texts = [decode(x) for x in d['skill_instructions'][:]]
                    window = grasp_window(d['actions'][:], texts)
                    if window['end'] - window['close'] < 5: raise ValueError('Too little closure history')
                    part = info['chain_params'][0]['params'].get('part_grasp', '')
                    object_id = decode(d['metadata/obj_id'][()])
                    groups[(part, object_id)].append({'object': obj, 'demo': key,
                        'episode_id': ep, 'object_id': object_id, 'part': part,
                        'instruction': texts[0], 'window': window})
                except (ValueError, KeyError) as exc:
                    skipped[str(exc)] += 1
        keys = sorted(groups); rng.shuffle(keys)
        for group in groups.values(): rng.shuffle(group)
        count = 0
        while count < per_object and keys:
            remaining = []
            for key in keys:
                if count >= per_object: break
                if groups[key]: selected.append(groups[key].pop()); count += 1
                if groups[key]: remaining.append(key)
            keys = remaining
        if count != per_object: raise ValueError(f'{obj}: only {count} eligible source demos')
    return selected, dict(skipped)


def manifest_cases(previous, indices=None):
    """Subset a fixed selection without changing any case's construction seed."""
    cases = previous['cases']
    chosen = list(range(len(cases))) if indices is None else indices
    if not chosen or len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(cases) for i in chosen):
        raise ValueError('Case indices must be nonempty, unique and within the source manifest')
    return [dict(cases[i], construction_seed=cases[i].get('construction_seed', previous['seed'] + i))
            for i in chosen]


class Scene:
    def __init__(self, data, case, seed):
        sys.path.insert(0, str(ROOT/'PartInstruct'))
        from PartInstruct.PartGym.env.backend.bullet_sim import BtWorld
        from PartInstruct.PartGym.env.backend.planner.panda_arm import PandaArm
        from PartInstruct.PartGym.env.backend.utils.scene_utils import load_grasping_setups, load_grasping_object
        from PartInstruct.PartGym.env.backend.utils.semantic_parser import SemanticParser
        random.seed(seed); np.random.seed(seed)
        with h5py.File(data/'demos'/f'{case["object"]}.hdf5', 'r') as h:
            d = h['data'][case['demo']]
            self.actions = d['actions'][:case['window']['end']].astype(float)
            self.tcp = d['obs/tcp_pose'][:len(self.actions)]
            self.fingers = d['obs/gripper_state'][:len(self.actions)].reshape(-1)
            meta = {k: v[()] for k, v in d['metadata'].items()}
        self.world = BtWorld(gui=False, control_hz=20.)
        self.world.set_gravity([0, 0, -9.81]); assets = data/'assets'
        self.table, self.floor = load_grasping_setups(self.world, assets/'urdfs/floor/floor.urdf', assets/'urdfs/wood/wood.urdf')
        self.robot = PandaArm(self.world, [.6, 0., .05], str(assets/'urdfs/robots/franka_panda/panda.urdf'))
        self.robot.reset()
        obj_path = assets/'partnet-grasping'/case['object']/case['object_id']
        pose = np.asarray(meta['obj_pose']).tolist()
        self.obj = load_grasping_object(self.world, obj_path/'mobility.urdf', pose[4:], pose[:4], float(meta['obj_scale']))
        parser = SemanticParser(self.world); parser.set_obj(self.obj)
        parser.load_part_hierarchy(str(obj_path), scale=float(meta['obj_scale']), num_points=3000)
        self.part = case['part'] or parser.obj_name
        if self.part not in parser.part_pcds: raise ValueError('Unsupported target part: ' + self.part)
        xyz, quat = self.world.p.getBasePositionAndOrientation(self.obj.uid)
        points = np.asarray(parser.part_pcds[self.part])
        self.local_part = Rotation.from_quat(quat).inv().apply(points - xyz)
        self.tree = cKDTree(self.local_part)
        self.initial_z = float(xyz[2])
        self.state = self.world.save_state()
        self.python_state = (self.world.sim_time, self.robot.t, copy.deepcopy(np.random.get_state()), random.getstate())
        self.initial = {'object_xyz': list(xyz), 'object_quat_xyzw': list(quat),
                        'robot_joints': self.robot.get_joint_states(),
                        'finger_position_m': float(self.robot.get_gripper_state())}
        self.parser = SimpleNamespace(obj_name=parser.obj_name, part_pcds={}, update_part_pcds=self.part_points)

    def part_points(self):
        xyz, quat = self.world.p.getBasePositionAndOrientation(self.obj.uid)
        points = Rotation.from_quat(quat).apply(self.local_part) + xyz
        self.parser.part_pcds = {self.part: points, self.parser.obj_name: points}
        return self.parser.part_pcds

    def native_grasp_equivalent(self):
        # Exactly the two world checks used by BulletPlanner.predicate_grasping;
        # no active predicate is allowed to advance physics here.
        points = self.part_points()[self.part]
        return bool(any(self.world.check_obj_gripper_collision(self.robot.panda, finger, points, .003, 20.)
                        for finger in [self.robot.LEFTFINGERIDX, self.robot.RIGHTFINGERIDX]))

    def restore(self):
        self.world.restore_state(self.state)
        self.world.sim_time, self.robot.t = self.python_state[:2]
        np.random.set_state(copy.deepcopy(self.python_state[2])); random.setstate(self.python_state[3])
        # No env/task/policy history is present. apply_action overwrites all arm
        # and finger motor targets before the first subsequent physics step.

    def close(self):
        if hasattr(self, 'state'): self.world.p.removeState(self.state)
        if hasattr(self, 'world'): self.world.close()


def contact_summary(scene, rows):
    """Force-qualified physical contacts; target membership is geometric proximity."""
    bilateral = []; targeted = []
    for row in rows:
        fingers = set(); target = False
        rot = Rotation.from_quat(row['object_quat_xyzw'])
        for c in row['contacts']:
            if not c['is_finger'] or c['normal_force_n'] < .1: continue
            fingers.add(c['robot_link'])
            local = rot.inv().apply(np.asarray(c['position_on_object']) - row['object_xyz'])
            target |= bool(scene.tree.query(local)[0] <= .005)
        bilateral.append(len(fingers) == 2); targeted.append(target)
    return {'bilateral_fraction': float(np.mean(bilateral)) if rows else 0.,
            'target_contact_fraction': float(np.mean(targeted)) if rows else 0.}


def replay(scene, case, condition, output, capture_frame=None, camera_config=None):
    from diagnostics.interaction_records import observe, candidate_vector, outcome_components
    scene.restore(); window = case['window']
    actions = perturb_actions(scene.actions, window, condition)
    schedule = controller_gains(scene.actions, window)
    logger = PhysicsCollisionLogger(scene)
    tcp = []; fingers = []; nominal_close_tcp = None; closure_native = False
    closure_rows_start = 0
    observations=[];observation_frames=[]
    try:
        onset = int(np.flatnonzero(actions[:, 6] <= .02)[0])
        for t, action in enumerate(actions):
            if capture_frame is not None and t in {max(0,capture_frame-1),capture_frame}:
                observations.append(observe(scene,camera_config));observation_frames.append(t)
            gain, gripper_gain, near = schedule[t]
            logger.action_index = t; logger.command = action.tolist()
            scene.robot.apply_action(action, gain=gain, gain_gripper=gripper_gain, gain_near_target=near)
            scene.world.step()
            pose = scene.robot.get_tcp_pose(); tcp.append(pose.to_list().tolist()); fingers.append(scene.robot.get_gripper_state())
            if t == window['close']: nominal_close_tcp = pose.translation.copy()
            if t == onset: closure_rows_start = len(logger.rows)
        closure_native = scene.native_grasp_equivalent()
        prelift_count = len(logger.rows)
        # Fixed, explicitly additional continuation: 5 cm world-Z lift over 1 s,
        # then hold for 0.5 s. It is not part of the source expert demonstration.
        pose = scene.robot.get_tcp_pose(); origin = pose.translation.copy()
        rotation = pose.rotation.as_euler('xyz'); last_grip = float(actions[-1, 6])
        for t in range(30):
            xyz = origin + np.array([0, 0, .05 * min(1., (t + 1) / 20)])
            action = np.r_[xyz, rotation, last_grip]
            logger.action_index = len(actions) + t; logger.command = action.tolist()
            scene.robot.apply_action(action, gain=.01, gain_gripper=.01, gain_near_target=.01)
            scene.world.step()
        hold_rows = logger.rows[-72:]  # Last 0.3 s, 240 Hz physics.
        contact = contact_summary(scene, hold_rows)
        xyz, quat = scene.world.p.getBasePositionAndOrientation(scene.obj.uid)
        rel = [Rotation.from_quat(r['tcp_quat_xyzw']).inv().apply(np.asarray(r['object_xyz']) - r['tcp_xyz']) for r in hold_rows]
        drift = float(np.max(np.linalg.norm(np.asarray(rel) - rel[0], axis=1)))
        lift = float(xyz[2] - scene.initial_z)
        stable = bool(lift >= .03 and contact['bilateral_fraction'] >= .8 and
                      contact['target_contact_fraction'] >= .8 and drift <= .01)
        metrics = tracking_metrics(tcp, scene.tcp, fingers, scene.fingers)
        metrics['closure_tcp_error_m'] = float(np.linalg.norm(nominal_close_tcp - scene.tcp[window['close'], 4:7]))
        row = {'condition': condition, 'tracking': metrics,
               'native_grasp_predicate_equivalent': closure_native,
               'lift_m': lift, 'hold_relative_position_drift_m': drift,
               'hold': contact, 'stable_target_grasp_lift': stable,
               'closure_contacts': contact_summary(scene, logger.rows[closure_rows_start:prelift_count]),
               'commanded_close_frame': onset, 'commanded_close_s': onset / 20.,
               'achieved_tcp_at_nominal_close': tcp[window['close']],
               'replay_fidelity_pass': replay_fidelity_passes(metrics, THRESHOLDS),
               'native_grasp_scope': 'Equivalent native force/part-distance predicate at segment end; not full-task success',
               'contact_scope': 'Physical finger/object contacts; target membership within 5 mm of sampled GT part surface'}
        row['outcome_components']=outcome_components(row)
        assert all(row['outcome_components']['passes'].values())==stable
        row['candidate_parameters']=candidate_vector(condition).tolist()
        row['result_file']=output.with_suffix('.json').name
        row['measured_finger_le_002_frame']=next((t for t,value in enumerate(fingers) if value<=.02),None)
        row['first_force_qualified_finger_object_contact_s']=next((r['elapsed_s'] for r in logger.rows if any(c['is_finger'] and c['normal_force_n']>=.1 for c in r['contacts'])),None)
        row['achieved_tcp_at_commanded_close']=tcp[onset]
        row['achieved_tcp_at_grasp_end']=tcp[-1]
        if capture_frame is not None:
            if not observations:raise RuntimeError('Pre-intervention observation was not captured')
            if len(observations)==1:
                observations.insert(0,observations[0]);observation_frames.insert(0,observation_frames[0])
            sensory=output.with_suffix('.observations.npz')
            np.savez_compressed(sensory,**{key:np.stack([x[key] for x in observations]) for key in observations[0]},
                                frame=np.asarray(observation_frames),time_s=np.asarray(observation_frames)/20.)
            row.update({'record_schema_version':1,'observation_file':sensory.name,
                        'observation_timing':'Two frames BEFORE the first changed command across this candidate set',
                        'decision_frame':capture_frame,
                        'primitive_context':np.r_[scene.actions[window['close']],(window['close']-capture_frame)/20.].tolist()})
        np.savez_compressed(output.with_suffix('.npz'), tcp=np.asarray(tcp), fingers=np.asarray(fingers), actions=actions)
        with gzip.open(output.with_suffix('.physics.json.gz'), 'wt') as f: json.dump(logger.rows, f)
        dump(output.with_suffix('.json'), row)
        return row, np.asarray(tcp)
    finally:
        logger.close()


def report(out, results, manifest):
    """Incremental report; missing/invalid replay never enters robustness counts."""
    dump(out/'results.json', results)
    eligible = sum(row.get('eligible', False) for row in results)
    trials = [trial for case in results for trial in case.get('perturbations', [])]
    summary = {'cases': len(results), 'planned_cases': len(manifest['cases']),
               'eligible_cases': eligible, 'perturbation_trials': len(trials),
               'status': 'in_progress',
               'conditions': {c['name']: {'trials':sum(t['condition']['name']==c['name'] for t in trials),
                                         'successes':sum(t['condition']['name']==c['name'] and t['stable_target_grasp_lift'] for t in trials)} for c in manifest['conditions'][1:]},
               'limits': manifest['limits']}
    dump(out/'summary.json', summary)
    with (out/'per_trial.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['object','demo','part','eligible','phase','condition','repeat','fidelity_pass','native_grasp','stable_lift','tcp_rmse_m','closure_tcp_error_m'])
        writer.writeheader()
        for case in results:
            for phase in ['baseline', 'perturbations']:
                for i,t in enumerate(case.get(phase, [])):
                    writer.writerow(dict(object=case['case']['object'], demo=case['case']['demo'], part=case['case']['part'],eligible=case.get('eligible',False),phase=phase,condition=t['condition']['name'],repeat=t['repeat'],fidelity_pass=t['replay_fidelity_pass'],native_grasp=t['native_grasp_predicate_equivalent'],stable_lift=t['stable_target_grasp_lift'],tcp_rmse_m=t['tracking']['tcp_rmse_m'],closure_tcp_error_m=t['tracking']['closure_tcp_error_m']))
    sys.path.insert(0, str(ROOT/'part2action/docs/slide_history/scripts'))
    from export_slide_tables import render
    date = manifest.get('media_date', '2026-10-30')
    media = out/'media'/date; media.mkdir(parents=True, exist_ok=True)
    render([['Replay-gated pilot','Count'],['Completed source cases',f'{len(results)} / {len(manifest["cases"])}'],
            ['Validated reference cases',str(eligible)],['Perturbation trials',str(len(trials))]], [.70,.30], media/'replay_status.png')
    text = f'<!doctype html><title>Interaction tolerance pilot</title><h1>Replay-gated grasp pilot</h1><img width="1000" src="media/{date}/replay_status.png"><p>Use this image: <a href="media/{date}/replay_status.png">replay_status.png</a></p><p><a href="per_trial.csv">Per-trial CSV</a> | <a href="summary.json">Summary</a> | <a href="manifest.json">Protocol</a></p><p>Invalid reference replay is excluded, not labeled as perturbation failure. No unvalidated case enters tolerance data.</p>'
    if trials:
        rows = [['Intervention', 'Stable grasp/lift repeats']]
        for c in manifest['conditions'][1:]:
            count = summary['conditions'][c['name']]
            if c['close_offset_steps']:
                label = f'Closure {c["close_offset_steps"]/manifest["control_hz_assumption"]:+g} s'
            else:
                kind = 'translation_m' if any(c['translation_m']) else 'rotation_deg'
                axis = next(i for i, value in enumerate(c[kind]) if value)
                scale, unit = (100, 'cm') if kind == 'translation_m' else (1, 'deg')
                label = f'{"Position" if kind == "translation_m" else "Wrist"} {"XYZ"[axis]} {c[kind][axis]*scale:+g} {unit}'
            rows.append([label, f'{count["successes"]} / {count["trials"]}'])
        render(rows, [.65, .35], media/'perturbation_outcomes.png')
        text += f'<h2>Single-variable outcomes</h2><img width="1000" src="media/{date}/perturbation_outcomes.png"><p>Use this image: <a href="media/{date}/perturbation_outcomes.png">perturbation_outcomes.png</a></p><p>Repeat counts check reproducibility in fixed scenes; they are not independent samples or estimates of general success probability. Fidelity gates apply to references; perturbed trajectories are expected to differ.</p>'
    text += '<ul>' + ''.join('<li>'+x+'</li>' for x in manifest['limits']) + '</ul>'
    (out/'index.html').write_text(text)
    if manifest.get('capture_observations'):
        records=[]
        for case in results:
            if not case.get('eligible'):continue
            info=case['case']
            directory=next(p for p in out.glob('case_*') if p.name.endswith(f'_{info["object"]}_{info["demo"]}'))
            for phase in ['baseline','perturbations']:
                for row in case[phase]:
                    records.append({'observation_file':str(directory.relative_to(out)/row['observation_file']),
                        'result_file':str(directory.relative_to(out)/row['result_file']),
                        'instance_group':f'{info["object"]}/{info["object_id"]}',
                        'source_group':f'{info["object"]}/{info["demo"]}',
                        'skill_instruction':info['instruction'],'phase':phase,'repeat':row['repeat']})
        dump(out/'records.json',{'schema_version':1,'source_split':manifest['split'],'records':records,
            'limits':['Prototype sensor collection; TinyRenderer and deterministic full-scene pixel sampling differ from the standard policy evaluator.',
                      'Nominal primitive is provided by the expert replay plan; later policy integration must supply its own nominal primitive.',
                      'Keep all records from an object instance in one split; one instance is insufficient for held-out training.']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/p/part2action/data/PartInstruct'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--objects', nargs='+', default=['bottle','mug','pliers','scissors'])
    p.add_argument('--per-object', type=int, default=5)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--fresh-scenes', action='store_true', help='Reconstruct the native scene for every trial instead of reusing a saved Bullet world')
    p.add_argument('--cases-manifest', type=Path, help='Reuse a previously selected training-only case list without selecting on replay outcomes')
    p.add_argument('--case-indices', type=int, nargs='+', help='Explicit zero-based subset of --cases-manifest; preserve original construction seeds')
    p.add_argument('--translation-m', type=float, default=.01)
    p.add_argument('--rotation-deg', type=float, default=10.)
    p.add_argument('--close-offset-steps', type=int, default=2)
    p.add_argument('--condition-names',nargs='+',help='Subset of non-identity conditions for a small instrumentation check')
    p.add_argument('--capture-observations',action='store_true')
    args = p.parse_args()
    if args.per_object < 1 or args.repeats < 2: p.error('Require positive cases and at least two exact-restoration repeats')
    if args.case_indices is not None and args.cases_manifest is None:
        p.error('--case-indices requires --cases-manifest')
    try:
        run_conditions = conditions(args.translation_m, args.rotation_deg, args.close_offset_steps)
    except ValueError as exc:
        p.error(str(exc))
    if args.condition_names:
        names={c['name'] for c in run_conditions[1:]}
        if len(set(args.condition_names))!=len(args.condition_names) or not set(args.condition_names)<=names:
            p.error('Condition names must be unique known non-identity conditions')
        run_conditions=[run_conditions[0]]+[c for c in run_conditions[1:] if c['name'] in args.condition_names]
    camera_config=None
    if args.capture_observations:
        from omegaconf import OmegaConf
        camera_config=OmegaConf.load(ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml')
    out = args.out_dir.resolve()
    if out.exists(): raise FileExistsError('Choose a new output directory')
    if args.cases_manifest:
        previous = json.loads(args.cases_manifest.read_text())
        if previous.get('split') != 'training only' or Path(previous['data_root']).resolve() != args.data_root.resolve() or previous['seed'] != args.seed:
            p.error('Case manifest must have matching training-only data and construction seed')
        try:
            cases = manifest_cases(previous, args.case_indices)
        except ValueError as exc:
            p.error(str(exc))
        skipped = previous.get('skipped_selection', {})
    else:
        cases, skipped = select_cases(args.data_root, args.objects, args.per_object, args.seed)
        cases = [dict(case, construction_seed=args.seed+i) for i, case in enumerate(cases)]
    out.mkdir(parents=True)
    date_match = re.search(r'(20\d{2})(\d{2})(\d{2})', out.name)
    media_date = '-'.join(date_match.groups()) if date_match else datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    manifest = {'media_date':media_date, 'seed':args.seed, 'cases':cases, 'skipped_selection':skipped,
        'data_root':str(args.data_root), 'split':'training only', 'repeats':args.repeats,
        'control_hz_assumption':20., 'physics_hz':240., 'fidelity_thresholds':THRESHOLDS,
        'state_restoration': 'fresh native scene for every trial' if args.fresh_scenes else 'saved Bullet state reused',
        'cases_manifest': str(args.cases_manifest) if args.cases_manifest else None,
        'cases_manifest_sha256': hashlib.sha256(args.cases_manifest.read_bytes()).hexdigest() if args.cases_manifest else None,
        'source_case_indices': args.case_indices,
        'conditions': run_conditions, 'baseline_only':args.baseline_only,
        'capture_observations':args.capture_observations,
        'sensor_protocol':'Native 300x300 camera optics; CPU TinyRenderer; 1024 deterministic visible full-scene points; no GT semantic filtering' if args.capture_observations else None,
        'record_schema':{'candidate':['world_dx_m','world_dy_m','world_dz_m','tcp_local_rx_rad','tcp_local_ry_rad','tcp_local_rz_rad','closure_offset_s'],
                         'robot_state':['world_x_m','world_y_m','world_z_m','qx','qy','qz','qw','finger_position_m'],
                         'primitive_context':['nominal_world_x_m','nominal_world_y_m','nominal_world_z_m','nominal_euler_x_rad','nominal_euler_y_rad','nominal_euler_z_rad','nominal_finger_command_m','time_to_nominal_close_s'],
                         'privileged_diagnostics':'Achieved motion, contacts, target membership, source IDs and outcomes are not predictor inputs'},
        'controller':'Native PandaArm.apply_action, one control step per stored action; native-inspired approach/closure gain schedule. Positional argument wiring preserved.',
        'continuation':'Fixed additional world-Z lift 5 cm/1 s plus hold 0.5 s, no replanning',
        'eligibility':'All reference repeats pass trajectory thresholds, native grasp-equivalent, stable target grasp/lift, and <=1e-6 restored TCP repeat difference.',
        'source_sha256': {str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [
            Path(__file__),ROOT/'part2action/diagnostics/interaction_tolerances.py',ROOT/'part2action/diagnostics/interaction_records.py',ROOT/'part2action/diagnostics/physics_collision.py',
            ROOT/'PartInstruct/PartInstruct/PartGym/env/backend/planner/panda_arm.py',ROOT/'PartInstruct/PartInstruct/PartGym/env/backend/utils/scene_utils.py',ROOT/'PartInstruct/PartInstruct/PartGym/config/config_oracle.yaml']},
        'limits':['Native per-frame gains/cadence are absent; reference replay is an explicit unverified controller reconstruction until gated.',
                  'Pilot grasp/lift criterion is not full-task success or a certified safe set.',
                  'Target membership uses sampled GT geometry and a 5 mm distance rule, not a semantic contact sensor.',
                  'Perturbations are conditional on this scene, fixed controller and continuation; failed references are excluded.',
                  'Captured observations, if enabled, are a prototype sensor protocol; multiple instances and held-out validation remain necessary.',
                  'No learned models, GPU allocation, training-stride change or existing evaluation modifications.']}
    dump(out/'manifest.json',manifest)
    if args.prepare_only:
        print(json.dumps({'selected_cases':len(cases),'skipped':skipped,'manifest':str(out/'manifest.json')}));return
    results = []
    for i, case in enumerate(cases):
        construction_seed = case['construction_seed']
        row = {'case':case,'baseline':[],'perturbations':[],'eligible':False}
        scene = None; directory = out/f'case_{i:03d}_{case["object"]}_{case["demo"]}';directory.mkdir()
        try:
            scene = Scene.__new__(Scene)
            scene.__init__(args.data_root,case,construction_seed)
            capture_frame=None
            if args.capture_observations:
                from diagnostics.interaction_records import decision_frame
                capture_frame=decision_frame(scene.actions,case['window'],run_conditions)
            dump(directory/'initial_state.json',scene.initial)
            reference = None; repeat_error = 0.
            for repeat in range(args.repeats):
                if args.fresh_scenes and repeat > 0:
                    scene.close()
                    scene = Scene.__new__(Scene)
                    scene.__init__(args.data_root,case,construction_seed)
                r, trajectory = replay(scene,case,run_conditions[0],directory/f'baseline_{repeat}',capture_frame,camera_config)
                r['repeat'] = repeat; row['baseline'].append(r)
                from diagnostics.interaction_records import motion_delta
                r['achieved_delta_vs_reference_at_nominal_close']=motion_delta(trajectory,reference if reference is not None else trajectory,case['window']['close'])
                dump(directory/f'baseline_{repeat}.json',r)
                if reference is not None: repeat_error = max(repeat_error,float(np.max(np.abs(trajectory-reference))))
                reference = trajectory
            row['restored_repeat_max_tcp_difference'] = repeat_error
            row['eligible'] = bool(repeat_error <= 1e-6 and all(r['replay_fidelity_pass'] and r['native_grasp_predicate_equivalent'] and r['stable_target_grasp_lift'] for r in row['baseline']))
            if row['eligible'] and not args.baseline_only:
                for condition in run_conditions[1:]:
                    for repeat in range(args.repeats):
                        if args.fresh_scenes:
                            scene.close()
                            scene = Scene.__new__(Scene)
                            scene.__init__(args.data_root,case,construction_seed)
                        r,trajectory = replay(scene,case,condition,directory/f'{condition["name"]}_{repeat}',capture_frame,camera_config)
                        r['achieved_delta_vs_reference_at_nominal_close']=motion_delta(trajectory,reference,case['window']['close'])
                        r['repeat']=repeat;row['perturbations'].append(r)
                        dump(directory/f'{condition["name"]}_{repeat}.json',r)
        except Exception as exc:
            import traceback
            row['error'] = repr(exc);row['traceback'] = traceback.format_exc()
        finally:
            if scene is not None: scene.close()
        results.append(row);report(out,results,manifest)
        print(f'[pilot] {i+1}/{len(cases)} {case["object"]}/{case["demo"]} eligible={row["eligible"]} perturbations={len(row["perturbations"])} error={row.get("error")}',flush=True)
    summary=json.loads((out/'summary.json').read_text())
    summary['status']='completed' if not any(r.get('error') for r in results) else 'completed_with_case_errors'
    summary['decision']='reference_validation_only' if args.baseline_only else ('perturbation_data_collected' if summary['perturbation_trials'] else 'no_perturbation_labels_reference_gate_not_passed')
    dump(out/'summary.json',summary)
    print('[pilot] finished',json.dumps(summary),flush=True)
    if any(r.get('error') for r in results):raise SystemExit(2)


if __name__ == '__main__': main()
