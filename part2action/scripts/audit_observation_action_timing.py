#!/usr/bin/env python3
"""Inspect released-demo timing without changing data or training labels.

Scan low-dimensional transitions, export flat annotated GIFs, and optionally
replay the opening prefix in CPU PyBullet. A positional target is not a measured
position: target-to-TCP distance alone is deliberately not an ordering test.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OBJECTS = ('bottle', 'mug', 'pliers', 'scissors')


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def inspect_transition(actions, tcp_pose, gripper, stride=2):
    """Use the first >5cm command departure and >1cm measured departure.

    Thresholds identify the initial approach, not contact. The lag is a
    descriptive statistic; only replay/source tracing can establish causality.
    """
    actions = np.asarray(actions, dtype=float)
    tcp = np.asarray(tcp_pose, dtype=float)[:, 4:7]
    gripper = np.asarray(gripper, dtype=float).reshape(-1)
    if len(actions) != len(tcp) or len(actions) != len(gripper):
        raise ValueError('Actions, TCP observations and gripper lengths differ')
    if len(actions) < 2 or not all(np.isfinite(x).all() for x in (actions, tcp, gripper)):
        raise ValueError('Need at least two finite recorded frames')
    command_departure = np.linalg.norm(actions[:, :3] - actions[0, :3], axis=1)
    measured_departure = np.linalg.norm(tcp - tcp[0], axis=1)
    command_indices = np.flatnonzero(command_departure > .05)
    motion_indices = np.flatnonzero(measured_departure > .01)
    command = int(command_indices[0]) if len(command_indices) else None
    motion = int(motion_indices[0]) if len(motion_indices) else None
    step = np.r_[0., np.linalg.norm(np.diff(tcp, axis=0), axis=1)]
    return {
        'frames': len(actions), 'first_approach_command': command,
        'first_measured_departure': motion,
        'motion_minus_command_frames': motion-command if command is not None and motion is not None else None,
        'motion_at_command_cm': float(step[command]*100) if command is not None else None,
        'finger_at_command_mm': float(gripper[command]*1000) if command is not None else None,
        'preceding_observation_index': command-1 if command is not None and command > 0 else None,
        # Loader starts at zero; restrict to the first target slot. Future
        # action-chunk slots can still include the command when this is false.
        'preceding_observation_sampled_stride2': bool((command-1) % stride == 0) if command is not None and command > 0 else None,
        'same_index_transition_sampled_stride2': bool(command % stride == 0) if command is not None else None,
    }


def load_demo(path, key, images=False):
    with h5py.File(path, 'r') as handle:
        demo = handle['data'][key]
        result = {
            'actions': demo['actions'][:], 'tcp': demo['obs/tcp_pose'][:],
            'gripper': demo['obs/gripper_state'][:].reshape(-1),
            'instructions': [decode(v) for v in demo['skill_instructions'][:]],
            'metadata': {k: v[()] for k, v in demo['metadata'].items()},
        }
        if images:
            result['rgb'] = demo['obs/agentview_rgb'][:]
    return result


def scan(data_root, objects):
    from ground_truth_reference import metadata_entries
    metadata = {}
    for name in ('episodes_meta_train.json', 'episodes_meta_test.json'):
        for entry in metadata_entries(json.loads((data_root/name).read_text())):
            metadata[str(entry['ep_id'])] = entry
    records, skipped, selected = [], [], []
    for obj in objects:
        path = data_root/'demos'/f'{obj}.hdf5'
        representatives = {}
        with h5py.File(path, 'r') as handle:
            keys = sorted(handle['data'], key=lambda k: int(k[5:]))
            for key in keys:
                demo = handle['data'][key]
                try:
                    actions, tcp, grip = demo['actions'][:], demo['obs/tcp_pose'][:], demo['obs/gripper_state'][:]
                    instructions = demo['skill_instructions']
                    instruction = decode(instructions[0])
                    stats = inspect_transition(actions, tcp, grip)
                    ep = decode(demo['metadata/ep_id'][()])
                    chain = metadata[ep]['chain_params']
                    skill = {'touch_obj': 'touch', 'grasp_obj': 'grasp'}.get(chain[0]['skill_name'], 'other')
                except (KeyError, ValueError, IndexError) as error:
                    skipped.append({'object': obj, 'demo': key, 'reason': str(error)})
                    continue
                records.append({'object': obj, 'demo': key, 'episode_id': ep,
                                'initial_skill': skill, 'instruction': instruction, **stats})
                if skill in ('touch', 'grasp') and stats['first_approach_command'] is not None:
                    representatives.setdefault(skill, key)
        # Include the exact expert reference from the reported stalled mug case.
        if obj == 'mug':
            representatives['touch'] = 'demo_1544'
            representatives['grasp'] = 'demo_1821'
        selected += [(obj, key) for key in representatives.values()]
        print(f'[scan] {obj}: {sum(r["object"] == obj for r in records)} valid demos', flush=True)
    return records, skipped, selected


def replay_prefix(data_root, obj, data, transition):
    """Run PartGym's real robot/controller/physics, without encoders/rendering.

    Reconstruct the source scene, including object settling and robot reset.
    Stop a few frames into approach, before interaction with the object. Gains
    follow the expert planner's finger-control and initial approach recipes;
    they are explicit assumptions, not fitted to the dataset.
    """
    sys.path.insert(0, str(REPO/'PartInstruct'))
    from PartInstruct.PartGym.env.backend.bullet_sim import BtWorld
    from PartInstruct.PartGym.env.backend.planner.panda_arm import PandaArm
    from PartInstruct.PartGym.env.backend.utils.scene_utils import load_grasping_setups, load_grasping_object

    meta = data['metadata']
    assets = data_root/'assets'
    world = BtWorld(gui=False, control_hz=20.)
    try:
        world.set_gravity([0., 0., -9.81])
        load_grasping_setups(world, str(assets/'urdfs/floor/floor.urdf'), str(assets/'urdfs/wood/wood.urdf'))
        robot = PandaArm(world, [.6, 0., .05], str(assets/'urdfs/robots/franka_panda/panda.urdf'))
        robot.reset()
        pose = np.asarray(meta['obj_pose']).tolist()
        load_grasping_object(world, assets/'partnet-grasping'/obj/decode(meta['obj_id'])/'mobility.urdf',
                            pose[4:], pose[:4], float(meta['obj_scale']))
        count = min(len(data['actions']), transition+4)
        rows = []
        for t in range(count):
            before = robot.get_tcp_pose().translation.copy()
            before_finger = float(robot.get_gripper_state())
            gain = .0001 if t < transition else .005
            # Equivalent to BulletEnv._take_action + its world.step.
            # Call the real apply_action to preserve even its argument wiring.
            robot.apply_action(data['actions'][t], gain=gain, gain_gripper=.01, gain_near_target=.2)
            world.step()
            after = robot.get_tcp_pose().translation.copy()
            after_finger = float(robot.get_gripper_state())
            recorded = data['tcp'][t, 4:7]
            recorded_finger = float(data['gripper'][t])
            rows.append({
                'frame': t, 'gain': gain, 'recorded_tcp_xyz': recorded.tolist(),
                'before_tcp_xyz': before.tolist(), 'after_tcp_xyz': after.tolist(),
                'recorded_finger_mm': recorded_finger*1000,
                'before_finger_mm': before_finger*1000, 'after_finger_mm': after_finger*1000,
                'before_tcp_error_mm': float(np.linalg.norm(before-recorded)*1000),
                'after_tcp_error_mm': float(np.linalg.norm(after-recorded)*1000),
                'before_to_previous_recorded_tcp_error_mm': float(np.linalg.norm(before-data['tcp'][t-1, 4:7])*1000) if t else None,
                'before_finger_error_mm': abs(before_finger-recorded_finger)*1000,
                'after_finger_error_mm': abs(after_finger-recorded_finger)*1000,
            })
        return {'backend': 'PyBullet DIRECT, CPU; real PartGym PandaArm/BtWorld',
                'scope': 'Initial prefix only; no full expert success replay',
                'interpretation': 'Ordering is verified at the first approach transition. One control step per released frame does not reproduce every touch-prefix state; native sampling/controller history remains unresolved.',
                'gains': {'hold': .0001, 'approach': .005, 'finger': .01, 'near_target_argument': .2},
                'rows': rows,
                'first_transition': rows[transition],
                'mean_before_tcp_error_mm': float(np.mean([r['before_tcp_error_mm'] for r in rows])),
                'mean_after_tcp_error_mm': float(np.mean([r['after_tcp_error_mm'] for r in rows])),
                'mean_before_finger_error_mm': float(np.mean([r['before_finger_error_mm'] for r in rows])),
                'mean_after_finger_error_mm': float(np.mean([r['after_finger_error_mm'] for r in rows]))}
    finally:
        world.p.disconnect()


def frame_rows(obj, key, data):
    rows = []
    for t, action in enumerate(data['actions']):
        tcp = data['tcp'][t, 4:7]
        previous_tcp = data['tcp'][max(0, t-1), 4:7]
        following = data['actions'][t+1] if t+1 < len(data['actions']) else None
        row = {'object': obj, 'demo': key, 'frame': t, 'sampled_stride2': t % 2 == 0,
               'instruction': data['instructions'][t],
               'tcp_step_cm': float(np.linalg.norm(tcp-previous_tcp)*100),
               'finger_measured_mm': float(data['gripper'][t]*1000),
               'current_action_index': t, 'next_action_index': t+1 if following is not None else None,
               'current_target_from_initial_cm': float(np.linalg.norm(action[:3]-data['actions'][0, :3])*100)}
        for axis, i in zip('xyz', range(3)):
            row[f'tcp_{axis}_m'] = float(tcp[i])
            row[f'current_action_{axis}_m'] = float(action[i])
            row[f'next_action_{axis}_m'] = float(following[i]) if following is not None else None
        row['current_gripper_command_mm'] = float(action[6]*1000)
        row['next_gripper_command_mm'] = float(following[6]*1000) if following is not None else None
        rows.append(row)
    return rows


def export_gif(out, obj, key, data, transition):
    import cv2
    import imageio.v2 as imageio

    def text(canvas, string, x, y, color=(230, 230, 230), scale=.5):
        cv2.putText(canvas, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def command_state(index):
        if index >= len(data['actions']):
            return 'END (no next target)'
        return 'APPROACH' if np.linalg.norm(data['actions'][index, :3]-data['actions'][0, :3]) > .05 else 'HOLD NEAR START'

    frames = []
    for t in range(max(0, transition-5), min(len(data['actions'])-1, transition+7)):
        previous = max(0, t-1)
        canvas = np.full((650, 1440, 3), (20, 25, 34), dtype=np.uint8)
        text(canvas, f'{obj} / {key} | expert frames only | observation index {t}', 20, 27, scale=.7)
        text(canvas, data['instructions'][t][:130], 20, 52)
        # Three panels deliberately show the same recorded RGB. These compare
        # label alignment, not policies or a newly executed action.
        for panel, heading in enumerate(('GROUND-TRUTH RECORDED FRAME', 'CURRENT TRAINING PAIR', 'NEXT-ACTION CANDIDATE')):
            x = panel*480
            text(canvas, heading, x+12, 81, (120, 220, 240), .55)
            canvas[95:485, x+45:x+435] = cv2.resize(data['rgb'][t], (390, 390))
            if panel == 0:
                text(canvas, f'TCP change since previous frame: {100*np.linalg.norm(data["tcp"][t,4:7]-data["tcp"][previous,4:7]):.2f} cm', x+12, 510)
                text(canvas, f'Measured finger: {1000*data["gripper"][t]:.3f} mm', x+12, 535)
            else:
                a = t if panel == 1 else t+1
                state = command_state(a)
                color = (255, 195, 95) if state == 'APPROACH' else (170, 190, 210)
                text(canvas, f'Input obs[{previous}], obs[{t}] -> action[{a}]', x+12, 510)
                text(canvas, state, x+12, 535, color, .7)
                text(canvas, f'Target XYZ: {np.array2string(data["actions"][a,:3],precision=3)}', x+12, 560)
                text(canvas, f'Finger command: {data["actions"][a,6]*1000:.1f} mm', x+12, 585)
        text(canvas, f'Stride 2: this observation is {"INCLUDED" if t%2==0 else "SKIPPED"} as a training sample (first action slot).', 20, 618, (255, 210, 120))
        text(canvas, 'Identical expert images in all panels. No predicted rollout; candidate labels have not been applied to training.', 20, 642, scale=.48)
        frames.append(canvas)
    gif = out/f'{obj}_{key}_timing.gif'
    imageio.mimsave(gif, frames, duration=700, loop=0)
    return gif.name


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_evidence():
    specs = [
        ('PartInstruct/PartInstruct/PartGym/env/bullet_env.py', 'def step(', 10),
        ('PartInstruct/PartInstruct/PartGym/env/bullet_env.py', 'def planning_step(', 145),
        ('part2action/data/partinstruct_loader.py', 'frame_indices =', 3),
        ('part2action/data/partinstruct_loader.py', 'actions_full = np.asarray', 3),
    ]
    result = []
    for relative, needle, length in specs:
        lines = (REPO/relative).read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if needle in line)
        result.append({'file': relative, 'line': start+1, 'excerpt': '\n'.join(lines[start:start+length])})
    return result


def write_review(out, report):
    cards = []
    for case in report['examples']:
        replay = case.get('replay')
        replay_text = ''
        if replay:
            r = replay['first_transition']
            replay_text = f'<p>At the approach transition, recorded TCP differs from replay BEFORE the command by {r["before_tcp_error_mm"]:.3f} mm and AFTER by {r["after_tcp_error_mm"]:.3f} mm. Whole-prefix finger error: before {replay["mean_before_finger_error_mm"]:.3f} mm; after {replay["mean_after_finger_error_mm"]:.3f} mm.</p>'
        s = case['stats']
        cards.append(f'<article><h2>{html.escape(case["object"]+" / "+case["demo"])}</h2><p>{html.escape(case["instruction"])}</p><img src="{case["gif"]}" alt="Expert frame and action-label comparison"><p>First approach command: {s["first_approach_command"]}; first measured departure: {s["first_measured_departure"]}; preceding frame included at stride 2: {s["preceding_observation_sampled_stride2"]}.</p>{replay_text}</article>')
    scan = report['scan_summary']
    explanation = ('The recorded TCP at the first approach transition matches the state AFTER execution of the same-index action in the verified CPU examples. For a causal next-action policy, use obs[t] to predict actions starting at t+1. This has not been applied to training.' if report['ordering_status'] == 'post_action_confirmed_in_replayed_examples' else 'The data transition patterns require further investigation; do not shift labels based on this report alone.')
    page = f'''<!doctype html><meta charset="utf-8"><title>Observation/action timing audit</title>
<style>body{{font:17px system-ui;background:#111923;color:#e4eaf2;max-width:1440px;margin:35px auto;padding:20px}}article{{margin-top:35px;border-top:1px solid #455265}}img{{width:100%}}a{{color:#90d5ff}}p{{line-height:1.5}}</style>
<h1>Observation/action timing audit</h1><p>{html.escape(explanation)}</p>
<p>Scanned {scan['valid_demos']} demonstrations across {len(scan['objects'])} objects. Among {scan['with_approach']} with an initial approach command, {scan['same_frame_motion']} first exceed 1 cm measured travel at that command index; {scan['preceding_frame_skipped_stride2']} preceding observations are omitted by stride 2. These counts describe data; replay establishes ordering only for the replayed examples.</p>
<p>The three GIF panels use identical expert images and compare target indexing. They are not three executed policies. Action chunks can contain transition commands in later slots even when the preceding observation is skipped as a first-slot training sample. No training data, model, or evaluations were changed.</p>
<p>Replay scope: first approach timing. Grasp prefixes reproduce closely; touch prefixes show later TCP/finger differences when replayed with one 20 Hz control step per released frame. Native sampling/controller history is unresolved, so this is not a complete expert-trajectory reproduction or proof that every stored frame has the same duration.</p>
<p><a href="transition_frames.csv">Frame table</a> · <a href="demo_transitions.csv">All-demo transition table</a> · <a href="timing_report.json">Full report / source evidence</a> · <a href="timing_review.zip">Download this review</a></p>{''.join(cards)}'''
    (out/'index.html').write_text(page)
    with zipfile.ZipFile(out/'timing_review.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.html', '.gif', '.json', '.csv'):
                bundle.write(path, path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=REPO/'datasets/PartInstruct')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--objects', nargs='+', choices=OBJECTS, default=list(OBJECTS))
    parser.add_argument('--replay', action='store_true', help='CPU replay of every selected prefix; use partgym_rollout environment')
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    out = args.out_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing review: {out}')
    out.mkdir(parents=True, exist_ok=True)
    records, skipped, selected = scan(data_root, args.objects)
    write_csv(out/'demo_transitions.csv', records)
    examples, all_frames = [], []
    for obj, key in selected:
        print(f'[example] {obj}/{key}', flush=True)
        data = load_demo(data_root/'demos'/f'{obj}.hdf5', key, images=True)
        stats = inspect_transition(data['actions'], data['tcp'], data['gripper'])
        case = {'object': obj, 'demo': key, 'episode_id': decode(data['metadata']['ep_id']),
                'instruction': data['instructions'][0], 'stats': stats,
                'gif': export_gif(out, obj, key, data, stats['first_approach_command'])}
        if args.replay:
            case['replay'] = replay_prefix(data_root, obj, data, stats['first_approach_command'])
            r = case['replay']['first_transition']
            print(f'[replay] {obj}/{key}: TCP error before {r["before_tcp_error_mm"]:.3f} mm, after {r["after_tcp_error_mm"]:.3f} mm', flush=True)
        examples.append(case)
        all_frames += frame_rows(obj, key, data)
    with_approach = [r for r in records if r['first_approach_command'] is not None]
    # Strong evidence: same-index post state within 0.1mm and pre state >10mm
    # at the command transition. Do not use the nearly-static holding frames.
    verified = [e for e in examples if e.get('replay') and
                e['replay']['first_transition']['after_tcp_error_mm'] < .1 and
                e['replay']['first_transition']['before_to_previous_recorded_tcp_error_mm'] < .1 and
                e['replay']['first_transition']['before_tcp_error_mm'] > 10.]
    report = {
        'data_root': str(data_root), 'ordering_status': 'post_action_confirmed_in_replayed_examples' if verified else 'not_confirmed',
        'input_fingerprints': [{'path': str(data_root/'demos'/f'{obj}.hdf5'),
                                'bytes': (data_root/'demos'/f'{obj}.hdf5').stat().st_size,
                                'mtime_ns': (data_root/'demos'/f'{obj}.hdf5').stat().st_mtime_ns}
                               for obj in args.objects],
        'verified_replay_examples': [e['object']+'/'+e['demo'] for e in verified],
        'training_pairing': {'history': ['obs[t-1]', 'obs[t]'],
                             'current_target_start': 'action[t]', 'candidate_target_start': 'action[t+1]',
                             'chunk_length': 8, 'sample_stride': 2},
        'scan_summary': {'objects': args.objects, 'valid_demos': len(records), 'skipped_demos': len(skipped),
                         'with_approach': len(with_approach),
                         'same_frame_motion': sum(r['motion_minus_command_frames'] == 0 for r in with_approach),
                         'preceding_frame_skipped_stride2': sum(r['preceding_observation_sampled_stride2'] is False for r in with_approach),
                         'motion_lag_histogram': dict(Counter(str(r['motion_minus_command_frames']) for r in with_approach))},
        'examples': examples, 'skipped': skipped, 'source_evidence': source_evidence(),
        'limits': ['Conversion code for the released HDF5 was not found in the local upstream checkout.',
                   'Replay checks initial prefixes, not full expert trajectories or task success.',
                   'Replay gain recipes are explicit and may differ for other planner modes.',
                   'Touch-prefix states do not fully match one 20Hz control step per stored frame. Native frame spacing or controller history needs further tracing.',
                   'The saved first frame is already post-action; its missing reset image cannot be recovered by shifting labels.',
                   'Observed ordering does not prove it is the sole cause of model stalls.',
                   'Statistics cover the four current training objects, not every released category.'],
        'recommended_follow_up': ['Add explicit action_target_offset=1 while retaining offset=0 for existing checkpoints.',
                                 'Use stride=1 or deliberately include transition observations in a controlled retraining.',
                                 'Check skill-boundary labels and final-frame/chunk handling when implementing the offset.',
                                 'Compare a small MLP retraining before interpreting diffusion or contact changes.'],
    }
    write_csv(out/'transition_frames.csv', all_frames)
    (out/'timing_report.json').write_text(json.dumps(report, indent=2)+'\n')
    write_review(out, report)
    print(json.dumps({'review': str(out/'index.html'), 'status': report['ordering_status'], 'scan': report['scan_summary']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
