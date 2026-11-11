"""Pure helpers for a replay-gated grasp perturbation pilot (not policy inputs)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def grasp_window(actions, instructions, close_width=0.02):
    """Require an initially open, first-skill grasp with a real close onset."""
    a = np.asarray(actions, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
        raise ValueError('Expected finite (T,7) actions')
    first = instructions[0]
    if not first.lower().startswith('grasp '):
        raise ValueError('Pilot supports initial grasp skills only')
    end = next((i for i, text in enumerate(instructions) if text != first), len(a))
    if a[0, 6] <= close_width:
        raise ValueError('Already closed at grasp start; no closure onset')
    closed = np.flatnonzero(a[:end, 6] <= close_width)
    if not len(closed):
        raise ValueError('No close onset inside initial grasp segment')
    close = int(closed[0])
    depart = np.flatnonzero(np.linalg.norm(a[:close, :3] - a[0, :3], axis=1) > .05)
    if not len(depart):
        raise ValueError('No distinct approach before closure')
    return {'start': int(depart[0]), 'close': close, 'end': end}


def conditions(translation_m=.01, rotation_deg=10., close_offset_steps=2):
    """One-factor screening; these offsets are a proposal, not known tolerances."""
    if not all(np.isfinite(x) and x > 0 for x in [translation_m, rotation_deg]):
        raise ValueError('Position and rotation magnitudes must be positive and finite')
    if type(close_offset_steps) is not int or close_offset_steps < 1:
        raise ValueError('Closure offset must be a positive integer number of steps')
    rows = [{'name': 'identity', 'translation_m': [0., 0., 0.],
             'rotation_deg': [0., 0., 0.], 'close_offset_steps': 0}]
    for kind, amount in [('translation_m', translation_m), ('rotation_deg', rotation_deg)]:
        for axis in range(3):
            for sign in [-1, 1]:
                row = {**rows[0], 'translation_m': [0., 0., 0.], 'rotation_deg': [0., 0., 0.]}
                row[kind][axis] = sign * amount
                row['name'] = f'{kind}_{"xyz"[axis]}_{"minus" if sign < 0 else "plus"}'
                rows.append(row)
    for shift in [-close_offset_steps, close_offset_steps]:
        rows.append({**rows[0], 'name': f'close_{"early" if shift < 0 else "late"}',
                     'close_offset_steps': shift})
    return rows


def perturb_actions(actions, window, condition, ramp_steps=8):
    """World translation and TCP-local rotation ramp in before closure, then hold.

    Positive timing offset delays ONLY the gripper command (edge padding, no
    wraparound). Arm commands/time budget are unchanged. No robot teleportation.
    """
    a = np.asarray(actions, dtype=float)[:window['end']].copy()
    original = a.copy()
    start, close = window['start'], window['close']
    finish = max(start + 1, close - 3)
    begin = max(start, finish - ramp_steps)
    phase = np.clip((np.arange(len(a)) - begin) / max(1, finish - begin), 0., 1.)
    weight = phase * phase * (3 - 2 * phase)
    a[:, :3] += weight[:, None] * np.asarray(condition['translation_m'])
    rotvec = np.deg2rad(condition['rotation_deg'])
    if np.any(rotvec):
        # Preserve the common prefix exactly. Euler -> rotation -> Euler can
        # change unperturbed commands by round-off even for an identity rotation;
        # contact dynamics and rendering can amplify that before intervention.
        active = weight > 0
        a[active, 3:6] = (Rotation.from_euler('xyz', a[active, 3:6]) *
                            Rotation.from_rotvec(weight[active, None] * rotvec)).as_euler('xyz')
    shift = int(condition['close_offset_steps'])
    indices = np.clip(np.arange(len(a)) - shift, 0, len(a) - 1)
    a[:, 6] = original[indices, 6]
    return a


def controller_gains(original_actions, window):
    """Fixed native-inspired schedule from the UNPERTURBED source commands.

    Perturbing closure timing must not also change arm/finger gains.
    """
    a = np.asarray(original_actions)[:window['end']]
    first = a[window['start'], :3]
    later = np.flatnonzero(np.linalg.norm(a[window['start']:window['close'], :3] - first, axis=1) > .002)
    second = window['start'] + int(later[0]) if len(later) else window['close']
    rows = []
    for t in range(len(a)):
        gain = .0001 if t < window['start'] or t >= window['close'] else (.005 if t < second else .01)
        finger = .1 if t >= max(window['close'], len(a)-20) else .01
        rows.append((gain, finger, .2 if t < second else .01))
    return rows


def tracking_metrics(tcp, recorded_tcp, fingers, recorded_fingers):
    tcp, recorded_tcp = np.asarray(tcp), np.asarray(recorded_tcp)
    xyz = np.linalg.norm(tcp[:, 4:7] - recorded_tcp[:, 4:7], axis=-1)
    angle = np.rad2deg((Rotation.from_quat(tcp[:, :4]).inv() *
                       Rotation.from_quat(recorded_tcp[:, :4])).magnitude())
    finger = np.asarray(fingers) - np.asarray(recorded_fingers).reshape(-1)
    return {'tcp_rmse_m': float(np.sqrt(np.mean(xyz ** 2))),
            'tcp_p95_m': float(np.quantile(xyz, .95)),
            'orientation_p95_deg': float(np.quantile(angle, .95)),
            'finger_rmse_m': float(np.sqrt(np.mean(finger ** 2)))}


def replay_fidelity_passes(metrics, thresholds):
    return bool(all(np.isfinite(metrics[k]) and metrics[k] <= value
                    for k, value in thresholds.items()))
