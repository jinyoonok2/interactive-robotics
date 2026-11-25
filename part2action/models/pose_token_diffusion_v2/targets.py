"""Training-only per-skill TCP goals, with explicit proxy quality and masks.

Recorded observations are post-action. Grasp uses measured TCP before closure;
touch uses closest fingertip-proxy-to-labelled-part geometry (not contact
verification). Move/rotate/release use demonstrated TCP endpoints, not predicted
object-part poses or assertions that the physical skill succeeded.
"""
from dataclasses import dataclass
import math

import numpy as np

from data.targets import _detect_grasp_contact_step, _detect_touch_contact_step

SKILLS = ('grasp_obj', 'touch_obj', 'move_gripper', 'rotate_obj', 'release_obj', 'unknown')


@dataclass(frozen=True)
class TargetSettings:
    touch_max_distance_m: float = .03
    # PartGym PandaArm.get_fingertip_pose(): finger_depth + finger_depth/8.
    # Robot-specific probe along TCP-local +Z, not an estimated contact event.
    touch_probe_offset_m: float = .05625

    def __post_init__(self):
        if not math.isfinite(self.touch_max_distance_m) or self.touch_max_distance_m <= 0:
            raise ValueError('Touch proxy distance must be finite and positive')
        if not math.isfinite(self.touch_probe_offset_m) or self.touch_probe_offset_m < 0:
            raise ValueError('Touch probe offset must be finite and nonnegative')


def fingertip_proxy_poses(poses, offset):
    """Copy TCP poses with XYZ shifted by the PartGym fingertip proxy offset."""
    proxy = np.asarray(poses,dtype=np.float32).copy()
    norms = np.linalg.norm(proxy[:,:4],axis=-1)
    valid = np.isfinite(proxy).all(-1) & (norms >= 1e-6)
    q = proxy[:,:4]/np.maximum(norms[:,None],1e-6)
    x,y,z,w = q.T
    z_axis = np.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],-1)
    proxy[:,4:7] += offset*z_axis
    proxy[~valid,4:7] = np.nan
    return proxy


def skill_kind(text):
    value = ' '.join(str(text).strip().lower().split())
    for prefixes, kind in [(('grasp ',), 'grasp_obj'), (('touch ',), 'touch_obj'),
        (('move ',), 'move_gripper'), (('reorient ', 'rotate '), 'rotate_obj'),
        (('release',), 'release_obj')]:
        if value.startswith(prefixes):
            return kind
    return 'unknown'


def segments(instructions):
    result = []
    for t, raw in enumerate(instructions):
        text = (raw.decode() if isinstance(raw, bytes) else str(raw)).strip()
        if not result or text != result[-1]['instruction']:
            result.append(dict(instruction=text, start=t, end=t, kind=skill_kind(text)))
        else:
            result[-1]['end'] = t
    return result


def phase_targets(actions, tcp_poses, instructions, *, part_pcd=None,
                  settings=TargetSettings(), max_slots=4):
    actions, poses = np.asarray(actions), np.asarray(tcp_poses)
    if actions.ndim != 2 or actions.shape[1] != 7 or poses.shape != actions.shape or len(instructions) != len(actions):
        raise ValueError('Aligned actions/TCP poses must be (T,7), with T skill annotations')
    phases = segments(instructions)
    if len(phases) > max_slots:
        raise ValueError('Demonstration exceeds configured skill slots')
    result = []
    for phase in phases:
        start, end, kind = phase['start'], phase['end'], phase['kind']
        target_t, quality, distance, reason = end, 'demonstrated_tcp_endpoint', None, ''
        mask = np.array([True, kind == 'rotate_obj', kind in {'move_gripper','rotate_obj','release_obj'}])
        if kind == 'grasp_obj':
            close = _detect_grasp_contact_step(actions, start, end)
            target_t = close-1 if close is not None else -1
            quality = 'measured_tcp_before_closure'
            mask[:] = True
            if target_t < start:
                reason = 'no_measured_preclosure_pose_in_phase'
        elif kind == 'touch_obj':
            quality = 'closest_fingertip_to_part_proxy'
            mask[:] = [True, False, False]
            if part_pcd is None:
                target_t, reason = -1, 'missing_touch_geometry'
            else:
                target_t, distance = _detect_touch_contact_step(part_pcd,
                    fingertip_proxy_poses(poses,settings.touch_probe_offset_m),start,end)
                if target_t is None or distance is None or distance > settings.touch_max_distance_m:
                    target_t, reason = -1, 'no_nearby_fingertip_part_proxy'
        elif kind == 'rotate_obj':
            quality = 'demonstrated_tcp_rotation_endpoint_proxy'
        elif kind == 'release_obj':
            quality = 'demonstrated_release_tcp_endpoint_proxy'
        elif kind == 'unknown':
            target_t, reason = -1, 'unsupported_skill'
        pose = np.array([0.,0.,0.,1.,0.,0.,0.], dtype=np.float32)
        if target_t is not None and 0 <= target_t < len(poses) and not reason:
            candidate = poses[target_t].astype(np.float32).copy()
            if np.isfinite(candidate).all() and np.linalg.norm(candidate[:4]) >= 1e-6:
                candidate[:4] /= np.linalg.norm(candidate[:4])
                pose = candidate
            else:
                reason = 'invalid_measured_tcp'
        else:
            reason = reason or 'invalid_target_timestep'
        opening = float(actions[end, 6])
        if not np.isfinite(opening) or not 0 <= opening <= .04+1e-6:
            mask[2] = False
            opening = 0.
        if reason:
            mask[:] = False
        result.append({**phase, 'pose': pose, 'component_valid': mask,
            'target_t': int(target_t) if target_t is not None else -1,
            'opening': opening, 'quality': quality, 'reason': reason,
            'touch_proxy_to_part_distance_m': distance})
    return result


def labels_at(phases, time, max_slots):
    pose = np.tile(np.array([0.,0.,0.,1.,0.,0.,0.], dtype=np.float32), (max_slots,1))
    valid = np.zeros((max_slots,3), dtype=bool)
    supervise = valid.copy()
    opening = np.zeros(max_slots, dtype=np.float32)
    kinds = np.full(max_slots, SKILLS.index('unknown'), dtype=np.int64)
    active = next((i for i, p in enumerate(phases) if p['start'] <= time <= p['end']), None)
    if active is None:
        raise ValueError('Sample time is outside the skill annotation')
    for i, phase in enumerate(phases):
        pose[i], valid[i], opening[i], kinds[i] = phase['pose'], phase['component_valid'], phase['opening'], SKILLS.index(phase['kind'])
        if i == active:
            supervise[i] = valid[i]
            # Reaching a contact goal ends its geometric supervision, but the
            # grasp's closed-opening intent remains valid for the active phase.
            if phase['kind'] in {'grasp_obj','touch_obj'} and time > phase['target_t']:
                supervise[i, :2] = False
    return dict(goal_poses=pose, goal_component_valid=valid, goal_supervision_valid=supervise,
        goal_opening=opening, goal_skill_kind=kinds, goal_active_phase=active)
