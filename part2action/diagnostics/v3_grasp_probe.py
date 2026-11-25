"""Explicit inference-only pose intervention and grasp-retention measurements."""
from dataclasses import replace
import hashlib

import numpy as np
from scipy.spatial.transform import Rotation
import torch


REFERENCE_ORDER = ('reference_replay', 'reference_native')
POLICY_ARMS = ('predicted', 'verified')
RETENTION_LIMITS = {'tcp_rise_m': .03, 'object_rise_m': .025,
                    'hold_contact_fraction': .8, 'relative_drift_m': .015}


def pose_matrix(xyzw_xyz):
    pose = np.asarray(xyzw_xyz, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[:4]) < 1e-8:
        raise ValueError('Expected a finite xyzw/XYZ pose')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[:4]).as_matrix()
    result[:3, 3] = pose[4:]
    return result


def relative_grasp_goal(state, opening):
    obj = pose_matrix([*state['object_xyzw'], *state['object_xyz']])
    tcp = pose_matrix(state['tcp_xyzw_xyz'])
    relative = np.linalg.inv(obj) @ tcp
    if not np.isfinite(opening) or not 0 <= opening <= .04:
        raise ValueError('Invalid reference opening')
    return {'object_to_tcp': relative.tolist(), 'opening_m': float(opening)}


def world_grasp_goal(reference, object_xyzw_xyz):
    relative = np.asarray(reference['object_to_tcp'], dtype=float)
    if relative.shape != (4, 4) or not np.isfinite(relative).all():
        raise ValueError('Invalid object-relative reference')
    world = pose_matrix(object_xyzw_xyz) @ relative
    return {'position_world': world[:3, 3].tolist(),
            'rotation_world': world[:3, :3].tolist(), 'opening_m': reference['opening_m']}


def goal_record(goal):
    return {'position_world': goal.position_world[0, 0].detach().float().cpu().tolist(),
            'rotation_world': goal.rotation_world[0, 0].detach().float().cpu().tolist(),
            'opening_m': float(goal.opening[0, 0]),
            'applicability_logits': None if goal.applicability_logits is None else
                goal.applicability_logits[0, 0].detach().float().cpu().tolist()}


class GoalIntervention:
    """Hook only the pose-head output, leaving the model's public API unchanged.

    The provider is privileged in the verified arm. Encoders, visual inputs,
    grounding, validity/confidence, skill logits, tokenizer and decoder remain
    unchanged. Both the raw prediction and the actually conditioned pose are logged.
    """
    def __init__(self, model, provider=None, *, force_geometry_confidence=False):
        self.model, self.provider, self.last = model, provider, None
        self.force_geometry_confidence = force_geometry_confidence

    def __enter__(self):
        self.handle = self.model.blocks['goal_head'].register_forward_hook(self._hook)
        return self

    def _hook(self, module, args, original):
        if original.position_world.shape[0] != 1:
            raise ValueError('This fixed-scene diagnostic requires batch size one')
        conditioned = original
        if self.provider is not None:
            reference = self.provider()
            position = torch.as_tensor(reference['position_world'], device=original.position_world.device,
                                       dtype=original.position_world.dtype).reshape(1, 1, 3)
            rotation = torch.as_tensor(reference['rotation_world'], device=original.rotation_world.device,
                                       dtype=original.rotation_world.dtype).reshape(1, 1, 3, 3)
            opening = original.opening.new_tensor([[reference['opening_m']]])
            if not 0 <= reference['opening_m'] <= .04:
                raise ValueError('Reference opening outside Panda limits')
            conditioned = replace(original, position_world=position, rotation_world=rotation, opening=opening)
            conditioned.validate(1, 1)
        if self.force_geometry_confidence and conditioned.applicability_logits is not None:
            logits = conditioned.applicability_logits.clone()
            # sigmoid(20) rounds to 1 in float32. Preserve opening confidence,
            # masks and skill identity; only the two geometry multipliers change.
            logits[..., :2] = 20.
            conditioned = replace(conditioned, applicability_logits=logits)
        self.last = {'raw_prediction': goal_record(original), 'conditioned_goal': goal_record(conditioned),
                     'privileged_target': self.provider is not None,
                     'force_geometry_confidence': self.force_geometry_confidence}
        return conditioned

    def __exit__(self, *args):
        self.handle.remove()


def inference_with_fixed_noise(model, observation, seed, prediction_index):
    device = observation.rgb.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    noise_seed = (int(seed)*1000003+int(prediction_index)) % (2**63-1)
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(noise_seed)
        return model(observation)


def policy_digest(model):
    from models.pose_token_diffusion_v3.checkpoint import checkpoint_state
    digest = hashlib.sha256()
    for name, value in sorted(checkpoint_state(model).items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def matching_states(first, second, tolerance=1e-6):
    return first.keys() == second.keys() and all(
        np.allclose(np.asarray(first[key]), np.asarray(second[key]), atol=tolerance, rtol=0)
        for key in first)


def retention_score(before, after, hold, *, native_grasp, commanded_opening, limits=None):
    limits = limits or RETENTION_LIMITS
    if not hold:
        raise ValueError('Retention needs measured hold observations')
    relative = []
    for row in hold:
        tcp = pose_matrix(row['state']['tcp_xyzw_xyz'])
        relative.append(tcp[:3, :3].T @ (np.asarray(row['state']['object_xyz'])-tcp[:3, 3]))
    drift = float(np.linalg.norm(np.asarray(relative)-relative[0], axis=1).max())
    contact = float(np.mean([bool(row['force_qualified_fingers']) for row in hold]))
    tcp_rise = after['tcp_xyzw_xyz'][6]-before['tcp_xyzw_xyz'][6]
    obj_rise = after['object_xyz'][2]-before['object_xyz'][2]
    passes = {'native_target_grasp_before_lift': bool(native_grasp),
              'closed_command_before_lift': bool(commanded_opening <= .02),
              'tcp_lift': bool(tcp_rise >= limits['tcp_rise_m']),
              'object_lift': bool(obj_rise >= limits['object_rise_m']),
              'hold_contact': bool(contact >= limits['hold_contact_fraction']),
              'relative_stability': bool(drift <= limits['relative_drift_m'])}
    return {'retained_grasp': all(passes.values()), 'passes': passes,
            'tcp_rise_m': float(tcp_rise), 'object_rise_m': float(obj_rise),
            'hold_contact_fraction': contact, 'relative_drift_m': drift,
            'limits': limits, 'scope': 'Added five-centimetre lift and hold diagnostic; not native benchmark success.'}


def choose_reference(records):
    """Predeclared source order, independent of either policy arm's outcomes."""
    for name in REFERENCE_ORDER:
        row = records.get(name)
        if row and row.get('status') == 'completed' and row.get('retention', {}).get('retained_grasp'):
            if not row.get('reference_goal'):
                raise ValueError('Eligible reference lacks a grasp pose')
            return name, row
    return None, None
