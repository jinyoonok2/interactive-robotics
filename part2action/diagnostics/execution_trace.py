"""Record native controller calls and exact physics cadence without changing them.

An action is a target, not a measured pose. Identity replay reissues the original
calls; it never teleports the robot to the recorded trajectory. Actual motor
targets/gains are retained to audit the upstream positional-argument wiring.
"""
from __future__ import annotations

import inspect
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def physical_state(env):
    p = env.world.p
    xyz, quat = p.getBasePositionAndOrientation(env.obj.uid)
    linear, angular = p.getBaseVelocity(env.obj.uid)
    joints = p.getJointStates(env.robot.panda, list(range(p.getNumJoints(env.robot.panda))))
    return {
        'tcp_xyzw_xyz': env.robot.get_tcp_pose().to_list().tolist(),
        'robot_q': [float(j[0]) for j in joints],
        'robot_qd': [float(j[1]) for j in joints],
        'object_xyz': list(xyz), 'object_xyzw': list(quat),
        'object_linear_velocity': list(linear), 'object_angular_velocity': list(angular),
        'object_joints': [[float(v) for v in p.getJointState(env.obj.uid, i)[:2]]
                          for i in range(p.getNumJoints(env.obj.uid))],
        'finger_m': float(env.robot.get_gripper_state()),
    }


class ExecutionTrace:
    """Instance-local wrappers; all arguments and return values pass through."""

    def __init__(self, env):
        self.env = env
        self.events = []
        self.phase_events = []
        self.phase = 'initial'
        self.physics_steps = 0
        self.original_apply = env.robot.apply_action
        self.original_motor = env.world.p.setJointMotorControl2
        self.original_step = env.world.p.stepSimulation
        self.original_mutations = {}
        self.apply_signature = inspect.signature(self.original_apply)
        self.active = None
        self.initial = physical_state(env)
        env.robot.apply_action = self.apply
        env.world.p.setJointMotorControl2 = self.motor
        env.world.p.stepSimulation = self.step
        for name in ['setCollisionFilterPair', 'setCollisionFilterGroupMask', 'changeDynamics',
                     'resetJointState', 'resetBasePositionAndOrientation', 'resetBaseVelocity',
                     'setPhysicsEngineParameter', 'setGravity']:
            if hasattr(env.world.p, name):
                original = getattr(env.world.p, name)
                self.original_mutations[name] = original
                setattr(env.world.p, name, self.mutation(name, original))

    def mutation(self, name, original):
        def call(*args, **kwargs):
            self.events.append({'kind': 'client_call', 'function': name,
                                'args': plain(args), 'kwargs': plain(kwargs),
                                'phase': self.phase, 'physics_start': self.physics_steps,
                                'physics_steps': 0})
            self.active = None
            return original(*args, **kwargs)
        return call

    def apply(self, *args, **kwargs):
        bound = self.apply_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        action = np.asarray(values.pop('action'), dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Native controller emitted an invalid action')
        event = {'kind': 'action', 'action': action.tolist(), 'kwargs': plain(values),
                 'phase': self.phase, 'physics_start': self.physics_steps,
                 'physics_steps': 0, 'motors': [], 'before': physical_state(self.env)}
        self.events.append(event)
        self.active = event
        return self.original_apply(*args, **kwargs)

    def motor(self, *args, **kwargs):
        if self.active is None:
            raise RuntimeError('Unlogged motor command outside an action')
        self.active['motors'].append({'args': plain(args), 'kwargs': plain(kwargs)})
        return self.original_motor(*args, **kwargs)

    def step(self, *args, **kwargs):
        if args or kwargs:
            raise RuntimeError('Unexpected stepSimulation arguments')
        if self.active is None:
            self.active = {'kind': 'physics', 'phase': self.phase,
                           'physics_start': self.physics_steps, 'physics_steps': 0}
            self.events.append(self.active)
        result = self.original_step(*args, **kwargs)
        self.active['physics_steps'] += 1
        self.physics_steps += 1
        return result

    def finish_action(self):
        """Called after each native env.step to retain its post-step state."""
        if self.active is not None:
            self.active['after'] = physical_state(self.env)

    @contextmanager
    def scope(self, name, arguments=None):
        previous = self.phase
        self.phase = name
        row = {'phase': name, 'arguments': plain(arguments or {}),
               'start_event': len(self.events), 'start_physics_step': self.physics_steps}
        self.phase_events.append(row)
        try:
            yield row
        finally:
            row.update(end_event=len(self.events), end_physics_step=self.physics_steps)
            self.phase = previous

    def close(self):
        self.env.robot.apply_action = self.original_apply
        self.env.world.p.setJointMotorControl2 = self.original_motor
        self.env.world.p.stepSimulation = self.original_step
        for name, original in self.original_mutations.items():
            setattr(self.env.world.p, name, original)


def trace_arrays(events):
    actions = [row for row in events if row['kind'] == 'action']
    return (np.asarray([r['action'] for r in actions], dtype=float),
            np.asarray([r['after']['tcp_xyzw_xyz'] for r in actions], dtype=float),
            np.asarray([r['after']['finger_m'] for r in actions], dtype=float))


def replay_agreement(reference, replay):
    """Compare measured states after the same commands, including orientation."""
    a, b = np.asarray(reference), np.asarray(replay)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Reference/replay TCP traces must be finite and aligned')
    position = np.linalg.norm(a[:, 4:7] - b[:, 4:7], axis=1)
    angle = (Rotation.from_quat(a[:, :4]).inv() * Rotation.from_quat(b[:, :4])).magnitude()
    return {'max_tcp_position_m': float(position.max()),
            'max_tcp_rotation_deg': float(np.rad2deg(angle).max())}


def state_trace_agreement(reference_events, replay_events):
    a = [r['after'] for r in reference_events if r['kind'] == 'action']
    b = [r['after'] for r in replay_events if r['kind'] == 'action']
    if len(a) != len(b) or not a:
        raise ValueError('Physical state traces are not aligned')
    result = replay_agreement([r['tcp_xyzw_xyz'] for r in a], [r['tcp_xyzw_xyz'] for r in b])
    result['max_finger_difference_m'] = float(np.max(np.abs(np.array([r['finger_m'] for r in a])-np.array([r['finger_m'] for r in b]))))
    result['max_robot_joint_difference_rad'] = float(np.max(np.abs(np.array([r['robot_q'][:7] for r in a])-np.array([r['robot_q'][:7] for r in b]))))
    result['max_object_position_m'] = float(np.linalg.norm(np.array([r['object_xyz'] for r in a])-np.array([r['object_xyz'] for r in b]),axis=1).max())
    angle = (Rotation.from_quat([r['object_xyzw'] for r in a]).inv()*Rotation.from_quat([r['object_xyzw'] for r in b])).magnitude()
    result['max_object_rotation_deg'] = float(np.rad2deg(angle).max())
    return result


def can_expand(rows, categories):
    """A successful pilot must cover every requested category, without hiding failures."""
    generated = [r for r in rows if r.get('generated_stable_success')]
    eligible = [r for r in generated if r.get('eligible')]
    covered = {r['case']['object'] for r in eligible}
    reasons = []
    if not set(categories) <= covered:
        reasons.append('Missing validated categories: ' + ', '.join(sorted(set(categories) - covered)))
    if len(eligible) < max(4, len(categories)):
        reasons.append('Fewer than four validated source executions')
    if generated and len(eligible) != len(generated):
        reasons.append('Some physically successful generated executions failed replay validation')
    if not any(r.get('variation_successes', 0) for r in eligible):
        reasons.append('No successful variation labels')
    if not any(r.get('variation_failures', 0) for r in eligible):
        reasons.append('No failed variation labels')
    return {'passed': not reasons, 'reasons': reasons,
            'generated_stable_successes': len(generated), 'validated_references': len(eligible),
            'covered_categories': sorted(covered)}
