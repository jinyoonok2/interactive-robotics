"""Versioned pose samples with separate observations, labels and privileges.

Both released HDF5 frames and replay captures use this contract. They remain
distinct observations: recorded images are never silently paired with replay
robot states. Missing sensors must be requested explicitly and cannot fall back
to simulator annotations. No original dataset or existing loader is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 'p2a.pose_sample.v1'
OBSERVATION_KEYS = frozenset({
    'rgb', 'depth_m', 'scene_pcd_camera_m', 'scene_pcd_world_m',
    'tcp_pose_xyzw_xyz', 'gripper_opening_m', 'task_instruction',
    'camera_intrinsic', 'world_to_camera_opengl',
})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(value):
    return value.copy() if isinstance(value, np.ndarray) else value


@dataclass
class PoseSample:
    observations: dict
    labels: dict
    privileged: dict
    provenance: dict

    def validate(self):
        unexpected = set(self.observations) - OBSERVATION_KEYS
        if unexpected:
            raise ValueError('Non-observation fields in input namespace: ' + str(sorted(unexpected)))
        if set(self.observations) & (set(self.labels) | set(self.privileged)):
            raise ValueError('Observation names overlap labels/privileged fields')
        if set(self.labels) & set(self.privileged):
            raise ValueError('Label names overlap privileged fields')
        for key in ('tcp_pose_xyzw_xyz', 'gripper_opening_m', 'task_instruction'):
            if key not in self.observations:
                raise ValueError('Missing observation: ' + key)
        tcp = np.asarray(self.observations['tcp_pose_xyzw_xyz'])
        if tcp.shape != (7,) or not np.isfinite(tcp).all() or not np.isclose(np.linalg.norm(tcp[:4]), 1., atol=1e-3):
            raise ValueError('Expected finite world-frame TCP quaternion xyzw + XYZ')
        opening = np.asarray(self.observations['gripper_opening_m'])
        if opening.size != 1 or not np.isfinite(opening).all():
            raise ValueError('Expected one measured gripper opening')
        if self.provenance.get('tcp_frame') != 'world' or self.provenance.get('units') != 'meters':
            raise ValueError('TCP coordinate frame/units must be explicit')
        if self.provenance.get('split') not in {'train', 'development', 'test'}:
            raise ValueError('Missing or unknown instance split')
        if not self.provenance.get('instance_group') or not self.provenance.get('sample_id'):
            raise ValueError('Missing source identity')
        return self

    def model_inputs(self, *, allow_privileged=(), require=()):
        """Labels never enter this view; diagnostic privileges need named opt-in."""
        allowed = tuple(allow_privileged)
        unknown = set(allowed) - set(self.privileged)
        if unknown:
            raise ValueError('Unknown privileged input request: ' + str(sorted(unknown)))
        result = {key: _copy(value) for key, value in self.observations.items()}
        result.update({key: _copy(self.privileged[key]) for key in allowed})
        missing = set(require) - set(result)
        if missing:
            raise ValueError('Required observations unavailable; no GT fallback: ' + str(sorted(missing)))
        return result


def save_pose_sample(path, sample):
    sample.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {namespace + '__' + key: np.asarray(value)
              for namespace, values in [('obs', sample.observations), ('label', sample.labels),
                                        ('privileged', sample.privileged)] for key, value in values.items()}
    arrays['schema_version'] = np.asarray(SCHEMA_VERSION)
    arrays['provenance_json'] = np.asarray(json.dumps(sample.provenance, sort_keys=True))
    # Exclusive creation protects existing captures, even outside the run launcher.
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def load_pose_sample(path):
    groups = {'obs': {}, 'label': {}, 'privileged': {}}
    with np.load(path, allow_pickle=False) as values:
        if str(values['schema_version']) != SCHEMA_VERSION:
            raise ValueError('Unsupported pose sample schema')
        provenance = json.loads(str(values['provenance_json']))
        for name in values.files:
            if name in {'schema_version', 'provenance_json'}:
                continue
            namespace, key = name.split('__', 1)
            if namespace not in groups:
                raise ValueError('Unknown sample namespace: ' + namespace)
            value = values[name].copy()
            groups[namespace][key] = str(value) if value.ndim == 0 and value.dtype.kind in 'US' else value
    return PoseSample(groups['obs'], groups['label'], groups['privileged'], provenance).validate()


def validate_manifest(manifest):
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported manifest schema')
    instances, sample_ids = {}, set()
    for row in manifest['records']:
        identity = row['sample_id']
        if identity in sample_ids:
            raise ValueError('Duplicate sample: ' + identity)
        sample_ids.add(identity)
        group, split = row['instance_group'], row['split']
        if split not in {'train', 'development', 'test'}:
            raise ValueError('Unknown split')
        if group in instances and instances[group] != split:
            raise ValueError('Object instance crosses splits: ' + group)
        instances[group] = split


class PoseSampleDataset:
    """Shared, hash-checked sample reader; also usable as a PyTorch map dataset."""
    def __init__(self, manifest_path, split=None):
        self.root = Path(manifest_path).resolve().parent
        self.manifest = json.loads(Path(manifest_path).read_text())
        validate_manifest(self.manifest)
        self.records = [row for row in self.manifest['records'] if split is None or row['split'] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        receipt = json.loads((self.root / record['receipt']).read_text())
        path = self.root / record['file']
        if receipt['status'] != 'completed' or file_sha256(path) != receipt['sample_sha256']:
            raise ValueError('Missing, failed or changed observation: ' + record['sample_id'])
        sample = load_pose_sample(path)
        for key in ('sample_id', 'instance_group', 'split'):
            if sample.provenance[key] != record[key]:
                raise ValueError('Sample/manifest identity mismatch: ' + key)
        return sample


def hdf5_pose_sample(path, demo_key, observation_index, *, task_instruction,
                     instance_group, split, action_chunk=8, goal_pose=None):
    """Adapt one recorded post-action frame; targets start at t+1.

    GT masks/clouds are labels, never observation inputs. A reference interaction
    pose can optionally be added as a training label. Camera-world transforms
    are not fabricated if the HDF5 does not record them.
    """
    import h5py
    if action_chunk < 1 or observation_index < 0:
        raise ValueError('Invalid chunk length or observation index')
    with h5py.File(path, 'r') as handle:
        demo = handle['data'][demo_key]
        obs = demo['obs']
        t = observation_index
        rgb = obs['agentview_rgb'][t]
        if rgb.shape[-1] == 3:
            rgb = rgb.transpose(2, 0, 1)
        observations = {'rgb': rgb, 'tcp_pose_xyzw_xyz': obs['tcp_pose'][t],
                        'gripper_opening_m': np.asarray(obs['gripper_state'][t]).reshape(1),
                        'task_instruction': str(task_instruction)}
        if 'agentview_pcd' in obs:
            observations['scene_pcd_camera_m'] = obs['agentview_pcd'][t, :, :3]
        actions = demo['actions'][t + 1:t + 1 + action_chunk]
        padded = np.zeros((action_chunk, 7), dtype=np.float32)
        padded[:len(actions)] = actions
        labels = {'action_chunk_world_xyz_euler_opening': padded,
                  'action_valid': np.arange(action_chunk) < len(actions)}
        for source, target in [('agentview_part_mask', 'part_mask'), ('agentview_part_pcd', 'part_cloud_camera_annotation')]:
            if source in obs:
                labels[target] = obs[source][t]
        if 'skill_instructions' in demo:
            value = demo['skill_instructions'][t]
            labels['skill_instruction'] = value.decode() if isinstance(value, bytes) else str(value)
        if goal_pose is not None:
            labels['interaction_pose_xyzw_xyz'] = np.asarray(goal_pose)
    return PoseSample(observations, labels, {}, {
        'sample_id': f'{instance_group}/{demo_key}/recorded/{t}', 'instance_group': instance_group,
        'split': split, 'source_kind': 'released_hdf5', 'source_path': str(Path(path).resolve()),
        'demo': demo_key, 'observation_index': t, 'next_action_index': t + 1,
        'timing': 'recorded observation after action t; target commands begin at t+1',
        'tcp_frame': 'world', 'scene_pcd_frame': 'camera', 'units': 'meters',
    }).validate()
