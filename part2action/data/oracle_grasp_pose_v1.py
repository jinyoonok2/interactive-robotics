"""Explicit GT first-grasp target labels; no clock or replay actions as inputs.

Recorded observations are post-action. The measured TCP immediately BEFORE a
close at action j is obs/tcp_pose[j-1]. A single first-grasp pose is supplied
throughout the task, including subsequent skills, so there is no GT phase gate.
Tasks not beginning with a grasp receive an invalid target and remain in data.
"""
from __future__ import annotations

import numpy as np
import torch

from .partinstruct_loader import PartInstructDataset
from .targets import _detect_grasp_contact_step, classify_skill_instruction


def first_grasp_target(actions, tcp_poses, instructions):
    invalid = np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)
    texts = [v.decode() if isinstance(v, bytes) else str(v) for v in instructions]
    if not texts or classify_skill_instruction(texts[0]) != 'grasp_obj':
        return {'pose': invalid, 'valid': False, 'reason': 'task_does_not_begin_with_grasp'}
    end = next((i-1 for i, value in enumerate(texts) if value != texts[0]), len(texts)-1)
    close = _detect_grasp_contact_step(actions, 0, end)
    if close is None or close < 1 or close-1 >= len(tcp_poses):
        return {'pose': invalid, 'valid': False, 'reason': 'no_measured_preclosure_pose'}
    pose = np.asarray(tcp_poses[close-1], dtype=np.float32).copy()
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[:4]) < 1e-6:
        return {'pose': invalid, 'valid': False, 'reason': 'invalid_measured_pose'}
    pose[:4] /= np.linalg.norm(pose[:4])
    return {'pose': pose, 'valid': True, 'reason': 'measured_TCP_before_first_close',
            'closure_action_index': close, 'pose_observation_index': close-1}


class OracleGraspPoseDatasetV1(PartInstructDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_tcp_pose or self.action_target_offset != 1:
            raise ValueError('Oracle visual experiment requires measured TCP and next-action targets')
        self._oracle_targets = {}

    def target_for(self, spec):
        key = (spec.file_path, spec.demo_key)
        if key not in self._oracle_targets:
            demo = self._open(spec.file_path)['data'][spec.demo_key]
            self._oracle_targets[key] = first_grasp_target(
                np.asarray(demo['actions']), np.asarray(demo['obs'][self.tcp_pose_key]),
                demo['skill_instructions'][:])
        return self._oracle_targets[key]

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        target = self.target_for(self._samples[index])
        sample['goal_pose'] = torch.from_numpy(target['pose'].copy())
        sample['goal_valid'] = torch.tensor(float(target['valid']))
        return sample
