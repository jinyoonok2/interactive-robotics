"""Skill-segment training with next-command masks and training-only pose labels."""
import inspect
from pathlib import Path

import numpy as np
import torch

from data.partinstruct_loader import PartInstructDataset, SampleSpec
from models.pose_token_diffusion_v2.dataset import SkillPoseDataset
from models.pose_token_diffusion_v2.targets import TargetSettings, labels_at
from models.pose_token_diffusion_v2.completion import CompletionDataset
from .contracts import SkillObservation
from .skills import annotated_skill


class CurrentSkillDataset(SkillPoseDataset):
    def _index_demos(self, max_demos_per_file, sample_stride):
        # Unlike the full-chunk v2 index, include short skills and final frames.
        # Padding is deterministic and masked; it never consumes another skill.
        for path in self.hdf5_paths:
            handle = self._open(path)
            keys = sorted(handle['data'], key=lambda k: int(k.split('_')[-1]))
            if max_demos_per_file is not None:
                keys = keys[:int(max_demos_per_file)]
            for key in keys:
                demo = handle['data'][key]
                if 'actions' not in demo or 'skill_instructions' not in demo or 'obs' not in demo:
                    continue
                required = [self.rgb_key, self.pcd_key, self.tcp_pose_key, self.gripper_key]
                if self.use_point_part_labels:
                    required.append(self.part_pcd_key)
                if self.require_part_mask:
                    required.append(self.mask_key)
                count = len(demo['actions'])
                if count < 1 or any(name not in demo['obs'] for name in required):
                    continue
                if any(len(demo['obs'][name]) != count for name in required) or len(demo['skill_instructions']) != count:
                    raise ValueError('Misaligned skill observations in ' + path + ':' + key)
                self._samples.extend(SampleSpec(path, key, t, count)
                                     for t in range(0, count, max(1, int(sample_stride))))

    def targets_for(self, spec):
        phases = super().targets_for(spec)
        # V2 omitted touch opening supervision. Use demonstrated opening, with
        # geometry still a distance proxy; never turn simulator predicates into inputs.
        if not hasattr(self, '_touch_opening_audited'):
            self._touch_opening_audited = set()
        key = spec.file_path, spec.demo_key
        if key not in self._touch_opening_audited:
            for phase in phases:
                if phase['kind'] == 'touch_obj':
                    demo = self._open(spec.file_path)['data'][spec.demo_key]
                    opening = float(demo['actions'][phase['end'], 6])
                    phase['component_valid'][2] = bool(np.isfinite(opening) and 0 <= opening <= .040001)
            self._touch_opening_audited.add(key)
        return phases

    def __getitem__(self, index):
        spec = self._samples[index]
        demo = self._open(spec.file_path)['data'][spec.demo_key]
        obs = demo['obs']
        phases = self.targets_for(spec)
        labels = labels_at(phases, spec.t, self.max_skill_slots)
        active = labels.pop('goal_active_phase')
        phase = phases[active]
        current = annotated_skill(phase['instruction'], previous=[p['instruction'] for p in phases[:active]],
                                  step_id=str(active))
        frames = [max(0, spec.t-i) for i in reversed(range(self.n_obs_steps))]
        rgb = np.stack([np.asarray(obs[self.rgb_key][t], dtype=np.uint8) for t in frames])
        commands = np.asarray(demo['actions'], dtype=np.float32)
        future = spec.t + 1 + np.arange(self.action_chunk)
        valid = (future <= phase['end']) & (future < spec.n_steps)
        action = np.zeros((self.action_chunk, 7), dtype=np.float32)
        if valid.any():
            action[valid] = commands[future[valid]]
            action[~valid] = action[valid][-1]
        if self.mask_key in obs:
            mask = np.asarray(obs[self.mask_key][spec.t])
            if mask.ndim == 3:
                mask = mask[..., 0]
        else:
            mask = np.zeros(rgb.shape[1:3])
        sample = {
            'rgb': torch.from_numpy(rgb).permute(0, 3, 1, 2).float()/255.,
            'agentview_pcd': torch.from_numpy(np.asarray(obs[self.pcd_key][spec.t], dtype=np.float32)),
            'tcp_pose': torch.from_numpy(np.asarray(obs[self.tcp_pose_key][spec.t], dtype=np.float32)),
            'gripper_state': torch.from_numpy(np.asarray(obs[self.gripper_key][spec.t], dtype=np.float32)).reshape(1),
            'skill_start_tcp': torch.from_numpy(np.asarray(obs[self.tcp_pose_key][phase['start']], dtype=np.float32)),
            'skill_spec': current, 'action_chunk': torch.from_numpy(action),
            'action_valid': torch.from_numpy(valid), 'part_mask': torch.from_numpy((mask > 0).astype(np.float32)),
            'phase_index': torch.tensor(active, dtype=torch.long),
            'meta': {'file': Path(spec.file_path).name, 'demo': spec.demo_key, 't': spec.t,
                     'skill': current.kind, 'skill_input_source': 'annotated_current_skill',
                     'phase_start': phase['start'], 'phase_end': phase['end'],
                     'goal_quality': phase['quality'], 'goal_reason': phase['reason'],
                     'valid_next_commands': int(valid.sum())}}
        sample.update({key: torch.from_numpy(value) for key, value in labels.items()})
        if self.use_point_part_labels:
            part = np.asarray(obs[self.part_pcd_key][spec.t], dtype=np.float32)
            if part.shape[0] != sample['agentview_pcd'].shape[0]:
                raise ValueError('Part supervision is not aligned with scene points')
            point_labels = part[:, 3] > .5 if part.shape[-1] >= 4 else np.zeros(len(part), dtype=bool)
            sample['point_part_labels'] = torch.from_numpy(point_labels.astype(np.float32))
            sample['point_part_valid'] = torch.tensor(bool(point_labels.any() and current.kind in {'grasp_obj', 'touch_obj'}))
        return sample


def build_dataset(data, goal_labels=None):
    accepted = set(inspect.signature(PartInstructDataset.__init__).parameters) - {'self'}
    own = {'validation_fraction', 'split_seed', 'split', 'demo_manifest'}
    ignored = {'train_hdf5', 'batch_size', 'num_workers', 'pin_memory'}
    unknown = set(data) - accepted - own - ignored
    if unknown:
        raise ValueError('Unknown v3 data options: ' + str(sorted(unknown)))
    kwargs = {k: v for k, v in data.items() if k in accepted | own}
    kwargs['hdf5_paths'] = data['train_hdf5']
    if not kwargs.get('use_pcd') or not kwargs.get('use_tcp_pose') or kwargs.get('use_part_pcd') or kwargs.get('use_contact_xyz'):
        raise ValueError('V3 uses observed scene/TCP inputs; GT part clouds/contact coordinates are supervision only')
    return CurrentSkillDataset(**kwargs, target_settings=TargetSettings(**(goal_labels or {})))


def collate(batch):
    return {key: torch.stack([row[key] for row in batch]) if isinstance(batch[0][key], torch.Tensor)
            else [row[key] for row in batch] for key in batch[0]}


def observation_from_batch(batch):
    return SkillObservation(batch['rgb'], batch['agentview_pcd'][..., :3], batch['tcp_pose'],
        batch['gripper_state'].reshape(-1, 1), tuple(batch['skill_spec']), batch['skill_start_tcp'])


class ProgressDataset(CompletionDataset):
    """Same audited boundary proxies as v2, now queried by an explicit skill spec."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase_cache = {}

    def __getitem__(self, index):
        from models.pose_token_diffusion_v2.targets import segments
        row = self._samples[index]
        sample = super().__getitem__(index)
        demo = self._files[row['file']]['data'][row['demo']]
        key = row['file'], row['demo']
        if key not in self._phase_cache:
            self._phase_cache[key] = segments(demo['skill_instructions'][:])
        phases = self._phase_cache[key]
        phase = row['query_phase']
        sample['skill_spec'] = annotated_skill(phases[phase]['instruction'],
            previous=[p['instruction'] for p in phases[:phase]], step_id=str(phase))
        sample['skill_start_tcp'] = torch.from_numpy(np.asarray(
            demo['obs']['tcp_pose'][phases[phase]['start']], dtype=np.float32))
        sample.pop('task_instruction')
        sample.pop('completion_query_phase')
        sample['progress_target'] = sample.pop('phase_termination')
        return sample
