"""Per-phase targets are training labels; observations use the v1 allowlist."""
import inspect
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from data.partinstruct_loader import PartInstructDataset
from models.pose_token_diffusion.dataset import observation_from_batch
from models.pose_token_diffusion.blocks.geometry import quaternion_matrix
from .contracts import SkillGoalBank
from .targets import TargetSettings, SKILLS, phase_targets, labels_at, skill_kind


class SkillPoseDataset(PartInstructDataset):
    def __init__(self, *args, target_settings=None, validation_fraction=.1, split_seed=42,
                 split='train', demo_manifest=None, **kwargs):
        super().__init__(*args,**kwargs)
        if not self.use_tcp_pose or not self.use_hierarchy or self.action_target_offset != 1:
            raise ValueError('V2 requires measured TCP, hierarchy labels, and next-action alignment')
        self.target_settings = target_settings or TargetSettings()
        self._goal_cache = {}
        if split not in {'train','validation','all'} or not 0 <= validation_fraction < 1:
            raise ValueError('Invalid complete-demonstration split')
        if demo_manifest is not None:
            manifest = json.loads(Path(demo_manifest).read_text())
            if manifest.get('schema_version') != 1:
                raise ValueError('Unsupported pilot demonstration manifest')
            records = manifest['demonstrations']
            keys = [(str(Path(r['file']).resolve()),r['demo']) for r in records]
            if len(keys)!=len(set(keys)) or any(r['split'] not in {'train','validation'} for r in records):
                raise ValueError('Pilot manifest contains duplicate demos or invalid split labels')
            available = {(s.file_path,s.demo_key) for s in self._samples}
            requested = {key for key,r in zip(keys,records) if split=='all' or r['split']==split}
            if not requested <= available:
                raise ValueError('Pilot manifest references unavailable demonstrations')
            self._samples = [s for s in self._samples if (s.file_path,s.demo_key) in requested]
            self.split_info = {'split':split,'manifest':str(Path(demo_manifest).resolve()),
                'manifest_sha256':hashlib.sha256(Path(demo_manifest).read_bytes()).hexdigest(),
                'demonstrations':sorted(requested),'samples':len(self._samples)}
            return
        by_file = {}
        for spec in self._samples:
            by_file.setdefault(spec.file_path,set()).add(spec.demo_key)
        validation = set()
        for path,keys in by_file.items():
            order = sorted(keys,key=lambda key:hashlib.sha256(f'{split_seed}:{path}:{key}'.encode()).hexdigest())
            count = max(1,int(len(order)*validation_fraction)) if len(order)>1 and validation_fraction else 0
            validation.update((path,key) for key in order[:count])
        self._samples = [spec for spec in self._samples if split=='all' or
            (((spec.file_path,spec.demo_key) in validation) == (split=='validation'))]
        self.split_info = {'split':split,'validation_fraction':validation_fraction,'split_seed':split_seed,
            'demonstrations':sorted({(s.file_path,s.demo_key) for s in self._samples}),
            'samples':len(self._samples)}

    def targets_for(self, spec):
        key = spec.file_path,spec.demo_key
        if key not in self._goal_cache:
            demo = self._open(spec.file_path)['data'][spec.demo_key]
            instructions = demo['skill_instructions'][:]
            has_touch = any(skill_kind(x.decode() if isinstance(x,bytes) else x) == 'touch_obj' for x in instructions)
            part = np.asarray(demo['obs'][self.part_pcd_key][:]) if has_touch and self.part_pcd_key in demo['obs'] else None
            self._goal_cache[key] = phase_targets(np.asarray(demo['actions']),
                np.asarray(demo['obs'][self.tcp_pose_key]),instructions,part_pcd=part,
                settings=self.target_settings,max_slots=self.max_skill_slots)
        return self._goal_cache[key]

    def __getitem__(self,index):
        sample = super().__getitem__(index)
        phases = self.targets_for(self._samples[index])
        labels = labels_at(phases,self._samples[index].t,self.max_skill_slots)
        active = labels.pop('goal_active_phase')
        if active != int(sample['phase_index']):
            raise ValueError('Goal and hierarchy phase annotations disagree')
        phase = phases[active]
        sample['meta'].update(goal_label_quality=phase['quality'],goal_label_reason=phase['reason'],
            goal_target_t=phase['target_t'],touch_proxy_distance_m=phase['touch_proxy_to_part_distance_m'])
        sample.update({key:torch.from_numpy(value) for key,value in labels.items()})
        return sample


def build_dataset(data, goal_labels=None):
    accepted = inspect.signature(PartInstructDataset.__init__).parameters
    own = {'validation_fraction','split_seed','split','demo_manifest'}
    unknown = set(data)-set(accepted)-own-{'train_hdf5','batch_size','num_workers','pin_memory'}
    if unknown:
        raise ValueError('Unknown v2 data options: '+str(sorted(unknown)))
    kwargs = {key:value for key,value in data.items() if (key in accepted or key in own) and key != 'self'}
    kwargs['hdf5_paths'] = data['train_hdf5']
    return SkillPoseDataset(**kwargs,target_settings=TargetSettings(**(goal_labels or {})))


def provided_goals_from_batch(batch):
    valid = batch['goal_component_valid'].bool()
    pose = batch['goal_poses'].float()
    identity = pose.new_tensor([0.,0.,0.,1.,0.,0.,0.])
    pose = torch.where(valid[:,:,:2].any(-1)[...,None],pose,identity)
    return SkillGoalBank(pose[...,4:7],quaternion_matrix(pose[...,:4]),batch['goal_opening'].float(),
        valid,20*F.one_hot(batch['goal_skill_kind'].long(),len(SKILLS)).float())
