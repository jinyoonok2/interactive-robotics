"""Observation/label boundary and instance-grouped splits for replay records."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def instance_split(records, validation_fraction=.2, seed=42):
    if not 0<validation_fraction<1:raise ValueError('Validation fraction must be between zero and one')
    groups=sorted({r['instance_group'] for r in records})
    if len(groups)<2:raise ValueError('Need at least two object instances for a held-out split')
    random.Random(seed).shuffle(groups)
    held=set(groups[:max(1,min(len(groups)-1,round(len(groups)*validation_fraction)))])
    return ([r for r in records if r['instance_group'] not in held],
            [r for r in records if r['instance_group'] in held])


class InteractionOutcomeDataset(Dataset):
    """No simulator-state labels are placed in the predictor input dictionary."""
    def __init__(self, index_path):
        self.root=Path(index_path).parent
        index=json.loads(Path(index_path).read_text())
        if index['source_split']!='training only':raise ValueError('Expected training-source collection')
        self.records=index['records']

    def __len__(self):return len(self.records)

    def __getitem__(self, index):
        record=self.records[index]
        row=json.loads((self.root/record['result_file']).read_text())
        with np.load(self.root/record['observation_file'],allow_pickle=False) as observation:
            tcp=observation['tcp_pose_xyzw_xyz'][-1]
            state=np.r_[tcp[4:7],tcp[:4],observation['finger_position_m'][-1]].astype(np.float32)
            inputs={'rgb':torch.from_numpy(observation['rgb'].copy()).float()/255.,
                    'pcd_camera_m':torch.from_numpy(observation['pcd_camera_m'].copy()),
                    'robot_state':torch.from_numpy(state),
                    'primitive_context':torch.tensor(row['primitive_context'],dtype=torch.float32),
                    'candidate':torch.tensor(row['candidate_parameters'],dtype=torch.float32),
                    'skill_instruction':record['skill_instruction']}
        labels={'success':torch.tensor(float(row['stable_target_grasp_lift'])),
                'criteria':torch.tensor([float(row['outcome_components']['passes'][k]) for k in ['lift','bilateral','target_contact','drift']]),
                'continuous':torch.tensor([row['lift_m'],row['hold']['bilateral_fraction'],row['hold']['target_contact_fraction'],row['hold_relative_position_drift_m']])}
        return {'inputs':inputs,'labels':labels,'instance_group':record['instance_group'],'source_group':record['source_group']}
