"""Training-only GT targets; observed inputs remain an explicit allowlist."""
import inspect
import torch
from data.oracle_grasp_pose_v1 import OracleGraspPoseDatasetV1
from data.partinstruct_loader import PartInstructDataset
from .contracts import Observation, TargetPose
from .blocks.geometry import quaternion_matrix


class PoseTokenDataset(OracleGraspPoseDatasetV1):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        target = self.target_for(self._samples[index])
        valid = bool(target['valid'] and sample['meta']['t'] < target['closure_action_index'])
        sample['pose_supervision_valid'] = torch.tensor(float(valid))
        # goal_pose/goal_valid remain static through the episode for GT inputs.
        # A separate validity label limits learned first-grasp supervision.
        return sample


def build_dataset(data):
    accepted = inspect.signature(PartInstructDataset.__init__).parameters
    unknown = set(data)-set(accepted)-{'train_hdf5', 'batch_size', 'num_workers', 'pin_memory'}
    if unknown:
        raise ValueError('Unknown pose-token data options: '+', '.join(sorted(unknown)))
    kwargs = {key: value for key, value in data.items() if key in accepted and key != 'self'}
    kwargs['hdf5_paths'] = data['train_hdf5']
    return PoseTokenDataset(**kwargs)


def observation_from_batch(batch):
    return Observation(batch['rgb'], batch['agentview_pcd'][..., :3], batch['task_instruction'],
                       batch['tcp_pose'], batch['gripper_state'].reshape(-1, 1))


def target_pose_from_tensor(pose, valid):
    valid = valid.reshape(-1).bool()
    identity = pose.new_tensor([0., 0., 0., 1., 0., 0., 0.])
    clean = torch.where(valid[:, None], pose, identity).float()
    return TargetPose(clean[:, 4:7], quaternion_matrix(clean[:, :4]), valid)


def provided_pose_from_batch(batch):
    return target_pose_from_tensor(batch['goal_pose'], batch['goal_valid'])
