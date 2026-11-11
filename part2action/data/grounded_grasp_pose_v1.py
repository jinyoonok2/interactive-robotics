"""Grasp-pose labels for shared visual features; no label is a policy input."""
import torch
from .oracle_grasp_pose_v1 import OracleGraspPoseDatasetV1


class GroundedGraspPoseDatasetV1(OracleGraspPoseDatasetV1):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        target = self.target_for(self._samples[index])
        # Current measured observation precedes or equals the pre-close frame.
        # A first-grasp target is not supervision for later skills or moving an
        # already-grasped object. All those frames still train the action policy.
        valid = bool(target['valid'] and sample['meta']['t'] < target['closure_action_index'])
        sample['grasp_pose_target'] = sample.pop('goal_pose')
        sample.pop('goal_valid')
        sample['grasp_pose_valid'] = torch.tensor(float(valid))
        return sample
