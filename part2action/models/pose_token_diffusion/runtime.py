"""Native rollout adapter; all executed actions come from the new visual policy."""
from torch import nn
from .contracts import Observation


class PoseTokenRolloutPolicy(nn.Module):
    use_action_deltas = False

    def __init__(self, policy, *, provided_pose=None):
        super().__init__()
        if policy.config.pose_source != 'provided' and provided_pose is not None:
            raise ValueError('Provided targets cannot enter predicted/disabled deployment')
        self.policy, self.provided_pose = policy, provided_pose
        self.last_output = None

    def forward(self, rgb, instructions, *, agentview_pcd, tcp_pose, gripper_state,
                plan_slots_override=None, slot_valid_logits_override=None,
                current_phase=None, force_current_phase=False):
        observation = Observation(rgb, agentview_pcd[..., :3], instructions, tcp_pose,
            gripper_state.reshape(rgb.shape[0], 1), plan_slots_override,
            slot_valid_logits_override, current_phase, force_current_phase)
        out = self.policy(observation, provided_pose=self.provided_pose)
        self.last_output = out
        result = {**out.actions, **(out.encoded.hierarchy or {})}
        if out.progress_logits is not None:
            result['termination_logits'] = out.progress_logits
        if out.predicted_pose is not None:
            result['predicted_pose_position_world'] = out.predicted_pose.position_world
            result['predicted_pose_rotation_world'] = out.predicted_pose.rotation_world
            result['pose_applicability_logits'] = out.predicted_pose.applicability_logits
        return result
