"""Learned phase/plan cache adapter; no expert phase or reference clock."""
from torch import nn
from .contracts import Observation


class PoseTokenV2RolloutPolicy(nn.Module):
    use_action_deltas = False

    def __init__(self,policy):
        super().__init__()
        if policy.config.pose_source == 'provided':
            raise ValueError('The v2 deployment adapter accepts predicted/disabled goals only')
        self.policy = policy
        self.last_output = None

    def forward(self,rgb,instructions,*,agentview_pcd,tcp_pose,gripper_state,
                plan_slots_override=None,slot_valid_logits_override=None,
                current_phase=None,force_current_phase=False):
        observation = Observation(rgb,agentview_pcd[...,:3],instructions,tcp_pose,
            gripper_state.reshape(rgb.shape[0],1),plan_slots_override,slot_valid_logits_override,
            current_phase,force_current_phase)
        output = self.policy(observation)
        self.last_output = output
        return {**output.actions,**output.encoded.hierarchy,
            'predicted_goal_position_world':output.predicted_goals.position_world,
            'predicted_goal_component_logits':output.predicted_goals.applicability_logits}
