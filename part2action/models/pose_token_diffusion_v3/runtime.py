"""Independent v3 execution loop: geometric refresh and semantic review are separate."""
import numpy as np
import torch

from .contracts import SkillObservation
from .planner import PlannerRequest, PlannerDecision


class HierarchicalRolloutPolicy:
    """Single-environment adapter; callers execute the returned action chunk.

    Review runs at chunk boundaries using the previous observation's monitor
    estimate. A high estimate requests planning; it never directly increments
    a phase. A repeated specification preserves the measured skill-start pose.
    """
    def __init__(self, policy, planner, *, review_every_chunks=8, min_skill_chunks=2,
                 completion_threshold=.5, max_skill_chunks=32):
        if review_every_chunks < 1 or min_skill_chunks < 1 or max_skill_chunks < min_skill_chunks or not 0 < completion_threshold < 1:
            raise ValueError('Invalid hierarchical runtime review settings')
        self.policy, self.planner = policy, planner
        self.review_every_chunks, self.min_skill_chunks = review_every_chunks, min_skill_chunks
        self.threshold, self.max_skill_chunks = completion_threshold, max_skill_chunks
        self.reset()

    def reset(self):
        self.task, self.skill, self.anchor = None, None, None
        self.skill_chunks, self.chunks_since_review = 0, 0
        self.progress_probability = None
        self.history = []
        self.stopped = False

    def _review_reason(self):
        if self.skill is None:
            return 'initial'
        if self.chunks_since_review == 0:
            return None
        if self.skill_chunks >= self.max_skill_chunks:
            return 'skill_timeout'
        if self.skill_chunks >= self.min_skill_chunks and self.progress_probability >= self.threshold:
            return 'monitor_requested_review'
        if self.chunks_since_review >= self.review_every_chunks:
            return 'periodic_review'
        return None

    @torch.inference_mode()
    def act(self, *, rgb, scene_points_camera, tcp_pose, gripper_opening, task=None):
        if rgb.shape[0] != 1:
            raise ValueError('The v3 planner adapter currently supports one environment at a time')
        if self.stopped:
            raise RuntimeError('Reset the runtime after a planner stop')
        if task is not None:
            if self.task is not None and task != self.task:
                raise ValueError('Reset before changing the task')
            self.task = task
        if not self.task:
            raise ValueError('The planner needs the task instruction')
        reason = self._review_reason()
        if reason is not None:
            frame = rgb[0, -1] if rgb.ndim == 5 else rgb[0]
            image = (frame.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
            request = PlannerRequest(self.task, image, tuple(tcp_pose[0].detach().cpu().tolist()),
                float(gripper_opening.reshape(-1)[0]), self.skill, self.progress_probability,
                tuple(self.history[-16:]), reason)
            decision = self.planner.plan(request)
            if not isinstance(decision, PlannerDecision):
                raise TypeError('Planner must return a validated PlannerDecision')
            self.history.append({'review_reason': reason, 'status': decision.status,
                'readiness_estimate': self.progress_probability,
                'skill': decision.skill.to_dict() if decision.skill else None})
            self.chunks_since_review = 0
            if decision.status != 'execute':
                self.stopped = True
                return {'status': 'planner_' + decision.status, 'action_chunk': None,
                        'physical_success_verified': False}
            # A language refinement of the same step preserves its reference.
            if self.skill is None or decision.skill.step_id != self.skill.step_id or decision.skill.kind != self.skill.kind:
                self.anchor = tcp_pose.detach().clone()
                self.skill_chunks = 0
            self.skill = decision.skill
        observation = SkillObservation(rgb, scene_points_camera[..., :3], tcp_pose,
            gripper_opening.reshape(1, 1), (self.skill,), self.anchor)
        self.policy.eval()
        output = self.policy(observation)
        self.progress_probability = float(output.progress_logits.float().sigmoid()[0])
        self.skill_chunks += 1
        self.chunks_since_review += 1
        return {**output.actions, 'status': 'execute', 'current_skill': self.skill.to_dict(),
            'progress_probability': self.progress_probability, 'planner_source': self.planner.source,
            'predicted_goal_position_world': output.predicted_goal.position_world.detach(),
            'predicted_goal_opening': output.predicted_goal.opening.detach(),
            'physical_success_verified': False}
