"""Candidate-conditioned outcome head; no rollout or pretrained weights loaded.

Prototype interface only. Perception/skill features may come from frozen existing
encoders. Post-action poses, contacts, GT part masks and outcomes are never inputs.
"""
import torch
from torch import nn


class InteractionSuccessHead(nn.Module):
    criterion_names=('lift','bilateral','target_contact','drift')

    def __init__(self, observation_dim, skill_dim, hidden_dim=128, detach_context=True):
        super().__init__()
        self.detach_context=detach_context
        # Robot: current world XYZ, quaternion xyzw, finger position (8).
        # Primitive: nominal world XYZ/Euler/finger command + time-to-close (8).
        self.context=nn.Sequential(nn.Linear(observation_dim+skill_dim+16,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.candidates=nn.Sequential(nn.Linear(7,hidden_dim),nn.GELU())
        self.score=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,5))
        self.register_buffer('candidate_scales',torch.tensor([.05,.05,.05,.785398,.785398,.785398,.5]))

    def forward(self, observation_features, skill_features, robot_state, primitive_context, candidates):
        if candidates.ndim!=3 or candidates.shape[-1]!=7:
            raise ValueError('Candidates must be (batch, count, 7): world metres, local rotation-vector radians, seconds')
        inputs=[observation_features,skill_features,robot_state,primitive_context]
        if any(x.ndim!=2 or x.shape[0]!=candidates.shape[0] for x in inputs) or robot_state.shape[-1]!=8 or primitive_context.shape[-1]!=8:
            raise ValueError('Context batch/dimensions do not match candidate batch')
        if self.detach_context:inputs=[x.detach() for x in inputs]
        context=self.context(torch.cat(inputs,dim=-1))[:,None].expand(-1,candidates.shape[1],-1)
        encoded=self.candidates(candidates/self.candidate_scales)
        logits=self.score(torch.cat((context,encoded),dim=-1))
        # Separate joint logit; no independence assumption between criteria.
        return {'success_logits':logits[...,0],'criterion_logits':logits[...,1:]}
