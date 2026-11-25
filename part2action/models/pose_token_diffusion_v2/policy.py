"""Independent v2 composition; no inheritance from a v1 policy or builder."""
from dataclasses import dataclass, replace
import torch
from torch import nn

from models.pose_token_diffusion.backbones import load_frozen_backbones
from .encoder import CompletionObservationEncoder
from models.pose_token_diffusion.blocks.grounding import TaskRegionGrounding
from models.pose_token_diffusion.blocks.action_decoder import PoseTokenDiffusionDecoder
from models.pose_token_diffusion.contracts import GroundedRegion
from . import ARCHITECTURE_ID
from .config import PoseTokenDiffusionV2Config
from .contracts import SkillGoalBank
from .blocks import SkillGoalHead, SkillPoseTokenizer


@dataclass(frozen=True)
class PolicyOutput:
    actions: dict
    encoded: object
    grounding: GroundedRegion
    grounding_bank: tuple
    predicted_goals: SkillGoalBank
    selected_goal: SkillGoalBank | None
    pose_memory: object


class PoseTokenDiffusionV2Policy(nn.Module):
    architecture_id = ARCHITECTURE_ID

    def __init__(self, config, *, backbones=None):
        super().__init__()
        if not isinstance(config,PoseTokenDiffusionV2Config):
            raise TypeError('V2 requires its own explicit configuration')
        self.config = config
        if backbones is None:
            backbones = load_frozen_backbones(config,img_size=config.img_size,text_device=config.text_device)
        self.blocks = nn.ModuleDict({'encoder':CompletionObservationEncoder(config,backbones),
            'grounding':TaskRegionGrounding(config),'goal_head':SkillGoalHead(config),
            'pose_tokenizer':SkillPoseTokenizer(config),'action_decoder':PoseTokenDiffusionDecoder(config)})

    def forward(self, observation, *, provided_goals=None, target_actions=None):
        observation.validate()
        batch, slots = observation.rgb.shape[0],self.config.max_skill_slots
        if self.config.pose_source == 'provided':
            if not isinstance(provided_goals,SkillGoalBank):
                raise ValueError('Provided v2 mode requires an explicit per-slot goal bank')
            provided_goals.validate(batch,slots)
        elif provided_goals is not None:
            raise ValueError('GT goals cannot enter predicted/disabled deployment')
        if target_actions is not None and target_actions.shape != (batch,self.config.action_horizon,self.config.action_dim):
            raise ValueError('Invalid v2 action supervision shape')
        encoded = self.blocks['encoder'](observation)
        encoded.validate(batch,self.config.hidden_dim)
        # The same grounding module processes each learned plan slot. GT phase
        # indices/skill texts are never needed for forward goal selection.
        regions = tuple(self.blocks['grounding'](replace(encoded,language_tokens=torch.cat([
            encoded.language_tokens[:,i+1:i+2],encoded.language_tokens[:,1:]],1))) for i in range(slots))
        predicted = self.blocks['goal_head'](encoded,regions)
        predicted.validate(batch,slots)
        phases = encoded.hierarchy['predicted_phase']
        indices = torch.arange(batch,device=phases.device)
        region = GroundedRegion(**{name:torch.stack([getattr(r,name) for r in regions],1)[indices,phases]
            for name in GroundedRegion.__dataclass_fields__})
        bank = predicted if self.config.pose_source == 'predicted' else provided_goals
        goal = bank.select(phases) if bank is not None else None
        memory = self.blocks['pose_tokenizer'](encoded,goal)
        memory.validate(batch,self.config.hidden_dim)
        actions = self.blocks['action_decoder'](encoded,memory,region=region,target_actions=target_actions)
        return PolicyOutput(actions,encoded,region,regions,predicted,goal,memory)


def build_policy(config, *, backbones=None):
    return PoseTokenDiffusionV2Policy(config,backbones=backbones)
