"""V3 wiring: explicit skill -> grounding -> current goal -> diffusion actions."""
from dataclasses import dataclass
from torch import nn

from models.pose_token_diffusion.backbones import load_frozen_backbones
from models.pose_token_diffusion.blocks.grounding import TaskRegionGrounding
from models.pose_token_diffusion.blocks.action_decoder import PoseTokenDiffusionDecoder
from models.pose_token_diffusion_v2.blocks import SkillPoseTokenizer
from . import ARCHITECTURE_ID
from .config import PoseTokenDiffusionV3Config
from .encoder import SkillObservationEncoder
from .blocks import CurrentSkillGoalHead, SkillProgressMonitor


@dataclass(frozen=True)
class PolicyOutput:
    actions: dict
    encoded: object
    grounding: object
    predicted_goal: object
    pose_memory: object
    progress_logits: object


class PoseTokenDiffusionV3Policy(nn.Module):
    architecture_id = ARCHITECTURE_ID

    def __init__(self, config, *, backbones=None):
        super().__init__()
        if not isinstance(config, PoseTokenDiffusionV3Config):
            raise TypeError('V3 requires its own configuration')
        self.config = config
        if backbones is None:
            backbones = load_frozen_backbones(config, img_size=config.img_size, text_device=config.text_device)
        self.blocks = nn.ModuleDict({
            'encoder': SkillObservationEncoder(config, backbones),
            'grounding': TaskRegionGrounding(config), 'goal_head': CurrentSkillGoalHead(config),
            'pose_tokenizer': SkillPoseTokenizer(config), 'action_decoder': PoseTokenDiffusionDecoder(config),
            'progress_monitor': SkillProgressMonitor(config)})

    def encode_and_ground(self, observation):
        observation.validate()
        encoded = self.blocks['encoder'](observation)
        encoded.validate(observation.rgb.shape[0], self.config.hidden_dim)
        region = self.blocks['grounding'](encoded)
        region.validate(observation.rgb.shape[0], self.config.hidden_dim)
        return encoded, region

    def progress(self, observation):
        encoded, region = self.encode_and_ground(observation)
        return self.blocks['progress_monitor'](encoded, region)

    def forward(self, observation, *, target_actions=None):
        if target_actions is not None and target_actions.shape != (observation.rgb.shape[0], self.config.action_horizon, 7):
            raise ValueError('Action target shape differs from the v3 configuration')
        encoded, region = self.encode_and_ground(observation)
        goal = self.blocks['goal_head'](encoded, region, observation.skills, observation.skill_start_tcp)
        goal.validate(observation.rgb.shape[0], 1)
        memory = self.blocks['pose_tokenizer'](encoded, goal if self.config.pose_source == 'predicted' else None)
        memory.validate(observation.rgb.shape[0], self.config.hidden_dim)
        actions = self.blocks['action_decoder'](encoded, memory, region=region, target_actions=target_actions)
        return PolicyOutput(actions, encoded, region, goal, memory,
                            self.blocks['progress_monitor'](encoded, region))


def build_policy(config, *, backbones=None):
    return PoseTokenDiffusionV3Policy(config, backbones=backbones)
