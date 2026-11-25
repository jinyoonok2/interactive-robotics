"""Independent pose-token diffusion policy composed from replaceable blocks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import ARCHITECTURE_ID
from .config import PoseTokenDiffusionConfig
from .contracts import EncodedObservation, GroundedRegion, Observation, PoseMemory, TargetPose


@dataclass(frozen=True)
class PoseTokenDiffusionBlocks:
    encoder: nn.Module
    grounding: nn.Module
    pose_tokenizer: nn.Module
    action_decoder: nn.Module
    pose_head: nn.Module | None = None
    progress_head: nn.Module | None = None


@dataclass(frozen=True)
class PolicyOutput:
    actions: dict[str, torch.Tensor]
    grounding: GroundedRegion
    target_pose: TargetPose | None
    pose_memory: PoseMemory
    progress_logits: torch.Tensor | None
    encoded: EncodedObservation | None = None
    predicted_pose: TargetPose | None = None


class PoseTokenDiffusionPolicy(nn.Module):
    architecture_id = ARCHITECTURE_ID

    def __init__(self, config: PoseTokenDiffusionConfig, blocks: PoseTokenDiffusionBlocks):
        super().__init__()
        self.config = config
        required = ('encoder', 'grounding', 'pose_tokenizer', 'action_decoder')
        if any(not isinstance(getattr(blocks, name), nn.Module) for name in required):
            raise ValueError('Every required block must be an explicit torch.nn.Module')
        if config.pose_source == 'predicted' and not isinstance(blocks.pose_head, nn.Module):
            raise ValueError('Predicted-pose mode requires an explicit pose_head')
        if config.progress_enabled and not isinstance(blocks.progress_head, nn.Module):
            raise ValueError('Enabled progress requires an explicit progress_head')
        modules = {name: getattr(blocks, name) for name in required}
        for name in ('pose_head', 'progress_head'):
            value = getattr(blocks, name)
            if value is not None:
                if not isinstance(value, nn.Module):
                    raise ValueError(name+' must be a module or None')
                modules[name] = value
        self.blocks = nn.ModuleDict(modules)

    def forward(self, observation: Observation, *, provided_pose: TargetPose | None = None,
                target_actions: torch.Tensor | None = None):
        observation.validate()
        batch_size = observation.rgb.shape[0]
        if self.config.pose_source == 'provided':
            if provided_pose is None:
                raise ValueError('Provided-pose mode requires an explicit target; no silent fallback')
            provided_pose.validate(batch_size)
        elif provided_pose is not None:
            raise ValueError('Provided target poses are forbidden in predicted/disabled modes')
        if target_actions is not None and target_actions.shape != (
                batch_size, self.config.action_horizon, self.config.action_dim):
            raise ValueError('Action supervision does not match the configured action chunk')
        encoded = self.blocks['encoder'](observation)
        if not isinstance(encoded, EncodedObservation):
            raise TypeError('encoder must return EncodedObservation')
        encoded.validate(batch_size, self.config.hidden_dim)
        region = self.blocks['grounding'](encoded)
        if not isinstance(region, GroundedRegion):
            raise TypeError('grounding must return GroundedRegion')
        region.validate(batch_size, self.config.hidden_dim)
        predicted = None
        if 'pose_head' in self.blocks:
            predicted = self.blocks['pose_head'](encoded, region)
            if not isinstance(predicted, TargetPose):
                raise TypeError('pose_head must return TargetPose')
            predicted.validate(batch_size)
        goal = predicted if self.config.pose_source == 'predicted' else provided_pose
        memory = self.blocks['pose_tokenizer'](encoded, goal)
        if not isinstance(memory, PoseMemory):
            raise TypeError('pose_tokenizer must return PoseMemory')
        memory.validate(batch_size, self.config.hidden_dim)
        if self.config.pose_source == 'disabled' and (memory.valid_mask.any() or memory.tokens.count_nonzero() or memory.relative_error.count_nonzero()):
            raise ValueError('Disabled pose mode must supply zero pose tokens/errors and an invalid mask')
        # Keep scene, language, robot and pose memory separate at this boundary.
        # Only the action decoder receives action targets, for its training loss.
        actions = self.blocks['action_decoder'](encoded, memory, region=region, target_actions=target_actions)
        if not isinstance(actions, dict) or not ({'action_chunk', 'action_noise_pred'} & actions.keys()):
            raise ValueError('action_decoder must return action_chunk and/or action_noise_pred')
        for name in ('action_chunk', 'action_noise_pred'):
            if name in actions and actions[name].shape != (batch_size, self.config.action_horizon, self.config.action_dim):
                raise ValueError(name+' does not match the configured action chunk')
        progress = None
        if self.config.progress_enabled:
            progress = self.blocks['progress_head'](encoded, memory)
            if progress.shape != (batch_size,):
                raise ValueError('Progress logits must be (B,)')
        return PolicyOutput(actions, region, goal, memory, progress, encoded, predicted)


def build_policy(config: PoseTokenDiffusionConfig, *, blocks: PoseTokenDiffusionBlocks | None = None, backbones=None):
    """Explicit new-family constructor; never redirects a legacy model builder."""
    if blocks is not None and backbones is not None:
        raise ValueError('Supply custom blocks or backbones, not both')
    if blocks is None:
        from .factory import build_default_blocks
        blocks = build_default_blocks(config, backbones=backbones)
    return PoseTokenDiffusionPolicy(config, blocks)
