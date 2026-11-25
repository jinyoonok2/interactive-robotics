"""Configuration for the new composition API; never accepted as a legacy recipe."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping
import math

from . import ARCHITECTURE_ID


@dataclass(frozen=True)
class PoseTokenDiffusionConfig:
    architecture: str = ARCHITECTURE_ID
    schema_version: int = 2
    implementation_stage: str = 'implemented'
    hidden_dim: int = 256
    action_horizon: int = 8
    action_dim: int = 7
    pose_source: str = 'provided'
    progress_enabled: bool = False
    visual_backbone: str = 'dinov2_vits14_reg'
    language_backbone: str = 'google/flan-t5-base'
    img_size: int = 252
    n_obs_steps: int = 2
    max_skill_slots: int = 4
    hierarchy_layers: int = 2
    temporal_layers: int = 1
    attention_heads: int = 4
    decoder_layers: int = 2
    diffusion_steps: int = 100
    visual_token_grid: int = 8
    scene_token_budget: int = 64
    local_token_budget: int = 8
    translation_scale_m: float = .1
    camera_fov_degrees: float = 60.
    diffusion_clip_sample: float = 2.
    action_scale: tuple = (.5, .5, .5, math.pi, math.pi, math.pi, .02)
    action_offset: tuple = (0., 0., 0., 0., 0., 0., .02)
    text_device: str = 'cpu'
    use_part_gate: bool = True

    def __post_init__(self):
        if self.architecture != ARCHITECTURE_ID or type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError('Expected implemented pose_token_diffusion_v1 configuration schema 2')
        if self.implementation_stage != 'implemented':
            raise ValueError('Scaffold configurations are not runnable model configurations')
        for name in ('hidden_dim', 'action_horizon', 'action_dim', 'img_size', 'n_obs_steps',
                     'max_skill_slots', 'hierarchy_layers', 'temporal_layers', 'attention_heads',
                     'decoder_layers', 'diffusion_steps', 'visual_token_grid', 'scene_token_budget', 'local_token_budget'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name+' must be a positive integer')
        if self.action_dim != 7:
            raise ValueError('The current action contract is world XYZ, RPY, per-finger opening')
        if self.pose_source not in {'provided', 'predicted', 'disabled'}:
            raise ValueError('pose_source must be provided, predicted, or disabled')
        if type(self.progress_enabled) is not bool:
            raise ValueError('progress_enabled must be a boolean')
        if type(self.use_part_gate) is not bool or self.hidden_dim % self.attention_heads or self.img_size % 14 or self.diffusion_steps < 2:
            raise ValueError('Invalid gate, attention width, DINO resolution or diffusion step count')
        for name in ('translation_scale_m', 'diffusion_clip_sample'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name+' must be finite and positive')
        if not math.isfinite(self.camera_fov_degrees) or not 0 < self.camera_fov_degrees < 180:
            raise ValueError('Camera FOV must be in (0,180) degrees')
        if len(self.action_scale) != 7 or len(self.action_offset) != 7 or any(not math.isfinite(v) or v <= 0 for v in self.action_scale) or any(not math.isfinite(v) for v in self.action_offset):
            raise ValueError('Action normalization requires seven positive scales and seven finite offsets')
        if self.visual_backbone != 'dinov2_vits14_reg' or self.language_backbone != 'google/flan-t5-base':
            raise ValueError('The proposed default retains DINOv2 ViT-S/14 registers and Flan-T5-base')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping):
        if not isinstance(values, Mapping):
            raise ValueError('Expected a pose-token configuration mapping')
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError('Unknown pose-token settings: '+', '.join(sorted(map(str, unknown))))
        if values.get('architecture') != ARCHITECTURE_ID:
            raise ValueError('Explicit architecture: pose_token_diffusion_v1 is required; legacy configs are not converted')
        values = dict(values)
        for key in ('action_scale', 'action_offset'):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def load_config(path: str | Path):
    # YAML is imported here so metadata/checkpoint helpers do not load it eagerly.
    import yaml
    return PoseTokenDiffusionConfig.from_mapping(yaml.safe_load(Path(path).read_text()))
