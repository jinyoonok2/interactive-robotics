"""Explicit v2 config; shared block settings are validated without changing v1."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from models.pose_token_diffusion.config import PoseTokenDiffusionConfig as BlockConfig
from . import ARCHITECTURE_ID


@dataclass(frozen=True)
class PoseTokenDiffusionV2Config(BlockConfig):
    architecture: str = ARCHITECTURE_ID
    schema_version: int = 1
    pose_source: str = 'predicted'
    progress_enabled: bool = False

    def __post_init__(self):
        if self.architecture != ARCHITECTURE_ID or type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Expected pose_token_diffusion_v2 config schema 1')
        if self.progress_enabled:
            raise ValueError('V2 initially retains the existing learned skill termination mechanism')
        # Validate the unchanged encoder/decoder contracts using the original
        # validator. This neither builds a v1 policy nor loads a checkpoint.
        values = asdict(self)
        values.update(architecture='pose_token_diffusion_v1', schema_version=2)
        BlockConfig(**values)

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict):
            raise ValueError('Expected a v2 configuration mapping')
        unknown = set(values)-{field.name for field in fields(cls)}
        if unknown or values.get('architecture') != ARCHITECTURE_ID:
            raise ValueError('Explicit v2 architecture required; unknown settings: '+str(sorted(unknown)))
        values = dict(values)
        for key in ('action_scale', 'action_offset'):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def load_config(path: str | Path):
    import yaml
    return PoseTokenDiffusionV2Config.from_mapping(yaml.safe_load(Path(path).read_text()))
