"""V3 identity and validated settings for independently instantiated shared blocks."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from models.pose_token_diffusion.config import PoseTokenDiffusionConfig as BlockConfig
from . import ARCHITECTURE_ID


@dataclass(frozen=True)
class PoseTokenDiffusionV3Config(BlockConfig):
    architecture: str = ARCHITECTURE_ID
    schema_version: int = 1
    pose_source: str = 'predicted'
    # The progress monitor is an independent v3 block, not the legacy head.
    progress_enabled: bool = True

    def __post_init__(self):
        if self.architecture != ARCHITECTURE_ID or type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Expected pose_token_diffusion_v3 schema 1')
        if self.pose_source not in {'predicted', 'disabled'} or self.progress_enabled is not True:
            raise ValueError('V3 supports predicted/disabled pose conditioning and a separate progress monitor')
        values = asdict(self)
        values.update(architecture='pose_token_diffusion_v1', schema_version=2, progress_enabled=False)
        BlockConfig(**values)

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict) or values.get('architecture') != ARCHITECTURE_ID:
            raise ValueError('An explicit v3 configuration is required')
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError('Unknown v3 settings: ' + str(sorted(unknown)))
        values = dict(values)
        for key in ('action_scale', 'action_offset'):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def load_config(path: str | Path):
    import yaml
    return PoseTokenDiffusionV3Config.from_mapping(yaml.safe_load(Path(path).read_text()))
