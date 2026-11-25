"""V2 goal bank; future labels are not fields of the shared Observation contract."""
from dataclasses import dataclass
import torch

from models.pose_token_diffusion.contracts import Observation, EncodedObservation, GroundedRegion, PoseMemory
from .targets import SKILLS


@dataclass(frozen=True)
class SkillGoalBank:
    position_world: torch.Tensor       # B,S,3
    rotation_world: torch.Tensor       # B,S,3,3
    opening: torch.Tensor              # B,S
    component_valid: torch.Tensor      # B,S,3: translation, rotation, opening/intent
    skill_logits: torch.Tensor         # B,S,len(SKILLS)
    applicability_logits: torch.Tensor | None = None

    def validate(self, batch, slots):
        shapes = {'position_world':(batch,slots,3), 'rotation_world':(batch,slots,3,3),
            'opening':(batch,slots), 'component_valid':(batch,slots,3),
            'skill_logits':(batch,slots,len(SKILLS))}
        for name, shape in shapes.items():
            value = getattr(self,name)
            if value.shape != shape:
                raise ValueError(name+' has an invalid goal-bank shape')
            if name != 'component_valid' and not torch.isfinite(value).all():
                raise ValueError('Nonfinite goal bank: '+name)
        if self.component_valid.dtype != torch.bool:
            raise ValueError('Goal component masks must be boolean')
        if self.applicability_logits is not None and (self.applicability_logits.shape != (batch,slots,3) or not torch.isfinite(self.applicability_logits).all()):
            raise ValueError('Invalid component applicability logits')

    def select(self, phases):
        index = torch.arange(phases.shape[0], device=phases.device)
        return SkillGoalBank(**{name: getattr(self,name)[index,phases][:,None]
            if getattr(self,name) is not None else None for name in self.__dataclass_fields__})
