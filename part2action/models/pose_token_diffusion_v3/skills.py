"""Validated planner-to-executor interface. No metric pose or simulator asset IDs.

Free-form current-skill text is retained for T5 grounding. Structured categories
carry explicit interaction intent; they do not replace visual part localization.
"""
from dataclasses import asdict, dataclass, fields
import json
import math
import re

from models.pose_token_diffusion_v2.targets import skill_kind

SKILLS = ('grasp_obj', 'touch_obj', 'move_gripper', 'rotate_obj', 'release_obj')
INTERACTIONS = ('unspecified', 'close', 'closed_contact', 'maintain_grasp', 'open')
DIRECTIONS = ('unspecified', 'up', 'down', 'left', 'right', 'forward', 'backward')


@dataclass(frozen=True)
class SkillSpec:
    kind: str
    instruction: str
    object_ref: str = ''
    target_part: str = ''
    interaction: str = 'unspecified'
    direction: str = 'unspecified'
    distance_m: float | None = None
    step_id: str = '0'

    def __post_init__(self):
        if self.kind not in SKILLS or self.interaction not in INTERACTIONS or self.direction not in DIRECTIONS:
            raise ValueError('Unsupported skill, interaction, or direction')
        for key in ('instruction', 'object_ref', 'target_part', 'step_id'):
            value = getattr(self, key)
            if not isinstance(value, str) or len(value) > 512:
                raise ValueError(key + ' must be a short string')
        if not self.instruction.strip() or not self.step_id.strip():
            raise ValueError('A current skill instruction and stable step ID are required')
        if skill_kind(self.instruction) != self.kind:
            raise ValueError('Skill kind and current instruction disagree')
        if self.kind in {'grasp_obj', 'touch_obj', 'rotate_obj'} and not (self.object_ref and self.target_part):
            raise ValueError('Contact/reorientation skills must name an observed object and part')
        expected = {'grasp_obj': 'close', 'touch_obj': 'closed_contact', 'release_obj': 'open'}
        if self.kind in expected and self.interaction != expected[self.kind]:
            raise ValueError('Skill and gripper interaction disagree')
        if self.distance_m is not None:
            if isinstance(self.distance_m, bool) or not isinstance(self.distance_m, (int, float)) or not math.isfinite(self.distance_m) or not 0 < self.distance_m <= 1:
                raise ValueError('Instruction-specified displacement must be in (0, 1] meters')
            if self.kind != 'move_gripper' or self.direction == 'unspecified':
                raise ValueError('A displacement requires a directed move skill')

    @property
    def language(self):
        # Include explicit bindings for planner outputs whose instruction uses
        # a pronoun. The same template is used for annotated training skills.
        return f'{self.instruction} Object: {self.object_ref or "unspecified"}. Part: {self.target_part or "unspecified"}.'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError('Planner output must contain only documented skill fields')
        if not {'kind', 'instruction', 'step_id'} <= set(value):
            raise ValueError('Planner output lacks its current skill or step ID')
        return cls(**value)

    @classmethod
    def from_json(cls, text):
        return cls.from_mapping(json.loads(text))


def annotated_skill(text, *, previous=(), step_id='0'):
    """Compile current/past annotation text only; never inspect future outcomes."""
    text = ' '.join(str(text).strip().split())
    kind = skill_kind(text)
    if kind not in SKILLS:
        raise ValueError('Unsupported annotated skill: ' + text)
    obj, part = '', ''
    match = re.match(r'(?:grasp|touch) the (.+?) at its (.+)$', text, re.I)
    rotation = re.match(r'(?:reorient|rotate) the (.+?) of the (.+?) to face .+$', text, re.I)
    if match:
        obj, part = match.groups()
    elif rotation:
        part, obj = rotation.groups()
    elif kind in {'grasp_obj', 'touch_obj', 'rotate_obj'}:
        raise ValueError('Cannot extract the annotated object/part binding: ' + text)
    interaction = {'grasp_obj': 'close', 'touch_obj': 'closed_contact', 'release_obj': 'open'}.get(kind, 'unspecified')
    if kind in {'move_gripper', 'rotate_obj'}:
        for previous_text in reversed(previous):
            prior = skill_kind(previous_text)
            if prior in {'grasp_obj', 'release_obj', 'touch_obj'}:
                interaction = 'maintain_grasp' if prior == 'grasp_obj' else 'unspecified'
                break
    direction = 'unspecified'
    if kind == 'move_gripper':
        for word, canonical in [('upwards', 'up'), ('downwards', 'down'), ('left', 'left'),
                                ('right', 'right'), ('forwards', 'forward'), ('backwards', 'backward')]:
            if re.search(r'\b' + word + r'\b', text, re.I):
                direction = canonical
                break
    return SkillSpec(kind, text, obj, part, interaction, direction, step_id=str(step_id))
