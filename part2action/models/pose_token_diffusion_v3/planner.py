"""Frozen VLM planner adapter with a strict semantic output boundary.

Importing this module neither downloads nor loads a VLM. A local, commit-pinned
Qwen checkpoint is required to instantiate the actual planner. The low-level
trainer does not instantiate it.
"""
from dataclasses import dataclass
import json
import re

from .skills import SkillSpec, SKILLS, INTERACTIONS, DIRECTIONS


@dataclass(frozen=True)
class PlannerRequest:
    task: str
    rgb: object                       # observed uint8 HWC image
    tcp_pose: tuple
    opening: float
    current_skill: SkillSpec | None
    progress_probability: float | None
    history: tuple
    reason: str


@dataclass(frozen=True)
class PlannerDecision:
    status: str
    skill: SkillSpec | None = None

    def __post_init__(self):
        if self.status not in {'execute', 'done', 'uncertain'}:
            raise ValueError('Unknown planner decision')
        if (self.status == 'execute') != isinstance(self.skill, SkillSpec):
            raise ValueError('Only execute decisions contain a validated skill')

    @classmethod
    def from_json(cls, text):
        value = json.loads(text)
        if not isinstance(value, dict) or set(value) != {'status', 'skill'}:
            raise ValueError('Expected exactly status and skill in planner response')
        skill = SkillSpec.from_mapping(value['skill']) if value['skill'] is not None else None
        return cls(value['status'], skill)


class JsonSkillPlanner:
    """Service-independent adapter; response generation can be tested separately."""
    source = 'predicted_planner'

    def __init__(self, generate):
        self.generate = generate

    def plan(self, request):
        return PlannerDecision.from_json(self.generate(request))


class QwenVLMPlanner(JsonSkillPlanner):
    MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'

    def __init__(self, *, revision, device='cuda', model_id=MODEL_ID, max_new_tokens=384):
        if not isinstance(revision, str) or not re.fullmatch(r'[a-f0-9]{40}', revision):
            raise ValueError('Pin the planner to a 40-character Hugging Face commit revision')
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.model_id, self.revision = model_id, revision
        self.max_new_tokens = max_new_tokens
        # No implicit download or mutation of the training environment.
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, revision=revision,
            local_files_only=True, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32).to(device).eval()
        self.model.requires_grad_(False)
        super().__init__(self._generate)

    def _generate(self, request):
        import torch
        from PIL import Image
        context = {'task': request.task, 'measured_tcp_xyzw_xyz': request.tcp_pose,
            'measured_finger_opening_m': request.opening,
            'current_skill': request.current_skill.to_dict() if request.current_skill else None,
            'transition_readiness_estimate': request.progress_probability,
            'review_reason': request.reason, 'execution_history': request.history}
        prompt = (
            'Choose only the current robot skill using the observed image and context. '
            'Return JSON only, with exactly {"status":"execute|done|uncertain", "skill":object_or_null}. '
            'A readiness estimate or timeout does not prove physical success. '
            'Use done only when the visible task outcome is achieved; use uncertain when evidence is insufficient. '
            'For execute, skill has fields kind, instruction, object_ref, target_part, interaction, direction, distance_m, step_id. '
            f'Allowed kinds: {SKILLS}; interactions: {INTERACTIONS}; directions: {DIRECTIONS}. '
            'Use current-skill instructions beginning Grasp, Touch, Move, Reorient, or Release, consistent with kind. '
            'Grasp requires close; touch requires closed_contact; release requires open. '
            'Name the observed object and its part for grasp/touch/reorientation; do not invent simulator IDs. '
            'Keep step_id stable while refining the same skill. Change it for a new skill or explicit retry. '
            'Set distance_m to null unless the task explicitly specifies a metric displacement. '
            'Do not output motor commands, poses, future skill lists, or hidden state. Context: ' +
            json.dumps(context, allow_nan=False))
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': Image.fromarray(request.rgb)}, {'type': 'text', 'text': prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt').to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.decode(generated[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
