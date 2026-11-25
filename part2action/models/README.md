# Named model families

Existing models and the separate v2 experiment have explicit namespaces here.

| Model family | Meaning | Status |
| --- | --- | --- |
| [`part_grounded_diffusion/`](part_grounded_diffusion/README.md) | Part-grounded visual features with diffusion action prediction; includes the original baseline, additive GT-pose extension and earlier learned-pose prototype | Named aliases to the preserved existing implementations |
| [`pose_token_diffusion/`](pose_token_diffusion/README.md) | Proposed explicit pose tokens and scene/robot context for an attention-based diffusion actor | Implemented independent model, training and native evaluation; full training/performance evaluation still separate |
| [`pose_token_diffusion_v2/`](../docs/POSE_TOKEN_DIFFUSION_V2_ARCHITECTURE.md) | Skill-specific pose goals selected by the learned phase, with position/orientation/interaction tokens | Initial runnable baseline; separate config, trainer and checkpoint identity; no full v2 training or benchmark yet |
| [`pose_token_diffusion_v3/`](../docs/POSE_TOKEN_DIFFUSION_V3_ARCHITECTURE.md) | Explicit external skill planning, current-skill grounding and pose goals, separate progress monitor and diffusion actions | Independent implementation and training path; annotated current skills condition executor training; VLM performance requires separate validation |

```text
models/
  part_grounded_diffusion/
    baseline.py                 # existing visual baseline
    gt_pose.py                  # evaluated additive pose extension
    learned_pose_prototype.py   # earlier additive learned-pose prototype
    verify_sources.py          # preservation check
  pose_token_diffusion/
    blocks/                    # actual encoder, grounding, pose and diffusion blocks
    policy.py                  # new model wiring
    contracts.py               # block inputs/outputs
    config.py
    configs/model.yaml + train.yaml
    backbones.py
    checkpoint.py
    tests/
  pose_token_diffusion_v2/
    targets.py                 # training-only per-skill goal definitions
    blocks.py                  # shared skill-goal head and component tokens
    policy.py                  # independent v2 model wiring
    dataset.py + losses.py
    config.py + configs/
    training.py + runtime.py
    checkpoint.py + tests/
  part2action_model.py          # preserved original implementation/import path
  visual_pose_policy_v1.py      # preserved GT-extension implementation/import path
  ...                          # other preserved existing model components
```

The remaining flat files are retained for existing imports, frozen experiment
source checks, and checkpoint compatibility. The named existing-family modules
refer to those classes rather than duplicating their implementations. New
development for the explicit planner/executor experiment belongs in `pose_token_diffusion_v3/`.
V2 creates fresh encoder/grounding/decoder instances from the preserved v1
blocks; it does not inherit either existing policy class or load their weights.

The existing pose model uses checkpoint identity `pose_token_diffusion_v1`;
v2 uses `pose_token_diffusion_v2` and rejects v1 checkpoints. Original checkpoint
tags stay unchanged. Both retain frozen DINOv2 and Flan-T5 through an explicit
adapter. There is no second top-level `architectures/` workspace.
