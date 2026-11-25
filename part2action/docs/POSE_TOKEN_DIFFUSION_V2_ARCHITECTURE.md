# Pose-token Diffusion v2

Separate Part2Action / Part2Gym experiment, started November 16, 2026.
This implements the **skill-specific goal baseline** in the
[proposal](SKILL_SPECIFIC_POSE_PROPOSAL.md). It is not evidence of a
task-success improvement or an established novel contribution.

## What changes

V1 predicts one first-grasp pose. V2 predicts a bank of goals using the learned
skill slots, then selects the goal belonging to the learned active phase.
The actor uses predicted goals during both training and deployment.

```text
RGB history + observed scene XYZ + task instruction + measured TCP/opening
  -> frozen DINOv2 / Flan-T5 + trainable observation encoder
  -> learned plan slots, phase and termination
  -> shared region grounding for each learned skill slot
  -> shared skill-goal head -> per-slot goal bank
  -> select goal using learned active phase
  -> TCP-relative position token + orientation token + interaction token
  -> cross-attention diffusion decoder, also attending to visual/scene/skill/robot context
  -> 8 world-frame action commands (XYZ, Euler rotation, finger opening)
  -> execute, observe, update phase and goal
```

The visual and language encoders and action decoder remain available throughout
execution. A goal is a conditioning signal, not a substitute for the action
sequence. Skill-specific component applicability is predicted; a touch goal
does not need an arbitrary orientation or closure target. The interaction token
contains predicted skill identity and opening intent, separate from rigid pose.
Action loss can backpropagate through goal prediction and the pose tokens.

## Goal labels and limits

Future demonstration information below is **training supervision only**.
The policy observation allowlist contains RGB, observed scene PCD, instruction,
measured robot state, and optional cached learned plan/phase estimates.

| Skill | Initial target | Components supervised |
| --- | --- | --- |
| Grasp | Measured TCP immediately before closure | Position, orientation, opening |
| Touch | Measured TCP when the fingertip proxy is closest to the labeled part, within 3 cm | Position only; explicitly a geometry proxy |
| Move/lift | Measured TCP at the demonstrated phase endpoint | Position and opening |
| Rotate/reorient | Measured TCP at the demonstrated phase endpoint | Position, TCP orientation, opening; endpoint proxy |
| Release | Measured TCP/opening at the phase endpoint | Position and opening; endpoint proxy |

Missing, distant or invalid targets are masked. Geometric supervision for a
grasp/touch approach ends after reaching its demonstrated event; grasp closure
intent remains supervised while that grasp phase is active. The later movement
and rotation phases have their own goals. Applicability learns these distinctions.

Touch proximity does not verify physical contact. TCP orientation is not an
object-part orientation estimate. Endpoint labels do not verify retention,
placement or task completion. Full target auditing and simulator testing remain
necessary before interpreting performance.

The touch probe follows the local PartGym
[`PandaArm.get_fingertip_pose()`](../../PartInstruct/PartInstruct/PartGym/env/backend/planner/panda_arm.py):
5.625 cm along TCP-local +Z. The touch expert itself offsets its TCP from the
surface, so TCP-to-surface distance is not a contact-distance test. Labels retain
the measured TCP pose at the selected event; the proxy only selects/filters
events. This robot-specific setting must change when the gripper geometry changes.

**Deferred:** explicit held-part versus orientation-target bindings, persistent
part-relative transforms, cross-skill consistency losses, multi-object relation
reasoning, and verified contact/support labels. V2 initially uses world-frame
TCP goals, expressed relative to the current TCP for action conditioning. It
does not repeatedly define a lift target as current height plus a fixed offset.

## Files and isolation

| File | Purpose |
| --- | --- |
| [targets.py](../models/pose_token_diffusion_v2/targets.py) | Per-phase target extraction, label quality and component masks |
| [blocks.py](../models/pose_token_diffusion_v2/blocks.py) | Shared goal predictor and three component memory tokens |
| [policy.py](../models/pose_token_diffusion_v2/policy.py) | Independent v2 module composition |
| [dataset.py](../models/pose_token_diffusion_v2/dataset.py), [losses.py](../models/pose_token_diffusion_v2/losses.py) | Label/observation boundary and objectives |
| [config.py](../models/pose_token_diffusion_v2/config.py), [configs/](../models/pose_token_diffusion_v2/configs) | Explicit v2 settings |
| [training.py](../models/pose_token_diffusion_v2/training.py) | Fresh-policy training and GPU smoke path |
| [runtime.py](../models/pose_token_diffusion_v2/runtime.py) | Learned-plan/phase cache adapter for rollout integration |
| [checkpoint.py](../models/pose_token_diffusion_v2/checkpoint.py) | V2-only checkpoint validation |

Build explicitly with `models.pose_token_diffusion_v2.build_policy(config)`.
V2 reuses unchanged v1 block implementations as new module instances. Its
configuration and checkpoint architecture ID is `pose_token_diffusion_v2`;
the initial config/checkpoint schema is 1. V1 and legacy checkpoints are rejected
before loading. No v1 builder, model source, configuration or active experiment
is redirected to v2.

## Training and checks

### Submitted development pilot — November 16

[Pilot record](../../results/diagnostics/pose_token_diffusion_v2_pilot_20261116_01/README.md):
64 demonstrations per object, 256 total. Skill-chain-stratified whole-demo
selection gives **226 train / 30 validation demonstrations**, with zero overlap,
30,029 training observations and 4,204 held-out observations. All five skill
types have valid goal labels in both splits. Six training and one validation
touch phases fail the geometric filter and remain masked.

- GPU smoke job **6940925**: one real pretrained-backbone update, component
  validation, sampled actions and image exports.
- Joint-training job **6940926**, dependent on successful smoke: two epochs,
  batch 16, AdamW LR 1e-4, two data workers; fresh trainable policy with frozen
  pretrained DINOv2 and Flan-T5. Predicted goals condition the actor throughout.
- The official benchmark and Stage 5 part-consistency mechanism are separate.

The [pilot recipe](../models/pose_token_diffusion_v2/configs/pilot.yaml) requires an explicit selection manifest;
`scripts/run_pose_token_diffusion_v2.py` prepares and submits the pinned study.
Data, source, configuration and initialization checks gate the jobs. Duplicate
submission to an existing study is refused.

Validation runs before optimization (epoch 0) and after each epoch. A single
held-out pass reports grounding, pose, applicability, phase/termination and
diffusion-action measurements per skill and object. Each report contains
`metrics.json`, `summary.csv`, `samples.csv`, `README.md`, `index.html` and
example PNG overlays. It also reports the actual learned-phase goal error
separately from the bank slot matched to the annotated phase for diagnostic
scoring. Annotations never become policy inputs.

Pilot validation covers every selected held-out observation. Five additional
single-observation action-sampling examples illustrate commanded trajectories;
they are not executed simulator rollouts. Missing labels are reported as null
or N/A. Validation errors measure agreement with one demonstration, not the
uniqueness or physical success of that pose. The
[pilot implementation receipt](../models/pose_token_diffusion_v2/validation_pilot_20261116.json) records 20 v2
checks and 21 unchanged v1 regression checks.

New evaluation uses the
[shared released execution settings](../configs/evaluation/partinstruct_native.json):
250 executed commands with eight commands per prediction for Test1, and native
physics thresholds. The final block is truncated at the action cap. Use the
separate `scripts/evaluate_pose_token_diffusion_v2.py --ckpt ... --out-dir ...`
entry point once a v2 checkpoint is ready. Grounding and progression stay learned;
this does not add GT part inputs or oracle skill advancement. Full v2 evaluation
has not been run. The November 16 implementation receipt below predates this
evaluation-protocol update and remains a dated implementation check.

### Matched simulator development evaluation — November 17

The [development comparison](../../results/diagnostics/pose_token_diffusion_v2_development_20261117_01/README.md)
evaluates both saved pilot epochs on the same 12 held-out training-release
scenarios and seed 42: one grasp, one touch, and one lift-containing sequence
per object. Selection excludes scenarios used by the pilot training split and
incomplete demonstration/metadata skill-chain matches. The official test
benchmark is reserved.

GPU array job **6943290** runs epoch 1 and epoch 2 separately; CPU job
**6943291** creates the paired report after both stop, retaining missing/error
coverage. Every rollout records MP4/GIF media, commanded actions, predicted
goals, learned phases/termination, and measured robot/object state. Only
completed trials with matching initial physical states enter paired counts.
Execution remains 250 actions with eight commands per prediction and learned
progression, with no new training or threshold tuning.

The lift group retains the original full grasp/lift/rotate instruction. A
grasp-to-lift milestone is reported separately from full-task success; neither
the grasp predicate nor that milestone establishes sustained retention. Native
reset initialization is not replay of the demonstration's recorded states.

Prepare/submit/report with `scripts/run_pose_token_v2_development.py`. The
separate runner in [development.py](../models/pose_token_diffusion_v2/development.py) does not modify the trained
policy or the existing v1 benchmark evaluator. Seven added selection/pairing
tests and the existing v2 suite passed (27 checks at submission). All 24
rollouts subsequently completed; neither checkpoint advanced beyond phase zero.

### Completion-supervision correction — November 17

The original action-chunk loader omitted final observations; its exact-boundary
completion labels supplied no positive rotation/release examples in this pilot.
This is a supervision defect, not an explanation of every execution failure:
touch-only tasks also failed without needing a phase transition.

The separate [completion pilot](../../results/diagnostics/pose_token_diffusion_v2_completion_20261117_01/README.md)
uses [completion_pilot.yaml](../models/pose_token_diffusion_v2/configs/completion_pilot.yaml). Main action batches,
226/30 demonstration split, 3,752 optimizer updates, frozen backbones, initial
random policy weights, and simulator settings remain matched. Each update adds
four completion-only observations: two positive and two negative queries,
balanced over skills. They have no future-action targets and do not enter the
action decoder. This adds computation, so it is not a compute-matched comparison.

[completion.py](../models/pose_token_diffusion_v2/completion.py) reads terminal observations independently of
action-chunk indexing. Annotation transitions supply boundary positives and up
to eight post-transition queries for the preceding skill. Pre-boundary frames
remain negative. Exact full-metadata-chain endpoints are explicitly weak
annotation proxies, not verified physical successes. The released HDF5s contain
no physical-success flags; 53 training and seven validation final phases with
incomplete/mismatched chains are masked. Proxy-positive counts are 44/7 for
release and 29/4 for rotation in training/validation. Positive frames following
one transition are correlated and are not independent completion events.

[encoder.py](../models/pose_token_diffusion_v2/encoder.py) queries the existing shared completion MLP for every
learned slot. It preserves parameter names/shapes and the original selected-slot
output. A training query selects its loss from that bank; it does not force the
actor phase or provide GT features. At runtime, the existing controller still
uses the selected phase's probability, threshold 0.5, and minimum two prediction
cycles. Simulator task-success predicates remain scoring only.

Held-out reports in `training/completion/epoch_*/` include per-skill recall,
false-positive rates, and a CSV of missed boundaries/premature triggers. The old
exact-frame completion metric is excluded from the ordinary component report
for this recipe. That report's validation total excludes completion loss;
completion scores are reported separately. Complete-chain endpoint proxies need
physical rollout validation, especially for terminal release/rotation.

Submitted jobs: GPU smoke **6948147**, fresh training **6948148**, evaluation
preparation **6948149**, matched simulator array **6948150**, report **6948151**.
Dependencies gate training on successful smoke, then repeat the same 24
development rollouts and compare before/after outcomes on matching resets.

36 v2 checks and 21 unchanged v1 regression checks passed, including query-label
isolation, original encoder equivalence, terminal loading without fabricated
actions, balanced sampling, validation state/RNG preservation, and comparison
pairing. A real HDF5 optimization step passed with production trainable blocks
and tiny test backbones. The GPU smoke uses actual pretrained backbones; CPU
checks do not establish physical success.

[train.yaml](../models/pose_token_diffusion_v2/configs/train.yaml) starts all **trainable policy components from
random initialization**, while retaining pretrained frozen DINOv2/Flan-T5.
It does not warm-start a prior policy. GT pose goals and GT phase labels are
loss targets, not actor inputs in this recipe. Entire demonstrations are split
deterministically into training and validation sets. This split does not itself
establish unseen-object or unseen-scenario generalization.

The default full-data recipe's two epochs and bounded validation pass are a
development recipe, not a demonstrated convergence budget. The separate pilot
above is submitted; full-data training has **not been launched**.
In a GPU allocation, run a one-update pretrained-backbone smoke check first:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=part2action:part2action/scripts \
  /p/part2action/envs/part2action_train/bin/python -B \
  part2action/scripts/train_pose_token_diffusion_v2.py \
  --out-dir /p/part2action/results/pose_token_diffusion_v2_smoke --smoke
```

Use a fresh output directory; the trainer refuses overwrites. Remove `--smoke`
and choose a separate directory for full training. Receipts record configuration,
source hashes, initialization digest, data split and optimizer steps.
Checkpoints store policy weights without frozen backbone weights; they are not
optimizer-resume snapshots.

CPU implementation checks from repository root:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=part2action:part2action/scripts
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/p/part2action/envs/part2action_train/bin/python -B -m unittest discover \
  -s part2action/models/pose_token_diffusion_v2/tests -v
/p/part2action/envs/part2action_train/bin/python -B \
  -m models.pose_token_diffusion_v2.tests.check_real_data \
  --out /tmp/v2_real_data_check.json
```

Tests cover phase-dependent goal selection, masks, action-to-goal gradients,
label leakage, frozen backbones, mixed precision and checkpoint isolation.
The opt-in real-data check uses actual HDF5 labels/observations and production
v2 trainable blocks with tiny frozen test encoders. It does not test pretrained
encoder semantics or simulator success. A native simulator rollout and full
pretrained-backbone GPU smoke remain separate validation gates.

The [November 16 implementation receipt](../models/pose_token_diffusion_v2/validation_20261116.json) records:

- 15 v2 tests and 21 unchanged v1 regression tests passed.
- 128 real demonstrations audited across bottle, mug, pliers and scissors.
- Valid labels for 71 grasp, 73 move, 8 rotation and 12 release phases; 51 of
  57 touch phases passed the fingertip-proxy filter. Six touch phases remain
  masked; these counts are label coverage, not successful policy trials.
- A real-observation optimization step and 100-step diffusion sampling produced
  finite action chunks of shape `(5, 8, 7)`, using tiny frozen test backbones.
- Whole-demo train/validation splits had no overlap in that sample; preserved
  source hashes for both existing experiments still matched.
