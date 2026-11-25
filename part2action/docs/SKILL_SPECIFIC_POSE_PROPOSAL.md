# Proposal: skill-specific, part-aware pose guidance

Created November 16, 2026. Project: **Part2Action / P2A**, Franka Panda in
PartGym with PartInstruct demonstrations.

**Status: exploratory proposal, separate from [RESEARCH_PLAN.md](RESEARCH_PLAN.md).**
The complete extensions below have not been implemented or established as novel.
This document records the candidate design, staged experiments, and evidence needed
before incorporating a supported direction into the main research plan.
Writing this proposal does not launch training or change existing experiments.

## Implementation update: November 16, 2026

The separate [Pose-token Diffusion v2 architecture](POSE_TOKEN_DIFFUSION_V2_ARCHITECTURE.md)
now implements the initial per-skill goal baseline: shared goal prediction from
learned plan slots, learned-phase goal selection, component-masked pose/interaction
tokens, predicted-goal action conditioning, and a fresh-policy training path.
Targets include preclosure grasp TCP, a clearly labeled touch geometry proxy,
and demonstrated TCP endpoints for later skills. These are implementation steps,
not validated physical contact labels or task-success gains.

Explicit held-part/goal-part bindings and cross-skill consistency remain proposed.
A [two-epoch v2 development pilot](../../results/diagnostics/pose_token_diffusion_v2_pilot_20261116_01/README.md)
is now submitted: smoke job 6940925 gates joint-training job 6940926. It uses
256 selected demonstrations (226 train, 30 validation), fresh trainable policy
weights, frozen DINOv2/T5, and predicted-goal action conditioning. Component
reports run before optimization and each epoch. Full-data training and benchmark
evaluation remain pending, and the main research plan remains separate.
Statements below about the existing trainer and checkpoints
refer to the v1 reference, not the new fresh-policy v2 trainer.

### November 17: completion-supervision follow-up

The first v2 pilot and its 24 development rollouts completed. Both checkpoints
remained in phase zero. The action-horizon sample filter excluded terminal
observations, leaving no positive completion examples for rotation/release in
that pilot. Touch-only failures additionally show an execution limitation.

The separate [completion correction](POSE_TOKEN_DIFFUSION_V2_ARCHITECTURE.md#completion-supervision-correction--november-17)
retains the original action batches, demonstrations, update budget and random
initialization, with frozen DINOv2/T5. Balanced completion-only queries include
terminal observations and previous-skill queries after annotated transitions;
no unavailable future-action targets are fabricated. Incomplete final chains
are masked, while complete-chain endpoints remain explicitly weak proxies.
Physical-success flags are absent from the released HDF5s.

[Follow-up study](../../results/diagnostics/pose_token_diffusion_v2_completion_20261117_01/README.md):
GPU smoke 6948147 passed; fresh training 6948148 started. Jobs 6948149–6948151
prepare and repeat the same 24 simulator rollouts and compare matched resets
after training. Per-skill boundary recall and premature triggers are reported
separately. This remains a development experiment, with additional completion
computation; no new success-rate or novelty claim is established.

## Research question and initial scope

Can pose goals conditioned on the active skill and relevant object part improve
whole-task execution? Beyond that baseline, does explicitly preserving required
part relationships across consecutive skills improve retention and composition?

Start with **one object, multiple parts, and a short skill sequence**. First
validate grasp and touch goals, then grasp-to-lift and grasp-to-lift-to-rotate
sequences. Multi-object instructions and unrestricted task planning are later
extensions, not requirements for the first milestone.

Pose prediction remains the geometric interface to action generation. A future
layer could specify desired object-part relationships above poses, but the
initial study should isolate skill-specific pose guidance and one consistency
mechanism.

## Verified implementation to build on

The reference is the independent
[Pose-token Diffusion model](../models/pose_token_diffusion/README.md), rather
than the older additive pose model described elsewhere in the research history.

| Component | Current behavior | Proposed extension |
| --- | --- | --- |
| Visual and scene context | DINOv2, Flan-T5, observed scene geometry, grounded region, measured TCP/opening | Reuse the verified observation paths |
| Skill hierarchy | Learned skill plan, phase selection, and termination | Reuse progression; connect its active skill to the goal definition |
| Action policy | Learns actions for grasp, touch, movement, rotation, and release | Retain the diffusion action interface |
| Pose input in GT mode | One first-grasp target supplied throughout the episode | Supply a goal appropriate to the active skill |
| Learned pose supervision | First-grasp regression before closure; applicability supervision on later/non-grasp samples | Define validated targets and applicability for additional skills |
| Touch pose guidance | No valid explicit target for touch-start tasks | Add a task/part-specific touch target |
| Pose-aware progress head | Implemented but disabled in the current recipe | Evaluate a change only if the existing termination mechanism is inadequate |

The completed provided/disabled training arms share initialization, data order,
and update counts. Both use auxiliary pose supervision; pose-off does not mean
that training has no GT labels. Compatible weights were transferred from an
earlier policy, while DINOv2/T5 remained frozen. These runs are not training from
scratch.

Use the dated [training plan](../../results/diagnostics/pose_token_diffusion_training_20261114_01/plan.json)
and [matched evaluation plan](../../results/diagnostics/pose_token_diffusion_evaluation_20261115_01/plan.json)
for experiment provenance. Their results must not be described as measurements
of the proposed skill-specific architecture.

## Pose goals, actions, and part relationships

A measured pose describes the current robot state. A goal pose describes a
desired configuration. In PartGym, each action commands world XYZ,
roll/pitch/yaw, and finger opening. The decoder predicts eight commands, and the
current runtime executes the first before observing and replanning. A goal pose
conditions this process; it does not replace the action sequence.

Pose information should remain available after grasping. What changes is the
goal associated with the active skill.

| Skill | Candidate geometric goal | Interaction requirement |
| --- | --- | --- |
| Grasp | TCP approach pose at the intended part | Establish the intended grasp and closure |
| Touch | TCP pose at a verified touch event, or a contact location with appropriate orientation constraints | Contact the specified part; do not require a grasp |
| Lift/move | Task-consistent TCP/object displacement in the specified reference frame | Preserve grasp or touch only when required by the instruction |
| Rotate | Desired object/part orientation, converted into a compatible TCP goal | Preserve the existing held-part relationship when required |
| Place/release | Placement pose and object-support relation | Coordinate placement with opening and verify the resulting state |

These are proposed target definitions. A desired object-part orientation and a
desired TCP orientation are different quantities. Rotation goals need a validated
mapping between them; the current pose head predicts TCP goals only. Finger
opening remains an action/skill requirement, rather than a component of rigid
pose itself.

For example, in **grasp the bottle body -> lift -> turn the cap toward the
observer**, the body is the held part and the cap is the orientation reference.
They must retain distinct roles. The original world-frame grasp pose stops being
the destination after grasping, but the gripper-to-held-part relationship remains
relevant.

Under a rigid, nonslipping grasp, that relative transform should remain
approximately constant as both move through the world. This is a proposed
geometric consistency objective, not proof that physical contact is retained.
Release transitions must relax the holding requirement, and touch tasks must not
inherit a grasp constraint.

## Proposed component boundaries

```text
Observed RGB/scene geometry + instruction + measured robot state
  -> existing encoders, skill hierarchy, and region grounding
  -> active skill + relevant part features
  -> shared skill-conditioned goal predictor
  -> relative-pose memory + existing scene/skill/robot memory
  -> cross-attention diffusion decoder -> action chunk
  -> execute, observe, and update the active goal
```

Use one shared goal predictor conditioned on skill and part features initially.
Separate complete policies for every skill are not required. Record the target's
frame, position/orientation applicability, validity, and confidence explicitly.
Do not impose an arbitrary full orientation label on a skill that permits many
orientations. Part-relative frames require a defined convention and a usable
estimate from observations, including treatment of symmetry and occlusion.

For the consistency experiment, add explicit held-part versus goal-part bindings
and a small persistent state if needed. Compare independently predicted goals
against coupled current/next-skill goals. This coupling and its learning objective
need a concrete specification before implementation.

## Incremental implementation and test gates

Keep a small runnable pipeline throughout development. After each component,
check its interface and its effect in the integrated model. Module tests and
successful smoke rollouts establish implementation behavior, not task gains.

| Stage | Build | Required evidence before expanding |
| --- | --- | --- |
| 0. Reference | Complete and audit the existing GT/off comparison | Identical completed trial pairs, explicit coverage gaps, grasp and full-task results |
| 1. Target definitions | Validate grasp and touch labels, then movement/rotation labels | Visual inspection of real demonstrations; correct event timing, frames, component masks, and invalid cases |
| 2. Goal predictor | Shared skill-conditioned pose head using grounded part features | Finite real-data optimization; held-out errors per skill; valid predictions for touch and grasp |
| 3. Pose-to-action connection | Feed predicted goals into the pose-token decoder during training and inference | Action-loss gradients reach the predictor; finite action sampling; no target-label leakage |
| 4. Phase updates | Refresh goals using learned skill progression | Small grasp-to-lift and touch rollouts; correct goal changes; no expert phase clock |
| 5. Part consistency | Preserve the specified part bindings/relationships across transitions | Matched ablation against independent goals; measured retention and full-task outcomes |

For Stage 1, retain the audited post-action observation convention: the current
grasp target uses measured TCP immediately before expert closure. Do not assume
the last frame of every phase is the appropriate target. A touch target needs
contact-event evidence; if only a geometric proxy is available, label and report
it as a proxy. Verify whether demonstrations provide sufficient labels before
committing to every skill type.

Use provided, known goals for inexpensive interface tests and debugging. The
main learned pipeline should use predicted goals during both action training
and deployment. A separate large GT-input campaign is not a prerequisite for
each stage. Any teacher-forcing or provided/predicted curriculum must be an
explicit experimental choice, with its deployment mismatch measured.

A lift destination must represent the intended displacement from the appropriate
phase reference. Recomputing “current height plus the lift distance” at every
control step would continually move the destination. Goal estimates may refresh
within a phase, while the intended task outcome remains consistent.

## Information boundary and preservation

At deployment, inputs are observations, instruction, measured robot state, and
learned skill/part/goal estimates. Expert actions, future demonstration frames,
GT part poses, and GT phase timestamps are not deployment inputs. Simulator
predicates may provide training labels or evaluation scores; they must not
silently become the deployed phase controller.

Keep existing policy sources, configurations, checkpoints, and active evaluation
inputs intact. Develop new components under a separate experimental model/config
identity with explicit checkpoint compatibility. Reuse verified encoders and
action interfaces without changing their historical implementations.

Before a new full training run, record the initialization decision explicitly.
Fresh-policy training would reset all trainable policy components while keeping
pretrained frozen DINOv2/T5 unless a separate backbone study is selected. The
current trainer always loads an initialization checkpoint, so fresh-policy
training needs an explicit, tested implementation path. Shared warm-start and
fresh-policy experiments answer different questions and must be labeled clearly.

## Comparisons and evaluation

The staged comparisons are:

| Comparison | Question |
| --- | --- |
| Matched first-grasp-only versus per-skill goals | Does broader goal coverage improve execution? |
| Per-skill world-frame versus part-relative goals | Does the chosen part representation help? |
| Independent part-aware goals versus goals with transition consistency | Does the proposed coupling add value? |

Run comparisons when the preceding stage is executable; there is no need to
launch every arm upfront. Match observations, initialization, training examples,
update budget, and evaluation starts. Match shared auxiliary supervision when
claiming an input/conditioning benefit. A historical first-grasp checkpoint can
provide context, but is not a matched control if its training protocol differs.

Use development data for target definitions and model choices. Hold out complete
demonstrations/scenarios rather than splitting adjacent trajectory frames across
training and validation. Reserve benchmark tests for the declared comparison.

Measure:

- Goal position/orientation errors and applicability per skill, with valid masks.
- Correct part selection and goal refresh across phase changes.
- Grasp acquisition and independently defined sustained retention during movement.
- Per-skill completion and full-task success on identical starts.
- Coverage gaps, runtime failures, inference latency, and trainable parameter cost.

The current native either-finger contact predicate is not sufficient evidence of
sustained retention. Define a retention measure using contact persistence and
object/robot motion, and verify that scoring does not advance simulation.
Reaching a pose alone is not evidence that touch, grasp, or placement succeeded.

Test1 evaluates changed initial states. Later use held-out part combinations and
task categories where dataset coverage permits; these are more relevant to a
composition claim. Repeat promising comparisons across training seeds before
making broad performance claims. Three evaluation seeds do not measure
training-seed variability.

## Evidence required before promotion into the main plan

The proposal is worth promoting when:

1. Target labels, information boundaries, and end-to-end execution are audited.
2. A matched comparison shows a practically useful improvement on the declared
   metric, with uncertainty and coverage reported. Specify the minimum useful
   effect and evaluation population before the experiment.
3. The relevant ablation isolates the added mechanism, including any cost or
   regressions on other skills. Changed actions or attention patterns alone do
   not establish an improvement.
4. The supported claim is distinguished from pose-guided hierarchy, part-frame
   representations, and relational constraints already present in prior work.
5. Results and limitations are reviewed before merging the validated scope into
   RESEARCH_PLAN.md. Unsupported extensions stay in this proposal.

If pose accuracy improves without task improvement, inspect goal usage, contact,
and progression. If only supplied goals help, diagnose prediction and the
provided/predicted mismatch. If multi-skill goals work but consistency adds no
benefit, retain the useful baseline and revise the candidate contribution.

## Novelty boundary and later extensions

Multi-task training, multiple objects, successive pose goals, cross-attention,
and part awareness are not individually new contributions. The candidate here is
the specific learned coupling between skill-specific part goals and relationships
that must persist across transitions. Its novelty and benefit remain hypotheses.

After the single-object study, consider explicit object-part-skill bindings in
multi-object scenes. A further level could predict desired relations such as
held, touching, supported, or oriented, then generate poses to realize them.
These extensions need additional representation, labels, and tests; they should
not be bundled into the first implementation.

| Related work | What it already establishes / what to inspect |
| --- | --- |
| [GHOST](https://arxiv.org/html/2606.10025v1) | Successive end-effector subgoals and goal-conditioned diffusion; its published goal interface does not explicitly encode semantic object-part-skill bindings, although visual features may learn such associations implicitly |
| [PartManip](https://arxiv.org/abs/2303.16958) | Part-based canonicalization, rewards, and state-to-vision policy distillation |
| [GAPartNet](https://arxiv.org/abs/2211.05272) | Actionable part semantics, poses, and cross-category manipulation |
| [AffordDP](https://arxiv.org/html/2412.03142v2) | Manipulable-part registration and transferable contact/post-contact trajectory guidance |
| [ReKep](https://arxiv.org/html/2409.01652v2) | Relational goals and constraints across stages; a close comparison for claims about preserving part relationships |
| [PALM](https://arxiv.org/html/2601.07060v2) | Interaction affordances and learned within-subtask progress |
| [Subgoal Transitions](https://proceedings.mlr.press/v305/jain25b.html) | Learned transition modeling, including an attention-based approach |
| [ManiSkillFormer](https://arxiv.org/abs/2609.16331) | Geometric contracts connecting skill schemas, perception, and motion templates; relevant to a later goal/relationship layer |

## Decisions to resolve before implementation or training

- Exact target events and component validity for each selected skill.
- Available part identity, frame, contact, and transition supervision.
- The concrete consistency objective and the transitions where it should apply.
- Initialization, training/validation split, budget, and development rollout set.
- The first comparison's success criterion and the claim it is intended to test.

No new model code, training configuration, or job submission is part of this
documentation change.
