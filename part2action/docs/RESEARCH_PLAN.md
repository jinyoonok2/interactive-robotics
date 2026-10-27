# Part2Action research plan

Updated October 27, 2026. Current decisions and evidence live here; this is not a
submission journal. Commands/storage: [SETUP](SETUP.md). Data/labels/splits:
[DATASET](DATASET.md). Earlier work is preserved in the
[documentation archive](archive/2026-10-22/README.md). The
[October 26 discussion](slide_history/2026-10-26_part2action_meeting_status.md)
introduces the new contribution direction.

## Research question and proposed contribution

**Can a policy learn which changes in interaction position, wrist orientation,
and gripper-closing time preserve task success, then use those learned tolerances
to choose more reliable actions?**

The proposed representation is a **part-, task-, and state-conditioned success
set**, rather than one demonstrated interaction point. Its supervision would
come from successful and failed simulator perturbations. Its value must come
from using the learned set in action selection or training, not only predicting
an additional label. This direction is proposed: neither perturbation collection,
tolerance learning nor tolerance-guided control is implemented or evaluated yet.

Use **interaction target point** for the existing predicted XYZ. Code/config names
such as `contact_xyz` and `contact_conditioning_enabled` remain unchanged for
compatibility. The current label selects a gripper-command event, reads the TCP
position at that event, and chooses the nearest annotated target-part point in
the current frame. It is a derived spatial proxy, not measured finger contact.
Already-closed and no-close fallbacks, and object motion between those frames,
limit its interpretation. `actions[:, 6]` already supervises the finger-position
command in every action; a separate closing-time head alone adds no new evidence
of which alternative interactions succeed.

The hypothesis is that demonstrations specify one solution while a task may
admit several combinations of position, orientation and timing. Learning their
outcomes could distinguish acceptable alternatives from failure regions. It
could also expose interactions between those variables that a point target
cannot express. This is a testable hypothesis, not a theoretical guarantee or
an established improvement over DP/DP3.

### What must distinguish this from existing work

Part guidance, keypoints, contact-relative coordinates, constraints, diffusion
and perturbation learning are not sufficient novelty claims on their own. Check
[PartInstruct](https://www.cs.jhu.edu/~alanlab/Pubs25/yin2025PartInstruct.pdf),
[KITE](https://arxiv.org/abs/2306.16605),
[ReKep](https://arxiv.org/abs/2409.01652) and
[GLiDE](https://arxiv.org/abs/2403.17124), alongside contact-based policies, before
claiming a new method. These are comparison targets, not a completed novelty audit.

The intended contribution is how **joint pose–timing success sets are learned
and used by the policy**, supported by matched-data evidence. Beating a baseline
with extra simulator supervision would not isolate the representation's value.

## Proposed experiment sequence and decision criteria

1. **Validate replay and measurements.** Restore demonstration states and replay
   unmodified actions on a small stratified pilot. Record replay success,
   commanded closure, physical finger–object contact, target-part identity and
   skill/task outcomes. Define coordinate frames, controller rate, wrist rotation
   convention, event selection and success criteria before perturbing anything.
   If reference replay fails, fix or delimit that mismatch first.
2. **Collect a pilot of intervention outcomes.** From the same saved starting
   states, vary translation, wrist orientation and closing-time offset around
   a defined interaction segment. Begin with single-variable sweeps, then joint
   changes. Specify whether continuation uses replay, a fixed policy or replanning;
   hold that rule fixed within a comparison. Record perturbations, seeds, outcomes
   and invalid/unreachable cases separately. Gripper-command time and physical
   contact time are separate measurements. Grasp closure is not the interaction
   event for every skill; initially scope the timing study to grasp.
3. **Test whether a useful success set exists.** Measure repeatability, coverage,
   failure boundaries and pose–timing dependence. A success set is conditional on
   the starting state and continuation controller, not an intrinsic property of
   an object point. Do not assume a box, a single connected region, or that all
   nearby changes preserve success. Stop or revise the hypothesis if replay or
   outcome noise prevents reliable supervision.
4. **Learn and use tolerances.** Only after the pilot, choose a representation for
   success likelihood over candidate interactions. Condition on observations,
   part/task information, robot state and the defined continuation protocol.
   Integrate it through a specified action-training objective or candidate-action
   selection rule; compare prediction alone with actual use in control. Keep
   sensors and frozen encoders fixed. GT simulator state may generate labels or
   score trials, but must not silently become a deployment input.
5. **Run controlled comparisons.** Use the matrix below, common data splits,
   perturbation budgets, evaluation scenarios and selection rules. Repeat selected
   conditions across training seeds. Report success, robustness to predefined
   shifts, per-stage outcomes, uncertainty and added inference cost. Evaluate
   unseen instances/states before claiming generalization beyond the existing
   repeatedly inspected test1 scenarios.

| Proposed condition | Question isolated |
| --- | --- |
| Existing point-conditioned policy | Reference for ordinary target conditioning |
| Ordinary diffusion policy trained on the same successful augmented demonstrations | Benefit from extra successful data alone |
| Matched outcome-supervised control with the same intervention records and candidate budget | Benefit from success/failure supervision without the proposed structured set |
| Spatial-only tolerance model | Contribution of acceptable spatial interactions |
| Joint position/orientation/timing tolerance model | Contribution of the joint representation |
| Joint model with tolerance-guided action use disabled | Whether the learned representation actually improves decisions |

All new conditions above are **unsubmitted proposals**. Finalize controls before
collection/training so that extra data, outcome labels, action sampling and
compute cannot explain a claimed representation gain. Split by source trajectory,
scenario and object instance as appropriate; related perturbations must not leak
across training and held-out evaluation. Separate equal-epoch from equal-update
or compute-matched comparisons. Report probability calibration if predicting a
success probability; do not call a thresholded prediction a certified safe set.

## Existing implementation and fixed screening protocol

This is the available experimental platform, not the proposed tolerance model:

- Frozen DINOv2 ViT-S/14-register and Flan-T5-base; two consecutive RGB frames,
  full scene PCD, measured TCP pose and gripper opening.
- Learned skill plan, phase selection/termination, visual part gate and a
  skill-conditioned interaction-target predictor. The visual gate itself is not
  skill-conditioned; the separate 3D point-part head is disabled.
- Transformer diffusion denoiser: eight noisy action-step tokens, two layers,
  four heads, width 256, pooled observation/skill context. The earlier diffusion
  baseline denoises the same eight actions using a flattened MLP. Separate
  modality-token cross-attention is not implemented.
- Eight world-XYZ/Euler/finger-opening commands; execute one, then observe again.
  DDPM uses 100 cosine steps and fixed physical-unit scaling.
- Screening defaults: **10 total epochs from scratch, stride 1, next-index
  targets (`t+1`), batch 16, LR 1e-4, training seed 42**, all 4,053 valid
  four-object demos. Stride changes sampled examples and optimizer updates per
  epoch; the two observation frames remain consecutive.
- Target auxiliary supervision and explicit target input are separate switches.
  GT masks, part clouds and derived targets are labels/diagnostic references,
  not Part2Action deployment inputs.

### Existing target-input comparison: supporting diagnosis

| Variant | Target auxiliary loss | Explicit target input | Status on October 27 |
| --- | --- | --- | --- |
| A | OFF | OFF | Not trained or queued |
| B: Transformer control | ON | OFF | Completed: 64/447 |
| C: Transformer target input | ON | ON | Interrupted/incomplete; job failed |

B/C tests explicit input; A/B would test auxiliary supervision. Neither tests
learned tolerances. C retains late normalized-global conditioning:

`r = (predicted_target_world - measured_TCP_world) / 0.1`

`action_features + P(0) + lambda * LayerNorm(P(r) - P(0))`

Coordinates are detached; lambda is a signed learned global scalar initialized
at 0.01, not confidence. B uses zero relative input and retains P(0). C matches
B's configuration except the input switch, name and output path; no early-fusion
change is included. Labels still use the existing heuristic derivation.

### Verified job state — October 27, 2026

Checked Slurm accounting/queue and recorded outputs after the server outage:

| Job | Purpose | Verified state |
| --- | --- | --- |
| 6734402 | Fresh target-input-ON training | FAILED, exit 2; latest launcher log rejects an existing output directory |
| 6734403 / 6734404 | ON evaluation / comparison report | Pending; evaluation dependency can no longer succeed |
| 6734186 / 6734187 | Released DP-S / DP3-S evaluations | COMPLETED; 29/176 and 81/176 |
| 6734999 / 6735000 / 6735004 | Oracle gate / no gate / gate report | COMPLETED; report available |
| 6735001 / 6735002 / 6735003 | ON target identity / zero / +5 cm world-X | Pending with DependencyNeverSatisfied on failed training |
| 6735005 | Target diagnostic report | Pending on blocked target evaluations |

The retained training log reaches epoch index 9 at approximately 58%; a partial
checkpoint exists. This is not a completed 10-epoch result. The latest launch
failed at the output-directory guard; do not infer that the architecture itself
failed. Recovery needs an explicit decision about partial-run continuation versus
a fresh replacement and corresponding dependencies. No recovery, cancellation or
new submission was performed during this documentation update. Do not duplicate
jobs blindly. Earlier 6734379/6734380/6734381 were cancelled before starting.

[ON config](../configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer_contact.yaml),
[submission ledger](../../results/part2action/runs/diffusion_transformer_contact_submission_20261022_retry1/submissions.json),
[ON review location](../../results/part2action/evaluations/diffusion_transformer_contact_comparison_20261022_retry1/index.html).
Ledgers preserve original submissions; the table above records current state.

## Completed evidence and its limits

| Experiment | Result | Interpretation |
| --- | --- | --- |
| 10 epochs, next-index, stride 1, input OFF: prior diffusion / Transformer | 46/447 / 64/447 | Decoder/capacity comparison, not a target-input gain |
| 10 epochs, next-index, Transformer, input OFF: stride 1 / 2 | 64/447 / 48/447 | Working default stride 1; equal epochs are not equal updates |
| 30 epochs, stride 1, input OFF: same-index / next-index diffusion | 10/447 / 16/447 | Shared untrained initialization; one training seed |
| Historical MLP timing continuations: same-index / next-index | 9/447 / 24/447 | Continuation evidence, not fresh training |
| 30 epochs, stride 2, same-index diffusion: control / late / early input | 33/447 / 22/447 / 40/447 | Early also changes grounding; matched early control absent |
| Frozen 6724878: predicted / oracle / no visual gate | 64/447 / 69/447 / 67/447 | Small aggregate differences; neither establishes a gate benefit |

Historical stride-2 and stride-1 30-epoch runs were not an isolated stride test;
common initialization was not verified. The 10- and 30-epoch studies differ in
decoder/target settings, so their scores do not isolate epoch count. Old 39-rollout
pilots and discontinued self-trained DP/DP3 runs remain historical references.

At 10 epochs, baseline/Transformer grasp-starting tasks score 40/339 versus
27/339, while touch-starting tasks score 6/108 versus 37/108. These are full-task
outcomes grouped by first skill, not isolated skill success. The overall +4.03
percentage-point difference has a scenario-cluster bootstrap 95% interval
[-0.22, +8.72] points. A40 policy-forward means are 80.5 versus 147.4 ms,
not end-to-end latency. One training seed does not measure training variability.
[Decoder report](../../results/part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/quantitative_report/index.html),
[stride report](../../results/part2action/evaluations/diffusion_transformer_stride_comparison_20261021/index.html).

### Frozen-policy diagnostics

Gate interventions reuse checkpoint 6724878 with target input OFF. Oracle GT
geometry is projected at the learned phase index (initially 0); unsupported
move/release/rotation phases use identity gating and log a reason. It supplies
no oracle phase progression and is not visibility-corrected segmentation.
Changing the gate affects phase/target features as well as actions. Oracle versus
predicted gains +1.12 points (95% scenario-cluster interval [-0.67, +3.13]); no gate
gains +0.67 points ([-1.12, +2.46]). These frozen, potentially out-of-distribution
interventions do not establish that part grounding is irrelevant or improved.
[Gate report](../../results/part2action/evaluations/frozen_gate_diagnostics_20261022/quantitative_report/index.html).

The blocked target diagnostics would freeze C and compare identity, zero relative
input, and a +0.05 m world-X shift. Predictions remain unchanged; actual intervened
inputs are logged separately. Zero retains projection bias and is not a retrained
OFF policy. All three use shared trial/step-specific forked PyTorch noise, including
a new identity evaluation. Closed-loop states can still diverge; this is not a
same-observation counterfactual or a physical-contact measurement. Source hashes
and per-job GPU/simulator preflights protect the submitted protocol.
[Diagnostic ledger](../../results/diagnostics/frozen_policy_submission_20261022/submissions.json),
[target review location](../../results/part2action/evaluations/frozen_target_diagnostics_20261022/index.html).

## Evaluation, diagnostics and comparison boundaries

Part2Action uses all 149 fixed test1 scenarios for bottle/mug/pliers/scissors,
seeds 42/43/44 (447 rollouts), max 120 executed actions, one action per cycle,
legacy termination, gripper clamp [0, 0.04] m, CPU point sampling and GPU inference.
Report object/task/first-skill/seed breakdowns, paired outcomes and scenario-cluster
uncertainty. This tests state generalization, not unseen categories. There is no
validation split in the current recipe; repeated development on test1 must be
acknowledged. New tolerance experiments need a predefined held-out protocol.

Existing logs measure task completion, sampled target-part surface distances,
phase transitions, gripper saturation and forward latency. Surface proximity is
not physical contact; command clipping is not grasp failure. Physical target-part
finger contact, acquisition/loss, duration, relative-pose retention and skill-event
timing need validated instrumentation. Fix criteria and denominators before using
those measurements for direction selection. Videos illustrate quantified findings.

Released DP-S/DP3-S native preflights and evaluations completed: **29/176 (16.5%)**
and **81/176 (46.0%)**. On our four categories the sampled subsets score **11/64**
and **30/64**, respectively. Their protocol uses 11 categories x16 sampled trials,
GT part inputs, oracle skill progression, 250-step budget and eight-action blocks;
no SAM/VLM. These are external references, not a matched ranking against our
447 fixed trials or an exact reproduction of every paper setting. Align scenarios,
action/observation schedules and information access before making direct claims.
[DP-S summary](../../results/partinstruct/evaluations/official_release_20261022/DP-S/full/summary.json),
[DP3-S summary](../../results/partinstruct/evaluations/official_release_20261022/DP3-S/full/summary.json).

## Immediate next decisions

First decide how to recover the interrupted C run and its blocked dependencies;
then finish the B/C and frozen target diagnosis without changing their question.
Use completed gate and released-policy results as supporting evidence with the
comparison limits above. In parallel, design the small replay/perturbation pilot
and prior-work comparison before building a new architecture. The contribution
plan is joint interaction-tolerance learning; existing point/gate studies diagnose
the platform and do not validate that proposed method.
