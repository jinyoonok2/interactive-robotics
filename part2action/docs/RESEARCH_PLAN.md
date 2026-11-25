# Part2Action research plan

Updated November 24, 2026. Status checked against Slurm accounting and saved run receipts.
Project: **Part2Action / P2A**, Franka Panda in PartGym with PartInstruct data.
The RB-Y1 PAP/LRL projects are separate; see [experiment ownership](EXPERIMENT_OWNERSHIP.md).
Commands: [SETUP](SETUP.md). Labels and splits: [DATASET](DATASET.md).

## Current status — November 24, 2026

No Part2Action jobs are queued. Sections after this one are dated history;
their job states and "next" items are superseded where they disagree.

| Generation | Result | Scope |
| --- | --- | --- |
| Original visual policy + GT pose | Initial-phase-end grasp 114/333 without, 160/333 with; full task 70/441 and 76/441 | Test1, three evaluation seeds, 120/1 protocol; six references missing |
| Pose-token v1, GT input | Grasp acquired 56/333 without, 41/333 with; full task 20/441 and 33/441 | Test1, three evaluation seeds |
| Pose-token v2, completion-corrected epoch 2 | Grasp 2/4, touch 0/4, sequence 0/4 | 12 development scenarios, one seed |
| Pose-token v3, epoch 10 | Grasp 2/4, touch 1/4, sequence 0/4 under learned and oracle progression | 12 development scenarios, one seed |
| V3 frozen grasp probe | Predicted/verified geometry × learned/forced gate: 0/4; reference replay succeeded | One mug scene, one seed |

These rows use different data, budgets and protocols and are not a ranking.
V3 held-out goal error stayed near 4 cm after epoch 2 while training goal loss
kept falling (0.051 at epoch 1, 0.003 at epoch 10). V3 trained on 226
demonstrations, and has no Test1 or VLM-planner evaluation.

**Working hypothesis, untested:** the pose-token diffusion decoder shared by
v1–v3 and the small training set limit execution more than skill progression.
**Proposed next step:** compare the original decoder with the pose-token decoder
under matched initialization, data, updates and protocol.

## Evaluation protocol update — November 16, 2026

New evaluations use the released PartInstruct execution settings: **250 executed
actions and eight actions per prediction for Test1**; Test4 uses 400 actions.
The [shared configuration](../configs/evaluation/partinstruct_native.json) and
current pose-token evaluators cover both v1 and the separate v2 checkpoint family.
Native camera/physics thresholds are retained, including the 0.15 m release
height. Learned grounding/progression and full-task scoring remain explicit
differences from the privileged released-policy evaluator.

The existing 120/1 Part2Action results and dated run descriptions below are
historical measurements, not new-protocol results. They are not relabeled or
merged with new runs. The superseded fixed 120/1 DP/DP3 evaluations and the old
self-trained DP3 pilot results were removed at the user's request. The completed
[native released-weight evaluations](../../results/partinstruct/evaluations/official_release_20261022)
and all model checkpoints are retained. This settings update does not submit
new training or full evaluation jobs.

## Objective and current scope

Test whether explicit grasp-pose conditioning improves **grasp acquisition and
full-task success while retaining the original visual policy**. First test
supplied GT pose in a matched comparison. Then evaluate a learned pose estimator
using shared visual and scene features. Both paths retain the original Transformer
diffusion action head and generate all task actions from reset.

The two supported paths are:

| Path | Implementation | Current state |
| --- | --- | --- |
| Original visual policy + supplied GT grasp pose | `VisualPosePolicyV1` | Both two-epoch fine-tuning arms completed; matched-training audit passed; evaluation recovery underway |
| Shared visual grounding + predicted grasp pose | `GroundedVisualPosePolicyV1` | CPU and full-backbone GPU smoke checks passed; full training not submitted |

The separate non-visual pose/controller pipeline is retired. Its eight result
roots were deleted at the user's request. Replay/tolerance/evaluator expansion
is deferred until useful pose-to-action behavior is demonstrated in the visual
policy. Earlier proposals and their results are historical context, not the
current training plan. [Cleanup and historical sources](archive/2026-11-11_scope_cleanup/README.md).

## Verified run status

| Stage | Job | Verified result / dependency |
| --- | --- | --- |
| Full-data GT-label audit | 6864649 | Completed: 4,053 demos / 502,431 frames |
| Matched GT/control GPU preflight | 6864650 | Completed, exit 0; both variants optimized and sampled valid actions |
| Exact simulation reset check | 6864687 | Completed; TCP, joints and object pose matched the separate reference reset |
| Native grasp-reference collection | 6864651 | 339 records: 333 completed, six unavailable |
| Two matched fine-tuning arms | 6864652, tasks 0–1 | Completed, exit 0: control 3h 53m; GT-pose 4h 33m; two epochs / 62,802 updates per arm |
| Failed V2 evaluation | 6864664, tasks 0–7 | All failed before any rollout: missing `oracle_path` import |
| V3 end-to-end smoke | 6910609, tasks 0–1 | Both completed, exit 0; control executed 120 actions, GT-pose completed its task in 72 |
| Recovered full evaluation | 6910610, tasks 0–7 | Queued after successful smoke checks; original checkpoints and protocol reused |
| Recovered report / statistics | 6910611 / 6910612 | Queued behind recovered evaluation/report; prior incomplete V2 reports preserved |
| New visual pose architecture GPU smoke | 6865392 | Completed, exit 0; one optimization step and observation-only action sampling |

Evaluation still plans **149 Test1 scenarios × seeds 42/43/44 = 447 rollouts
per arm, 894 total**. Training completed with identical shared initial weights,
per-epoch minibatch digests and update counts. V2 then failed before producing any
policy measurement because the adapter did not import its reference-path helper.
These missing measurements are not a 0% success result.

V3 adds the missing import, four regression checks and a full-budget end-to-end
rollout smoke check for each trained arm. Both smoke checks now passed. On that
one matched start, control timed out at 120 actions and GT guidance completed
the task in 72. These runtime checks are stored separately and do not establish
a benchmark success-rate gain. Full evaluation verifies that both passed smoke
receipts still match the trained checkpoint hashes before running. The
same 120-step budget, native scoring, reset checks and learned action/phase logic
are retained. Scheduler reservations are reduced to four CPUs and 24 GB host RAM;
the existing two-thread compute settings are unchanged. No retraining is needed.
Prior sources, failed logs and incomplete reports remain preserved in
`evaluation_recovery_v3/previous_v2/`. The six unavailable expert references remain
explicit coverage gaps. There is no valid benchmark grasp/task success rate yet.
[Recovery revision](../../results/diagnostics/visual_oracle_pose_20261110_01/evaluation_revision_v3.json),
[recovery jobs](../../results/diagnostics/visual_oracle_pose_20261110_01/evaluation_recovery_v3/submissions.json).

[GT run](../../results/diagnostics/visual_oracle_pose_20261110_01/README.md),
[job ledger](../../results/diagnostics/visual_oracle_pose_20261110_01/submissions.json),
[GT preflight receipt](../../results/diagnostics/visual_oracle_pose_20261110_01/preflight.json),
[new architecture smoke receipt](../../results/diagnostics/grounded_visual_pose_20261111_01/preflight/preflight_checks.json).
Scheduler states are a dated snapshot; recheck Slurm before reporting progress.

## Stage 1: matched GT-pose comparison

Both arms initialize from the same completed ten-epoch original visual control
checkpoint, `world_geometry_10epoch_diffusion_next_action_transformer_6724878/last.pt`.
One arm disables the new pose signal; the other supplies a GT target TCP pose.
Both receive **two additional epochs**, all four-category training demonstrations,
seed 42, batch 16, LR 1e-4, the same RGB/PCD observations, action targets and
existing auxiliary losses. DINOv2 and T5 stay frozen. Existing trainable policy
modules fine-tune, and the pose-conditioning MLP learns jointly from action loss.

```text
RGB history + observed scene PCD + task + measured TCP/opening
  -> original encoders, visual gate, history, hierarchy and fusion -> context
GT target TCP pose + current TCP -> relative-pose MLP -------------> context
  -> original Transformer diffusion action head -> 8 × 7 action chunk
```

The added MLP is **10 → 256 → 256**. It embeds relative translation, relative
rotation and validity before the original action decoder. Its final projection
starts at zero, so both arms initially reproduce the original policy. It does
not correct or replace already generated robot commands. Direct contact-coordinate
conditioning stays disabled in both arms; the existing contact loss remains
auxiliary supervision.

### Target definition and information boundary

For a grasp-starting demonstration, closure command `j` uses measured
`obs/tcp_pose[j-1]` as its target: released observations are post-action.
The pose is **xyzw quaternion followed by world XYZ in meters**. The supplied
first-grasp goal stays fixed throughout the task, including later skills.
No GT phase, closure clock, reference action prefix/tail or GT part cloud is
added to the policy inputs. Non-grasp-starting tasks remain with invalid targets.

The audit found 2,478 grasp-start demonstrations with valid targets and 1,575
non-grasp-start demonstrations retained for training. Test references are obtained
in separate native-expert executions, not by driving the evaluated policy with
expert commands. Only successful, unambiguous pre-closure references qualify;
independent policy resets are checked for physical agreement.

### Evaluation and claims

The original learned skill progression and termination logic remain in control.
Every evaluated action comes from the policy from reset through the full task.
Record separately:

- Initial-phase part-grasp acquisition with a closed command.
- Native part grasp at the final action of the initial learned phase.
- Full native task success.
- Missing references, runtime errors, completed coverage and paired coverage.

The native grasp predicate permits contact with either finger; it does not prove
sustained retention. Primary comparisons use identical completed scenario/seed
pairs, retaining all planned denominators. Do not count missing references as
physical failures or silently substitute a pose. Report per-seed means/std only
with complete declared coverage, and identify any incomplete comparison.

Before attributing a difference to GT guidance, verify shared initialization,
minibatch-order digests and optimizer-step counts from the completed arms.
This initial screen uses one training seed; three evaluation seeds do not measure
training-seed variability. The fixed expert-reference goal may become stale after
object motion and is not an assured performance upper bound.

[Model](../models/visual_pose_policy_v1.py),
[conditioner](../models/pose_conditioning_v1.py),
[labels](../data/oracle_grasp_pose_v1.py),
[trainer](../scripts/train_visual_pose_v1.py),
[evaluation adapter](../scripts/evaluate_visual_oracle_pose_v3.py),
[config](../configs/pose_learning/visual_oracle_pose_v1.json),
[GT-stage diagram](slide_history/media/2026-11-10/visual_gt_pose_finetuning.pdf).
The V1 submitter and V2 adapter remain historical frozen sources. The recovery
uses V3 with its own source hashes and a smoke gate; preserve all trained artifacts.

## Stage 2: shared grounding and predicted pose

The implemented architecture keeps global visual/scene/skill context in the
original action path and adds a shared grounding-to-pose branch:

```text
Same RGB history + observed scene PCD + task + measured TCP/opening
  -> shared encoders / history / inferred skill
       -> original global visual/scene/skill context -------------------+
       -> skill-conditioned soft target-part grounding                 |
            -> weighted geometry + visual attention + TCP/opening      |
            -> target grasp TCP pose + predicted applicability         |
            -> differentiable relative-pose MLP -----------------------+
       -> original Transformer diffusion action head -> 8 × 7 chunk
```

The baseline contact scorer supplies soft scene-region weights. Weighted point
features and covariance plus a geometry/skill query attending to RGB tokens
feed the pose estimator. This is learned visual attention, not a calibrated
image crop. It predicts desired TCP position/orientation, not object pose.
The predicted relative-pose features and applicability condition the same action
decoder; action gradients can reach the estimator and grounding.

Global visual context, language/skill, history and robot state remain available.
Contact XYZ and aligned part membership are supervision, not a second independent
contact-to-action target. Unused heatmap, approach and contact-token heads are
omitted in this version. The original contact conditioner's learned zero-input
constant is retained when loading the baseline checkpoint for action equivalence.

Pose regression labels cover observations before the first closure in grasp-start
demonstrations. Later/non-grasp frames still train actions and supervise zero
pose applicability. Deployment predicts grounding, pose and applicability from
observations and inferred skill; no GT grasp pose, part cloud or event clock is
needed. A single demonstrated pose does not represent all equivalent grasps.
Later skills currently rely on the original action policy.

### Validation completed; full training deferred

Fifteen CPU/regression checks passed across both visual paths. A real HDF5 loader
check retained two RGB frames and 1,024 observed scene points. GPU job **6865392**
then used the actual pretrained DINO/T5 encoders with batch 16, completed one
optimization step with finite gradients, and sampled valid **16 × 8 × 7** action
chunks using observation-only predicted hierarchy, grounding and pose. Gradients
reached pose translation/rotation, applicability, grounding and conditioning;
DINO/T5 stayed frozen. This was one smoke step, not a completed training epoch
or evidence of grasp/task performance. Its checkpoint is a smoke artifact only.

The prepared recipe fine-tunes the original visual checkpoint for two epochs;
new pose/attention/applicability layers initialize anew. Full training and physical
evaluation have not been submitted. This design adds both pose and region
supervision, so the next measured study needs a matched control and a
supervision-only ablation before isolating the effect of predicted pose input.

[Model](../models/grounded_visual_pose_policy_v1.py),
[pose/grounding modules](../models/grounded_pose_head_v1.py),
[labels](../data/grounded_grasp_pose_v1.py),
[trainer](../scripts/train_grounded_visual_pose_v1.py),
[rollout adapter](../scripts/evaluate_grounded_visual_pose_v1.py),
[config](../configs/pose_learning/grounded_visual_pose_v1.yaml),
[comparison diagram](slide_history/media/2026-11-11/visual_pose_architectures.pdf),
[CPU validation](records/grounded_visual_pose_v1_validation.json).

## Next decisions

1. Let the V3 end-to-end smoke checks gate the recovered GT/control evaluation.
   Training is complete; preserve both checkpoints and avoid duplicate training.
2. Audit matched training, reference gaps and paired evaluation coverage, then
   compare grasp acquisition, phase-end grasp and full-task success.
3. If GT guidance is useful, train/evaluate the shared visual pose estimator with
   matched initialization, data, update budgets and evaluation starts. Otherwise
   diagnose target usage and failure modes before expanding the pipeline.
4. Compare predicted-pose input against pose-supervision-only and baseline
   controls. Consider contact-auxiliary removal only through a matched ablation.
   Defer outcome critics/tolerance-based action selection.

## Baseline context and cleanup boundaries

The [config index](../configs/README.md) now lists three current regular recipes.
Eighteen historical configs were moved into a checksummed archive, with one
compatibility symlink retained for frozen defaults. Model settings and source
code must be read together; learned weights live in checkpoints.

The historical original visual Transformer scored **64/447** with direct contact
input disabled; the earlier matched contact-input model scored **63/447**.
Those are 3D-contact studies, not the current GT-grasp-pose comparison. The old
64/447 checkpoint supplies initialization; the newly fine-tuned control is the
primary comparator. [Historical visual-policy results](../../results/README.md).

Active code supports both visual paths, original policy dependencies and their
training/evaluation/reporting tools. Unused replay/tolerance/evaluator launchers,
reports and dedicated tests were retired from active directories after checking
imports and queued jobs. Historical source is compressed and checksummed;
generated Python caches were deleted. Frozen model/data/helper files remain
where required by the current run's broad source manifest. In particular,
`generate_logged_references.py` still imports `run_interaction_tolerance_pilot.py`.
Do not remove these files or edit their hashes while the queue depends on them.

[This cleanup manifest](archive/2026-11-11_scope_cleanup/manifest.json),
[previous non-visual result deletion](records/cleanup_20261111_nonvisual_pose.json),
[archived research history](archive/2026-11-11_scope_cleanup/README.md).
Original visual checkpoints, released data, current result folders, other projects
and dated slide history remain intact.
