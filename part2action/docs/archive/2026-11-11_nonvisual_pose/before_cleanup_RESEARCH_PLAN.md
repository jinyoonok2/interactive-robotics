# Part2Action research plan

Project identity: **Part2Action / P2A**, Franka Panda in PartGym. The separate
RB-Y1 Part-Aware Pose predictor and Live Robotics Lab policy runs are listed in
[the experiment ownership index](EXPERIMENT_OWNERSHIP.md). Their 20,000-update
training and door-opening results are not Part2Action experiments.

Updated November 10, 2026. Current decisions and evidence live here; this is not a
submission journal. Commands/storage: [SETUP](SETUP.md). Data/labels/splits:
[DATASET](DATASET.md). Earlier work is preserved in the
[documentation archive](archive/2026-10-22/README.md). The
[October 26 discussion](slide_history/2026-10-26_part2action_meeting_status.md)
introduces the new contribution direction.

[November 2 meeting slides](slide_history/2026-11-02_robustness_training_update.md)
now follow recovered policy results, released DP/DP3 references, GPC, the proposed
tolerance architecture, implemented reference collection, remaining work and
adopted/different components. The dated evidence snapshot distinguishes ongoing
jobs from completed results; no new tolerance-policy improvement is claimed.

## First priority: make interaction geometry useful to the actor — November 5

### Active stage: original visual policy with supplied GT grasp pose — November 10

The user approved testing **GT pose guidance in the original visual model first**,
then adding a learned pose estimator. New versioned code implements that first
stage: `VisualPosePolicyV1` inherits the original construction and preserves its
RGB/scene/language/state encoders, grounding, hierarchy, shared parameter names,
and Transformer diffusion action head. A new module embeds a target TCP pose
relative to the current TCP and adds its signal to the existing action features.
It does not output or overwrite commands. Its zero-initialized final projection
starts exactly at the original policy. The old implementation is unchanged.

The [current GT-stage diagram](figures/2026-11-10_visual_gt_pose_finetuning/visual_gt_pose_finetuning.png)
separates frozen DINO/T5 encoders, fine-tuned existing policy modules and the
new pose-conditioning MLP. [Vector PDF](figures/2026-11-10_visual_gt_pose_finetuning/visual_gt_pose_finetuning.pdf),
[editable SVG](figures/2026-11-10_visual_gt_pose_finetuning/visual_gt_pose_finetuning.svg)
and a [LaTeX figure/caption](figures/2026-11-10_visual_gt_pose_finetuning/report_figure.tex)
are available for reports. The earlier estimator-branch figure remains the
future-stage design; it does not describe this supplied-GT-pose experiment.

The initial screen fine-tunes two arms from the same completed original visual
control checkpoint (`6724878`, ten training epochs): **pose input disabled** and
**GT pose input enabled**. Both receive two additional epochs on all 4,053
four-category demonstrations, seed 42, batch 16, LR 1e-4, the same existing
auxiliary losses, visual observations and action targets. Shared initial-weight
digests, epoch minibatch-order digests and optimizer-step counts are saved for a
matched-training audit. The old 64/447 score is context for the initialization;
the new fine-tuned control is the primary comparator. This is an initial
fine-tuning screen, not a multi-training-seed result or an assured upper bound.

For tasks starting with grasp, the supplied target is the measured TCP pose
immediately before the first commanded closure. Released observation `j-1` is
the pre-action state for closure command `j`. The target remains fixed throughout
the full episode, including subsequent skills: no GT phase, closure time, or
skill-end clock is supplied. Non-grasp-starting tasks receive an invalid target
in both arms and remain in the data/evaluation population. The loader does not
add part clouds, expert actions, or event indices to model inputs.

Evaluation plans the same **149 Test1 scenarios at seeds 42/43/44: 447 rollouts
per arm**. A separate native expert scene attempts a reference for each of the
339 grasp-starting scenario/seed pairs. Only a successful native grasp with an
unambiguous measured pre-closure pose qualifies. The evaluated policy resets
independently to the same physical state and generates **all** its actions from
reset through the full task; no recorded prefix or continuation is executed.
The original learned phase/termination logic remains in control. Native part
grasp acquisition and grasp at the end of the initial learned phase are scored
separately from full-task success. The native grasp predicate permits one-finger
part contact; sustained retention is not implied. Failed or missing expert
references are explicit coverage gaps, never policy failures or silent fallback
poses. Primary tables use identical completed case/seed pairs and retain all
planned denominators.

Six CPU checks passed, including baseline/noise equivalence, retained visual
input dependence, action-loss gradients to the pose module, geometry-frame
invariance, measured-preclosure label timing and honest missing-trial reporting.
A real released HDF5 sample also passed through the unchanged RGB/action loader
with the added target fields. Slurm stages perform a full-data label audit and
real-data GPU optimization/sampling preflight before either training arm starts.
Source/config/checkpoint hashes and snapshots are stored in the new run. The
learned pose estimator is deliberately not trained in this stage.

Startup verification: the full label audit completed with **4,053 demos and
502,431 frames**, including 2,478 valid grasp-start targets and 1,575 non-grasp-
starting demos retained with invalid targets. The first full-vision simulation
reset matched its separate expert reference exactly in TCP, robot joints, object
position and orientation; RGB and the 1,024-point scene cloud remain available.
The first reference generations succeeded. This is setup evidence, not policy
success-rate evidence. Submitted jobs: audit **6864649**, GPU preflight
**6864650**, oracle reference array **6864651**, matched training array
**6864652**, evaluation **6864664**, automatic report **6864665**, and simulation
reset check **6864687**. Evaluation uses a versioned adapter calling the original
`rollout_one` helper; its unstarted earlier jobs 6864653/6864654 were superseded,
with the original model/trainer/source snapshot preserved. The GPU preflight
was pending scheduling priority at the last startup check.

[Run report](../../results/diagnostics/visual_oracle_pose_20261110_01/README.md),
[jobs](../../results/diagnostics/visual_oracle_pose_20261110_01/submissions.json),
[versioned model](../models/visual_pose_policy_v1.py),
[pose conditioner](../models/pose_conditioning_v1.py),
[label adapter](../data/oracle_grasp_pose_v1.py),
[runner](../scripts/run_visual_oracle_pose_v1.py),
[config](../configs/pose_learning/visual_oracle_pose_v1.json).

### Scope correction: retain the visual baseline and add pose estimation — November 10

The user's intended comparison is **the original visual policy versus the same
visual policy with an added target-grasp-pose estimator**. The separate
geometry-MLP/learned-controller studies below do not answer that question. They
change the action model and use diagnostic privileges, and their grasp scores
must not be presented as an improvement over the visual Transformer. Preserve
their artifacts as component studies; the next architecture work should address
the matched visual-policy extension, rather than expand the small-controller
pipeline.

The intended forward paths are:

```text
Baseline:
RGB history + scene PCD + task + measured TCP/opening
  -> existing encoders, part gate, hierarchy and fusion
  -> existing Transformer diffusion action head -> 8 x 7 action chunk

With pose estimation:
Same observations -> same encoders, part gate, hierarchy and fusion -> h_t
  h_t + measured TCP -> new target-grasp-pose estimator -> pose context
  h_t + pose context -> same Transformer diffusion action head -> 8 x 7 chunk
```

The new module predicts a desired **gripper/TCP position and orientation** for
the interaction, not merely the object's pose. Its prediction conditions the
existing action generator before action decoding. It does not replace that
generator with the small MLP controller or correct an already generated action.
Reuse the learned visual/scene/skill features: do not substitute GT part clouds
or handcrafted geometry statistics for the baseline's normal input pipeline.
The existing implementation forms `action_fused` immediately before
`self.action_head` in `models/part2action_model.py`; this is the intended
insertion point. The current 3D contact-point head is not a 6D pose estimator.

| Condition | Visual policy input and action head | Pose use |
| --- | --- | --- |
| Original baseline | Original RGB/scene/state/language pipeline and diffusion Transformer | No new pose branch |
| Baseline + predicted pose | Same pipeline and action-head architecture | New estimator predicts pose from the policy's observation features and conditions actions |
| Oracle-pose diagnostic | The pose-augmented visual policy and its same action head | Replace the predicted target with a GT target; label this privileged condition separately |

An additional supervision-only control should retain the pose estimator and its
pose loss but disconnect pose conditioning from the action path. This separates
the effect of extra supervision from actually using the predicted pose. The
primary comparison remains baseline versus baseline plus the pose module.

Match demonstration IDs/splits, RGB history, point sampling, shared initial
weights, minibatches, optimizer/update budget, action representation, diffusion
settings, training seeds, and evaluation starts. Use either the same shared
untrained initialization, or the same pretrained initialization and fine-tuning
budget in both arms; do not compare an old checkpoint with a newly fine-tuned
variant and attribute the whole difference to pose estimation. Keep existing
contact conditioning and auxiliary-loss settings matched outside the new module.
Normal rollout must use predicted grounding/pose, the original learned skill
progression, and policy actions throughout. No reference-event clock, recorded
continuation, GT target-part cloud, or GT target pose enters the normal arm.

Construct and audit target-pose supervision before training: coordinate frame,
quaternion convention, measured pre-closure TCP versus commanded pose,
observation/action time alignment, valid interaction/skill coverage, and missing
labels must be explicit. Targets supervise the pose loss; they are not normal
action-model inputs. First validate the added path, original RGB feature use,
gradients, and baseline equivalence with pose conditioning disabled. Use a common
small debugging subset only for smoke checks, then the same full predefined
evaluation population in both arms. The existing four-category Test1 protocol is
149 scenarios at seeds 42/43/44, or 447 rollouts per model. Report pose error,
initial part grasp, and full-task success separately. Pilot 24-case/72-attempt
scores and the historical 447-rollout policy scores are different populations.

**Status:** this section and the
[corrected architecture figure](figures/2026-11-10_matched_visual_pose_comparison/visual_baseline_plus_pose.png)
record the intended eventual design and comparison. The **learned estimator**
branch has not yet been integrated or evaluated; the newly implemented supplied-
GT-pose conditioning stage above precedes it. The earlier
[implemented-system figure](figures/2026-11-10_architecture_comparison/visual_policy_vs_pose_guided_pilot.png)
remains a historical description of the separate pilot, not the desired matched
architecture. Preserve all original models, configs, checkpoints and figures.

The user relayed the professor's priority: establish that the actor uses the
interaction target to improve control before advancing the outcome evaluator.
The evaluator/tolerance proposal below remains a later stage. Existing results
show no demonstrated aggregate target-input benefit; equal success totals do
not establish identical actions or complete independence from the target.

Workflow preference, November 5: queue authorized long experiments on Slurm,
save durable job/checkpoint/report records, and return with status links after
submission/startup checks rather than waiting in chat for completion. The
small learned decoder below has established supplied-pose use in a controlled
pilot, and the subsequent point-to-pose pilot has completed. Its report retains
the annotated-input and recorded-continuation limits; it does not validate the
full visual architecture. Establish useful part-guided control before evaluator
expansion.

### Current implementation: common pose samples and point necessity — November 8

The next decision is **whether an explicit interaction point is needed at all**.
All earlier geometry heads received a future-reference-selected point, so their
results cannot establish point necessity. The earlier point-first architecture
below is a hypothesis, not a requirement. First unify data access, then run one
matched comparison within that interface rather than creating another unrelated
dataset/model pipeline.

Added `data/pose_samples.py` provides versioned observations, labels, privileged
diagnostic fields and provenance. It adapts both released HDF5 frames and fresh
replay captures; recorded RGB is never silently combined with replay robot
states. HDF5 post-action frame `t` predicts commands beginning at `t+1`; fresh
captures after action 0 predict commands beginning at action 1. Units, frames,
source IDs, instance splits and hashes are explicit. Normal model inputs exclude
all labels and privileges; named privileges require explicit diagnostic opt-in.

Reuse the fixed 95-source population (63 Train, eight development, 24 Test) and
the existing instance split. Capture RGB, depth, unsegmented visible scene PCD,
calibration and measured robot state together, without advancing physics.
The initial small-head comparison uses the same frozen annotated part/object
clouds in both conditions; camera data are stored for future visual integration,
not encoded by this MLP experiment.

| Version-1 condition | Extra information beyond common part/object geometry and robot state |
| --- | --- |
| `without_point` | None; the last three input channels are zero |
| `with_oracle_point` | Reference-selected interaction point relative to current TCP |

Both heads have 52 input channels, width 64 and 8,137 parameters. The first 49
features and all neighborhood selection use current geometry only. Both decode
position relative to the **same part centroid**, with a linear position residual,
and predict rotation identically. Neither the output origin nor cropping reads
the oracle point. This differs from the prior point-anchored/bounded decoder;
historical 23/24 is context, not a matched score for this new architecture.

Train two variants at three seeds, 3,000 updates, batch 64, matched initial
weights/minibatches, optimizer/loss and development-only checkpoint selection.
Execute both variants for all three head seeds on the same 24 Test starts plus
the retained Train calibration: 150 new executions. Reuse the 25 unchanged
reference-oracle-pose controls read-only. The same learned actor, source-event
clock, original action-0 prefix and recorded continuation remain diagnostic
conditions. Missing/blocked/error trials are unknown, not task failures.

This comparison asks whether the point helps **given this annotated geometry,
data and model capacity**. It does not prove universal point necessity or
eliminate GT geometry, the source clock or recorded continuation. If no-point
control is adequate, omit a separate point predictor. If the point helps,
investigate what information it supplies before training a deployed localizer.
After choosing the interface, reuse the existing observation encoders and connect
the pose head to the original action Transformer in a matched architecture study.

Existing model implementations, configs and checkpoints are preserved. This
version adds a separate config and implementation; launches save exact source,
config, actor and manifest hashes and refuse existing output directories.
Some existing files are uncommitted, so Git alone is not a complete experiment
archive. Do not claim a config name alone reproduces changing implementation code.

[Common loader](../data/pose_samples.py),
[pose head](../models/part_conditioned_pose.py),
[versioned config](../configs/pose_learning/point_ablation_v1.json),
[runner](../scripts/run_pose_point_ablation.py),
[run commands](SETUP.md#common-pose-samples-and-point-necessity--november-8).

Seven data-boundary/timing/point-leakage/reporting checks passed, as did launcher
syntax and an installed HDF5 frame read through the common adapter. All 95
parent geometry/reference hashes and unchanged implementation hashes verified.
Slurm submission: single-case camera preflight **6845314**, remaining 94 common
captures **6845317**, six-head training **6845318**, 150 executions **6845319**,
automatic report **6845320**. The main collection excludes case 0, so the
preflight observation is not collected twice. All stages have after-any
dependencies and explicit failed/missing receipt handling. These are queued
jobs, not completed training or physical results.

[New background report](../../results/diagnostics/pose_point_ablation_20261108_01/README.md),
[common manifest](../../results/diagnostics/pose_point_ablation_20261108_01/manifest.json),
[job ledger](../../results/diagnostics/pose_point_ablation_20261108_01/submissions.json).

### Completed point ablation and next visible-geometry experiment — November 10

The common-record point ablation completed: 95/95 current observations, six
heads and 150/150 new physical executions, with no runtime errors. At head
seeds 42/43/44, Test initial grasp was **23/24, 22/24, 20/24 without point** and
**20/24, 22/24, 21/24 with oracle point**. Across the same 24 cases at three
training seeds, totals were 65/72 and 63/72. Position error improved from
21.07 to 17.49 mm with the oracle point, but rotation was similar (25.35 versus
25.03 degrees), and no grasp benefit was established. Full-task successes after
recorded continuation were 58/72 versus 57/72. These repeated attempts are not
72 independent object instances. Retain the point-free interface for the next
bounded component comparison, without claiming universal point redundancy.

The next experiment removes **full-shape coordinates**, while keeping diagnostic
target-region selection fixed. Existing RGB, depth and camera calibration in
all 95 common records suffice; no new demonstrations or simulator captures are
needed. Backproject current depth pixels into world coordinates. Use a fixed
5 mm proximity to GT object/part annotation clouds only to assign region
membership, then compute shape features from the retained **observed depth
coordinates**. Hidden asset points never enter the visible shape or its output
origin. These are approximate GT-derived region labels, not learned perception
or exact simulator masks. The tolerance is fixed before the new evaluation;
minimum 16 visible points is required, with no GT fallback or silent case drops.

Reuse `PartConditionedPoseHead`, the common loader, original training routine
and physical executor **unchanged**. Train only three new point-free heads from
visible geometry: same data/splits, initialization, minibatches, optimizer,
3,000-update budget, loss and development selection. The old full-geometry
point-free heads and 75 executions are reused read-only; compare the new 75
executions at the same three seeds and 25 starts. Audit matched initialization,
minibatches, update budget, parameter count and training indices against the
parent. The 25 oracle-pose controls are also retained read-only.

This isolates an observation/representation change under privileged region
labels. The frozen actor, reference closure/phase clock and original prefix/tail
remain. Only after visible-geometry feasibility is established should region
membership be supplied by the learned visual part module in a matched test.

[Depth conversion](../data/visible_pose_geometry.py),
[configuration](../configs/pose_learning/visible_geometry_v1.json),
[runner](../scripts/run_visible_pose_geometry.py).

Five camera-coordinate/visibility/leakage/reporting checks passed. Three actual
Train captures had 264, 263 and 64 visible part points, above the fixed 16-point
minimum. Existing parent source/capture/model/execution hashes verified. Slurm
chain submitted: conversion **6859526**, training **6859527**, evaluation
**6859528**, automatic report **6859529**. Submission is not completed training
or a new grasp result. Existing baseline executions are not rerun.

[Visible-geometry report](../../results/diagnostics/pose_visible_geometry_20261110_01/README.md),
[job ledger](../../results/diagnostics/pose_visible_geometry_20261110_01/submissions.json).

**Train-only formatting repair, same day.** Initial conversion found Train
`demo_790` (mug handle) had two actual visible points. The existing feature
constructor requires 16 rows; that is an input-format constraint, not proof
that the two measured points must be replaced by full GT shape. Preserve this
strict conversion audit. Its still-unstarted training/evaluation jobs
6859527/6859528 were cancelled; report 6859529 now depends on conversion only.

Version 2 repeats every observed point equally when a region has 2–15 pixels,
meeting the unchanged head's row-count requirement while retaining actual
observed counts in receipts/provenance. It adds no coordinates or information;
pose performance under sparse observations is measured rather than assumed.
Empty/single-point regions remain unavailable, with no GT fallback. The 5 mm
annotation tolerance, data/splits, model, training and evaluation settings are
unchanged; the repair was chosen using the Train case before examining new Test
outcomes. Seven checks, including sparse-coordinate provenance and no-fallback
behavior, passed; the two-point Train case now produces finite features.
Conversion runs in one CPU job to avoid 95 repeated interpreter launches.

[Active version-2 config](../configs/pose_learning/visible_geometry_v2.json),
[sparse adapter](../data/sparse_visible_pose_geometry.py),
[version-2 runner](../scripts/run_sparse_visible_pose_geometry.py).

Active version-2 Slurm chain: conversion **6859674**, three-head training
**6859675**, 75 evaluations **6859676**, automatic report **6859677**.
The version-1 strict audit remains separate; no existing architecture or
checkpoint was overwritten.
[Active report](../../results/diagnostics/pose_visible_geometry_20261110_02/README.md),
[jobs](../../results/diagnostics/pose_visible_geometry_20261110_02/submissions.json).

### Motivation and working hypothesis

The frozen target-input diagnostic produced 66/447 successes with the original
target, 66/447 with zero relative geometry and 68/447 with a +5 cm world-X shift.
These are the paired diagnostic protocol's scores, not the separately evaluated
63/447 standard ON run. The aggregate results do not demonstrate that supplying
the predicted point improves control. They also do not establish identical
successful episodes or identical motion across conditions.

The proposed repair is **contact-anchored interaction pose and motion**. A point
identifies where to interact; a gripper pose describes how the robot should
interact there. Use the point to construct that pose, then make the approach
trajectory reach it. The hypothesis is that an explicit geometric relationship
and supervision for each stage will make the point useful for action generation.
This must improve correct-target control; sensitivity to wrong targets alone
would not establish a useful policy.

### What the current architecture permits

The evaluated Transformer denoiser consumes a pooled observation condition and
predicts absolute world-frame action poses. `ContactGeometryConditioner` adds a
projection of detached target-minus-TCP geometry to that condition. In the
normalized-global variant, a learned scalar starts at 0.01; this initialization
is not evidence that its trained value is zero. Action loss trains the
conditioner and decoder but cannot train the target coordinates through this
detached branch. Other image/scene features provide a route to predict absolute
actions without using explicit target geometry. Adding a token or moving the
same input earlier alone does not establish a useful geometric relationship.

The existing target is a derived target-part surface proxy, not a measured
finger contact or a desired TCP pose. A correct surface point does not specify
gripper orientation, opening, approach clearance, or interaction time. The older
contact-relative MLP head and distance-gated near-contact variants already exist;
reintroducing contact-plus-offset for every action is not a new method and does
not resolve the earlier TCP/surface and phase issues.

### Proposed actor change: interaction subgoal and approach trajectory

For a first controlled approach-to-grasp/touch experiment, make the selected
part point the origin of an explicit interaction subgoal. Use local scene
geometry, skill and measured robot state to predict the gripper orientation,
opening and the offset between that surface point and the desired TCP. Rebuild
the world-frame interaction pose from that geometry instead of predicting an
independent absolute interaction position. Account for actual gripper/finger
kinematics; do not place TCP directly on the surface. Audit offset targets and
labels against demonstration/replay contact events before bounding them.

Generate the approach toward this subgoal. One testable construction is a
trajectory bridge from the measured initial TCP pose to the interaction pose,
with learned path residuals that vanish at the endpoints. The fixed geometric
construction links the contact subgoal to the approach endpoint; diffusion can
model alternative approach paths and orientations. Preserve scene context for
obstacles and reachability. This link is an architectural constraint, not proof
of task success or a guarantee that an erroneous target is safe/reachable.

The interaction event need not fall inside an eight-action chunk. Begin with
event-aligned approach segments and explicit subgoal supervision; execute a
prefix and observe again. Do not force every short chunk to end at contact or
keep move/release/retraction actions at the initial surface point. Establish the
approach experiment before extending the representation to those phases.

Relative-position attention between action tokens and the local target-part
point features is a subsequent mechanism to test, rather than another pooled
XYZ input. Do not change the keypose representation, attention, labels and
training budget simultaneously when attributing a result.

### Proposed heads and losses

Retain shared observation encoders and the existing skill context. Reuse the
point-selection head, add an interaction-pose head, and adapt the diffusion
action decoder to the approach subgoal. These are prediction heads within one
policy; separate heads do not require separate visual/language backbones.

```text
RGB history + scene point cloud + skill + measured robot state
                         |
                 Shared observation features
                         |
                 Interaction-point head -> c
                         |
       Local part geometry + skill + current robot state
                         |
          Interaction-pose head -> offset d, rotation R, opening g
                         |
              Target TCP pose: position c + d, rotation R
                         |
    Scene context + target pose + current TCP -> approach decoder
                         |
                Execute a prefix, observe again
```

| Component | Head and output | Supervision / proposed loss |
| --- | --- | --- |
| Interaction-point localization | Existing point-selection head; target-part point `c` | Validity-masked localization loss against an audited reference point; start with Smooth L1 on XYZ and retain applicable existing point-label supervision |
| Interaction pose | New head; surface-to-TCP offset `d`, rotation `R`, and opening `g` | Smooth L1 on reconstructed position `c + d`, rotation geodesic error on SO(3), and Smooth L1 on opening at the defined interaction event |
| Approach motion | Adapted diffusion action decoder; approach action sequence | Existing diffusion noise-prediction objective on valid action steps, with the approach tied to the interaction subgoal |

Conceptually, `L_total = lambda_point * L_point + lambda_pose * L_pose +
lambda_action * L_diffusion`. Within `L_pose`, weight position, rotation and
opening separately because they use different units. Select loss weights and
normalization using training/development data, before final evaluation.

Supervise the reconstructed TCP pose against a verified demonstrated/replayed
interaction pose. A ground-truth point alone does not define that pose. Audit
event timing and distinguish measured TCP states from commanded action poses
before building labels. Reject or explicitly mask unsupported events instead
of interpreting every closest surface point as a successful contact.

For a hard endpoint construction, endpoint consistency is enforced by the
trajectory representation. If a soft construction is compared, add a separately
reported interaction-event consistency loss; do not constrain every eight-step
chunk endpoint to the surface. Avoid a new unrestricted absolute-position branch
that bypasses the point. An unrestricted offset can also cancel the anchor;
derive offset limits from gripper geometry and audited events, then check this
failure mode explicitly. Do not claim that the sum `c + d` alone guarantees use.

Keep the existing frozen vision/language encoders initially. For the first
verified-point experiment, train the new pose head and adapted motion decoder
without dependence on localization errors. When connecting the learned point
head, declare whether downstream losses backpropagate into it. Initially freeze
or detach the point predictor to isolate its accuracy; joint fine-tuning is a
later ablation with localization supervision retained so a nominal surface point
does not drift into an unconstrained latent control variable.

### Distinguish three failure stages

| Failure | Example | Diagnostic |
| --- | --- | --- |
| Localization | Selects mug body instead of handle | Compare predicted and verified interaction points/target-part membership |
| Pose construction | Selects handle but proposes an unsuitable gripper orientation or TCP offset | Supply a verified point; compare constructed pose with a verified interaction pose |
| Motion/control | Receives a suitable interaction pose but misses it, closes at the wrong time, or fails to acquire contact | Supply a verified pose; measure executed approach, opening and grasp/touch outcome |

Develop in that order of isolation: first verify motion generation/execution
with a verified pose, then train pose construction with verified points, then
connect the learned point predictor. Supplying a verified pose tests the
approach stage; it does not validate the geometric point-to-pose relationship.
For the initial diagnostic, fix the intended skill so phase-selection errors do
not obscure these stages; subsequently evaluate learned skill progression.

### First experiment and criteria for proceeding

Use an instance-held-out development split and audited approach events. Check
coordinate frames, observation/action timing, surface-to-TCP offsets and valid
grasp/touch labels. Compare the existing late-input actor with the proposed
subgoal actor under matched data and training updates. Train a verified-target
diagnostic version to separate low-level control capacity from localization,
then test predicted targets. GT diagnostic information is not a deployment input.
Use predicted targets in training/deployment consistently or explicitly measure
the teacher-forcing gap; do not attach a newly predicted target at test time and
claim a fair comparison.

Before a large rollout sweep, freeze actual observations, skill, robot state
and diffusion noise. Change only the supplied target and measure the resulting
interaction pose and action trajectories. Record the old branch's learned
weight, feature scales and action sensitivity. A forced endpoint shift in the
new actor is a plumbing check; it is not learned robustness evidence. An
unconstrained residual can cancel an anchor, so inspect offsets and their bounds.

Evaluate correct, zero, shuffled and plausible shifted targets with the same
trained actor. Use fresh simulator executions to measure first approach,
orientation/opening, target-part contact acquisition, grasp/touch success and
full-task success, including paired successes/failures. Wrong-target sensitivity
alone is insufficient: correct-target control must improve over the matched
baseline. Also test genuine scene changes with updated observations/targets;
changing an internal point alone does not test scene adaptation.

If verified-target control works but predicted-target control fails, prioritize
localization/label reliability. If both fail, repair the pose/path representation
or controller interface before adding an evaluator. A single point cannot
represent the complete task; measure whether the explicit subgoal helps the
phase where its geometric meaning is valid.

This is a design proposal, not an implemented actor or a new training result.
Relevant precedents include
[HDP: Manipulation Trajectory Generation via Contact Guidance](https://arxiv.org/abs/2411.12982)
(contact subgoals plus contact-directed low-level learning) and
[3D Diffuser Actor, CoRL 2024](https://3d-diffuser-actor.github.io/)
(spatial interaction between action and scene tokens). Contact-guided actions
and relative coordinates are established ideas; novelty is not claimed for
this repair.

### Three related papers and architecture lessons — November 7

These papers inform the current actor repair. They are precedents for explicit
interaction goals and structured control, not evidence that our architecture
has already improved or that a part-to-pose hierarchy is novel.

| Paper and verified venue | Mechanism relevant to this plan | What to adopt or test |
| --- | --- | --- |
| Wang et al., **Hierarchical Diffusion Policy: Manipulation Trajectory Generation via Contact Guidance (HDP)**, IEEE Transactions on Robotics, 2025, vol. 41, pp. 2086–2104. DOI: `10.1109/TRO.2025.3547272`. [Publication](https://ieeexplore.ieee.org/document/10912754), [paper, Section III-B](https://arxiv.org/html/2411.12982v1). | A diffusion guider predicts contact goals; a diffusion actor generates motion from observations and contacts. Ordinary diffusion imitation loss did not sufficiently link the generated trajectory to contact attainment. Its actor adds a contact-directed Q-learning objective. | Give the interaction goal an explicit relationship to motion. Our first comparison can use supervised pose-reaching or a constrained approach representation; adopting HDP's critic/RL is not required. HDP is a journal paper and is primarily contact-guided, rather than a semantic part-mask architecture. |
| Goyal et al., **RVT-2: Learning Precise Manipulation from Few Demonstrations**, Robotics: Science and Systems (RSS), 2024. DOI: `10.15607/RSS.2024.XX.055`. [Proceedings](https://www.roboticsproceedings.org/rss20/p055.html), [paper, Sections III-A/B](https://www.roboticsproceedings.org/rss20/p055.pdf). | Predict the next keyframe gripper pose, pass it to a motion planner, then observe again for the next keyframe. A coarse-to-fine scene representation supports localization; features at the selected position condition rotation prediction. | Use part-local geometry to predict gripper orientation and offset, and treat pose as a repeated subgoal. RVT-2 does not validate our semantic part selection; pose prediction plus separate execution is already established. |
| Rana et al., **Learning from 10 Demos: Generalisable and Sample-Efficient Policy Learning with Oriented Affordance Frames**, Conference on Robot Learning (CoRL), 2025, PMLR 305, pp. 5464–5482. [Proceedings and final paper](https://proceedings.mlr.press/v305/rana25a.html), [authors' explanation](https://affordance-policy.github.io/). | Define task frames at functional object regions and orient them toward the tool at subtask start. Train diffusion subpolicies in these coordinates, with a self-progress output for composition. Perception tracks the frames; the work evaluates spatial/object variation and disturbances. | Treat this as a close comparator for part-relevant geometry in control. Neither functional-region guidance, relative frames, modular policies nor robustness to moved objects alone distinguishes our proposal. |

**Meaning of connecting a goal to motion.** Predicting a correct point or pose
with an auxiliary loss does not require the action decoder to use it. An
additional learning objective can evaluate whether the generated approach
reaches the intended interaction pose. Alternatively, an event-aligned path
representation can attach its endpoint to that pose while learning the route.
The latter constrains commanded motion, not successful physical execution.
Neither approach should force every short action chunk to terminate at contact.

The recent neural pilot used relative goal-pose inputs and demonstrated learned
input dependence; it did **not** include a hard endpoint constraint or HDP's
contact-directed Q objective. The point-to-pose pilot used annotated geometry
and a reference-selected surface point, privileged timing, and original recorded
continuation after the first skill. See its
[completed report](../../results/diagnostics/point_to_pose_20261105_01/README.md).
It establishes component feasibility, not deployed part localization or a
matched improvement over the original visual actor.

### Differences to test against Oriented Affordance Frames

OAF already uses task-relevant regions, including functional object parts. Its
reference frame is a coordinate system for a policy, not simply an object
centroid or a desired TCP pose. Its policy already predicts gripper motion;
claiming that only our method considers robot pose would be incorrect.

For a close reading, use the [final paper](https://raw.githubusercontent.com/mlresearch/v305/main/assets/rana25a/rana25a.pdf):
Section 3.1, pp. 4–5 (frame/state/action definitions), Section 3.2 (progress),
Appendix A.1, pp. 13–14 (perception), and A.2, pp. 14–15 (assumptions/limits).
Humans specify the target object and three reference points for the task frame;
its deployment pipeline uses visual correspondence and object-pose tracking.
SAM supplies a whole-object mask in that pipeline, rather than our proposed
instruction-conditioned semantic part mask. This does not make OAF's deployed
localization an oracle. Its compact controller uses frame-relative state and
already outputs end-effector poses, opening and progress.

Our proposed distinction is an **explicit, supervised transformation from the
selected part's local geometry to a robot interaction pose**, followed by
scene-aware execution. This is a design hypothesis. Adding a pose head alone
does not establish a research contribution.

| Dimension | OAF precedent | Our proposed experiment and present limit |
| --- | --- | --- |
| Functional region | Human-specified task frames are localized through visual correspondence and tracking; they supply the reference for a subpolicy. | Learn instruction/skill-conditioned part masks and point selection, then construct the pose. Learned selection across alternative parts remains to be demonstrated; current pose pilots supply annotations. |
| Region to gripper geometry | Policy states/actions are represented relative to the affordance/tool frames. | Explicitly supervise surface-to-TCP offset and wrist orientation from local part shape and robot state. Test against an OAF-style relative-frame policy without this intermediate pose head. OAF already predicts poses, so this is a difference in decomposition and supervision. |
| Scene information | The demonstrated low-dimensional policy uses tracked frame-relative state. | Retain scene context in the controller for obstacles and reachability. Test matched scenes where the target region is unchanged but the feasible approach changes. This is not a claim that scene-aware control itself is novel. |
| Alternative successful interactions | Diffusion represents multiple possible actions; reported robustness is not an explicitly supervised pose–closure-time success set. | Later learn which combinations of offset, orientation and closure timing succeed for a part/task/state. Demonstrate useful outcome-dependent selection beyond generic diffusion, augmentation or success scoring. This stage remains deferred until useful geometry-guided control works. |
| Progress and timing | Self-progress prediction supports transitions between subpolicies. | Learn interaction/closure timing and phase updates from observations, rather than keeping the pilot's source-event clock. A progress head alone would repeat an established idea. |

For example, both approaches can place a useful reference on a mug handle. Our
test is whether **handle shape and current robot configuration determine a better
interaction pose**, then whether the controller reaches and uses it. A narrow
handle may need a different wrist orientation or TCP offset from a wide one.
Later, outcome data could describe several acceptable pose/timing combinations.
These are proposed capabilities to measure, not limitations we have proved in
OAF or capabilities already established in our system.

**Working question:** does an explicitly supervised part-to-interaction-pose
interface improve requested-part execution on held-out shapes and robot states,
beyond a policy operating directly in a task-relevant affordance frame?

Use matched observations, annotations, object splits, training data/updates and
execution budgets. An OAF-style adaptation to our simulator must be labelled as
our reproduction, not the authors' released result. Compare:

| Comparison | What it isolates |
| --- | --- |
| Existing direct actor with part/contact conditioning | Whether the repair improves the current architecture. |
| Scene-to-pose-to-motion, without explicit part/point structure | Whether pose supervision alone explains any gain. |
| OAF-style relative-frame policy, without the separate interaction-pose head | Whether the extra decomposition helps beyond a functional coordinate frame. |
| Proposed part/point-to-pose-to-motion policy | Whether explicit part geometry provides additional control value. |

Start with verified regions/points to isolate pose/control; then evaluate learned
localization separately. Test different valid requested parts in the same scene,
held-out local geometries, and actual object/robot-state changes with updated
observations. Score requested-part compliance as well as task success. Reaching
a mug by the wrong part must not count as satisfying a handle-specific command.
Report first-skill and full-task outcomes separately, and identify any recorded
continuation. Internal target shifts diagnose dependence; moved-object success
alone does not distinguish us from OAF's existing spatial generalization.

Do not claim novelty for part awareness, relative coordinates, pose prediction,
separate training, self-progress or diffusion multimodality alone. The immediate
priority remains proving useful geometry-guided control; joint pose–timing
tolerance is a later contribution hypothesis requiring matched outcome data.

### Next bounded experiment: matched pose-geometry features — November 7

The first comparison isolates **which pose-head inputs help**, before changing
the controller or implementing an OAF-style baseline. Every variant receives
the same privileged surface point and predicts a bounded TCP offset/orientation
around that point. This is not a comparison against a localization-free or
fully visual scene-to-pose model.

| Pose-head variant | Information retained from the same 52D input |
| --- | --- |
| `point_state` | Supplied point, point relative to TCP, current TCP orientation and opening |
| `object_geometry` | Above plus whole-object centroid relative to the point and whole-object extents |
| `part_geometry` | Above plus target-part centroid/extents and local part-neighborhood statistics/covariance |
| `part_geometry_semantics` | Above plus the semantic part-name one-hot |

All heads keep the same architecture/parameter count and point-plus-offset
decoder. Excluded channels are zero during both training and execution. Train
three seeds (42/43/44), 3,000 updates each, batch 64, with matched initialization,
minibatches, loss and development-only checkpoint selection. Hashes of initial
parameters and sampled minibatches audit the comparison. Execute preselected
seed 42 for every variant, with the same frozen neural actor and source-event
clock. The last two variants separate local geometry from added semantic identity.

Reuse all 75 validated geometry observations. Keep **63 training demonstrations
from six instances and eight development demonstrations from four instances**
unchanged. Add 20 full-replay-eligible sources from the original four held-out
mug instances, deterministically selected before training (selection seed
20261107). This yields **24 test demonstrations, six per held-out instance**, not
24 independent object shapes or an untouched final benchmark. Collect actual
initial geometry separately for each new source. The known training calibration
failure remains in the execution comparison.

For each of 24 test cases plus that calibration case, execute four predicted
poses and a reference-oracle pose through the same frozen actor: **125 trials**.
Measure TCP position/orientation error, native target-part grasp before the
recorded continuation, original full-task success after continuation, and paired
per-case/per-instance outcomes. The oracle-pose control identifies cases where
the frozen actor itself is a bottleneck. Coarse centroid/extents are not a full
scene encoder. This pilot does not establish deployed part selection or learned
tolerance, even if richer features improve execution.

Queued CPU Slurm chain: collection **6837582** (20 new observations), training
**6837583** (12 heads), execution **6837584** (125 trials), report **6837585**.
All arrays use concurrency four and after-any dependencies; missing/blocked/
runtime-error records remain distinct from physical task failures. Submission
is not completion. The report is written automatically; no dataset-wide
expansion or evaluator training follows this bounded study.

Checks before submission: six feature-mask/invariance tests, exact reconstruction
of all 75 parent feature vectors in the training environment, split/source/case
hash checks, launcher syntax, and a missing-collection training/report test.

[Background report](../../results/diagnostics/pose_geometry_ablation_20261107_01/README.md),
[frozen plan](../../results/diagnostics/pose_geometry_ablation_20261107_01/plan.json),
[job IDs](../../results/diagnostics/pose_geometry_ablation_20261107_01/submissions.json),
[runner](../scripts/run_pose_geometry_ablation.py),
[feature masks](../models/pose_geometry_ablation.py).

Check later with `/usr/bin/python3 part2action/scripts/check_background_experiment.py
results/diagnostics/pose_geometry_ablation_20261107_01`. The report includes
matched-budget audits and per-instance/paired execution CSVs. A successful
result justifies the next matched OAF-style/direct-pose comparison; failure
directs attention to pose labels, features or downstream control.

### Completed geometry comparison and seed replication — November 8

The bounded geometry study finished: 95/95 observations, 12/12 trained heads,
and 125/125 executions, with no runtime errors. At this stage both the pose head
and downstream action component are neural models. The original full visual
Part2Action policy has not been retrained with this hierarchy. This is not the
programmed MolmoBot pose-correction experiment in the separate PAP project.

Physical execution used predetermined pose-head seed 42. On 24 held-out mug
demonstrations from four previously inspected instances:

| Pose/features | First-skill target-part grasp | Full task after recorded continuation |
| --- | ---: | ---: |
| Reference oracle pose | 24/24 | 23/24 |
| Point and robot state | 10/24 | 19/24 |
| Added whole-object geometry | 19/24 | 22/24 |
| Added part-local geometry, no semantic one-hot | 23/24 | 22/24 |
| Added semantic part identity | 20/24 | 22/24 |

Relative to object geometry, part geometry recovered five initial-grasp
failures and lost one success (`demo_144`). Full-task outcomes recovered
`demo_144` but lost `demo_1465`: the totals tie. Thus the observed benefit is
initial grasp acquisition, not an established full-task improvement over the
object-geometry variant. Native grasp at the skill boundary does not certify
sustained contact. All four predicted-pose variants failed the existing
training calibration task despite passing its boundary grasp check; retain it.

Offline pose metrics from all three training seeds favor part geometry over
point/state and coarse object geometry, especially orientation. Physical
execution had covered only seed 42. The next bounded action therefore evaluates
the **existing seed-43/44 checkpoints**, with the same frozen actor, original
world/reset seed 42, supplied points, event clock and recorded continuation.
No checkpoint or feature variant is selected by the replication's test scores.

The replication reuses the original 125 execution directories read-only and
adds only **200 missing executions**: 25 starts (24 held-out plus calibration)
× four feature variants × two remaining head seeds. Original oracle controls
are reused once. Twelve trained heads and all 95 observations are shared
unchanged; no new training or data collection is submitted. The final report
separates head seeds, object instances and paired wins/losses; 72 attempts per
variant do not represent 72 independent object shapes.

CPU Slurm jobs: execution **6844268** (array 0–199, concurrency four), automatic
report **6844269** (after-any dependency). Seed routing, missing-result handling,
read-only reuse, all source/checkpoint hashes and 325 unique trial assignments
passed preflight checks. Submission is not completion.

[Replication report](../../results/diagnostics/pose_geometry_seed_replication_20261108_01/README.md),
[plan](../../results/diagnostics/pose_geometry_seed_replication_20261108_01/plan.json),
[job IDs](../../results/diagnostics/pose_geometry_seed_replication_20261108_01/submissions.json),
[runner](../scripts/run_pose_geometry_seed_replication.py).

Use `scripts/check_background_experiment.py` with that result directory for
progress. If the geometry benefit persists, proceed to the matched direct-pose
and OAF-style controller comparisons before claiming a distinctive hierarchy.
If it is seed-dependent, inspect the paired pose/contact failures first.
Observation-only point selection, learned visual part localization, removal of
the privileged clock, and fully learned task continuation remain open stages.

### Temporary pose-guided prototype and first execution — November 5

The first implementation isolates supplied-pose/controller feasibility before
training a learned decoder. It uses the successful `mug/demo_1` replay (episode
91152, object 8605), whose instruction is to grasp the mug at its **right** and
slide it right. This source is not an explicitly labeled handle grasp.

The measured preclosure TCP pose is supplied as an oracle interaction pose.
Replay the original prefix through action 51, replace only arm poses during
actions 52–99, preserve closure onset at 81 and every original finger command,
then execute the original full-task continuation from action 100. Conditions
are unchanged replay, successful commanded-pose hold, measured-pose hold,
measured-pose smooth bridge, and a bridge toward a +5 cm world-X shifted pose.
The commanded/achieved-pose distinction is explicit rather than assumed away.
Controller defaults, source action count and initial seed 42 remain fixed.

Two fresh-process executions per condition give ten trials. Record supplied-pose
tracking before closure and at the grasp boundary, native target-part grasp,
left/right physics contacts, original full-task success, repeated robot/object
motion, command traces and videos. Score the grasp boundary before continuation
can recover from an approach error. Native grasp accepts either finger; bilateral
contact is reported separately, not substituted for native task success. Pose
reach uses predefined 5 mm / 5 degree tolerances at closure end. No added lift,
controller search, source-data edits, learned actor or evaluator are involved.

Geometry/boundary checks passed (three checks). All ten trials in CPU array
**6826598** completed with exit 0; report **6826601** also completed with exit 0.

| Condition | Native original-task success | TCP position error at closure end |
| --- | --- | --- |
| Unchanged recorded-action replay | 2/2 | 0.51 mm from measured preclosure reference |
| Successful commanded-pose hold | 2/2 | 0.39 mm from supplied command pose |
| Measured-pose hold | 2/2 | 0.39 mm from supplied measured pose |
| Measured-pose smooth bridge | 2/2 | 1.93 mm from supplied measured pose |
| +5 cm world-X shifted-pose bridge | 0/2 | 3.23 mm from shifted pose; 47.61 mm from unshifted reference |

All conditions passed the native target-part grasp predicate and had bilateral
finger contact at the closure boundary, including the shifted condition. Those
measurements alone therefore do not establish the intended successful
interaction. The shifted approach produced a different physical trajectory and
failed the unchanged original task; do not reinterpret it as zero contact.

All ten trials preserved the original prefix commands, finger commands and
full-task continuation exactly. Their measured initial and approach-start
physical states matched to 1e-10. Within every condition, the two repetitions
had zero observed differences in TCP positions, object positions and finger
opening. These are repetitions of one scene at one seed, not independent
generalization trials. The analytic construction makes pose dependence explicit
by design; it is not evidence that the existing learned actor uses contact.

This establishes supplied-pose/controller feasibility on this source. Next,
prototype the point-to-interaction-pose head with verified points and explicit
pose supervision, retaining this approach execution as a diagnostic. The
learned approach decoder still requires a separate training and validation
experiment. This pilot neither trains nor validates that decoder, and is not a
comparison against the existing learned policy.

The smooth bridge still had 13.16 mm of measured target-position error just
before the original closing command began, reducing to 1.93 mm by closure end.
Successful completion therefore does not establish that the robot settled at
the target before closing. Preserve this observation when designing temporal
supervision and validating the learned approach/closing sequence; the initial
pilot did not adjust timing after seeing outcomes.

[Experiment plan and outputs](../../results/diagnostics/pose_guided_approach_20261105_01/),
[integrity checks](../../results/diagnostics/pose_guided_approach_20261105_01/integrity_checks.json),
[runner](../scripts/run_pose_guided_approach.py),
[pose bridge](../diagnostics/pose_guided_approach.py),
[geometry checks](../tests/test_pose_guided_approach.py),
[CPU launcher](../slurm/pose_guided_approach.sbatch).

### Learned goal-pose actor: controlled experiment — November 5

The user requested testing an actual trained neural actor after clarifying that
the preceding prototype generated its path analytically. Implement a separate
state-only low-level actor first; it does not retrain the full visual
Part2Action policy or the point-to-pose head. The test asks whether a learned
decoder uses a supplied interaction pose, and whether this improves executed
grasp/move behavior over a matched decoder without that pose.

Use actual pre-action robot states and original commands from accepted successful
mug task-3 (grasp then move) full replays. Preparation froze 36 demonstrations,
3,860 state/action examples: 24 training demonstrations from six object instances,
eight development demonstrations from four other instances, and four test
demonstrations from four further instances. Whole object instances and all their
frames remain in one split. Instances and candidate ordering were randomized
with fixed seed 20261105 before model training; invalid sources are logged. This
is conditional on successful reference replay, not coverage of all original data.

The prototype is a three-hidden-layer MLP (width 128). Inputs are measured TCP
position/orientation, finger opening, source-event clock, and the supplied goal's
relative position/orientation. The source-event clock is privileged and identical
across interventions; this experiment does not discover closing time or skill
progression. The neural outputs are TCP-relative command translation and rotation
plus a learned open/close logit. Decode commands using current TCP state only:
the supplied goal is never copied or added to outputs and no interpolation/path
generator is called. Predict all seven commands and observe again at every step.

Train matched pose-enabled and pose-disabled models from the same initialization
and minibatch sequence (seed 42), 4,000 optimizer updates, batch 256. The control
zeros only the six pose-input features; model capacity, data, state/clock features
and training budget match. Loss combines scaled Smooth L1 command-position error,
relative-rotation-vector error and opening BCE. Select checkpoints using
development loss only; report held-instance test errors after selection. This
MLP feasibility experiment does not implement the proposed diffusion decoder.

First measure same-observation interventions on frozen test robot states and
clocks: correct pose, +5 cm world-X pose, and zero relative pose. Report predicted
command changes rather than infer dependence from task-success totals. Then
execute five conditions on four held-out sources plus one explicitly labeled
training-source calibration case: recorded replay, neural correct pose, neural
shifted pose, neural zero-relative pose, and the separately trained no-pose model.
Replay only source action 0 before neural control, predict through the first
skill, then retain the original full-task continuation. The learned approach
therefore starts earlier than the previous analytic test's action 52; do not
treat their task counts as a matched comparison.

All 25 executions use fresh scenes and fixed seed 42. Record actual commands,
grasp-boundary target error and contact, original native task success, and videos.
Keep training-calibration and held-instance scores separate. Correct-pose benefit,
wrong-pose sensitivity and closed-loop success are distinct findings; none alone
establishes that contact localization or the full existing architecture works.

Three model checks passed, including that zero network weights cannot copy a
distant goal to its output. CPU training **6830842** completed with exit 0
(approximately one minute). All 25 trials in execution array **6830843**,
report **6830844**, and neural-command audit **6830863** completed successfully.

Development selection retained update 1,200 for the pose-enabled model and 2,600
for the matched control, from the same 4,000-update budget. On 462 held-instance
test frames, command-position mean error was **7.57 mm with pose versus 11.20 mm
without pose**; mean rotation error was **13.86 versus 18.41 degrees**. These are
teacher-forced command errors, not physical trajectory errors or task success.
They are frame-level measurements from four sources, not 462 independent trials.

With the test state and clock fixed, a +50 mm world-X goal shift changed neural
command XYZ by a mean **[35.51, -0.14, -2.87] mm**; median command-displacement
magnitude was **39.58 mm**. Zeroing relative pose changed commands by a median
**12.01 mm**. The pose-disabled model changed by exactly zero under the supplied
goal shift. Goal-dependent command changes therefore occur in an actual trained
network; output decoding has no goal argument. This establishes input dependence
for the prototype; closed-loop task success was then checked separately below.

#### Completed neural execution results

The following counts concern four held-out mug instances, one source episode
per instance, fixed simulation seed 42. They exclude the separate training-source
calibration case. Native target-part grasp is measured at the end of the learned
initial skill, before the original task continuation can recover an error.

| Condition | Native original-task success | Native target-part grasp before continuation |
| --- | --- | --- |
| Recorded-action control | 4/4 | 3/4 |
| Trained pose-enabled actor, correct pose | **4/4** | **4/4** |
| Same trained actor, zero relative pose | 0/4 | 0/4 |
| Same trained actor, +5 cm world-X pose | 2/4 | 2/4 |
| Matched actor trained without pose | 1/4 | 0/4 |

On the training-source calibration case, recorded replay and neural correct-pose
control succeeded; the zero, shifted and separately trained no-pose conditions
failed. That case is a fit/execution check, not evidence of generalization.

The pose-enabled actor both responds to the supplied geometry and successfully
controls these held-out examples. The no-pose control's one full-task success
occurred without a target-part grasp at the initial boundary; the unchanged
continuation can recover. The shifted-pose actor still succeeded on two sources:
a shifted pose can permit a different successful interaction, so target
dependence does not imply every corrupted pose must fail. Pose interventions
can also change predicted orientation/opening and hence closing time; the
closed-loop comparison does not isolate translational error alone.

The independent audit recomputed **2,336 neural commands** from frozen weights
and logged actual pre-action states/pose/clock inputs, matching executed command
positions within 0.01 mm, rotations within 0.001 degree, and finger commands
exactly. The original action-0 prefix and post-grasp continuations were unchanged;
common starting robot/object poses matched to 1e-10 within each source. All
recorded-action controls reproduced. No analytic bridge generated neural actions.

This is a positive feasibility result for a small state-only supplied-pose actor,
with one training seed and four held-out instances. It is not an improvement
claim for the existing visual Transformer, learned contact prediction or the
proposed point-to-pose head. Next connect explicit point-to-interaction-pose
prediction and shared observation features to a decoder that has demonstrated
pose use, then evaluate the combined actor under a larger predefined held-out
protocol before returning to the evaluator/tolerance proposal.

[Learned experiment](../../results/diagnostics/learned_pose_actor_20261105_01/),
[neural-command audit](../../results/diagnostics/learned_pose_actor_20261105_01/neural_command_audit.json),
[neural actor](../models/pose_guided_actor.py),
[training/evaluation runner](../scripts/run_learned_pose_actor.py),
[audit runner](../scripts/audit_learned_pose_actor.py),
[model checks](../tests/test_pose_guided_actor.py),
[CPU launcher](../slurm/learned_pose_actor.sbatch).

### Queued next stage: contact-point-to-pose head — November 5

The supplied-pose actor is ready to serve as a **frozen downstream component**.
The next question is whether a learned head can convert a supplied target-part
point into a useful TCP pose. This is a controlled module experiment, not a claim
that deployed localization or the full visual policy is ready.

The background pipeline retains the prior instance split and freezes a copy of
the successful neural actor/checkpoint. Expand only pose-head training sources,
up to 12 eligible mug task-3 demonstrations per existing training instance.
The frozen selection contains **75 sources: 63 training, eight development,
four test**, still six/four/four object instances respectively. The actor itself
is not retrained on these extra demonstrations. The four pilot evaluation
instances have already been used for the preceding component test, so this is
continued pilot evaluation rather than a fresh final benchmark.

First, collect actual scene geometry after original action 0 for every source.
Check robot/object alignment against the successful replay. Query the annotated
target-part and whole-object clouds without advancing physics. Select the part
surface point nearest the successful reference TCP goal. This is a privileged
derived surface proxy, **not measured finger contact or deployed localization**.
Future pose information selects the oracle point for this diagnostic; the pose
head receives the point, current robot state, semantic part label and geometry
descriptors, not the reference pose itself.

The new head receives 52 fixed geometric/state features and predicts a bounded
surface-to-TCP offset (10 cm per-axis bound) plus a continuous 6D orientation.
Reconstruct position as `surface point + learned offset`; project orientation
to SO(3). Supervise position, 6D rotation reconstruction and rotation-matrix
alignment; report rotation error as a geodesic angle. The head does not predict
closing time or replace the actor's opening output. Train seeds 42/43/44, 3,000
updates each, with development-only checkpoint selection. Execute preselected
head seed 42, not whichever seed scores best on test outcomes.

Run the same frozen actor with four supplied-pose conditions on one training
calibration source plus the four held-instance sources (**20 executions**):
reference oracle pose, learned pose from the annotated point, learned pose after
a +5 cm world-X point intervention, and a point-plus-training-mean-offset/mean-
orientation baseline. The point intervention recomputes geometry features from
the unchanged cloud; it does not move the physical object. Compare pose error,
grasp before continuation, and native original-task success. This distinguishes
the pose-construction problem from the already tested low-level actor.

CPU Slurm chain submitted: collection **6830968** (75 tasks, concurrency four),
pose-head training **6830969**, execution **6830970** (20 tasks, concurrency four),
automatic report **6830971**. Submission is not a completed result. Collection
errors or missing upstream models block downstream execution and appear in the
final report; skipped jobs are not negative physical examples. Three head/
geometry checks and the blocked-training/report path passed before submission.
Source snapshots, frozen actor hash and job ledger are retained; no automatic
dataset-wide expansion or evaluator training follows this bounded study.

[Background report](../../results/diagnostics/point_to_pose_20261105_01/README.md),
[job IDs](../../results/diagnostics/point_to_pose_20261105_01/submissions.json),
[status snapshot](../../results/diagnostics/point_to_pose_20261105_01/status.json),
[pose head](../models/interaction_pose_head.py),
[pipeline runner](../scripts/run_point_to_pose.py),
[progress checker](../scripts/check_background_experiment.py).

For a later check, run `/usr/bin/python3 part2action/scripts/check_background_experiment.py
results/diagnostics/point_to_pose_20261105_01` or open the generated report.

## Successful reproducible full-trajectory baselines — November 2

The user clarified that a tolerance anchor does **not** need to match the
released HDF5 path exactly. Accept a full recorded-action execution if it
succeeds under the fixed native task/controller and repeats reliably from the
same source scene and seed. Compare repeated executions with each other, not
with the original measured HDF5 trajectory. The original HDF5 demonstrations
remain unchanged. Call these **successful replay baselines**, distinguishing
them from the released expert ground truth.

Collection scope is all valid bottle/mug/pliers/scissors training demonstrations
(currently 4,053); this is the existing training scope, not all 11 released
categories. Each source gets two fresh-process full executions at seed 42,
native controller defaults, nominal 20 Hz and full task context. A native task
success event is required in both, following native first-success termination;
final-frame contact retention is not an added requirement. The whole source
sequence is still executed for measurement. Native touch predicates advance
physics internally; retain and count those steps rather than assuming every
action always produces exactly 12 substeps. Repeated robot/object positions and finger opening
must agree within 1 mm, arm/object joint positions within 0.001 rad or m as
appropriate, and TCP/object rotations within 1 degree. These are repeatability
criteria, not accuracy requirements relative to the old HDF5 path. Raw continuous
differences are saved. Two repeats are a reproducibility screen, not a success
probability estimate over different initial states or seeds.

First technical preflight: first source per category plus the already successful
`mug/demo_1` positive control, two executions each. Once all five pairs finish
without runtime errors and reproduce their own trajectories, with at least one
successful pair, screen the remaining sources automatically. A category need
not pass its first source to be included in full screening. No grasp-only cut,
regenerated expert, extra lift/hold, per-case controller search or HDF5-fidelity
gate is used. Failed baseline executions and collection errors remain separate
from negative perturbation labels. This stage submits no perturbation or model
training jobs.

Inventory completed: **4,053 sources**, with two entries missing actions kept in
the exclusion ledger. Initial preflight **6783619** and coordinator **6783620**
were stopped after the native touch predicate exposed an incorrect fixed-total-
physics-step assertion. Its outputs remain separate; the corrected preflight
uses the same source inventory and native behavior. Native first-success task
semantics are explicit before full screening. Current status is recorded in the
collection ledger, not inferred from submission.
Corrected preflight array **6783628** and coordinator **6783629** use the frozen
`preflight_v2/plan.json`; `current_study.json` identifies the current run. The
stopped preflight remains an instrumentation-debug record, not baseline data.

Subsequent tolerance tests use the same initial scene, controller, seed and
full-task scoring as their accepted baseline, changing only declared action
factors. Their observations must come from the actual replay state rather than
assuming the old HDF5 observation matches it. Report source coverage and baseline
exclusions so results describe tolerance conditional on successful baselines.

[Collection ledger](../../results/diagnostics/full_trajectory_baselines_20261102/),
[collector](../scripts/collect_full_trajectory_baselines.py),
[repeatability checks](../diagnostics/full_trajectory_baseline.py).

### First full recorded-action replay

The user requested restarting from the original dataset trajectory, before
further grasp-only diagnostics, expert regeneration, or robustness training.
Earlier 1/20 validation results concern a reconstructed **initial grasp segment
plus an added lift test**, not full original-task replay or a demonstration
failure rate.

CPU job **6783613** completed (exit 0, 1 min 18 s). The first full replay uses
`mug/demo_1`, episode `91152`, object `8605`: grasp the mug at its right and move
it to the right. All **122/122 stored actions** were applied unchanged, in order,
once each through native `env.step(action)`, with full task context and native
defaults (20 Hz control, 12 physics substeps per action). No learned policy,
expert planner, phase-specific gain reconstruction, action modification, added
lift, appended hold or success-based early stop was used.

The native original-task criterion succeeded at action index **105** and remained
successful at the final stored action. This establishes task completion for this
one full replay, **not exact trajectory reproduction**: post-action TCP-position
RMSE is **101.28 mm**, maximum **245.26 mm**, final **9.47 mm**. The initial
post-action state matches closely; the first >10 mm deviation occurs at index 1.
Recorded and replayed approach timing differs, while grasp/move states become
much closer. Native current defaults are not claimed to recover the original
per-frame execution settings. Source data are unchanged, and all before/after
states, effective motor calls and command-equality assertions are saved.

[Full-replay results](../../results/diagnostics/full_dataset_trajectory_replay_20261102/mug_demo_1/README.md),
[recorded/replayed video](../../results/diagnostics/full_dataset_trajectory_replay_20261102/mug_demo_1/recorded_vs_full_replay.gif),
[worker](../scripts/replay_full_dataset_trajectory.py),
[launcher](../slurm/replay_full_dataset_trajectory.sbatch).

## Research question and proposed contribution

**Can a policy learn which changes in interaction position, wrist orientation,
and gripper-closing time preserve task success, then use those learned tolerances
to choose more reliable actions?**

The proposed representation is a **part-, task-, and state-conditioned success
set**, rather than one demonstrated interaction point. Its supervision would
come from successful and failed simulator perturbations. Its value must come
from using the learned set in action selection or training, not only predicting
an additional label. A replay-gated CPU perturbation collector is now implemented.
Legacy initial-grasp replay plus added lift validation passed one of 20 cases;
this is not a full-trajectory success rate. Fresh native-expert
generation subsequently validated five of eight pilot references plus a separate
mug smoke, with exact robot/object replay. A small candidate-success head has now
been trained on six instances; held-instance performance did not improve over
simple baselines. Tolerance-guided policy control is not implemented or evaluated.

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

**Focused prior-work check, November 1:** the following primary sources are more
direct overlaps with the current candidate-success prototype. This is a reading
map, not an exhaustive novelty audit.

- [Where2Act (ICCV 2021)](https://arxiv.org/abs/2101.02692): simulated interaction
  outcomes supervise actionability, action proposals and action scoring. A
  learned success score for a candidate interaction is established prior work.
- [VAT-Mart (ICLR 2022)](https://arxiv.org/html/2106.14440): task-conditioned
  pointwise affordances, trajectory proposals and trajectory-success scoring
  learned from interaction data. Task conditioning and trajectory scoring alone
  cannot distinguish our proposal. Read Sections 4.2 and 5.2 for data, losses
  and use of scores in actual manipulation.
- [Dex-Net 2.0 (RSS 2017)](https://goldberg.berkeley.edu/pubs/dex-net-2.0-Camera-Ready-RSS-2017.pdf):
  grasp-quality prediction and candidate selection with analytic robustness
  labels that account for pose/friction uncertainty, using synthetic depth data.
  Repeating one deterministic scene is not equivalent to robustness under a
  distribution of uncertainty.
- [DART (CoRL 2017)](https://proceedings.mlr.press/v78/laskey17a.html): inject
  noise into the supervisor during demonstrations to collect recovery behavior
  and reduce covariate shift. Our fixed-replay outcome collection does not yet
  collect corrective expert actions or train recovery behavior.
- [QT-Opt (CoRL 2018)](https://research.google/pubs/qt-opt-scalable-deep-reinforcement-learning-for-vision-based-robotic-manipulation/):
  learned action values guide closed-loop grasp decisions. Our outcome head is
  a supervised fixed-continuation prototype, not a Bellman-trained Q function
  or a deployed recovery controller.
- [Dynamics randomization (ICRA 2018)](https://xbpeng.github.io/projects/SimToReal/):
  vary simulated dynamics during training to improve transfer. Our command
  sweeps currently leave physics, sensing and initial scene fixed.

The current collector/head are feasibility infrastructure. A contribution would
require evidence beyond generic action-success classification: e.g., useful
joint pose–closure-time dependence and improved candidate selection under
predefined uncertainty, with spatial-only and unstructured outcome predictors
matched for labels, candidate budget and observations. Include ordinary policy
augmentation trained on the same successful data. Specific novelty of timing,
part conditioning or their combination remains unestablished.

### Priority recent main-conference reading — 2025–2026

At the user's request, prioritize these recent papers alongside the foundational
work above. Venues were checked against official proceedings or conference programs.

1. [HAVE: Learn from What We HAVE — CoRL 2025](https://liy1shu.github.io/HAVE_CoRL25/).
   A history-aware verifier selects among diffusion-generated candidate actions.
   This is a direct conceptual comparator: candidate generation plus learned
   verification is not our novelty. Interaction history is an explicit difference
   from the current pre-intervention-only prototype.
2. [Steering Your Diffusion Policy with Latent Space Reinforcement Learning — CoRL 2025](https://diffusion-steering.github.io/).
   DSRL learns to steer a frozen diffusion policy through its initial latent noise.
   A practical comparator for improving a base policy without changing its weights.
3. [Policy Decorator — ICLR 2025](https://policydecorator.github.io/).
   Online residual-policy learning with bounded actions and controlled exploration;
   relevant to evaluating whether a separate correction mechanism suffices.
4. [Diffusion Policy Policy Optimization — ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html).
   Policy-gradient fine-tuning of diffusion policies; a reference for direct
   reward-based policy improvement instead of a separately trained outcome head.
5. [Demystifying Robot Diffusion Policies: Action Memorization and a Simple Lookup Table Alternative — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/122ea6470232ee5e79a2649243348005-Abstract-Conference.html).
   Examines an action-retrieval interpretation in sparse-data settings. Relevant
   to mechanism analysis and simpler controls; its findings should not be treated
   as a universal characterization of all diffusion policies.

HAVE and DSRL are the first reads for the current direction. This selection does
not establish that joint pose–timing modeling is novel, nor does it imply a change
to the running experiments or implementation of these baselines.

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

## Replay-gated robustness pilot — October 30

The user authorized starting the pilot. CPU smoke **6761054** and full pilot
**6761091** completed successfully as software jobs. No GPU/model/encoder is used, and the
existing policy evaluations are unchanged. This is an instrumentation/replay
feasibility study, not a new policy comparison.

- Select five initially-open, first-skill grasp demos per object from training
  episodes only, round-robin over part/instance groups before replay outcomes.
  Twenty cases were found; seven unusable candidates were recorded as skipped.
- Reconstruct native scene/reset and replay the initial grasp segment at an
  explicitly assumed 20 Hz. Per-frame native gains/cadence are absent from HDF5;
  the native-inspired gain schedule is a reconstruction requiring validation.
  Preserve the upstream controller's argument wiring, without modifying it.
- Restore physics state, simulation time, robot time and Python/NumPy RNG between
  trials. No BulletEnv skill state or learned-policy history is instantiated.
  Each first action reissues all arm and finger targets. Require restored TCP
  repeat differences <=1e-6 as a reproducibility check.
- Three unmodified references must meet predefined replay thresholds: TCP RMSE
  <=1 cm, p95 <=2 cm, closure-position error <=1 cm, orientation p95 <=10 degrees,
  and finger-position RMSE <=5 mm. Also require the native grasp-equivalent
  predicate and a fixed additional 5 cm lift/hold test. These are initial pilot
  criteria, not previously validated benchmark tolerances.
- Only eligible cases get 14 one-factor conditions: +/-1 cm on each world axis,
  +/-10 degrees about each TCP-local rotation axis, and +/-2 control steps of
  gripper-command timing. Smoothly introduce spatial offsets before closure and
  hold them through the grasp. Keep arm command timing, gain schedule and
  continuation rule fixed; timing changes only the gripper-command sequence.
- The lift test requires >=3 cm object rise and, over the final 0.3 s, >=80% bilateral
  force-qualified finger contact, >=80% target-part proximity-qualified contact,
  and <=1 cm TCP-relative object-position drift. Target membership uses sampled
  GT surface geometry within 5 mm; it is an approximation, not a semantic contact
  sensor. The test is not full-task success.
- At most 60 reference plus 840 perturbation trials (900 total). Failed reference
  replay is excluded, never counted as perturbation failure. Stop at the replay
  feasibility finding if no cases pass; do not manufacture tolerance labels.

The runner writes incremental CSV/JSON, compressed physics traces, command/TCP
arrays, a dated PNG status table, and an HTML review. This pilot records
low-dimensional outcomes and source references; matched deployable observation
collection, grouped held-out splits and learned tolerance models remain future
work. Native predicate-equivalent scores and the additional lift test are
reported separately from trajectory fidelity.

[Submission ledger](../../results/diagnostics/interaction_tolerance_submission_20261030/submissions.json),
[smoke output](../../results/diagnostics/interaction_tolerance_smoke_20261030/),
[full output](../../results/diagnostics/interaction_tolerance_pilot_20261030/).
Source: [runner](../scripts/run_interaction_tolerance_pilot.py),
[intervention helpers](../diagnostics/interaction_tolerances.py),
[CPU launcher](../slurm/interaction_tolerance_pilot.sbatch).
Five focused tests passed for identity preservation, local rotation composition,
edge-padded command-time shifts, unchanged gains during timing intervention,
reference-fidelity rejection and training-only selection with invalid metadata.
Smoke 6761054 completed without runtime errors. Bottle demo_1029 reproduced
identical TCP trajectories across two restores and passed the native grasp and
additional lift test (4.61 cm object rise). It was nevertheless excluded: TCP p95
error 2.14 cm exceeded the predefined 2 cm limit (RMSE 0.81 cm; closure error
0.027 cm). The gate was not relaxed. Full pilot 6761091 completed 20 cases /
60 references with zero eligible cases and zero perturbation trials. This means
no case passed the conjunction of all gates; it does not mean every grasp failed.
These repeated restores test reproducibility, not independent training or rollout
seed uncertainty; repeated deterministic outcomes must not be counted as
independent evidence.

## Replay investigation and reset-method confirmation — October 31

The user authorized investigating the pilot. A read-only breakdown and 24 CPU
probe trials (four diagnostic cases x three controller profiles x two fresh-scene
repeats) are complete: jobs 6779328/6779329/6779330/6779331. The original pilot
outputs and thresholds are preserved; probe-source snapshots are retained.

**Original failures:** considering any of the three repeats, 15/20 cases exceeded
finger RMSE, 12/20 exceeded TCP p95 and 3/20 exceeded TCP RMSE (overlapping counts).
Only one case passed fidelity on all repeats, while 18 passed the native grasp
predicate and seven passed the added lift test on all repeats. Mug demo_404
passed all three categories but failed the original repeatability check.

**Replay reconstruction issue:** the first collector guessed the strong gripper
settling phase from the number of frames remaining in the segment. Native
`switch_gripper` can follow a different termination path, so this inference is
not justified by saved actions. Testing 0.01 gripper gain throughout reduces
finger RMSE from 3.25 to 0.22 mm (bottle), 4.08 to 0.075 mm (mug), and 7.08 to
0.94 mm (pliers), but worsens the scissors case from 4.64 to 7.03 mm. This is
controller calibration evidence, not a universal corrected release recipe.
Per-frame gains/cadence remain absent from demonstration metadata.

**Restoration issue:** original snapshot repeats show genuine post-contact XYZ
divergence. Fresh native scenes reproduce identical TCP/finger trajectories in
all 12 tested case/profile pairs. This validates a repeatable reconstruction
alternative; the specific internal simulator state responsible is not isolated.
The previous repeat maximum also mixed metres and quaternion components; the
new audit reports XYZ millimetres and rotation degrees separately and treats
opposite quaternion signs as the same orientation.

**Approach calibration:** testing a 15 cm pregrasp threshold lowers bottle p95
error from 21.35 to 0.063 mm but worsens pliers from 5.85 to 19.36 mm. It is an
inferred, exploratory parameter, not known release metadata. Do not apply it as
an unconditional global fix or select per-case gains using task-success scores.
Scissors lift still fails under all three tested profiles.

**Controlled reference check completed:** fresh-reference job **6779359** finished
all original 20 cases and three repeats in 33 minutes (exit 0). Only the scene-reset method changed;
commands, legacy gain schedule, construction seeds, physical continuation and
all original eligibility thresholds were retained. This was baseline-only
validation, with no perturbation-label collection or GPU/model training. Mug
`demo_404`, object instance `8614`, handle, was the only eligible case (1/20);
its three fresh reference trajectories were identical. This isolates restoration
before changing controller reconstruction. A usable
controller-conditioned robustness dataset ultimately needs repeatable nominal
success plus actual executed observations; expert-replay agreement should remain
explicitly measured rather than conflated with every physical failure.

[Investigation report, CSV and dated PNGs](../../results/diagnostics/interaction_replay_audit_20261031/report/index.html),
[probe ledger](../../results/diagnostics/interaction_replay_audit_20261031/submissions.json),
[fresh-reference ledger](../../results/diagnostics/interaction_tolerance_fresh_submission_20261031/submissions.json).
Eight focused tests pass, including quaternion-sign equivalence, separated
controller hypotheses and preservation of construction seeds when selecting a subset.
No eligibility limit was loosened to obtain a passing case.

### First validated-case perturbations — October 31

At the user's request, CPU job **6779656** completed for the validated mug
handle case in 22 minutes 55 seconds (exit 0). All three identity references
passed, followed by 14 single-variable conditions with three fresh-scene repeats each:
world XYZ translations +/-1 cm, TCP-local XYZ rotations +/-10 degrees, and
gripper closure +/-2 steps (0.1 s at 20 Hz). All **42/42 perturbation trials**
passed the stable target grasp/lift criterion. Controller gains, arm timing, continuation,
eligibility thresholds and original construction seed **50** (42 + original
case index 8) are preserved. No calibration profile is substituted.

This case was selected for reference validity, not perturbation success. Exact
repeats assess reproducibility, not independent-scene success probability. This
is a one-case feasibility pilot, not generalization evidence or model training.
The report exports per-condition stable grasp/lift counts as JSON, CSV and a PNG
under `media/2026-10-31/perturbation_outcomes.png`. All tested offsets retained
success; failure boundaries and interactions between variables remain unmeasured.
The doubled-offset follow-up below is submitted. Joint conditions and expansion
to more validated cases remain proposals.

[Perturbation submission ledger](../../results/diagnostics/interaction_tolerance_mug_submission_20261031/submissions.json),
[perturbation outputs](../../results/diagnostics/interaction_tolerance_mug_perturbations_20261031/).

### First boundary-search step — completed November 1

The user asked to begin step by step. CPU job **6779925** tests the same validated
mug handle grasp with doubled single-variable offsets: world XYZ +/-2 cm,
TCP-local XYZ +/-20 degrees, and gripper closure +/-4 control steps (0.2 s).
Three fresh identity references must pass before 14 conditions x three repeats
(42 perturbation trials). Job 6779925 completed in 16 minutes 26 seconds, exit 0:
**39/42** perturbations passed. World-X -2 cm failed all three repeats; the other
13 conditions passed all three repeats. Construction seed 50, legacy controller gains, arm
timing, spatial ramp duration, continuation and every reference/success threshold
remain unchanged. This changes perturbation magnitude, not the learned policy.

The question is whether the previous all-success result used offsets smaller
than the local failure boundary. If all new conditions pass, a larger sweep is
the next proposal; if failures appear, bracket those directions and then test
selected combinations. The finite follow-up below was subsequently authorized. The
collector now accepts configurable magnitudes and prints their actual units in
the PNG outcome table; original defaults remain 1 cm, 10 degrees and two steps.
Nine focused replay/intervention tests pass, including preservation of the
identity reference and single-variable semantics at the larger magnitudes.

This remains one-case data-collection feasibility. Training a useful success
predictor additionally requires multiple validated source grasps, failure labels,
deployable RGB/point-cloud observations and grouped held-out cases. The new
candidate-success branch was subsequently prototyped below; candidate selection
is not implemented and the branch is untrained.

[Boundary-search ledger](../../results/diagnostics/interaction_tolerance_mug_boundary_submission_20261031/submissions.json),
[boundary-search outputs](../../results/diagnostics/interaction_tolerance_mug_boundary_20261031/).

### Automatic finite offset sweep — completed November 1

The user authorized queuing successive levels without individual submissions.
CPU job **6779946** completed in 1 hour 9 minutes 36 seconds, exit 0, after
successful completion of 6779925. It ran four
levels sequentially: position +/-3/4/6/8 cm; TCP-local wrist rotation
+/-30/40/60/80 degrees; closure timing +/-0.3/0.4/0.6/0.8 s. Every condition still
changes only one variable. Each level repeats three identity checks and 14
conditions x three repeats: **168 perturbation plus 12 reference trials** in
total, after the existing 1 cm and 2 cm studies. Case, construction seed 50,
controller, temporal/spatial command rules, continuation and success thresholds
are retained. Timing offsets are validated to keep a real closure onset inside
the source segment.

The sequence completes all predefined levels even after physical failures, to
measure direction dependence and possible return to success at larger offsets.
It halts on process errors or invalid references. Its combined JSON/CSV and dated
PNG table report first sampled failures, the nearest smaller sampled all-success
offset, and any larger sampled success after failure. These observations do not
certify a continuous or monotone success boundary. A new reusable orchestrator
calls the existing collector, rather than duplicating replay or scoring logic.
Eleven focused tests pass, including finite-range validation and nonmonotonic
outcome reporting; combined report generation was checked with synthetic data
in a temporary directory. No architecture training or GPU allocation is added.

All references passed and all four levels completed: **129/168** new perturbation
trials passed. Including the earlier 1 cm and 2 cm studies, the record contains
210 successful and 42 failed trials across 84 unique single-variable settings,
with three exact repeats each (252 perturbation trials, one source grasp).
First sampled failures include positive local wrist X at 30 degrees, negative
local wrist X at 60 degrees, and local wrist +/-Y and -Z at 80 degrees. Closing
time shifts through +/-0.8 seconds passed all repeats. World-X -2 cm failed but
some larger negative-X offsets passed; local wrist +Z also returned to success
at a larger offset after failures. Thus the observed outcomes are nonmonotonic
in some directions and cannot be summarized as one scalar radius. These are
controller-conditioned replay outcomes, not architecture performance or a
continuous failure boundary. Achieved motion and the cause of each failure
still require inspection before using these records as learning supervision.

**Initial achieved-motion audit:** offsets above describe requested commands,
not guaranteed physical grasp displacements. In repeat zero, a -8 cm world-X
command changes TCP X by about -2.16 cm versus the identity trajectory at the
nominal closing frame; +8 cm changes it by about +2.68 cm. Do not claim an
8 cm achieved-position tolerance from command magnitudes. Full executed paths
and closure/contact events must be retained separately from candidate commands.
The +30-degree local-Z trial keeps bilateral/target contact and passes drift but
has 2.934 cm object rise, below the unchanged 3 cm threshold. Thus some binary
failures reflect the chosen lift criterion rather than object loss. Preserve
continuous outcome components and report failure causes before learning a
success set; do not relax the threshold after inspecting these outcomes.

### Observation records and candidate-success interface — November 1

At the user's request, the collector now has opt-in pre-intervention RGB-D and
full-scene point-cloud capture, an outcome dataset interface, an instance-grouped
split helper, and an untrained `InteractionSuccessHead` prototype. These do not
alter any trained policy or install action selection into rollouts.

- Capture two frames immediately before the first command changed by any
  candidate in the selected set. Every candidate shares the same unmodified
  prefix; frame zero is padded if necessary. Save RGB, depth, camera/world XYZ
  clouds, TCP quaternion/world position, measured finger position, joints and
  camera calibration. No part masks or semantic point labels are model inputs.
- Optics follow the native 300x300 agent-view configuration. This prototype
  explicitly uses CPU TinyRenderer and deterministic full-scene pixel sampling
  (1024 points), rather than the standard policy evaluator's renderer/foreground
  FPS path. It is not silently claimed to have identical sensor preprocessing.
- Preserve requested candidate world translation, TCP-local rotation vector and
  closure offset separately from achieved pose changes at nominal closure,
  measured closure-threshold time, physical contact time and execution traces.
  Record all four continuous outcome components, margins, thresholds and causes
  of failure. Success thresholds and the controller remain unchanged.
- The dataset separates predictor inputs from labels/diagnostics. Grouping by
  object category/instance keeps every trajectory and repeat for that instance
  in one split; requesting a held-out split from only one instance raises an
  error. Repeated trials are not independent source demonstrations.
- The head consumes frozen observation/skill features, current robot state,
  nominal primitive command/time-to-close and candidate adjustments. It predicts
  a joint success logit plus four criterion logits without assuming independence
  of the criteria. Context features are detached by default. The nominal
  primitive is currently supplied by the expert replay plan, not a deployed
  policy; later integration must define that interface and test distribution
  transfer explicitly. Achieved motion, contacts and outcome labels are excluded
  from its inputs. No pretrained features are downloaded or new models trained.

CPU check **6782141** completed and passed: three identity replays with observation
capture disabled, followed by three captured identities and two candidate
conditions x three repeats (+3 cm world X, previously successful; +30 degrees
local wrist X, previously failed). It checks capture-on/off replay equivalence,
identical pre-intervention observations across candidates, usable RGB/point
clouds, label separation and presence of both success and failure outcomes.
It produced nine captured records (six success, three failure), identical
pre-intervention observations across candidates, and zero capture-on/off replay
difference. Seventeen focused tests pass, including grouped splits, candidate units,
pre-intervention timing, outcome decomposition, dataset input boundaries and
candidate-head gradients/permutation behavior. One mug instance is not a held-out
learning dataset.

[Collection-check ledger](../../results/diagnostics/interaction_records_check_20261101/submissions.json),
[collector](../scripts/run_interaction_tolerance_pilot.py),
[record helpers](../diagnostics/interaction_records.py),
[dataset interface](../data/interaction_outcomes.py),
[candidate head](../models/interaction_success.py).

### Fresh logged expert-reference pilot — November 1

The user authorized a small fresh-expert generation/replay/variation pilot and
conditional expansion if it works. This is a separate dataset; released HDF5s,
trained policies and existing benchmarks are not replaced. The first adapter
supports initially-open **first-skill grasps**, not all stages or full-task
expert regeneration. Inventory of the four installed training HDF5s finds
**2,478 supported sources and 1,577 explicitly unsupported/malformed entries**
(4,055 raw entries; the policy training recipe uses 4,053 valid demos). Expanding
this adapter is therefore not equivalent to covering the entire dataset.

`generate_logged_references.py` uses the released native BulletPlanner and local
VGN weights, with no new expert-network training. The environment supplies the
expert only its used low-dimensional observations to avoid unnecessary rendering
inside wall-time-limited controller loops; native VGN TSDF views use TinyRenderer.
Raw instruction strings and GT geometry guide the expert, not a learned policy.
Source scenes are initialized from pinned **training** episode metadata using
native settling/reset semantics. New executions need not match old HDF5 paths.

The recorder passes through each native `apply_action`, captures actual joint
motor targets/gains/forces and collision-filter/physics-state mutation calls,
counts exact physics substeps, and logs phase events,
measured robot/object states and force-qualified contacts. The original controller
argument wiring is preserved. Each generated case also records the resolved
configuration, physical settings/dynamics, asset hashes, seeds and versions.
Identity replay reissues these calls in fresh native scenes; it does not teleport
the robot along saved states or substitute measured poses for action targets.

Every generated reference must pass the unchanged additional grasp/lift criteria
and three fresh replays with maximum TCP-position/finger/object-position
differences <=1e-6 m, robot-joint differences <=1e-6 rad, TCP/object rotation
differences <=0.001 degrees and initial-state difference <=1e-9.
One replay has capture disabled; two capture pre-intervention RGB-D/point clouds.
An invalid generation or identity replay never supplies a negative variation
label. Only validated references receive 14 small one-factor conditions, two
predefined joint pose/time conditions and one +60-degree local-X probe. Raw
continuous outcomes and failure causes remain available; these are fixed-controller
grasp/lift labels, not full-task or uncertainty-distribution guarantees.

Initial smoke **6782662** generated a successful fresh mug grasp (3.87 cm lift,
bilateral/target contact fractions 1.0), but the recorder initially omitted native
precondition finger-collision-filter changes. Its first replay differed by up to
0.64 mm TCP position. A subsequent scene also lost its Bullet connection when
old copied-client objects were collected after a connection ID was reused.
The blocked eight-case descendant **6782672** was cancelled. All outputs remain
preserved; these are collection-adapter errors, not negative variation labels.
Recovery records client mutation calls and collects closed-scene cycles before
constructing the next connection. Object/joint agreement checks were also added;
no physical success limit was relaxed. Fresh smoke **6782678**, followed by
pilot **6782679** after successful validation, uses a separate recovery directory.
It regenerates the same mug `demo_404` source, then validates and perturbs it.
The corrected smoke has completed all three successful identity replays with
zero recorded TCP, finger, arm-joint and object-pose differences. Its first two
variations produced a failure at -1 cm X and success at +1 cm X for this newly
generated execution; the remaining conditions are still running at this snapshot.
This new reference uses a different expert-selected grasp from the old HDF5 replay,
so these labels must not be merged as repeats of the earlier mug sweep.

Recovery-1 smoke **6782678** subsequently stopped at the strict common-observation
check after six accepted variations (four successes, two failures). Its first
rotation trial changed pre-intervention RGB/depth because the helper converted
every Euler command to a rotation and back, including zero-offset prefix rows.
Tiny command round-off amplified in contact/rendering; all translation trials
retained identical prefix observations. The fix composes rotations only where
the intervention weight is positive, preserving the untouched commands exactly.
A nontrivial-Euler regression test now requires exact prefix equality. No RGB,
depth, replay or outcome threshold was relaxed. Blocked pilot **6782679** was
cancelled; outputs/source snapshots remain intact and no record index was
published for the interrupted case. New smoke **6782742** and dependent pilot
**6782743** use `logged_expert_references_20261101_recovery2`. Automatic expansion
has not started. The combined focused test suite now passes 24 tests.

A separate eight-case pilot is selected before generation outcomes,
two sources per category. Automatic expansion requires validated references in
all four categories, at least four validated sources, no physically successful
generation rejected by replay, and both successful and failed variation labels.
The expansion controller also checks measured pilot storage requirements. If
the gate passes, supported training sources are collected in CPU array shards,
12 cases per shard and at most four concurrent shards, with per-case validation
retained. Unsupported sources are counted, never silently labeled as collected.
No robustness-head training is submitted by this workflow.

Six focused tests pass for pass-through logging/cadence and collision filters,
quaternion-aware robot/object replay comparison, expansion gates, joint
interventions and training-source inventory.
Operational status is in the study ledger and summaries, not this snapshot.

[Study plan, coverage and submissions](../../results/diagnostics/logged_expert_references_20261101/),
[recovery study](../../results/diagnostics/logged_expert_references_20261101_recovery1/),
[current recovery study](../../results/diagnostics/logged_expert_references_20261101_recovery2/),
[generator](../scripts/generate_logged_references.py),
[recorder](../diagnostics/execution_trace.py),
[conditional expansion](../scripts/run_logged_reference_study.py).

[Sweep plan/source ledger](../../results/diagnostics/interaction_tolerance_mug_sweep_20261101/sweep_plan.json),
[combined sweep review](../../results/diagnostics/interaction_tolerance_mug_sweep_20261101/),
[orchestrator](../scripts/run_interaction_tolerance_sweep.py).

## Controlled lift diagnosis and first predictor training — November 2

The user authorized fixing the execution/validation issue and preparing concrete
report/training material. The corrected fresh-expert smoke completed (14/17
variations successful); the eight-case pilot completed without runtime errors,
with five validated references and 72/84 successful variations. The native expert
and instantaneous grasp predicate reported success for the three rejected
references. Rejection concerns **our added fixed lift/hold**, not a reproduction
of the original full expert task. Both scissors lose contact during the added
lift; the rejected pliers retains bilateral target contact but rises 29.233 mm,
0.767 mm below the unchanged 30 mm limit. Automatic dataset expansion was not
submitted because no scissors reference passed the additional criterion.

**Handoff diagnosis completed:** CPU job **6783263** tested the same five logged grasp
prefixes (three rejected, two successful controls), with original fixed lift,
preserved actual final closing gain, 0.5 s additional settling, 2 s lift, and
preserved gain plus settling (25 trials). Job **6783267** completed preserved
gain + slower lift, with/without settling (10 more trials). All 35 trials
reproduced the original grasp prefix; legacy continuations reproduced their
source outcomes. None of seven recipes rescued all criteria for any of the three
rejected references; both successful controls passed every recipe.
All physical criteria stayed unchanged. This is a diagnostic
selection after observed failures, not an unbiased category success estimate.
The continuation recipe is explicit in each output; probe labels are not mixed
into the earlier fixed-controller training corpus.

Scissors-top results show that preserving the final 0.1 finger gain
instead of lowering it to 0.01 maintains bilateral target contact but gives only
6.53 mm rise. Preserving gain plus settling raises the object 40.68 mm, but
bilateral-contact fraction is zero, so it still fails the original criterion.
Thus a gain handoff affects outcomes but is not a demonstrated universal fix.
Continuous components and target contact must be inspected separately; a
two-finger pinch criterion may reject other retained interaction modes. Do not
change its threshold after seeing results or relabel old records as a new task.

**Original task-context audit completed:** the first collector truncated the
chain to the initial grasp and used skill-level termination. Audits preserve the
full original task metadata and use task-level termination, while still executing
only the initial grasp plus our added lift. Instrumented CPU execution hit the
native 30-second wall-clock guard in two cases. Job **6783306** therefore disabled
only that wall-clock guard, retaining native iteration limits and controller
settings. The bottle control passed exact replay and lift validation; the three
rejected references remained rejected (scissors approximately -3 and 0 mm rise;
pliers approximately 26 mm). This is not a replay of the complete original expert
task and does not establish that the original demonstrations failed.

**Next bounded collection step:** array job **6783341** tests the second and third
native-ranked grasp proposals for each of the three rejected sources (six trials,
two concurrent). Controller settings, task context and physical criteria are
fixed. Eligibility requires native grasp validation, stable lift and exact
identity replay. No automatic dataset expansion or action-policy training follows.
Report refresh job **6783348** runs after the array finishes; current status lives
in the [proposal study ledger](../../results/diagnostics/alternative_expert_proposals_20261101/).

**Outcome training completed:** CPU job **6783264** finished in 4 min 50 s.
The corpus contains **107 records: 92 successes, 15 failures, six source
instances** (two bottle, three mug including the smoke, one pliers). These are
101 variations plus six captured identity references. Only six unique
pre-intervention contexts exist. No scissors outcome is included in training.
Frozen local DINOv2 ViT-S/14-register patch means, Flan-T5-base skill features,
full-scene point moments, robot state and expert nominal primitive context feed
the candidate-success head. No GT masks, object IDs, contacts or post-action
measurements are predictor inputs. Expert nominal target/time-to-close remains
a prototype interface, not a deployed-policy observation.

Leave-one-instance-out evaluation uses six folds, three training seeds and 250
fixed full-batch updates; no held-out early stopping or hyperparameter tuning.
Candidate-only and full-context heads share architecture, optimizer and loss;
the candidate-only control zeros context inputs. A training-fold empirical-prior
constant is also reported. Mean held-instance results across training seeds:

![Held-instance predictor results](slide_history/media/2026-11-02/robustness_pilot/archive_grasp_lift_pilot/slide03_predictor_results.png)

Use this image: [slide03_predictor_results.png](slide_history/media/2026-11-02/robustness_pilot/archive_grasp_lift_pilot/slide03_predictor_results.png).

The context head fits training outcomes better, but there is **no held-instance
gain**. The 86% majority-class accuracy is not useful evidence of failure
prediction. Nine of fifteen failures come from the single pliers instance;
holding it out also removes that category from training. Pooled AUROC compares
scores produced by separate folds and is sensitive to cross-instance
calibration; retain per-instance metrics and raw predictions. These are pilot
classifier results, not an architecture contribution or policy robustness gain.
All 36 final fold/seed/head checkpoints, loss curves, corpus hashes and out-of-fold
predictions are saved. No existing action policy was retrained or changed.

[Historical grasp/lift pilot slides](slide_history/media/2026-11-02/robustness_pilot/archive_grasp_lift_pilot/slides.md),
[training results/checkpoints](../../results/diagnostics/interaction_outcome_training_20261101/),
[handoff probes](../../results/diagnostics/lift_handoff_probe_20261101/),
[factorial continuation probes](../../results/diagnostics/lift_handoff_factorial_20261101/).

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
| C: Transformer target input | ON | ON | Completed: 63/447 |

B/C tests explicit input; A/B would test auxiliary supervision. Neither tests
learned tolerances. C retains late normalized-global conditioning:

`r = (predicted_target_world - measured_TCP_world) / 0.1`

`action_features + P(0) + lambda * LayerNorm(P(r) - P(0))`

Coordinates are detached; lambda is a signed learned global scalar initialized
at 0.01, not confidence. B uses zero relative input and retains P(0). C matches
B's configuration except the input switch, name and output path; no early-fusion
change is included. Labels still use the existing heuristic derivation.

### Verified recovered policy jobs — complete by October 31, 2026

Duplicate Slurm accounting confirms that 6734402 first ended in **NODE_FAIL**.
Its automatic requeue on October 26 then exited with code 2 because the original
output directory already existed. Eight epochs completed; the log stopped about
58% through epoch nine (one-based display), not after a completed ten-epoch run.
The checkpoint does not contain optimizer/scaler/RNG state, so recovery uses a
fresh ten-epoch run rather than treating weights-only continuation as equivalent.

The user authorized recovery on October 27. The partial run and two stale review
placeholders were moved out of active results into the
[interruption archive](../../results/part2action/archive/interrupted_contact_20261027_recovery1/README.md),
with the checkpoint checksum, original ledgers and scheduler history preserved.
The six blocked descendants 6734403/6734404/6735001/6735002/6735003/6735005 were
cancelled. Completed controls, gate diagnostics and released DP/DP3 results remain
intact. Earlier 6734379/6734380/6734381 were cancelled before starting.

| Job | Purpose | Verified state after recovery submission |
| --- | --- | --- |
| 6744368 | Fresh target-input-ON training, ten epochs | COMPLETED, A100 PCIe 40 GB, 51h 42m |
| 6744369 | Standard 447-rollout ON evaluation | COMPLETED: 63/447 |
| 6744370 | Expert/OFF/ON GIFs and quantitative report | COMPLETED, 447 GIFs |
| 6744371 / 6744372 / 6744373 | Frozen ON: identity / zero / +5 cm world-X | COMPLETED: 66/447, 66/447, 68/447 |
| 6744374 | Target diagnostic report and GIFs | COMPLETED, 447 GIFs |
| 6734186 / 6734187 | Released DP-S / DP3-S evaluations | COMPLETED; 29/176 and 81/176 |
| 6734999 / 6735000 / 6735004 | Oracle gate / no gate / gate report | COMPLETED; report available |

Replacement jobs use `--no-requeue` to avoid retrying these fresh-output launchers
into existing directories after another interruption. The training config is
byte-identical to the interrupted submission: stride 1, next-index targets, two
frames, seed 42, late normalized-global target input ON and auxiliary loss ON.
Source fingerprints and dependencies are recorded in the
[recovery ledger](../../results/part2action/runs/diffusion_transformer_contact_submission_20261027_recovery1/submissions.json).
No new architecture, tolerance-learning experiment, or completed-control rerun
was added. Statuses are snapshots; query Slurm before further action.

[ON config](../configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer_contact.yaml),
[replacement ON review](../../results/part2action/evaluations/diffusion_transformer_contact_comparison_20261027_recovery1/index.html).

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

The completed target diagnostics froze replacement C and compared identity, zero relative
input, and a +0.05 m world-X shift. Predictions remain unchanged; actual intervened
inputs are logged separately. Zero retains projection bias and is not a retrained
OFF policy. All three use shared trial/step-specific forked PyTorch noise, including
a new identity evaluation. Closed-loop states can still diverge; this is not a
same-observation counterfactual or a physical-contact measurement. Source hashes
and per-job GPU/simulator preflights protect the submitted protocol.
[Original diagnostic ledger](../../results/diagnostics/frozen_policy_submission_20261022/submissions.json),
[replacement target review](../../results/part2action/evaluations/frozen_target_diagnostics_20261027_recovery1/index.html).

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

### Fixed-scenario released-policy references — submitted October 31

The user requested aligning DP/DP3 evaluations with our scenario list. Each
released model now has a four-case preflight (one case per category, seed 42)
and a dependent 447-trial evaluation: DP-S **6779898 -> 6779899**; DP3-S
**6779900 -> 6779901**. Full runs reuse the exact saved 149-scenario manifest and
seeds 42/43/44, 120 executed-action budget, one action per prediction, [0,0.04] m
finger bounds and our oracle-config camera/robot/task thresholds. Strict EMA
loading and native observation preprocessing are reused; no retraining occurs.
Checkpoint action chunks/denoising remain intact; only the executed prefix is
one action. The separate T5 preprocessing encoder is explicitly in eval mode.

These are **scenario/budget-matched privileged references**, not a fair ranking
of architectures. GT part inputs and oracle skill progression remain. The
native evaluator uses sequential-skill success (`skill_mode=True`), while our
existing 447-rollout evaluator uses full-task predicates (`skill_mode=False`).
The new scores retain native skill scoring and explicitly label this difference;
matching denominators does not remove it. A common scorer and a specified
grounding/progression interface are required for an isolated architecture study.
Identical seed numbers also do not guarantee identical observation RNG streams
across different sensor modalities. Old native results remain untouched.

[Fixed-evaluation ledger](../../results/partinstruct/evaluations/official_fixed_20261031/submissions.json),
[adapter and protocol](../scripts/official_partinstruct/README.md#fixed-scenario-privileged-reference).
Three new tests cover command clipping/budget, timeout-versus-success and exact
scenario validation; the five shared scenario tests also pass. GPU preflights
validate checkpoint loading and rollout interfaces before dependent full runs.

**November 1 recovery:** both original preflights failed before the first scored
trial. Direct stepping retained an extra tokenizer batch axis, unlike the native
AsyncVectorEnv shared-memory conversion; T5 rejected the input shape. This was
an adapter error, not a policy failure. The adapter now applies the native
declared observation shapes and dtypes to tokens, part masks and numeric inputs
before batching. Four focused adapter tests pass, plus CPU integration checks
through both native DP/DP3 preprocessing paths with a shape-checking language
encoder stub. Full checkpoint inference still requires the GPU preflight.

Blocked, never-started jobs 6779899/6779901 were cancelled. Fresh replacements
are DP-S **6781661 -> 6781662** and DP3-S **6781663 -> 6781664**, preserving the
same 149-scenario manifest, seeds, execution settings and scoring limitations.
Original failed outputs and source snapshots remain intact. **November 2,
10:13 America/New_York snapshot:** DP3-S completed **58/447 (13.0%)**; DP-S was
still running with **408/447 trials finished and 2 successes**. The DP-S fraction
is a partial inventory-order result, not a final 447-trial score. GT-part/oracle-
progression and native sequential-skill-scoring differences remain, so these
numbers do not establish a matched architecture ranking. The earlier native
176-trial evaluations remain completed and unchanged.
[Recovery ledger](../../results/partinstruct/evaluations/official_fixed_20261101_recovery1/submissions.json).

## Immediate next decisions

B/C and frozen-target evaluations are complete: 64/447 OFF versus 63/447 ON
provides no demonstrated aggregate input benefit; the scenario-cluster 95%
interval for ON minus OFF is [-3.58, +3.13] percentage points. Identity/zero/shift
diagnostics score 66/66/68, with difference intervals including zero. Read the
replay audit and fresh-reference validation when interpreting tolerance labels.
Use completed gate and released-policy results as supporting evidence with the
comparison limits above. The first validated mug pilot succeeded on all 14
conditions (42 repeated trials); the expanded sweep now has successful and failed
single-variable outcomes on one grasp. Audit achieved motion and failure causes,
then design joint conditions and expand validated source cases. These records
are not learned-policy robustness evidence. Complete
and a prior-work comparison before building a new architecture. The contribution
plan is joint interaction-tolerance learning; existing point/gate studies diagnose
the platform and do not validate that proposed method.
