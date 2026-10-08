# Part2Action Research Plan

Updated: 2026-10-05. Status: unified MLP variants implemented and training submitted; performance evaluation outstanding.
Environment and current runs: [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md).

## Goal and strongest recommendation

**Next: test one unified MLP action policy conditioned on predicted contact
geometry throughout the motion.** Replace distance-based XYZ switching and the
separate bounded contact-offset branch in a new configurable variant. Keep the
visual part gate, contact predictor, skill hierarchy, temporal encoder, and
existing action conventions. Preserve current variants and checkpoints.

```text
Task + RGB history + scene cloud + TCP pose + finger opening
  -> existing part grounding, contact prediction, and skill hierarchy
  -> observation/skill features + predicted contact geometry relative to TCP
  -> one MLP action head -> XYZ, wrist, gripper throughout the motion
```

The question is whether predicted geometry improves actions without the hard
switch and bounded surface-relative output. This differs from the current
baseline, whose auxiliary contact point is unused by the action command.
The unified MLP implementation and paired configs now exist. No training jobs
have been submitted for them; this is not a demonstrated improvement.

## Immediate experiment: contact-conditioned unified MLP

### What to replace first

| Component | First experiment |
|---|---|
| Visual part gate and contact predictor | Keep, including auxiliary supervision |
| Skill hierarchy | Keep its existing interface and behavior |
| Temporal encoder | Keep two frames in both comparison variants |
| 3 cm distance gate | Disable in both unified variants |
| Contact-plus-bounded-offset action branch | Disable; one action head directly predicts the existing action representation |
| MLP action head | Add optional predicted-contact conditioning |
| Backbone, dataset subset, contact-label mode | Keep fixed across the comparison |

Do not confuse the contact distance gate with the visual part gate or skill
selector. Existing hierarchy phase/termination modules are not the proposed new
contact-branch selector. A learned selector is a later alternative, not required
for the unified experiment.

### Matched comparison

- **U0: unified MLP without explicit contact conditioning.** Two-frame temporal
  encoder; retain the contact head and its supervision but do not feed its output
  to the action head.
- **U1: unified MLP with predicted contact conditioning.** Same model, training
  split, losses, action targets, and budget; supply predicted contact geometry
  through a small conditioning projection. Match action-head input capacity by
  using zeroed conditioning features in U0.

Start with the relative vector `predicted_contact_xyz - current_tcp_xyz` in a
specified common coordinate frame and normalization. A confidence/uncertainty
feature is optional: first inspect what the existing head actually exposes.
Attention concentration alone must not be called calibrated contact confidence.
Define handling of absent/degenerate scene geometry consistently at training
and rollout; do not use ground-truth label validity as a deployment input.

Use **predicted geometry in both training and rollout**. Ground-truth part masks
and derived contact points remain auxiliary supervision only. For the first
comparison, detach explicit contact-conditioning features from action-loss
backpropagation in both variants, so action gradients cannot distort the contact
predictor through this new path. Shared feature learning otherwise stays as-is;
a later ablation can test joint gradients.

Train U0 and U1 from scratch with matched seeds; do not reuse incompatible action
head weights silently. Existing one-frame baseline and switched two-frame model
are historical references, not the matched control for this question.

### Preparation, verification, and decision

1. Finish reviewing contact-gate GIFs and document the geometry limitation.
   Completed CPU audit 6675263 found 0/57 sampled touch segments and 0/16 scissors
   grasp segments ever eligible at 3 cm. This measures demonstration gate
   coverage, not rollout success or dataset corruption. The 3/5/7/10 cm gallery
   and GIF comparisons are diagnostics, not trained threshold ablations.
2. Carve a fixed episode-level validation split from training demos, stratifying
   by object/task where possible. Keep official test metadata out of tuning.
   Use the original four-object subset initially; expanding to 11 categories is
   a separate experiment after release verification.
3. Implement an optional contact-conditioning module and explicit configuration
   flags. Validate that U0 ignores contact predictions, U1 uses them, training
   and rollout apply identical transforms, and neither receives privileged part
   inputs. Preserve existing near-contact behavior when its config is selected.
4. Run a short Slurm smoke run before matched full training. Check finite losses,
   conditioning scales, auxiliary contact quality, and action outputs. Then use
   a development rollout protocol with matched seeds, steps, and termination.
5. Report success by object/skill, TCP/action errors, wrist/gripper failures,
   contact localization, and compute. On the trained U1 model, zero/shuffle the
   conditioning as a diagnostic intervention to check whether it is used;
   distribution-shift effects mean this does not replace the U0 comparison.
6. Repeat promising results across seeds. Advance when conditioning helps
   closed-loop behavior without a material regression in contact prediction.
   If it does not, investigate localization and whether one point is adequate
   before assuming a more powerful decoder will solve the problem.

Index outputs under `results/part2action/runs/` and
`results/part2action/evaluations/`; large physical outputs remain on bigtemp.
Contact diagnostics and GIFs are under `results/diagnostics/contact_label_audit/`.

## Second experiment: action chunks with a stronger decoder

If U1 supports the conditioning hypothesis, compare MLP against a diffusion or
flow-matching decoder that predicts a sequence of actions. Keep contact
conditioning, observations, hierarchy, split, and evaluation protocol fixed.
Define prediction horizon, execution horizon, temporal alignment, normalization,
and gripper/wrist action semantics before integration. Include an MLP chunk
baseline with the same horizons to distinguish chunking gains from decoder gains.
Measure inference latency and use receding-horizon execution. This is a separate
implementation and experiment, not a near-contact YAML toggle.

Candidate contacts, interaction poses, spatial correspondence, and a learned
branch selector remain follow-up directions. Surface contacts and usable TCP
poses differ; explicit pose proposals need orientation, gripper geometry, and
opening width. Keep collision and representability checks for those directions.

## Current problems to investigate

| Observation in current code/configs | Possible consequence or unresolved question |
|---|---|
| Approach head exists, but approach loss weight is zero and its output does not control actions | We are not testing an explicit learned approach controller |
| Contact XYZ is a weighted average of scene points | Multiple modes can average into empty space or between surfaces |
| Contact correction starts only within 3 cm | Large approach errors may never activate it |
| Training switch uses labeled contact; rollout uses predicted contact | Prediction errors can change switching behavior |
| Baseline and near-contact differ in both frame history and contact control | Their comparison cannot isolate either change |
| Contact targets are derived proxies, not measured finger contacts | Label validity and surface-to-TCP geometry need auditing |
| Maintained configs omit contact_label_mode; loader defaults to legacy | Skill-specific label support exists but is not active; do not assume labels are skill-specific |
| Action head is a deterministic MLP | Different valid demonstrated motions may be averaged |
| Contact scorer receives pooled visual context | Fine correspondence between image regions and scene points may be lost |
| No held-out validation HDF5 split | Offline metrics currently assess training data, not generalization |
| Hierarchy predicts skill slots and phases | Bad actions may originate in skill selection rather than low-level control |
| Two-frame temporal model takes longer to train | Profile encoders, data loading, and simulator inference before optimizing |

## Four longer-term candidate methods

| Method | Proposed change | Why test it | Evidence to collect |
|---|---|---|---|
| 1. Contact candidates and confidence | Score scene points, retain several candidates, predict candidate-specific pose/width; supervise valid neighborhoods | Avoid averaging distinct contact modes; preserve alternative interactions | Surface distance, target localization, candidate recall, confidence calibration, rollout success |
| 2. Contact-and-pose approach | Predict a pre-contact pose and interaction pose; condition approach/local control on geometry and skill | Make contact guidance useful before the hand is already close | Pre-contact reach rate, collision rate, final pose error, grasp/touch success, switch stability |
| 3. Spatially grounded action features | Associate point tokens with calibrated image features; compare structured 3D action representations if needed | Preserve which image region belongs to each scene point | Part/contact localization, pose error, success under object-pose variation, compute cost |
| 4. Multimodal action generation | Compare MLP with a contact-conditioned diffusion action head using repeated observations | Represent several valid motion sequences instead of their average | Success, motion consistency, inference latency, recovery after perturbation |

Method 4 is not a simple near-contact YAML toggle: the current near-contact
branch requires an MLP and monolithic action head. New integration is needed.
Methods 1 and 2 should be developed incrementally rather than together in an
uninterpretable large change. Never restore the discarded whole-path offset
or enlarge it as a substitute for approach control.

## Locate the bottleneck before retraining

Initial label pilot: [CONTACT_LABEL_AUDIT.md](CONTACT_LABEL_AUDIT.md).
User review found consistent touch timing disagreement across the reviewed
objects: legacy selects motion onset, skill-specific selects the final approach.
The per-skill/object 3 cm switch audit is complete for the first 32 demos per
object (job 6675263). Inspect its report and the threshold-comparison GIFs;
expand coverage before making broader claims about contact supervision.
Investigate simulator contacts only with faithful replay, and demonstrated
pre-contact/interaction TCP poses as targets separate from surface contacts.

1. Verify complete baseline and near-contact evaluations and their episode IDs,
   checkpoints, seeds, action conventions, step limits, and controller settings.
2. Audit contact labels by skill: event timing, valid fraction, coordinate frame,
   nearest surface, gripper geometry, and object motion. Compare legacy versus
   skill-specific derivation on training demos before selecting new targets.
   If enabling approach supervision, verify directions come from actual TCP
   displacement or correctly interpreted deltas, not averaged absolute positions.
3. Measure contact error in centimeters, prediction-to-surface distance, and
   contact probability spread. Compare to an appropriate label neighborhood.
4. Log distance to contact, approach pose error, local-controller activation,
   switch frequency, wrist error, and gripper close timing.
5. Separate failures into skill/phase selection, wrong part, wrong contact,
   failure to approach, wrist/collision issues, close timing, and post-contact motion.
6. Run controlled diagnostic interventions one component at a time. Any oracle
   part/contact/skill experiment is diagnosis only, labeled separately; it is
   not a deployment policy or headline no-mask benchmark result.
7. Profile time in data loading, frozen visual/text encoders, trainable modules,
   and rollout inference. Consider caching frozen features only when preprocessing
   is fixed; measure cache size and loading cost, and preserve augmentation behavior.

## Evaluation and experiment order

- Retain the four-object PartInstruct/PartGym setup for initial controlled work.
- Split training demonstrations by episode (not adjacent timesteps), stratified
  by object/task where possible. Keep test demos completely out of training and
  validation. Preserve the existing full-data checkpoints as historical references.
- For new ablations, hold training split, labels, action representation, budget,
  and rollout protocol fixed. Compare one frame/no contact, two frames/no contact,
  one frame/near contact, and two frames/near contact to separate the two factors.
- Prioritize the matched U0/U1 unified MLP experiment above. The four candidate
  methods below remain a longer-term menu; the earlier contact-and-pose-first
  recommendation is superseded by this smaller controlled experiment.
- Repeat promising experiments across seeds and report per-object/task success
  counts with uncertainty. The 39-trial protocol is a pilot, not strong evidence
  of broad generalization. Use a separate development protocol for repeated tuning.
- Compare official policies on matched objects, tasks, trial counts, and limits.
  Report their privileged part inputs explicitly. DP3's existing scissors-only
  five-trial result cannot be ranked against our four-object 39-trial protocol.
- Proceed with larger training only when diagnostics and controlled rollouts
  support the proposed improvement. Replacing the backbone is not the first step.

## Dataset and environment suitability

PartInstruct provides sequences, task/skill language, robot observations,
actions, and part labels; PartGym enables closed-loop success evaluation.
This is appropriate for a part-to-action proof of concept. It does not alone
establish real-camera robustness, broad object generalization, or true contact
pose supervision. Four objects and one observed camera stream constrain coverage.
Check coordinate transforms, action semantics, cloud quality, and task balance.
Do not assume extra sensors or force/contact annotations exist without inspection.

Reuse the installed separate training and simulator environments, Slurm jobs,
and bigtemp storage. Record dependencies, checkpoint/config provenance, and
resource use for reproducibility; current env suitability is supported by
completed runs, not a fresh dependency audit.

## Existing modularity and implementation boundary

- `models/heads.py`: separate fusion, part gate, point-cloud encoder, contact
  attention, approach, MLP/diffusion action, and contact-offset classes.
- `models/temporal.py`: identity versus temporal transformer encoders.
- `models/hierarchical.py`: skill planning, phase selection, termination modules.
- `models/part2action_model.py`: conditional module construction and orchestration.
- `scripts/train.py`: YAML model/data builders, losses, freezing, and learning-rate groups.

Existing switches support ablations, but dependencies constrain combinations.
A head in `heads`, its loss weight, and its use in executed actions are three
separate choices: zero loss does not remove a module or connect its output to control.
Changing module flags can invalidate checkpoint compatibility. Keep data/model
flags aligned and validate configs. New candidate/pose controllers require new
classes and matching training/rollout integration; they are not implemented yet.

## Research precedents

These papers motivate representations; none validates our exact 3 cm switch
or guarantees gains on PartGym.

- [Contact-GraspNet](https://arxiv.org/abs/2103.14127): roots grasp proposals in observed surface contacts to reduce the search space.
- [Where2Act](https://openaccess.thecvf.com/content/ICCV2021/papers/Mo_Where2Act_From_Pixels_to_Actions_for_Articulated_3D_Objects_ICCV_2021_paper.pdf): predicts point actionability, interaction orientations, and proposal success.
- [VAT-Mart](https://hyperplane-lab.github.io/vat-mart/): predicts affordances and contact-conditioned trajectory proposals for articulated objects.
- [PerAct](https://peract.github.io/) and [RVT](https://proceedings.mlr.press/v229/goyal23a.html): structured spatial representations for action prediction.
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/): multimodal action-sequence generation and receding-horizon execution.

## Unified MLP implementation (2026-10-05)

`ContactGeometryConditioner` in `models/heads.py` projects the detached predicted
world contact minus world TCP, scaled by 0.1 m, into the existing action features.
It is enabled with `use_unified_contact_conditioning`; U0 uses
`contact_conditioning_enabled: false` to feed zeros through the same projection.
Nonfinite geometry feeds zeros. Confidence is omitted until its semantics and
calibration are established. Training and rollout builders use the same flags.

Paired configs under `configs/architecture_update/` are
`hierarchical_world_geometry_30epoch_unified_control.yaml` and
`hierarchical_world_geometry_30epoch_unified_contact.yaml`. Both retain the
existing two-frame temporal encoder, hierarchy, labels, and auxiliary losses,
and disable near-contact/residual/relative action branches. The existing MLP
already outputs eight actions; this experiment preserves that chunk length.
The later decoder comparison should therefore match the existing MLP horizon,
rather than describe it as a single-action output. Current rollout executes one
action per observation cycle.

`slurm/train_world_geometry_unified_h100.sbatch` accepts `VARIANT=control` or
`VARIANT=contact`, stores large outputs on bigtemp and registers them under
root results/. It has not been submitted. These configs still inherit empty
`val_hdf5`; create the fixed episode validation split before full training.
CPU synthetic checks cover geometry influence, detached gradients, disabled
control, translation invariance, nonfinite geometry, and a model forward path
with stub vision/text encoders. Real GPU training and simulator evaluation
remain required.

## Training decision — 2026-10-06

User explicitly chose train/test only, without a train/validation split. This
supersedes the validation-split prerequisite above for this pair. Both variants
use the original four training files, no held-out validation (`val_hdf5: []`),
30 epochs, seed 42, and no checkpoint initialization. Official test episodes
remain excluded from training. Offline fallback metrics refer to training data,
not held-out validation. Evaluate the final checkpoint with matched rollouts.

Submitted control job 6687620 and contact-conditioned job 6687621 using
`train_world_geometry_unified_h100.sbatch`. Large outputs are
`/p/part2action/results/part2action/runs/world_geometry_30epoch_unified_control_6687620`
and `world_geometry_30epoch_unified_contact_6687621`; launcher registers both
under repository-root `results/part2action/runs/` at startup. Both runs completed 30 epochs on 2026-10-08. Matched PartGym rollouts6701228–6701231 are submitted after all four models
passed H100 smoke6701227 with CPU FPS. Full task-success results remain pending.
