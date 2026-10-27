# Unified Diffusion Ablation

Updated October 16, 2026. This is a supporting implementation reference.
Diffusion is the default action head for future experiments. MLP support and
completed results are retained for historical comparison; further MLP training
requires an explicit request. This decision does not change queued jobs or
existing checkpoint configurations.
Current experiment decisions: [RESEARCH_PLAN.md](RESEARCH_PLAN.md).
Commands: [SETUP.md](SETUP.md#train). Paths and job status:
[PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md).

The unified policy supports either an MLP or a small conditional DDPM action
decoder. Contact conditioning is applied to shared action features before the
decoder, so it can guide either decoder throughout the motion. The visual and
language encoders, temporal encoder, hierarchy, part gate, and contact predictor
remain separate modules.

## Matched variants

The new configs are
`configs/architecture_update/hierarchical_world_geometry_30epoch_unified_diffusion_control.yaml`
and `hierarchical_world_geometry_30epoch_unified_diffusion_contact.yaml`.
They differ only in explicit predicted-contact conditioning, experiment name,
and output directory. Both retain the existing four training categories,
train/test separation without a validation split, seed 42, 30 epochs, two-frame
history, eight-action horizon, hierarchy, and auxiliary supervision. Both start
from scratch.

The contact variant uses normalized-global conditioning: detached predicted
contact minus TCP is scaled by 0.1 m, projected, bias-subtracted, normalized,
and weighted by a learned global scalar initialized at 0.01. The control
disables that contribution and retains the shared projection bias. Neither
uses the distance-based XYZ switch or bounded offset. Ground-truth part/contact
labels remain auxiliary supervision, not deployment inputs.

## Diffusion decoder

The existing `DiffusionActionHead` is a small MLP noise predictor conditioned on
pooled policy features and the diffusion timestep. It jointly denoises the
eight-by-seven action chunk. This is a custom decoder ablation, separate from
official DP/DP3, and does not introduce a temporal U-Net or diffusion transformer.

New configs use 100 cosine diffusion steps. The cosine schedule leaves the final
forward distribution close to Gaussian noise; the older 50-step linear defaults
left substantial action signal at the terminal timestep. Reverse sampling uses
the DDPM posterior variance and float32 arithmetic under mixed precision.

Action targets are normalized inside the diffusion head, and generated samples
are returned in the original world-action units. Fixed scales are 0.5 m for
XYZ, pi radians for roll/pitch/yaw, and 0.02 m for gripper opening, with a 0.02 m
gripper offset. These are unit scales, not limits or dataset-fitted bounds;
the estimated clean action is clipped to ±2 in normalized coordinates during
reverse sampling. This prevents extreme reverse-process estimates; it is not a
collision check or a certified robot workspace constraint. No test-set statistics
are used.
Normalization buffers are saved with the checkpoint. All model builders pass
the same schedule and normalization settings to training and inference.

Training minimizes noise-prediction MSE for actions while retaining the existing
auxiliary losses. MLP action loss and diffusion noise loss have different
meanings and numerical scales; compare rollout performance, not their raw loss
values. Euler angles remain the original action representation, including their
wrapping limitation. Sampling requires 100 denoiser evaluations per policy cycle,
so inference latency must be measured before assessing a decoder improvement.

## Verification

Focused checks cover normalized action recovery with an oracle denoiser,
terminal noise level, finite gradients, predicted geometry rather than contact
label conditioning, disabled control geometry, inference shapes, matched configs,
and invalid normalization scales. Existing MLP conditioning and near-contact
checks also pass.

The real-data GPU preflight is
`slurm/smoke_unified_diffusion.sbatch`. It checks one optimization step for each
variant with actual frozen backbones, AMP, finite gradients, and deployment-input
sampling. Job6701391 completed successfully on H100 on 2026-10-08. Both variants passed
real-data loss, finite-gradient, optimizer-step, and bounded sampling checks.
Verification is indexed under
`results/diagnostics/preflight_checks/unified_diffusion_smoke_6701391/verification.json`. This preflight does not
measure task success or establish a useful policy after one optimization step.

## Launch training

The GPU preflight passed. From the repository root:

```bash
sbatch --export=ALL,VARIANT=control \
  part2action/slurm/train_unified_diffusion.sbatch
sbatch --export=ALL,VARIANT=contact \
  part2action/slurm/train_unified_diffusion.sbatch
```

The dedicated diffusion launcher uses the existing durable training environment.
It accepts A40,
A100, or H100 GPU allocations and registers unique run folders under root
`results/part2action/runs/`, with physical outputs in `/p/part2action/results/`.
Full diffusion training was submitted on October 12, 2026: control **6712229**
and contact **6712230**, using `slurm/train_unified_diffusion.sbatch` and
submission-time configuration snapshots. Do not repeat the launch commands
for this pair. Submission records are in
`results/part2action/runs/diffusion_submissions_20261012.json`.
Both completed all30 epochs successfully on H100.
Both use fresh seed 42 initialization, 30 epochs, stride 2, and same-index targets
(offset 0), matching the original 30-epoch two-frame MLP control protocol.
The prior MLP contact references used additive conditioning or continuation
training, so those comparisons also differ in conditioning or training history.
The extra MLP from-scratch timing pair was skipped; the new diffusion timing pair below trains from scratch.
Evaluate both on all 149 test1 scenarios for evaluation seeds42, 43, 44
(447 rollouts per model). Report per-seed success, its mean and sample standard
deviation, approach/grasp behavior, and inference latency. These evaluation
seeds do not measure variability across independently trained checkpoints. Save one expert-left combined GIF per trial.
Evaluation jobs6717932/6717933 are submitted, alongside original30-epoch
two-frame MLP control6717931. Combined export6717934 follows all three, with
expert reference left. Outputs: `results/part2action/evaluations/diffusion_30epoch_20261013/`.
Policy-forward latency and allocated GPU names are recorded; compare timing
with the hardware differences in mind. For later matched rollouts,
pass the new training run as `RUN_DIR` to the existing full-evaluation launcher;
recording is enabled by default.

## Diffusion-control timing from scratch — October 15, 2026

The user replaced the continuation design with matched30-epoch scratch runs.
Configs `hierarchical_world_geometry_30epoch_diffusion_timing_same_index.yaml`
and `hierarchical_world_geometry_30epoch_diffusion_timing_next_action.yaml`
share the same untrained policy tensors, frozen pretrained encoders, fresh
optimizer,30 epochs, LR1e-4, batch16, stride1, two frames, and contact disabled.
Only action-target offset0/1 differs. The initialization is newly generated,
marked epoch-1/zero training steps, and fingerprint-verified before optimization;
it is not the old30-epoch diffusion-control checkpoint.

Training6719765/6719766, automatic evaluations6719767/6719768, and export6719769
are submitted. Evaluations each use149 test1 scenarios ×seeds42,43,44.
Review panels: expert → same-index scratch → next-action scratch.
Artifacts: `results/part2action/evaluations/diffusion_timing_scratch_20261015/`.
Launcher: `slurm/train_diffusion_timing.sbatch`,96-hour maximum and flexible
A40/A100/H100 GPUs. Initialization/config/source hashes and submission records:
`results/part2action/runs/diffusion_timing_scratch_submissions_20261015.json`.
Do not duplicate these submissions. The superseded ten-epoch continuation jobs,
configs, result folders, links, and logs were cancelled and removed; the older
completed30-epoch diffusion models and their evaluations remain intact.

Contact impact has an offline outcome-group review in the completed decoder
comparison folder. It compares separately trained policies and does not disable
contact within one frozen diffusion model. A frozen-model intervention must
control diffusion sampling noise and preserve RNG state before making causal
claims about contact-input effects.

## Early contact fusion — October 16, 2026

Implemented as an opt-in `model.contact_conditioning_stage: early_temporal`.
The default remains `late`, preserving existing checkpoint architectures.

The early path uses the latest frame's DINO features, scene point cloud, and
selected skill to predict contact before action temporal encoding. A projected
token contains predicted contact relative to TCP, point-attention peak, TCP
position, and measured finger opening. Each patch's two-frame temporal sequence
attends to this token; the latest visual frame is then read out and passed through
part gating, point-cloud fusion, language fusion, and the diffusion decoder.
The attention peak is a concentration cue, not a calibrated probability that the
contact is correct. Only the latest frame has a contact prediction; historical
contact points are not fabricated or copied from ground truth.

There is no late geometry addition, learned global scalar, or distance switch
in this mode. Geometry and attention concentration are detached on the action
path; the contact head still learns from its auxiliary contact loss. Training
and deployment both use predicted geometry. Ground-truth contact is supervision
only. The existing measured-opening input to the action features remains.

Separate configs in `configs/architecture_update/`:

| Config suffix | Contact input to temporal token |
| --- | --- |
| `30epoch_diffusion_early_control.yaml` | Relative contact and attention peak zeroed; robot state retained |
| `30epoch_diffusion_early_contact.yaml` | Predicted relative contact and attention peak; same robot state |

Both filenames begin `hierarchical_world_geometry_`. They have identical model
parameters and training settings apart from the contact enable flag and output
names. Both train from scratch for 30 epochs, with two frames, stride 2,
same-index targets (offset 0), the same four-object training set, and no validation
split. This matches the completed original diffusion experiment's data settings;
it does not adopt an alignment result from the separate pending timing study.

The early path also uses latest-frame features for skill/phase selection and
contact grounding, before its temporal action processing. Therefore an early vs
old late-fusion comparison includes that representation change. The matched
early control/contact pair isolates the added predicted-contact information
within the new architecture. It should be trained and evaluated before claiming
that earlier fusion improves performance.

Launcher (contact was submitted October 16; do not duplicate):

```bash
sbatch --export=ALL,VARIANT=contact part2action/slurm/train_early_contact_diffusion.sbatch
```

The launcher supports A40/A100/H100 GPUs and a 96-hour limit. Use the existing
149-scenario, three-seed evaluation and combined-GIF export after training.
Existing checkpoints cannot be used as initial weights for this configuration;
the new temporal input projection must be learned from scratch.

Training job: contact `6721407` completed 30 epochs successfully. Early control `6721406` was
cancelled before starting at the user's request; use existing models for comparison.
The unused control config remains available but should not be submitted without
a new request. Evaluation `6724354` and dependent GIF export `6724355` were
submitted October 18. The review reuses completed control/late-contact recordings:
`results/part2action/evaluations/diffusion_early_contact_20261018/`.
Submission/config snapshots and source hashes:
`results/part2action/runs/diffusion_early_submission_20261016/submissions.json`.

## Transformer denoiser comparison — October 18, 2026

The first Transformer experiment changes the diffusion denoiser while keeping
the existing pooled feature input. It does not introduce separate visual,
language, robot-state, or contact memory tokens. Those are a later information-flow
experiment. The observation-history Transformer and skill hierarchy are retained.

Config:
`configs/architecture_update/hierarchical_world_geometry_30epoch_diffusion_transformer_control.yaml`.
New model options are `diffusion_denoiser_type: transformer`, two Transformer
layers, four attention heads, width 256, and dropout 0. No causal mask is used:
all eight noisy action tokens are available together during denoising. Learned
action-step positions distinguish the steps; the same pooled policy context and
diffusion-time embeddings condition every action token. The existing DDPM
normalization, noise loss, 100-step cosine schedule, clipping, and sampling
equations are shared with the original denoiser.

Training is fresh for 30 epochs, seed 42, two observation frames, stride 2,
same-index targets, identical four-object data and losses, with contact conditioning
disabled. This matches the completed original diffusion control and is independent
of the stride-1 timing study. The RNG stream for shared-module initialization is
preserved across denoiser types, including modules constructed after the action
head. The Transformer itself has fresh weights. The decoder parameter count
differs; this is not an equal-capacity proof that attention alone helps.

The visual part-mask gate remains enabled and supervised (`use_part_gate: true`,
gate loss 0.5). It influences the visual input to action features. The separate
3D point-part head is disabled (`use_point_part_head: false`); the heatmap head
is instantiated but its loss weight is zero. No new part prediction module is
added by this decoder change.

Launcher: `slurm/train_diffusion_transformer.sbatch`. A real-data GPU preflight
checks an AMP optimization step, finite gradients, and action sampling before
the full training process starts. Its temporary optimized model is discarded;
the training process initializes from scratch. Existing model defaults remain
unchanged so older diffusion checkpoints still use their original denoiser.

Submitted pipeline: training `6724615`, evaluation `6724616`, GIF export
`6724617`. The CPU checks passed (15 checks across Transformer, early-contact,
original diffusion, normalization, and fresh-initialization protections); the
Transformer checks also verified that shared initial tensors match across
architectures. Denoiser parameters: original 489,016; Transformer 1,323,783.
GPU preflight remains part of the queued job, not a completed verification.
Snapshots and commands:
`results/part2action/archive/superseded/diffusion_transformer_30epoch_20261018/submission/submissions.json`.
Review: `results/part2action/archive/superseded/diffusion_transformer_30epoch_20261018/review/`.

## Current 10-epoch next-index screening — October 18, 2026

This supersedes the previously queued 30-epoch Transformer pipeline, which was
cancelled before starting (`6724615`/`6724616`/`6724617`). The Transformer denoiser
implementation above remains the same; the new comparison changes the common
training protocol for both candidates together.

Configs `hierarchical_world_geometry_10epoch_diffusion_next_action_baseline.yaml`
and `hierarchical_world_geometry_10epoch_diffusion_next_action_transformer.yaml`
are under `configs/architecture_update/`. Both use 10 epochs from scratch,
two frames, next-index targets (offset 1), stride 1, no contact conditioning,
and identical dataset/seed/optimizer/losses. Only denoiser type and output names
differ. The baseline is the previous diffusion head, not the old direct MLP
policy. No checkpoint continuation is used.

Training `6724877`/`6724878`, evaluations `6724879`/`6724880`, GIF export `6724881`.
Launcher `slurm/train_diffusion_screening.sbatch` accepts VARIANT=baseline or
transformer; these jobs are already submitted, so do not duplicate them.
Review: `results/part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/`.
Records: `results/part2action/runs/diffusion_screening_10epoch_submission_20261018/submissions.json`.
