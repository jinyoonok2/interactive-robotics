# Part2Action

Part2Action predicts robot actions from visual observations, robot state, and
instructions naming an object part. Training uses PartInstruct demonstrations;
closed-loop evaluation uses the PartGym Franka Panda environment.

This README describes the model and implementation. The repository
[README](../README.md) provides the wider project overview. Read
[PROJECT_ENVIRONMENT.md](docs/PROJECT_ENVIRONMENT.md) for current runs, paths,
environments, and Slurm rules, and [SETUP.md](docs/SETUP.md) for run commands.

## Inputs and outputs

At deployment the policy receives RGB, the full scene point cloud, the task
sentence, the TCP pose, and measured gripper opening. Current recipes exclude
joint angles. Dataset part masks, target-part clouds, and per-step skill labels
provide training supervision; they are not deployment inputs.

The model predicts an **8 × 7 action chunk**:

```text
world x, world y, world z, roll, pitch, yaw, gripper
```

Current rollouts execute the first action, observe the scene again, and predict
another chunk. The contact head separately predicts a 3D location on the scene
cloud. Its influence on actions depends on the architecture variant.

## Shared architecture

- Frozen DINOv2 ViT-S/14 with register tokens encodes RGB; frozen Flan-T5-base
  encodes task language.
- A learned hierarchy predicts skill slots, selects the current skill from
  observations/state, and predicts termination.
- A skill-conditioned part gate reweights visual tokens. Ground-truth masks
  supervise that prediction; predicted gating influences the policy features.
- Point-cloud features and contact attention predict a world-frame contact point.
- An MLP action head predicts the action chunk. The near-contact and unified
  recipes also use a two-frame temporal transformer.

The unified variants retain the hierarchy, part gate, and auxiliary contact
head. “Unified” refers to using the action policy throughout the motion without
switching XYZ commands at a contact-distance threshold.

## Architecture variants

| Variant | Observation history | How predicted contact coordinates affect actions |
| --- | --- | --- |
| Baseline | One frame | Auxiliary supervision; coordinates do not directly enter the action head or replace XYZ |
| Near-contact | Two frames | Within 3 cm, replace XYZ with predicted contact plus a bounded 3 cm offset; otherwise use MLP XYZ |
| Unified control | Two frames | Same conditioning projection as unified contact, supplied with zeros; no XYZ switching |
| Unified contact | Two frames | Feed predicted contact relative to the TCP into action features throughout motion; no XYZ switching |

Unified contact projects the detached vector
`(predicted_contact_world - TCP_world) / 0.1` into action features. It uses
predictions during both training and deployment. Ground-truth contact targets
remain auxiliary supervision. Contact confidence is not an implemented input.
The old bounded contact-relative action offset and residual are disabled in the
unified pair.

The near-contact training gate uses demonstration contact labels; rollout gating
uses predictions. The unified control/contact pair is the matched comparison for
explicit geometry conditioning. The original baseline also differs in temporal
history, so comparisons against it involve more than that single change.

Configs are under [configs/architecture_update/](configs/architecture_update/),
with names ending in `no_residual`, `near_contact`, `unified_control`, and
`unified_contact`. Model construction uses config flags; keep matched experiments
consistent outside the module being tested.

## Dataset and evaluation

The full released dataset contains 11 categories at
`/p/part2action/data/PartInstruct`. Current trained variants use only bottle,
mug, pliers, and scissors, with train/test separation and no held-out validation
split. Training losses and offline training-data checks are not test success rates.

All four variants completed 30 epochs and passed five-action H100 smoke tests on
2026-10-08. Their matched evaluation uses 13 supported object/task combinations
and three trials each: **39 test1 rollouts per model**, with a 120-step limit,
legacy learned termination, and one action executed per policy cycle.
Point-cloud sampling uses deterministic PyTorch3D CPU FPS; policy inference uses
GPU. Detailed progress and job IDs belong in the environment guide.

Official PartInstruct DP3-S is a separate privileged-input baseline: its completed
scissors pilot used ground-truth part clouds and succeeded in 1 of 5 trials.
Its protocol differs from the 39-trial Part2Action evaluation. DP-S training was
stopped at the user's request; its latest epoch2500 checkpoint is retained.

## Implementation layout

```text
part2action/
  configs/                    Experiment recipes and module flags
  data/                       HDF5 loading, coordinate transforms, target derivation
  models/                     Encoders, fusion, hierarchy, contact and action heads
  diagnostics/                Rollout diagnostic helpers
  scripts/
    train.py                  Offline training
    evaluate.py               Offline metrics
    rollout_partgym.py        Closed-loop simulator evaluation
    register_result.py        Register external outputs in root results/
  slurm/                      Batch launchers and GPU smoke checks
  tests/                      Geometry, conditioning, gate, and action checks
  docs/                       Environment guide, setup, research plan, slides
  storage -> /p/part2action    Dataset, caches, environments, heavy outputs, logs
```

The framework clone is [../PartInstruct/](../PartInstruct/). Outputs are indexed
under root [../results/](../results/README.md); source/configs and small reports
remain in the repository. Follow that guide for new experiments.

## Results and research

- [DP3 success GIFs](../results/partinstruct/visualizations/dp3_success/) and
  [failure GIFs](../results/partinstruct/visualizations/dp3_failures/).
- [Research plan](docs/RESEARCH_PLAN.md): first measure the unified contact
  conditioning effect, then consider a diffusion or flow-matching action decoder.
- [Meeting slides](docs/slide_history/) preserve dated research discussions.
