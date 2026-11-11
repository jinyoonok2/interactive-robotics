# Part2Action setup and run commands

Updated October 22, 2026. This is the operational guide for the existing UVA CS
installation. Decisions, results and active job IDs: [RESEARCH_PLAN](RESEARCH_PLAN.md).
Dataset schema, labels and split counts: [DATASET](DATASET.md).
[Archived documentation](archive/2026-10-22/README.md) is historical, not a launch plan.

## Existing installation and storage

Repository: `/u/xna8aw/workspace/research_projects/interactive-robotics`.
Use existing environments rather than rebuilding them. Heavy outputs belong in
`/p/part2action`, accessible through `part2action/storage`; root `results/` contains
registered browsing links. Do not create new heavy outputs under
`part2action/results/`, bigtemp or liverobotics.

| Purpose | Location |
| --- | --- |
| Training/config validation/report Python | `/p/part2action/envs/part2action_train/bin/python` |
| Part2Action PartGym rollout/GIF Python | `/p/part2action/envs/partgym_rollout/bin/python` |
| Released PartInstruct DP/DP3 Python | `/p/part2action/envs/partinstruct/bin/python` |
| Data, demos, assets and split JSON | `/p/part2action/data/PartInstruct` |
| Released checkpoints | `/p/part2action/data/official_checkpoints/PartInstruct` |
| Generated DP/DP3 caches | `/p/part2action/caches/partinstruct` |
| Runs/evaluations/diagnostics | `/p/part2action/results/` |
| Slurm logs | `/p/part2action/logs/` via repository log links |

`datasets/PartInstruct`, `PartInstruct/data`, and `results/data/PartInstruct`
resolve to the same dataset. The framework source is root `PartInstruct/`, not
removed `part2action/third_party/PartInstruct`. Register a new external result
with `part2action/scripts/register_result.py` using the training Python (3.10+).
The official environment uses Python 3.9 and cannot import that registration helper.
See [results organization](../../results/README.md).

```bash
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONNOUSERSITE=1
```

GPU training and inference run inside Slurm allocations, never on portal/login
nodes.

User preference, November 5: run authorized long data-collection, training and
simulation work as background Slurm jobs. Save source/config snapshots, job IDs,
checkpoints and an automatic report; return after checking submission/startup
instead of keeping the chat waiting for all results. Use dependency chains and
failure-aware reporting so results can be checked later. A submitted job is not
a completed experiment.

Optional environment activation:

```bash
export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX=/u/xna8aw/workspace/.micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action_train
```

Use `partgym_rollout` or `partinstruct` for their respective jobs. Reuse cached
backbones/offline mode where available. Keep tokens and API keys out of logs.
Do not commit/push, cancel unrelated jobs, or resume discontinued training
without the user's instruction. PartInstruct edits remain on `partinstruct-edited`.

## Dataset access and download helper

The full release is already installed; do not download another copy. For a new
installation, accept terms at the [official dataset](https://huggingface.co/datasets/SCAI-JHU/PartInstruct).
The helper reads `HF_TOKEN` or the existing Hugging Face token file.

```bash
cd /u/xna8aw/workspace/research_projects/interactive-robotics/part2action
DATA_DIR=/p/part2action/data/PartInstruct \
OBJECTS="bottle mug pliers scissors" bash download_subset.sh
```

It fetches selected HDF5s and shared metadata, not simulator assets. It skips
nonempty files without proving completeness; `FORCE=1` deliberately replaces
files. Installed-data verification is linked from [DATASET](DATASET.md#storage-and-verification).
Do not alter masks, contact labels or data scope while testing contact input.
DP3 generated caches require suffix `+dataset.cache_suffix=dp3s`, separate from DP.

## Train

### Visible depth geometry without interaction points — November 10

Use `configs/pose_learning/visible_geometry_v2.json` and
`scripts/run_sparse_visible_pose_geometry.py`. The completed point ablation provides
all 95 current RGB-D/calibration records, the frozen actor and full-geometry
point-free baselines. Only a depth-to-region converter, config and comparison
orchestrator are new; the common sample loader, pose head, training routine and
physical executor are reused unchanged.

Object/part membership uses a fixed 5 mm GT surface proximity label. Every
feature point itself comes from observed depth, not hidden GT shape. Region
labels are approximate, not a learned detector or exact simulator segmentation.
The initial version-1 audit exposed a Train handle with two visible pixels.
Version 2 repeats each observed point equally to meet the old head's 16-row
format when 2–15 actual points exist; receipts retain both actual and stored
counts. No new coordinates are invented. Empty/single-point regions still block
training rather than falling back or dropping Test cases. Three new heads and
75 new executions compare against 75 read-only baseline executions and 25
oracle-pose controls. Existing source/config/checkpoint files remain preserved.

```bash
PYTHONPATH=part2action:part2action/scripts /p/part2action/envs/part2action_train/bin/python \
  part2action/tests/test_visible_pose_geometry.py
bash -n part2action/slurm/sparse_visible_pose_geometry.sbatch
PYTHONPATH=part2action:part2action/scripts /p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/run_sparse_visible_pose_geometry.py --mode prepare \
  --out-dir /p/part2action/results/diagnostics/NEW_VISIBLE_STUDY
PYTHONPATH=part2action:part2action/scripts /p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/run_sparse_visible_pose_geometry.py --mode submit \
  --out-dir /p/part2action/results/diagnostics/NEW_VISIBLE_STUDY
```

The Slurm dependency chain converts existing records, trains, simulates, and
writes a failure-aware automatic report. No simulator data recapture is needed.

The strict version-1 visibility audit is retained as
`pose_visible_geometry_20261110_01` (conversion 6859526, report 6859529).
Its unstarted training/evaluation descendants were cancelled after the Train
formatting issue. The version-2 study uses a fresh `pose_visible_geometry_20261110_02`
directory and preserves the old config and source unchanged.
Active version-2 jobs: conversion **6859674**, training **6859675**, evaluation
**6859676**, report **6859677**. See
[the active report](../../results/diagnostics/pose_visible_geometry_20261110_02/README.md).

### Common pose samples and point necessity — November 8

New implementation/config only; existing Transformer, geometry pilots and their
running source files remain unchanged. The config is
`configs/pose_learning/point_ablation_v1.json`; model type is
`part_conditioned_pose_v1`. Preserve a used version and create a new config/run
for changes. Configs do not replace source versioning: each run snapshots the
implementation, manifest, config and frozen actor and checks their hashes.

Checks use the training environment with
`PYTHONPATH=part2action:part2action/scripts`:

```bash
/p/part2action/envs/part2action_train/bin/python part2action/tests/test_common_pose_samples.py
/p/part2action/envs/part2action_train/bin/python part2action/tests/test_pose_point_reporting.py
bash -n part2action/slurm/pose_point_ablation.sbatch
```

The runner prepares a **fresh** output directory, verifies all 95 existing
references, then queues common camera/state capture, six matched pose heads,
150 physical evaluations and an automatic report. Both variants receive GT
part/object geometry; the sole optional feature is the oracle point. The new
centroid-relative decoder is shared and does not structurally privilege a point.
Actual camera captures and labels are kept separate; camera features are not
encoded by these small heads. The old actor and privileged clock/prefix/tail
remain frozen. Reused 25 oracle controls are read-only.

```bash
PYTHONPATH=part2action:part2action/scripts /p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/run_pose_point_ablation.py --mode prepare \
  --out-dir /p/part2action/results/diagnostics/NEW_POINT_STUDY
PYTHONPATH=part2action:part2action/scripts /p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/run_pose_point_ablation.py --mode submit \
  --out-dir /p/part2action/results/diagnostics/NEW_POINT_STUDY
```

An optional single-case Slurm collection preflight may populate observation 0
first. Submission validates and reuses completed common captures and queues only
missing ones; failed/incomplete directories are never overwritten. A second
submission is refused. Run the standard background checker on the result path.
Missing collection/training/executions are distinguished from physical failures.

Submitted run: `pose_point_ablation_20261108_01`, preflight **6845314**, capture
**6845317**, training **6845318**, simulation **6845319**, report **6845320**.
Read [the result](../../results/diagnostics/pose_point_ablation_20261108_01/README.md)
or run:

```bash
python3 part2action/scripts/check_background_experiment.py \
  results/diagnostics/pose_point_ablation_20261108_01
```

### Temporary oracle-pose approach diagnostic — November 5

This is an analytic pose-hold/trajectory-bridge controller test, not training or
evaluation of a learned pose-guided actor. It uses the known-successful mug/right
source, with unchanged closing commands and original full-task continuation.
The submitted first run and its results are registered at
[pose-guided approach pilot](../../results/diagnostics/pose_guided_approach_20261105_01/).

Pure geometry/intervention checks run without a simulator:

```bash
PYTHONPATH=part2action PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /p/part2action/envs/part2action_train/bin/python \
  part2action/tests/test_pose_guided_approach.py
```

For a separately authorized rerun, prepare a **fresh** storage directory with
`scripts/run_pose_guided_approach.py --mode prepare --out-dir /p/part2action/results/diagnostics/NEW_NAME`.
This freezes the source hashes, measured preclosure pose and ten-trial plan.
Submit `slurm/pose_guided_approach.sbatch` with `--array=0-9%2` and
`--export=ALL,MODE=worker,POSE_STUDY=/absolute/fresh/output`; use explicit output/error
paths under that directory. Schedule the same launcher with `MODE=report` and
`--dependency=afterany:ARRAY_JOB_ID` for the final JSON, Markdown, plot and videos.
Each array element runs in a fresh process. Do not modify fingerprinted source
files while the trial/report jobs are active, and do not reuse an existing output.

### Learned pose-guided actor diagnostic — November 5

The next experiment trains a small state-only neural actor and a matched no-pose
control on successful replay states/commands, then performs target-input
interventions. This is distinct from the analytic bridge and from the full
visual Part2Action policy. Current artifacts:
[learned pose actor](../../results/diagnostics/learned_pose_actor_20261105_01/).

Run the three neural output/geometry checks with:

```bash
PYTHONPATH=part2action PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /p/part2action/envs/part2action_train/bin/python \
  part2action/tests/test_pose_guided_actor.py
```

`scripts/run_learned_pose_actor.py --mode prepare --out-dir /p/part2action/results/diagnostics/NEW_NAME`
freezes instance splits, accepted source cases, actual pre-action states and
command targets. Use `slurm/learned_pose_actor.sbatch` with `MODE=train` and
`POSE_STUDY=/absolute/output` for compute-node training. After successful training,
submit `MODE=evaluate` with the array bounds in `plan.json` (first study: 0–24,
concurrency four), followed by `MODE=report` after the evaluation array.
Training uses `part2action_train`; simulator/report jobs use `partgym_rollout`.
All pose interventions reuse a frozen trained checkpoint. Reference poses and
source-event clocks are privileged diagnostic inputs; these runs do not train
localization or infer the closing event. Use a fresh output directory and retain
source/checkpoint hashes. Do not edit fingerprinted sources during active jobs.

### Background contact-point-to-pose pipeline — November 5

Current study:
[point-to-pose pilot](../../results/diagnostics/point_to_pose_20261105_01/README.md).
The learned action decoder is frozen; the new pose head learns a bounded TCP
offset and orientation from an annotated target-part surface point and actual
scene geometry. Annotated geometry/point and source clock are privileged inputs
for this diagnostic; deployed localization and the evaluator are later stages.

The queued chain is collection **6830968** (75 sources, concurrency four),
training **6830969** (three pose-head seeds), execution **6830970** (20 trials,
concurrency four), and report **6830971**. The report writes `README.md`,
`summary.json`, learning curves and execution videos automatically. Data/pose
validation failures block downstream learning and are reported, rather than
counting as failed physical trials.

Check progress later without loading ML libraries:

```bash
/usr/bin/python3 part2action/scripts/check_background_experiment.py \
  results/diagnostics/point_to_pose_20261105_01
squeue -j 6830968,6830969,6830970,6830971
```

The checker reads completed source/rollout records, training progress, final
status and submission IDs. The scheduler command reports live queue state.
After jobs leave the queue, consult the generated summary/report and `sacct`
for exit status. The initial `status.json` is a submission snapshot; the final
report updates it.

For separately authorized repeats, use a fresh output directory with
`scripts/run_point_to_pose.py --mode prepare --out-dir /p/part2action/results/diagnostics/NEW_NAME`,
then `--mode submit` to queue all four stages through `slurm/point_to_pose.sbatch`.
The runner captures source and actor hashes and refuses changed inputs. Do not
edit those source files while jobs are queued/running. The report uses after-any
dependencies and upstream status checks to record failures without manual
intervention. Three geometry/head checks and the blocked-upstream report path
passed before this submission.

### Matched pose-geometry ablation — November 7

[Current background report](../../results/diagnostics/pose_geometry_ablation_20261107_01/README.md).
Compare point/state, added object geometry, added part-local geometry, and added
semantic identity using identical 52D pose heads and a frozen neural actor.
Every variant retains the same privileged point; this is not deployed semantic
localization or an OAF reproduction. Training/development examples are unchanged;
evaluation expands to six demonstrations per original held-out mug instance.

CPU jobs: geometry collection **6837582** (20 new sources), training **6837583**
(four variants × three seeds), execution **6837584** (125 trials including
reference-pose controls and training calibration), automatic report **6837585**.
Reuse the 75 original observations; all 95 sources retain individual input hashes.
Each head receives 3,000 matched updates, with development checkpoint selection;
execution uses fixed seed 42. Report first-skill grasp separately from full-task
success after the original recorded continuation.

```bash
/usr/bin/python3 part2action/scripts/check_background_experiment.py \
  results/diagnostics/pose_geometry_ablation_20261107_01
squeue -j 6837582,6837583,6837584,6837585
```

For an authorized repeat, use a new directory with
`scripts/run_pose_geometry_ablation.py --mode prepare --out-dir /p/part2action/results/diagnostics/NEW_NAME`,
then `--mode submit`. Run those commands in the existing training environment
with `PYTHONPATH=part2action:part2action/scripts:PartInstruct`. The Slurm launcher
uses CPU-only training/simulation interpreters and source hashes; do not edit
its runner, feature utility or imported frozen sources while queued/running.
Final outputs include `README.md`, `summary.json`, matched-training audit,
per-case/per-instance execution tables, learning curves and videos. Missing or
failed upstream stages remain readable in the final report.

### Geometry-head training-seed replication — November 8

[Current Part2Action report](../../results/diagnostics/pose_geometry_seed_replication_20261108_01/README.md).
Reuse the previous 125 completed executions and existing 12 pose-head models;
execute only the 200 missing seed-43/44 trials. All other actor, data, point,
physics, timing and recorded-continuation settings remain fixed. This is an
exploratory replication on the same four held-out mug instances, not a new
visual policy, fresh final benchmark, or any MolmoBot experiment.

CPU evaluation array **6844268**, report **6844269**. The report separates head
seeds and retains per-instance/paired outcomes. Reused records are symlinked
read-only and their hashes checked; no training or recollection is scheduled.

```bash
/usr/bin/python3 part2action/scripts/check_background_experiment.py \
  results/diagnostics/pose_geometry_seed_replication_20261108_01
squeue -j 6844268,6844269
```

For an authorized repeat, use a new output directory with
`scripts/run_pose_geometry_seed_replication.py --mode prepare --out-dir /p/part2action/results/diagnostics/NEW_NAME`,
then `--mode submit` in the existing training environment and project PYTHONPATH.
The launcher is `slurm/pose_geometry_seed_replication.sbatch`. All model and
simulation sources remain fingerprinted; do not edit them while queued/running.

### Existing actor training

Current defaults: fresh 10 epochs, two consecutive frames, next-index targets,
stride 1, training seed 42, batch 16, LR 1e-4. The current contact experiment uses
a Transformer diffusion denoiser and changes only explicit contact conditioning.
Contact auxiliary prediction remains active in the ON and OFF models.

```bash
/p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/validate_configs.py \
  part2action/configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer_contact.yaml \
  --check-files
```

`part2action/slurm/train_diffusion_screening.sbatch` accepts `VARIANT=transformer`
(or the previous diffusion `baseline`) and an absolute `CONFIG` snapshot. It
performs real-data GPU AMP/sampling preflight in a separate process, then starts
fresh training with no trained-policy/optimizer continuation. Active jobs already
exist: consult the research plan before submitting anything again.

For a newly authorized experiment only, after creating a reviewed config snapshot:

```bash
sbatch --export=ALL,VARIANT=transformer,CONFIG=/absolute/path/to/NEW_CONFIG.yaml \
  part2action/slurm/train_diffusion_screening.sbatch
```

Snapshot configs and record source hashes, seed, run directory and dependencies
in a submission ledger. Editing a source YAML does not modify an already loaded
job. Historical MLP, near-contact, continuation and early-fusion launchers are
not the current recipe. Do not silently change other settings with a tested flag.

## Evaluate

Part2Action standard: fixed 149 four-object test1 scenarios x seeds 42/43/44 =447
rollouts per model, max 120 actions, execute one, legacy termination, gripper
clamp [0, .04] m, deterministic CPU point sampling and GPU inference. This is
not all test splits. GT masks/part clouds/contact targets never enter the policy.
Report rollout-seed variability separately from training-seed variability.

Single-policy launcher requires absolute `RUN_DIR` and a fresh `OUT_DIR`:

```bash
sbatch --export=ALL,RUN_DIR=/absolute/run,OUT_DIR=/absolute/NEW_EVAL \
  part2action/slurm/rollout_bigtemp_full_eval_a40.sbatch
```

Defaults include all 149 scenarios and three seeds. For small smoke tests, use
`NUM_EPISODES`, matching `EXPECTED_EPISODES`, and optionally one `EVAL_SEEDS` value.
`RECORD=0` disables video; `FLAT_REVIEW=0` retains raw outputs for a combined
export. Current contact comparisons use `CONTACT_DIAGNOSTICS=1,CONTACT_REPORT=1`.
Do not substitute those distance diagnostics for physical contact/grasp metrics.

For new multi-policy comparisons, use `scripts/submit_evaluation_comparison.py`
with repeated `--run LABEL=/absolute/run`, a fresh `--out-dir`, and `--dry-run`
first. It validates scenarios and submits GPU evaluations with CPU GIF export.
Dependencies should reuse completed matched controls instead of rerunning them.

### Quantitative reports and videos

```bash
/p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/compare_rollout_metrics.py \
  --source contact_off=/absolute/CONTROL_EVAL/_raw/contact_off \
  --source contact_on=/absolute/CONTACT_EVAL/_raw/contact_on \
  --data-root /p/part2action/data/PartInstruct \
  --out-dir /absolute/NEW_REPORT
```

Each source requires its `hierarchical/rollout_results_*.json` and
`contact_motion_report/per_trial.csv`; output must be new. The report validates
matched keys/protocol fields and produces grouped JSON, per-trial/paired CSV,
scenario-cluster bootstrap interval, and dated PNG tables with an HTML index.
The existing renderer is reused; do not introduce a redundant table generator.

Combined GIFs use a `comparison_request.json` with `sources`, `out_dir`,
`data_root`, `speed`, and `fps`. Set `quantitative_report: true` to run the
quantitative report after GIF export in the existing CPU job:

```bash
sbatch --export=ALL,COMPARISON_REQUEST=/absolute/comparison_request.json \
  part2action/slurm/export_evaluation_gifs.sbatch
```

Use one flat gallery with expert reference left, then matched policies; raw data
belongs under `_raw/`. Task-matched expert demos may have different starting
scenes and are illustrative, not exact test ground truth. Playback aligns time,
not task phase. Preserve scenario lists, configs, fingerprints and raw traces.
Do not rerun an exporter blindly into an existing quantitative-report directory.

### Frozen-policy gate and target diagnostics

`slurm/evaluate_policy_diagnostic.sbatch` runs the isolated
`scripts/run_policy_diagnostic.py` adapter. Modes are `gate_oracle`, `gate_none`,
`target_identity`, `target_zero`, and `target_shift_x`. The standard trainer,
model and rollout runner are unchanged. Required environment variables: `MODE`,
`RUN_DIR`, fresh `OUT_DIR`, separate fresh `SMOKE_OUT`, and `MANIFEST` containing
source fingerprints. Existing jobs are listed in the research plan; do not
submit duplicates. Each job checks source hashes, completes a five-step smoke,
then evaluates 447 trials and summarizes contact-motion logs. Dependent export
jobs create quantitative reports and GIFs.

Oracle projection is privileged and supports GT grasp/touch parts at the learned
phase index, with identity fallback for unsupported phases. Target identity/zero/
shift runs share trial/step-specific DDPM noise; reuse that identity control for
comparisons. These are frozen-policy interventions with distribution-shift limits,
not new training or physically verified contact measurements.

### CPU replay-gated interaction-tolerance pilot

The November 2 baseline protocol uses `scripts/collect_full_trajectory_baselines.py`
and `slurm/collect_full_trajectory_baselines.sbatch`. It replays **all** stored
actions with native defaults and full-task scoring. `--no-media` suppresses RGB
rendering while retaining commands, before/after physical states and motor-call
logs; it does not change controller execution. Original-HDF5 tracking errors are
diagnostics, not eligibility gates. Two fresh processes at seed 42 must both
receive the original task's native success event and reproduce each other's
measured state trajectories. Continue through the full stored sequence for
measurement, but do not impose an added final-frame retention condition. Native
touch checks themselves advance physics; count and retain those internal steps.

`--mode prepare --data-root /p/part2action/data/PartInstruct --out-dir FRESH_DIR`
inventories the four training categories and freezes the source/config hashes.
The `pilot` array has five items: first source per category and the successful
`mug/demo_1` control, two executions per item. `finish` runs after the pilot array
and submits the remaining full screening only if the technical repeatability
preflight passes. Unlike the earlier grasp collector, this does not require
every category's first source to succeed. Full screening uses eight sources per
shard, at most four concurrent CPU jobs; each worker launches fresh subprocesses
for both repetitions. Collection failures and unsuccessful baselines remain in
the coverage report and never become perturbation failure labels. This stage
does not submit perturbations or training. The current ledger is
`results/diagnostics/full_trajectory_baselines_20261102/`.

The older grasp-only protocols below remain historical and use different tasks
and criteria; their labels must not be merged into the full-task baseline pool.

Controlled continuation diagnosis uses `scripts/probe_lift_handoff.py` and
`slurm/probe_lift_handoff.sbatch`, with immutable `PLAN` and fresh `OUT_DIR`.
`PROBE_SCRIPT=probe_lift_factorial.py` supplies the planned preserved-gain x
duration combinations. It replays the exact original grasp prefix, then changes
only declared continuation factors. The legacy continuation is checked against
the source outcome. These diagnostic labels remain separate from the legacy
training corpus. Do not overwrite reference data or change success thresholds.

The small classifier experiment uses `slurm/train_interaction_pilot.sbatch`
and `scripts/train_interaction_pilot.py` with `PLAN`/`OUT_DIR`, four CPUs/16 GB,
local cached frozen backbones and no GPU/network downloads. It verifies
reference-valid completed record indices, hashes inputs, caches
pre-intervention features, trains candidate-only/full-context outcome heads,
and saves per-instance held-out predictions, checkpoints, learning curves and
a PNG report. This is **outcome-head training, not action-policy training**.
The completed plan/results are at
`results/diagnostics/interaction_outcome_training_20261101/`.
The expert nominal target/time-to-close input is a prototype primitive context;
connecting it to a deployed policy remains future work.

Task-context diagnosis uses `generate_native_context_audit.py` and
`audit_native_context.sbatch`. It preserves complete task metadata, still
extracting only the first grasp plus the added lift. The runtime audit disables
only the controller wall-clock guard; native physical and iteration limits stay
fixed. Alternative proposals use `generate_alternative_expert.py` and
`generate_alternative_expert.sbatch`, selecting a declared native-ranked grasp
with unchanged controller/validation. Each array item checks its rank against
the immutable plan. No automatic expansion follows these diagnostic jobs.

The November 2 slide deck now covers recovered policy/released-policy results,
GPC, the full-task tolerance proposal, implementation status and planned controls.
Regenerate its numbered PNG tables and Markdown with the training Python:

```bash
python part2action/docs/slide_history/scripts/build_20261102_research_update.py \
  --out-dir part2action/docs/slide_history/media/2026-11-02/robustness_pilot \
  --slides part2action/docs/slide_history/2026-11-02_robustness_training_update.md
```

The default refreshes running-job counts from source records and saves a dated
`evidence.json` snapshot. Add `--use-snapshot` to reproduce that exact snapshot.
`tables.json` retains every table value and numbered image mapping. The old
`scripts/build_robustness_training_report.py` path delegates to this generator;
it cannot restore the superseded grasp-only deck. Earlier slides, figures and
evidence are retained under `archive_grasp_lift_pilot/`. The uploaded GPC paper
stays in place; no PDF, HTML or ZIP presentation is generated.

Fresh fully logged expert generation is separate from legacy HDF5 replay:
`scripts/generate_logged_references.py` accepts an immutable `--plan`, fresh
`--out-dir` and `--case-index`. It generates an initial grasp with native VGN /
BulletPlanner, records actual motor calls and physics cadence, verifies identity
replays, and only then executes perturbations. The adapter currently supports
initially-open first-skill grasps from the four-object training dataset; its
inventory explicitly reports unsupported sources. It is not full-task expert
regeneration. New outputs never replace the released HDF5s.

`slurm/generate_logged_references.sbatch` uses the existing rollout Python on
four CPU cores/16 GB with no GPU or network downloads. Required variables:
`PLAN`, `OUT_DIR`, plus `CASE_INDEX` (or array index) for a single case.
`REQUIRE_ELIGIBLE=1` makes a smoke job fail if generation or replay is invalid.
`MODE=study` runs preselected pilot indices and the conditional expansion
controller; `MODE=shard` processes the recorded remaining indices in shards.
Expansion is authorized only after the plan's all-category reference and outcome
gates pass, with the same per-case gates retained at scale. Source fingerprints
are checked before every case. Each case has generated/replay/variation traces,
generation context, outcomes and a gated observation-record index. Consult
`results/diagnostics/logged_expert_references_20261101_recovery2/submissions.json` and
`expansion_decision.json` for actual job/gate status.

`slurm/interaction_tolerance_pilot.sbatch` runs the isolated
`scripts/run_interaction_tolerance_pilot.py` without GPUs or learned encoders.
Completed smoke/full jobs and their investigation are recorded in the research
plan; consult it before submitting follow-ups.
It requests four CPUs/12 GB, four hours, and disables automatic requeue.

For a newly authorized run, provide fresh `OUT_DIR` and `MANIFEST` pointing to a
submission JSON with `source_sha256`. Optional `PER_OBJECT` (default 5), `REPEATS`
(default 3) and `SMOKE_ONLY=1` (one-object baseline-only mode) control scope. The
job verifies source fingerprints before executing. `FRESH_SCENES=1` reconstructs
the native scene for each trial; `CASES_MANIFEST=/absolute/manifest.json` reuses
a fixed training-only selection, and `BASELINE_ONLY=1` validates references
without perturbations. Keep those choices in the submission ledger.
`CASE_INDICES="8"` (CLI `--case-indices 8`) selects zero-based indices from
`CASES_MANIFEST`. Selection preserves the original per-case construction seed;
filtering or reordering cases must not change the reconstructed scene. The
written manifest records each `construction_seed` and the selected indices.
Only reference-valid cases enter perturbations, with fresh identity rechecks.
Perturbation magnitudes are reusable launcher settings: `TRANSLATION_M` (default
0.01), `ROTATION_DEG` (10) and `CLOSE_OFFSET_STEPS` (2). The corresponding CLI
options are `--translation-m`, `--rotation-deg` and `--close-offset-steps`.
The boundary-search follow-up uses 0.02, 20 and 4. They change one variable per
condition, preserve the identity reference and controller schedule, and are
recorded in the manifest and PNG labels. Use a fresh output directory per sweep.

`CAPTURE_OBSERVATIONS=1` (CLI `--capture-observations`) enables two
pre-intervention RGB-D/point-cloud snapshots per trial. `CONDITION_NAMES` (CLI
`--condition-names`) selects a small list of non-identity conditions; identity
references remain mandatory. `records.json` indexes only reference-valid cases.
Each record links a `.observations.npz` file and result JSON. Capture frames are
chosen before the first changed command across the entire selected candidate
set, and include the previous frame (padded at frame zero). This is an explicit
prototype sensor path: CPU TinyRenderer, native camera optics and deterministic
visible full-scene pixel sampling, not the standard evaluator's foreground FPS.

`slurm/check_interaction_records.sbatch` requires a fresh study `OUT_DIR` with a
source-fingerprinted `submissions.json`. It runs paired capture-off/on identity
replays and two known-outcome candidate conditions. The final validator checks
replay equality, identical pre-intervention inputs and observation contents;
inspect `captured/record_validation.json` before consuming the dataset.

`data/interaction_outcomes.py` loads raw observation inputs separately from
outcome labels. `instance_split` keeps all source trajectories/repeats of each
object instance together and rejects a one-instance held-out split. The
untrained `models/interaction_success.py:InteractionSuccessHead` accepts frozen
observation/skill features, robot state, nominal primitive context and candidate
adjustments, then returns joint-success and four criterion logits. It detaches
context by default; callers must also freeze their encoders and put them in eval
mode. No current policy rollout selects candidates with this head. The nominal
primitive is supplied by expert replay in this prototype, so policy integration
must define its replacement and validate transfer before deployment claims.
Use an `afterok` smoke
dependency for a full pilot. Never bypass the reference-fidelity gate to fill
failed perturbation labels. Native control cadence/gains are reconstruction
assumptions, and grasp/lift outcomes are not full-task benchmark success.

For a finite sequential sweep, reuse `scripts/run_interaction_tolerance_sweep.py`:

```bash
PYTHONPATH=part2action:part2action/scripts /p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/run_interaction_tolerance_sweep.py submit \
  --previous-run /p/part2action/results/diagnostics/interaction_tolerance_mug_boundary_20261031 \
  --prior-run /p/part2action/results/diagnostics/interaction_tolerance_mug_perturbations_20261031 \
  --out-dir /p/part2action/results/diagnostics/NEW_SWEEP_DIRECTORY \
  --scales 3 4 6 8 --dependency PREVIOUS_JOB_ID --dry-run
```

Remove `--dry-run` only for a newly authorized submission and use a fresh output
directory. Scales multiply the original 1 cm / 10 degree / two-step offsets,
with one variable changed per condition. The CPU launcher runs all levels in
one job, three reference repeats and 42 perturbations per level. It preserves
per-case construction seeds and verifies controller/source fingerprints.
Invalid references or process errors halt the sweep; physical failures are
labels and do not stop later levels. The range is finite and timing offsets must
preserve closure onset inside the segment. Combined `outcomes.csv`,
`sampled_boundaries.csv`, `sweep_summary.json`, `index.html` and
`media/YYYY-MM-DD/sweep_outcomes.png` update after completed levels. First sampled
failure is not a certified boundary, and exact repeats are not independent scenes.

Results contain `manifest.json`, `results.json`, `summary.json`, `per_trial.csv`,
per-trial NPZ/JSON/compressed physics traces, `index.html`, and PNG media.
`media/YYYY-MM-DD/perturbation_outcomes.png` reports per-condition stable
grasp/lift repeat counts after perturbations are collected. These repeats are
fixed-scene reproducibility checks, not independent statistical samples. Keep
these outputs in `/p/part2action/results/diagnostics/`; root `results/diagnostics/`
links expose them for review. Existing model/training/rollout code is untouched.

## Released DP-S and DP3-S

Use downloaded author-released weights for the reproduction/reference objective;
our earlier self-trained DP/DP3 checkpoints are historical, not released weights.
The installed release revision and sizes are in
`/p/part2action/data/official_checkpoints/PartInstruct/download_manifest.json`.

Launcher: `part2action/slurm/evaluate_official_partinstruct.sbatch`, with
`MODEL=DP-S` or `DP3-S`, and absolute `OUT`. The October 22 submissions completed;
consult the research plan before repeating them.
Each job strictly loads EMA weights and runs a native scissors preflight before
native test1 sampling: 11 categories x16 trials =176, max250 steps, eight-action
blocks, GT part observations and oracle skill progression. This is not the fixed
447-rollout protocol, not SAM/VLM evaluation, and not every paper setting.

[Adapter documentation](../scripts/official_partinstruct/README.md) explains
legacy class/mask aliases and DP input channels recovered from actual weights.
No learned tensor is resized, omitted or replaced. Native checkpoint model
parameters remain intact. Use absolute paths because Hydra may change directories.

For the separate fixed-scenario reference, use
`part2action/slurm/evaluate_official_fixed_scenarios.sbatch` with `MODEL`, fresh
`OUT`, source-fingerprinted `MANIFEST` and our saved `SCENARIOS_MANIFEST`.
`SCENARIO_INDICES="0 46 96 128"` and `EVAL_SEEDS="42"` run the four-case
preflight; omit both for 447 trials. Full runs depend on their model's successful
preflight. This matches scenario selection, 120-action budget and one-step
execution, but **retains privileged inputs and native sequential-skill scoring**.
Consult the adapter README before comparing scores with our full-task evaluator.

## Slurm and status

All current launchers allow `a40|a100_40gb|a100_80gb|h100_94gb`, one GPU, without
node pinning. Current requests: screening train 8 CPUs/40G/96h; Part2Action
rollout 8 CPUs/32G/24h; official native evaluation 16 CPUs/100G/48h; CPU GIF/report
export 4 CPUs/12G/3h. GPU model/hardware differences must accompany latency claims.

```bash
squeue -u xna8aw -o '%.18i %.38j %.12T %.12M %.35R'
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode -X
scontrol show job JOBID
```

`Priority` and `Resources` are scheduler waits; `Dependency` waits on another job.
After a job leaves the queue, inspect accounting and logs before claiming success.
A timeout may leave a checkpoint. If tool sandboxing blocks scheduler sockets,
use the supported escalation mechanism. Do not cancel/resubmit automatically.

## Presentation format

Keep future meeting decks within six to seven slides (maximum seven unless
requested otherwise), concise `Part2Action:` titles, no speaker notes. Use one
block of short individual lines per slide without leading bullet dashes, before
media; do not split explanation above and below a table.

Slide tables are PNGs under `docs/slide_history/media/YYYY-MM-DD/`, embedded at
their intended position with an explicit "Use this image" download link. Reuse
`docs/slide_history/scripts/export_slide_tables.py --slides PATH`; editable values
remain in the dated `tables.json`. Keep code under `slide_history/scripts/` and
do not recreate the removed `slide_history/figures/` folder. Reports also provide
PNG tables and their underlying CSV/JSON rather than requiring video inspection.


Replay investigations use `scripts/build_interaction_replay_audit.py` for existing
trial failure breakdowns/dated PNGs and `scripts/probe_interaction_replay.py` with
`slurm/probe_interaction_replay.sbatch` for fresh-scene controller hypotheses.
The probe requires `MANIFEST`, `PILOT_DIR`, `CASE_INDEX` and fresh `OUT_DIR`.
It preserves original eligibility limits and is exploratory calibration, not
an independent policy or robustness benchmark. Do not select gains by success
or silently substitute a calibrated profile into an existing comparison.

## Original visual policy with oracle grasp-pose input — November 10

This first stage retains RGB, scene PCD, language, measured state, learned
hierarchy and the Transformer diffusion action head. GT pose changes only the
new feature-conditioning path. A separately fine-tuned control receives no pose
signal. The learned estimator is a later stage. The submitted screen has two
fine-tuning epochs per arm from the same original visual control checkpoint.

The installed run is
`results/diagnostics/visual_oracle_pose_20261110_01`, linked to
`/p/part2action/results/diagnostics/visual_oracle_pose_20261110_01`.
Open its `README.md`, `submissions.json`, `label_audit.json`, `preflight.json`,
`simulation_reset_preflight.json` and eventual paired results. Policy actions
run from reset through the full task. Native experts execute only in separate
reference scenes; failed references remain explicit coverage gaps.

New experiments use a fresh output directory, not the already submitted one:

```bash
export PYTHONPATH=part2action:part2action/scripts
/p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/run_visual_oracle_pose_v1.py --mode prepare \
  --out-dir /p/part2action/results/diagnostics/CHOOSE_NEW_NAME
```

The submitted run preserves its original V1 source snapshot and records the
versioned evaluation correction in `evaluation_revision.json`; use the recorded
jobs and `evaluate_visual_oracle_pose_v2.py` for that run. The V1 submitter is
historical; do not submit it again or edit its frozen sources. Reuse unchanged
models/training/data for a new versioned submission, with a validated evaluation
entry point. Scheduler commands require the usual cluster access.
