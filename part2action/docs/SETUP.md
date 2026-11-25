# Part2Action setup and run commands

Updated November 19, 2026. This is the operational guide for the existing UVA CS
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

### Storage verification and enforcement — November 19

Use **`/p/part2action/results/` for physical run outputs**, not the home-backed
workspace. Browse the same files through `part2action/storage/results/` or the
registered root `results/` links. Moving files into the repository's physical
`part2action/` directory does not move them off the home filesystem.

The v3 prepare/train/evaluation entry points now normalize fresh repository
`results/...` destinations onto the project allocation and reject arbitrary
home output directories. V3 Slurm launchers additionally verify the resolved
study and stdout/stderr paths before running. Put logs under the project run
directory or `/p/part2action/logs/`.

`slurm/project_storage_env.sh` and `diagnostics/storage.py` put Hugging Face,
Torch, DINOv2, Matplotlib, CUDA and compiler caches under
`/p/part2action/caches/`. Cached Flan-T5-base and DINOv2 files were copied and
checksum-verified there; no download is necessary. Global caches used by other
projects are not reconfigured by these project-local settings.

The cleanup migrated remaining local Part2Action run artifacts with verified
checksums and preserved their original browsing paths as links. Actual trained
checkpoints, demonstration data, historical records, and report assets are
retained. Redundant one-update v3 smoke weights were removed, and the source
tree was checked for regenerable Python/test caches. See the
[cleanup receipt](records/storage_cleanup_20261119.json).

The ten-epoch job 6957500 had already failed during epoch 3; checkpoints for
epochs 1 and 2 are retained. The failure log does not establish quota as the
cause. The failed run and cancelled dependent evaluation are not completed
training/benchmark results. Operational storage changes require a new matching
smoke receipt before another training launch; old source snapshots are preserved.

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

### Shared visual grounding and predicted pose — November 11

The original visual GT-pose study below remains queued independently. The new
architecture is implemented in `models/grounded_visual_pose_policy_v1.py`; the
training recipe is `configs/pose_learning/grounded_visual_pose_v1.yaml`.
It fine-tunes the original visual Transformer, with newly initialized pose
modules and frozen DINO/T5. Dedicated non-visual study launchers and results have
been retired; see [the cleanup receipt](records/cleanup_20261111_nonvisual_pose.json).

CPU checks use the existing training environment:

```bash
PYTHONPATH=part2action:part2action/scripts:part2action/tests \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
/p/part2action/envs/part2action_train/bin/python - <<'PYTEST'
import runpy
for name, check in runpy.run_path('part2action/tests/test_grounded_visual_pose_v1.py').items():
    if name.startswith('test_'):
        check()
        print('PASS', name)
PYTEST
```

The Slurm entry accepts `DRY_RUN=1` for one real-data optimization step and a
sampled diffusion action chunk. Set `TASK_OUTPUT` to a fresh physical directory
under `/p/part2action/results/`, register it using `register_result.py`, and use
`slurm/grounded_visual_pose_v1.sbatch`. Full training is deliberately not launched
before the currently authorized GT-stage comparison has been examined.

```bash
sbatch --partition=gpu --constraint="a40|a100_40gb|a100_80gb|h100_94gb" \
  --output=/p/part2action/logs/part2action/slurm/grounded_pose_%j.out \
  --error=/p/part2action/logs/part2action/slurm/grounded_pose_%j.err \
  --export=ALL,DRY_RUN=1,TASK_OUTPUT=/p/part2action/results/diagnostics/NEW_GROUNDED_POSE_PREFLIGHT \
  part2action/slurm/grounded_visual_pose_v1.sbatch
```

The implementation smoke test **6865392 completed successfully** with the actual
pretrained encoders. Its frozen source plan, logs and receipts are under
[grounded_visual_pose_20261111_01](../../results/diagnostics/grounded_visual_pose_20261111_01/README.md).
It completed one optimization step and observation-only diffusion sampling; its
checkpoint is a smoke artifact, not a completed training run. Full new-model
training is not submitted. The independent GT-stage GPU preflight also passed;
both fine-tuning arms 6864652 finished. V2 evaluation failed before any rollout
because of a missing helper import; V3 smoke 6910609 now gates evaluation 6910610.

After a trained checkpoint exists, the observation-only rollout adapter is
`scripts/evaluate_grounded_visual_pose_v1.py --config CONFIG --ckpt CHECKPOINT
--out-dir FRESH_PER_SHARD_DIRECTORY`. Defaults cover all 149 four-object Test1
scenarios at seeds 42/43/44, with the original learned phase/termination loop.
`--num-episodes` explicitly selects a debugging subset; `--shards/--shard-index`
partition the full population. Run visual inference inside a Slurm GPU allocation
using `partgym_rollout`. Runtime errors remain unknown, not policy failures.

### Original visual baseline training

Historical baseline recipe: fresh 10 epochs, two consecutive frames, next-index
targets, stride 1, seed 42, batch 16 and LR 1e-4. The completed original visual
checkpoint initializes the current pose study. These commands describe that
baseline recipe, not a request to launch another training run.

```bash
/p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/validate_configs.py \
  part2action/configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_transformer.yaml \
  --check-files
```

`part2action/slurm/train_diffusion_screening.sbatch` accepts `VARIANT=transformer`
(or the previous diffusion `baseline`) and an absolute `CONFIG` snapshot. It
performs real-data GPU AMP/sampling preflight in a separate process, then starts
fresh training with no trained-policy/optimizer continuation. The pose study
already has its own queue; consult the research plan before submitting anything
again.

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

Use the versioned pose adapters for the two pose architectures: the current
GT-stage recovery jobs use `evaluate_visual_oracle_pose_v3.py`, and predicted-pose
checkpoints use `evaluate_grounded_visual_pose_v1.py` in `partgym_rollout`.
The generic baseline launch commands below load original visual-policy
checkpoints; they do not construct the added pose modules.

As of November 16, the Part2Action standard uses fixed 149 four-object Test1
scenarios x seeds 42/43/44 =447 rollouts per model, **250 executed actions and
eight actions per prediction**, learned termination, gripper
clamp [0, .04] m, deterministic CPU point sampling and GPU inference. This is
not all test splits. GT masks/part clouds/contact targets never enter the policy.
Report rollout-seed variability separately from training-seed variability.
The shared [execution configuration](../configs/evaluation/partinstruct_native.json)
uses the released evaluator's 400-action budget for Test4 and 250 for other splits.
Camera/physics thresholds follow its native environment configuration, including
the 0.15 m release height. The last action block is shortened at the budget cap;
250 commands with eight-command blocks require at most 32 policy predictions.
This matches execution settings, not input privilege or skill/success scoring.
Historical Part2Action 120/1 results retain their original protocol and are not
merged into new 250/8 reports. Historical recovery adapters remain frozen;
use `evaluate_pose_token_diffusion.py` or `evaluate_pose_token_diffusion_v2.py`
for the current pose-token families. Do not reuse a historical pinned plan as
a new-protocol submission. No new full evaluation was submitted by this change.

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

### Historical replay and evaluator tools

Replay/tolerance/evaluator launchers and their dedicated report/training commands
are retired from the active tree. Do not submit their old commands. Checksummed
source is available in the [scope-cleanup archive](archive/2026-11-11_scope_cleanup/README.md).
The native-reference generator and its imported legacy helper remain because
the queued GT-pose evaluation depends on them.

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

## Slurm and status

Training and evaluation run on the UVA CS Slurm cluster. The baseline launchers
allow `a40|a100_40gb|a100_80gb|h100_94gb`, one GPU, without node pinning. V3
launchers contain no resource directives, so pass them at submission. V3 training
and phase evaluation also allowed `a4000|a4500|rtx_4000_ada`; the November 22
grasp probes used `a16`. Each v3 study keeps its exact sbatch commands in
`submissions.json`; the epoch-10 phase evaluation commands are in the restart
training study. Baseline requests: screening train 8 CPUs/40G/96h; Part2Action
rollout 8 CPUs/32G/24h; official native evaluation 16 CPUs/100G/48h; CPU GIF/report
export 4 CPUs/12G/3h. GPU model/hardware differences must accompany latency claims.

```bash
squeue -u xna8aw -o '%.18i %.38j %.12T %.12M %.35R'
sacct -j JOBID --format=JobID,JobName,State,Elapsed,ExitCode -X
scontrol show job JOBID
```

V3 training, after a successful `MODE=smoke` submission for the same study:

```bash
STUDY=/p/part2action/results/diagnostics/MY_V3_STUDY
sbatch --no-requeue --kill-on-invalid-dep=yes --partition=gpu --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=12:00:00 \
  --constraint="a40|a100_40gb|a100_80gb|h100_94gb|a4000|a4500|rtx_4000_ada" \
  --output="$STUDY/%x_%A_%a.out" --error="$STUDY/%x_%A_%a.err" \
  --export=ALL,STUDY="$STUDY",MODE=train \
  part2action/slurm/pose_token_diffusion_v3.sbatch
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

Keep dated slide documents under `docs/slide_history/slides/` and paper PDFs
under `docs/slide_history/papers/YYYY-MM-DD/`.
Slide tables are PNGs under `docs/slide_history/media/YYYY-MM-DD/`, embedded at
their intended position with an explicit "Use this image" download link. Reuse
`docs/slide_history/scripts/export_slide_tables.py --slides PATH`; editable values
remain in the dated `tables.json`. Keep code under `slide_history/scripts/` and
all figure exports in the dated media folders. Reports also provide
PNG tables and their underlying CSV/JSON rather than requiring video inspection.


Historical replay reports/probes are retained in the
[retired source archive](archive/2026-11-11_scope_cleanup/README.md). Their old
commands are not active launch instructions.

## Original visual policy with oracle grasp-pose input — November 10

This first stage retains RGB, scene PCD, language, measured state, learned
hierarchy and the Transformer diffusion action head. GT pose changes only the
new feature-conditioning path. A separately fine-tuned control receives no pose
signal. The learned estimator is a later stage. The submitted screen has two
fine-tuning epochs per arm from the same original visual control checkpoint.

Verified November 13: both fine-tuning arms **6864652** completed with identical
initial shared weights and matched minibatches/update counts. All V2 evaluation
shards **6864664** failed before any rollout because `oracle_path` was not imported.
The trained checkpoints are reused unchanged. Both V3 smoke jobs **6910609**
passed; they gate full evaluation **6910610**, report **6910611**, and statistics
**6910612**. Their one-case outcomes are runtime checks, not benchmark estimates. Reference
coverage remains 333 completed / 339 grasp starts; six unavailable records are
coverage gaps. No benchmark grasp/task rate is available yet. Do not retrain or
submit duplicate evaluations.

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
jobs and `evaluate_visual_oracle_pose_v3.py` for the current recovery. V3 is pinned
by `evaluation_revision_v3.json` and requires passed smoke receipts for both arms. The V1 submitter is
historical; do not submit it again or edit its frozen sources. Reuse unchanged
models/training/data for a new versioned submission, with a validated evaluation
entry point. Scheduler commands require the usual cluster access.
