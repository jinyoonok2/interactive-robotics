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
nodes. Optional environment activation:

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
