# Part2Action setup and run commands

Updated October 15, 2026. This is the command guide for the existing UVA CS
server installation. Read [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md)
for paths, architecture settings, and current job records;
read [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for findings and next experiments.

## Existing installation

Repository: `/u/xna8aw/workspace/research_projects/interactive-robotics`.
Heavy data, environments, model caches, checkpoints, results, and Slurm logs
live under `/p/part2action`. `part2action/storage` links to that allocation;
repository-root `results/` is the entry point for experiment outputs.
The full released PartInstruct dataset is already installed and verified.
Reuse it and the existing environments.

```bash
cd /u/xna8aw/workspace/research_projects/interactive-robotics
```

Use these interpreters for commands that require their respective environments:

| Purpose | Python executable |
| --- | --- |
| Part2Action training and configuration checks | `/p/part2action/envs/part2action_train/bin/python` |
| PartGym rollouts and GIF export | `/p/part2action/envs/partgym_rollout/bin/python` |
| Official PartInstruct DP/DP3 | `/p/part2action/envs/partinstruct/bin/python` |

GPU training and rollouts belong in UVA CS Slurm allocations. Lightweight
configuration checks, metadata counts, and presentation exports can run on CPU.

## Dataset access and download helper

Dataset contents, category counts, train/test formats, and the meanings of
all five benchmark splits are in [DATASET.md](DATASET.md). Server locations are
in [the environment guide](PROJECT_ENVIRONMENT.md#dataset-storage). This section
retains the access and download commands formerly in `DATA_SUBSETS.md`.

For a new installation, accept the terms at the
[PartInstruct dataset](https://huggingface.co/datasets/SCAI-JHU/PartInstruct)
and obtain read access. The helper reads `HF_TOKEN` or the existing
`~/.cache/huggingface/token` file; keep credentials out of source and logs.

```bash
cd /u/xna8aw/workspace/research_projects/interactive-robotics/part2action
DATA_DIR=/p/part2action/data/PartInstruct \
OBJECTS="bottle mug pliers scissors" bash download_subset.sh
```

The helper downloads selected `demos/<object>.hdf5` files and shared object,
part, train, and test metadata. It uses IPv4, skips existing nonempty files,
and accepts `FORCE=1` for deliberate replacement. It does not download simulator
assets or prove that an existing file is complete. The existing full installation
has separate download and migration verification under root `results/data/`.

Choose category names from [the dataset category table](DATASET.md#full-release-counts-by-category).
Downloading files does not automatically expand the object list in a training
configuration.

`python scripts/make_object_configs.py --objects mug bottle scissors --tag TAG`
can generate older subset recipes from inside `part2action/`. Those generated
recipes are separate from the maintained architecture configurations below;
use the latter for current experiments. Reusing a tag overwrites its recipes.

## Validate configuration

```bash
PYTHONPATH=part2action /p/part2action/envs/part2action_train/bin/python \
  part2action/scripts/validate_configs.py \
  part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_unified_diffusion_contact.yaml \
  --check-files
```

## Train

Both diffusion runs completed: control 6712229 and contact 6712230.
They trained from scratch for 30 epochs with two observation frames, stride 2, eight-action
chunks, same-index targets, and training seed 42. Contact uses normalized-global
conditioning; control disables its contact-dependent contribution. Do not
repeat these launch commands unless another pair is requested:

```bash
sbatch --export=ALL,VARIANT=control part2action/slurm/train_unified_diffusion.sbatch
sbatch --export=ALL,VARIANT=contact part2action/slurm/train_unified_diffusion.sbatch
```

The launcher accepts A40/A100/H100 GPUs and creates unique durable run folders.
Submission-time config snapshots and source hashes are recorded in
`results/part2action/runs/diffusion_submissions_20261012.json`.
Technical details: [DIFFUSION_ABLATION.md](DIFFUSION_ABLATION.md).

The original two-frame MLP recipes use `train_world_geometry_unified_h100.sbatch`
with `VARIANT=control` or `contact`. They use additive conditioning and are
historical recipes, rather than the latest normalized diffusion pair.

The two timing MLP continuations6705831/6705832 completed ten additional epochs
from the same 30-epoch control checkpoint, with stride 1 and contact disabled.
Same-index uses offset 0; next-action uses offset 1. Their launcher is
`train_action_timing.sbatch`. The additional MLP from-scratch timing experiment
was skipped; diffusion now has its own from-scratch timing comparison below. Changing a queued or running config does
not change an already-loaded model; use explicit snapshots and separate runs.

### Diffusion timing from scratch

The requested fair timing comparison uses `slurm/train_diffusion_timing.sbatch`
with `VARIANT=same_index` or `next_action`. Both train for30 epochs, LR1e-4,
stride1, two frames, and contact conditioning disabled. Only action-target
offset0/1 differs. They use the same newly randomized policy weights, not the
completed30-epoch model. Frozen pretrained DINOv2/T5 remain unchanged.

`create_diffusion_timing_initialization.py` creates the shared epoch-1, zero-step
policy state on CPU. Both configs require this untrained artifact; the trainer
rejects trained/modified initialization and verifies identical loaded policy
weights before optimization. Each run saves an initialization verification.

Training6719765/6719766 is already submitted; evaluations6719767/6719768 and
GIF export6719769 follow automatically. Do not repeat submissions.
Config snapshots and job records:
`results/part2action/runs/diffusion_timing_scratch_submissions_20261015.json`.
Outputs: `results/part2action/evaluations/diffusion_timing_scratch_20261015/`.
The launcher requests up to96 hours with flexible A40/A100/H100 eligibility.
The old diffusion continuations and their output directories were cancelled
and removed. `train_action_timing.sbatch` remains the older MLP continuation
launcher; do not use it for the new from-scratch diffusion pair.

## Evaluate

The standard evaluation covers **all 149 test1 scenarios** for bottle, mug,
pliers, and scissors across all 32 object–task groups. Each scenario is selected
explicitly once per fixed evaluation seed 42, 43, 44: **447 rollouts per model**.
There is no random scenario sampling. This is the complete four-object **test1**
split, not all four-object test scenarios; [DATASET.md](DATASET.md#how-the-authors-define-the-splits)
explains test2–test5. Report each seed's success rate and its
mean and sample standard deviation. This measures evaluation randomness;
current checkpoints still come from one training seed each.

Runtime settings: max 120 executed actions, one action per observation,
legacy learned termination, gripper commands bounded to0–0.04m, deterministic
CPU point-cloud FPS, and GPU policy inference. Training data remains separate
from test scenarios. Historical sampled39-rollout scores cover fewer task types
and must be reported separately from the full-set scores.

Same-index6714294 and next-action6714295, plus GIF export6714296, completed.
Their full-set scores are9/447 and24/447 respectively. New comparison jobs
6717931/6717932/6717933 evaluate MLP control, diffusion control, and diffusion
contact; GIF export6717934 depends on all three. Results destination:
`results/part2action/archive/mlp/evaluations/action_timing_full_test1_20261012/`.

For a new matched comparison, use a fresh output directory:

```bash
PYTHONPATH=part2action:part2action/scripts \
/p/part2action/envs/partgym_rollout/bin/python \
  part2action/scripts/submit_evaluation_comparison.py \
  --run same_index=/p/part2action/results/part2action/archive/mlp/runs/world_geometry_10epoch_timing_same_index_6705831 \
  --run next_action=/p/part2action/results/part2action/archive/mlp/runs/world_geometry_10epoch_timing_next_action_6705832 \
  --out-dir /p/part2action/results/part2action/evaluations/NEW_EXPERIMENT_DATE \
  --dry-run
```

Remove `--dry-run` only when submitting another requested comparison. Add more
`--run LABEL=PATH` arguments for additional policies. The helper validates all 149
scenario definitions, records the scenario list and code/metadata fingerprints,
submits flexible GPU jobs, and schedules a dependent CPU GIF export.

The launcher `rollout_bigtemp_full_eval_a40.sbatch` uses `EXPECTED_EPISODES=149`
and `EVAL_SEEDS="42 43 44"` by default. Its filename is historical: physical
outputs go to `/p/part2action`. It requests24 hours; the CPU exporter requests
three hours. For an explicit smoke test, set `NUM_EPISODES` and matching
`EXPECTED_EPISODES`, and optionally use one `EVAL_SEEDS` value.

Single-policy evaluations accept `RUN_DIR` and `OUT_DIR` and automatically export
reference-plus-policy GIFs. `RECORD=0` disables recordings; `FLAT_REVIEW=0`
retains raw results for a separate comparison export. To retry export without
rerunning policies:

```bash
sbatch --export=ALL,COMPARISON_REQUEST=/p/part2action/results/part2action/evaluations/EXPERIMENT/comparison_request.json \
  part2action/slurm/export_evaluation_gifs.sbatch
```

## Review format and status

Keep one flat review folder with one combined GIF per scenario/seed, `index.html`,
metrics, manifest, and portable ZIP. Raw videos and traces belong under `_raw/`.
Panels are always expert reference on the left, then matched policies. When an
exact test-episode recording is unavailable, use an expert demonstration with
the same complete skill chain and parameters, preferring the same object
instance. Label its actual source episode and different initial scene. Reference
videos are review-only; no ground-truth geometry enters the policy. Playback
aligns elapsed sample time, rather than task phase. Missing matching reference
data stops export instead of substituting an unrelated task.

Check current status rather than rerunning launch commands:

```bash
squeue -u xna8aw
sacct -j 6719765,6719766,6719767,6719768,6719769 \
  --format=JobID,JobName,State,Elapsed,ExitCode
```

Official DP/DP3 wrappers are under `PartInstruct/scripts/slurm_scripts/` and use
the `partinstruct` environment. DP remains stopped. The official DP3 scissors
pilot used privileged part-cloud input and a different evaluation protocol.
Keep those results separate from Part2Action comparisons.

## Presentation format

Export slide tables as PNGs under `docs/slide_history/media/YYYY-MM-DD/` and
embed each image where the table belongs, with an explicit “Use this image”
download link. Use `docs/slide_history/scripts/export_slide_tables.py --slides PATH`.
Keep table sources in that dated media folder and presentation code under
`slide_history/scripts/`; do not recreate the removed `slide_history/figures/` folder.

Keep future meeting decks within six to seven slides; seven is the maximum
unless another length is explicitly requested. Combine related investigations
and results instead of shortening each slide into a separate page. Use concise
`Part2Action:` titles and no speaker notes. Use one concise block of individual lines
per slide, without leading bullet dashes, before its images and media links; do
not split the explanation into separate blocks above and below a table.

## Current decoder screening: 10 fresh epochs

Use previous diffusion as baseline, two frames, next-index targets, stride 1,
contact conditioning off. Compare with the matched Transformer diffusion config.
Both use the full current four-object training set and fresh optimizer/model
initialization, with pretrained DINOv2/T5 frozen. This is 10 total epochs, not
10 extra epochs on a trained checkpoint.

Config paths:
`configs/architecture_update/hierarchical_world_geometry_10epoch_diffusion_next_action_baseline.yaml`
and `hierarchical_world_geometry_10epoch_diffusion_next_action_transformer.yaml`.
The launcher is `part2action/slurm/train_diffusion_screening.sbatch`, with
`VARIANT=baseline` or `VARIANT=transformer` and an optional CONFIG snapshot path.
The pair is already submitted as `6724877`/`6724878`; do not submit duplicates.
Automatic evaluations `6724879`/`6724880` and export `6724881` produce
expert → previous diffusion baseline → Transformer diffusion GIFs in
`results/part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/`.
