# Part2Action setup and run order

Read [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md) first for current model
behavior, run status, storage, and university Slurm rules. This file lists the
maintained commands. The old workstation/synthetic heatmap recipe is retired.

## Existing installation

Source is `/u/xna8aw/workspace/research_projects/interactive-robotics`.
Dataset, caches, environments, checkpoints, and Slurm logs live under
`/p/part2action`; browse them through `part2action/storage` and root `results/`.
The full released dataset is already downloaded and verified. Do not reinstall
environments or download data merely to run a current experiment.

```bash
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX=/u/xna8aw/workspace/.micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action_train
```

## Validate configuration

```bash
PYTHONPATH=part2action python part2action/scripts/validate_configs.py \
  part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_unified_contact.yaml
```

## Submit a new matched experiment when requested

The existing control6687620 and contact6687621 runs already completed 30 epochs.
Submitting these commands creates new training runs; it does not resume them.
Both use the original four training categories, with train/test only.

```bash
sbatch --export=ALL,VARIANT=control part2action/slurm/train_world_geometry_unified_h100.sbatch
sbatch --export=ALL,VARIANT=contact part2action/slurm/train_world_geometry_unified_h100.sbatch
```

Launchers accept A40/A100/H100 GPUs and register outputs in root `results/`.
Do not train directly on login/portal nodes. Inspect the resolved config, logs,
and saved checkpoint after the job exits. Training loss is not rollout success.

## Evaluate

Activate `partgym_rollout` inside an allocated GPU job. The maintained launcher
is `part2action/slurm/rollout_bigtemp_full_eval_a40.sbatch`; its name is historical.
Supply absolute RUN_DIR and OUT_DIR under `/p/part2action/results/part2action/`.
The 39-trial protocol covers bottle, mug, pliers, and scissors. Sampling now uses deterministic PyTorch3D CPU FPS; policy inference remains
on GPU. H100 smoke6701227 passed for all four checkpoints. Full evaluations
6701228–6701231 are submitted; inspect their status before resubmitting.
OUT_DIR defaults to a unique source-run/job-ID directory in personal storage.
Full evaluation scores remain pending.

Official DP/DP3 wrappers live under `PartInstruct/scripts/slurm_scripts/` and
submit from the inner repository using the `partinstruct` environment. DP remains
stopped at the user's request. Keep official privileged-input evaluations distinct
from Part2Action evaluations.

## Review diagnostics

Contact audit and offline comparison gallery:
`results/diagnostics/contact_label_audit/review/index.html`.
Gate comparison GIFs and DP3 success/failure GIFs are retained under root results.
Download the self-contained HTML or GIFs for viewing on your local computer.
