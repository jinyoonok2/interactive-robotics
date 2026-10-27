# Research results

This repository-root directory is the entry point for Part2Action and
PartInstruct outputs. Browse results here even when their physical files live
under `/p/part2action`. A linked run can be running, partial, or
finished: check its logs and scheduler status before interpreting it.

Current work uses diffusion action heads. Completed MLP experiments are retained
in the [MLP archive](part2action/archive/mlp/README.md).

- [10-epoch next-index diffusion comparison](part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/index.html): fresh previous-diffusion baseline `6724877` versus Transformer `6724878`, both with two frames; evaluations `6724879`/`6724880`, GIF export `6724881`.
- [Completed decoder comparison](part2action/evaluations/diffusion_30epoch_20261013/index.html): 447 combined GIFs, with expert reference, historical MLP control, diffusion control, and diffusion contact. This shared comparison stays active.
- [Contact outcome review](part2action/evaluations/diffusion_30epoch_20261013/contact_impact_review.html): diffusion control/contact comparisons grouped by success outcome.
- Diffusion timing from scratch: `part2action/evaluations/diffusion_timing_scratch_20261015/`; dependent evaluations and GIF export follow the two training runs.
- [Early-contact comparison](part2action/evaluations/diffusion_early_contact_20261018/index.html): completed: control **33/447**, late contact **22/447**, early contact **40/447**; 447 expert-left GIFs ready.
- Early-contact training submission: `part2action/runs/diffusion_early_submission_20261016/submissions.json`.
- [Observation/action timing](diagnostics/observation_action_timing_20261010/index.html): dataset timing evidence remains relevant to diffusion.

The standard evaluation protocol uses all 149 four-object test1 scenarios at
seeds 42, 43, and 44, producing 447 rollouts per model. Pending gallery pages
are replaced after their evaluation jobs finish. Setup checks remain under
`diagnostics/preflight_checks/`. Dataset label audits remain separate from the
archived MLP policy results.

| Section | Contents |
| --- | --- |
| `part2action/runs/` | Training runs, saved configs, checkpoints, and training logs |
| `part2action/evaluations/` | Rollout metrics, diagnostics, and recorded media |
| `part2action/archive/mlp/` | Archived MLP runs, evaluations, diagnostics, and submission records |
| `partinstruct/runs/` | Official DP/DP3 training and continuation runs |
| `partinstruct/archive/` | Earlier DP training histories; superseded checkpoints removed |
| `partinstruct/evaluations/` | Official baseline rollout results and media |
| `diagnostics/` | Local preflights, dataset timing checks and policy diagnostics |
| `data/PartInstruct/` | Link to durable raw demonstrations, assets, and split metadata |
| `data/full_dataset_download/` | Download verification manifest; complete only when status says complete |
| `data/gt_demo_videos/` | Exported ground-truth demonstration videos |

Historical baseline/near-contact shortcuts now live under
`part2action/archive/mlp/shortcuts/`. External checkpoint and recording storage
paths remain intact, including MLP inputs used by shared diffusion comparisons.
The DP shortcut identifies stopped run 6668506 (epoch2500); official DP/DP3
results remain in their existing PartInstruct sections.

## Future experiments

Evaluation reviews default to **one folder of combined GIFs**: one GIF per
matched trial, with two or more labeled policies side by side. Keep `index.html`,
the GIFs, summary/CSV/manifest files, and `comparison_bundle.zip` directly in
the evaluation directory. Supporting source recordings and traces belong in
`_raw/`; do not create separate visible policy galleries or nested GIF folders.
Use another review format only when the task requires it.

Always prepend the **ground-truth expert reference as the leftmost panel**.
An exact test-episode demonstration takes priority. When it is unavailable,
select a dataset expert demonstration with the same complete skill chain and
parameters, preferring the same object instance, and label it as a different
episode. Record its demo/episode IDs and match type in the manifest. Do not
substitute a different task or imply identical starting scenes or task-phase
alignment. A single-policy review therefore has reference + policy panels.

Use `part2action/scripts/submit_evaluation_comparison.py` for matched evaluations.
It submits flexible GPU jobs and a dependent CPU GIF-export job automatically;
see [the setup commands](../part2action/docs/SETUP.md#evaluate). The standalone
rollout launcher also exports flat GIFs by default. `RECORD=0` opts out of media;
`FLAT_REVIEW=0` is for raw jobs whose GIFs are exported by the comparison job.

Use a descriptive name plus Slurm job ID (or date for manual work). Keep every
run's configs, metrics, logs, and media together. Evaluations get a separate
named directory identifying the source training run and evaluation protocol.
Never silently overwrite a previous run.

Large generated outputs live under `/p/part2action/results/`; raw data stays on
`/p/part2action/data/PartInstruct`. Small reports may live directly
in the corresponding results section. The maintained Slurm training,
evaluation, and contact-audit launchers register their output directories at
startup, making running and failed jobs discoverable too. Slurm stdout/stderr
retain their scheduler paths under each project's `logs/slurm/` directory.

For a new launcher, create its physical output directory and register it:

```bash
python3 part2action/scripts/register_result.py \
  --section part2action/runs \
  --target /p/part2action/results/part2action/runs/experiment_JOBID
```

Other sections accepted by the helper are listed in `--help`. It is safe to
repeat registration for the same target and refuses conflicting names. Use
absolute output paths for Slurm/Hydra jobs. CPU visualization scripts default
to `results/diagnostics/contact_label_audit/` when run from the outer repo root.

Project storage is accessible through `part2action/storage -> /p/part2action`.
Environments are under `envs/`, generated caches under `caches/partinstruct/`,
and Slurm logs under `logs/<project>/slurm/` in that allocation. Maintained
launchers use these paths. Saved historical checkpoint configs are preserved
verbatim and may contain obsolete paths. The finished downloader's temporary
`part2action/results/` redirect has been removed.

Cleanup verification: [data/storage_cleanup_20261008/](data/storage_cleanup_20261008/).
Dataset links used by training remain available. Repository source, small reports,
slide figures, and local galleries stay in the repository.
