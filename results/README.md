# Research results

This repository-root directory is the entry point for Part2Action and
PartInstruct outputs. Browse results here even when their physical files live
under `/p/part2action`. A linked run can be running, partial, or
finished: check its logs and scheduler status before interpreting it.

| Section | Contents |
| --- | --- |
| `part2action/runs/` | Training runs, saved configs, checkpoints, and training logs |
| `part2action/evaluations/` | Rollout metrics, diagnostics, and recorded media |
| `part2action/archive/` | Historical retained runs; not current comparisons |
| `partinstruct/runs/` | Official DP/DP3 training and continuation runs |
| `partinstruct/archive/` | Earlier DP training histories; superseded checkpoints removed |
| `partinstruct/evaluations/` | Official baseline rollout results and media |
| `diagnostics/contact_label_audit/` | Contact-label pilot, overlays, and offline review gallery |
| `diagnostics/contact_switch_audit_6675263/` | Completed CPU switch audit: REPORT.md, summary.json, segments.jsonl |
| `data/PartInstruct/` | Link to durable raw demonstrations, assets, and split metadata |
| `data/full_dataset_download/` | Download verification manifest; complete only when status says complete |
| `data/gt_demo_videos/` | Exported ground-truth demonstration videos |

Open the contact visualization at
[diagnostics/contact_label_audit/review/index.html](diagnostics/contact_label_audit/review/index.html).
It is self-contained and can be downloaded for local browser review.

Existing `part2action/baseline`, `part2action/near_contact`, `partinstruct/dp`,
`partinstruct/dp3`, and `partinstruct/dp3_eval` shortcuts are retained. The DP
shortcut identifies the latest stopped run, 6668506 (epoch2500). Earlier DP
training histories are under `partinstruct/archive/`; their checkpoints were removed.

## Future experiments

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
