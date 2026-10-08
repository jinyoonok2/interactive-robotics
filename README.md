# Interactive Robotics

This repository contains **Part2Action**, a policy for robot manipulation
conditioned on instructions about object parts, such as “grasp the bottle at
its neck.” It learns from PartInstruct demonstrations and runs in the PartGym
simulator with a Franka Panda robot.

The current research question is whether predicted contact geometry improves
action prediction when it conditions the policy throughout the motion.
The controlled experiment compares a unified MLP policy with and without that
geometry input. A diffusion or flow-matching action decoder is a later experiment.

## Start here

| Document | Purpose |
| --- | --- |
| [Part2Action README](part2action/README.md) | Model inputs, architecture, variants, and source layout |
| [Project environment](part2action/docs/PROJECT_ENVIRONMENT.md) | Maintained run status, storage paths, environments, Slurm, and agent instructions |
| [Setup and run order](part2action/docs/SETUP.md) | Commands for the existing UVA CS server installation |
| [Research plan](part2action/docs/RESEARCH_PLAN.md) | Experiments, bottlenecks, and next steps |
| [Results guide](results/README.md) | Where to find outputs and how to organize new experiments |

Read the project environment guide before changing Part2Action or PartInstruct.
This root README provides repository orientation; the Part2Action README describes
the model. Run history and operational details belong in the maintained guide.

## Current experiment

Four Part2Action models completed 30 epochs: baseline, near-contact, unified
control, and unified contact. The unified pair keeps the same temporal encoder,
action decoder, and auxiliary supervision, and varies explicit predicted-contact
conditioning. All four passed an H100 simulator smoke test on 2026-10-08, and
matched 39-trial evaluations were launched afterward. Consult the environment
guide and Slurm for current progress; smoke tests do not establish task success.

The full released PartInstruct dataset is installed: **11 object categories,
69.09 GiB** of raw demonstrations, assets, and metadata. These four trained models
use only bottle, mug, pliers, and scissors. Expanding training to the full release
is a separate planned experiment. Training and test episodes are kept separate;
there is no held-out validation split in the current recipes.

## Repository layout

```text
interactive-robotics/
  README.md                   Repository overview and navigation
  part2action/                Active policy implementation and experiment configs
    docs/                     Environment guide, setup, research plan, slides
    storage -> /p/part2action  Personal project storage
  PartInstruct/               Upstream framework, PartGym, and official DP/DP3 baselines
  datasets/PartInstruct       Link to the full released dataset
  results/                    Unified experiment and visualization index
  ManiSkill3/                 Retained source for possible future experiments
  affordance-pipeline/        Historical affordance experiments
  SUBPROJECT.txt              Separate course-project notes
```

## Storage and execution

Heavy project files live under **`/p/part2action`**, accessible through
`part2action/storage`:

| Folder | Contents |
| --- | --- |
| `data/PartInstruct/` | Full raw dataset, simulation assets, and split metadata |
| `caches/partinstruct/` | Generated DP/DP3 training caches |
| `envs/` | Existing training and simulator environments |
| `results/` | Retained checkpoints, training histories, rollouts, and diagnostics |
| `logs/` | Slurm stdout/stderr |

Source, configs, slides, small reports, and local galleries remain in this
repository. Browse outputs through root [results/](results/README.md), including
[DP3 report GIFs](results/partinstruct/visualizations/).

This project runs on the **University of Virginia (UVA) CS servers**, using
**Slurm** to schedule GPU training and evaluation jobs. Reuse the installed
environments. New heavy outputs
belong in personal project storage and must be registered in root `results/`.
Follow the results guide; do not create outputs under `part2action/results/`.
