# Interactive Robotics

This repository contains **Part2Action**, a policy for robot manipulation
conditioned on instructions about object parts, such as “grasp the bottle at
its neck.” It learns from PartInstruct demonstrations and runs in the PartGym
simulator with a Franka Panda robot.

**Project scope:** this checkout is Part2Action (P2A). The RB-Y1 pose-head
20,000-update run belongs to the separate Part-Aware Pose project; Live Robotics
Lab has its own MolmoBot policy fine-tuning runs. See the
[cross-project experiment map](part2action/docs/EXPERIMENT_OWNERSHIP.md) before
interpreting a job or result as Part2Action work.

Current work is **pose-token diffusion v3**: an explicit skill specification,
learned part grounding, a predicted gripper goal and a diffusion action decoder.
V3 is trained and diagnosed on a small development set; no improvement over a
baseline has been established.

## Start here

| Document | Purpose |
| --- | --- |
| [Part2Action README](part2action/README.md) | Model inputs, architecture, variants, and source layout |
| [Current configs](part2action/configs/README.md) | Three current recipes, model-code roles and historical config archive |
| [Research plan](part2action/docs/RESEARCH_PLAN.md) | Current question, evidence, active experiments and next decisions |
| [Documentation index](part2action/docs/README.md) | Which documents are current, proposals, records or archives |
| [Dataset](part2action/docs/DATASET.md) | Annotations, split counts and evaluation scope |
| [Setup and run order](part2action/docs/SETUP.md) | Commands for the existing UVA CS server installation |
| [Results guide](results/README.md) | Where to find outputs and how to organize new experiments |

Read the research plan and setup guide before changing Part2Action or PartInstruct.
The three maintained documents are research plan, setup and dataset; older
analysis and run journals are archived. This README provides repository orientation.

## Current status

No Part2Action jobs were queued as of November 24, 2026. The
[research plan](part2action/docs/RESEARCH_PLAN.md) holds results by model
generation and the next decision. The
[v3 architecture](part2action/docs/POSE_TOKEN_DIFFUSION_V3_ARCHITECTURE.md)
holds the design and run history.

The full released PartInstruct dataset is installed: **11 object categories,
69.09 GiB** of raw demonstrations, assets, and metadata. Current Part2Action training uses only bottle, mug, pliers, and scissors. Expanding training to the full release
is a separate planned experiment. Training and test episodes are kept separate;
there is no held-out validation split in the current recipes.

## Repository layout

```text
interactive-robotics/
  README.md                   Repository overview and navigation
  part2action/                Active policy implementation and experiment configs
    docs/                     Research plan, setup, dataset, slides and archives
    storage -> /p/part2action  Personal project storage
  PartInstruct/               Upstream framework, PartGym, and official DP/DP3 baselines
  datasets/PartInstruct       Link to the full released dataset
  results/                    Unified experiment and visualization index
  ManiSkill3/                 Retained source for possible future experiments
  affordance-pipeline/        Historical affordance experiments
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
