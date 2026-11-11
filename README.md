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

Current work tests whether grasp-pose guidance improves the original visual
Transformer diffusion policy. A matched GT-pose experiment comes first, followed
by a learned pose estimator using shared RGB and scene features. Both paths
retain vision and the original action decoder. Replay/tolerance/evaluator
expansion is deferred.

## Start here

| Document | Purpose |
| --- | --- |
| [Part2Action README](part2action/README.md) | Model inputs, architecture, variants, and source layout |
| [Current configs](part2action/configs/README.md) | Three current recipes, model-code roles and historical config archive |
| [Research plan](part2action/docs/RESEARCH_PLAN.md) | Current question, evidence, active experiments and next decisions |
| [Dataset](part2action/docs/DATASET.md) | Annotations, split counts and evaluation scope |
| [Setup and run order](part2action/docs/SETUP.md) | Commands for the existing UVA CS server installation |
| [Results guide](results/README.md) | Where to find outputs and how to organize new experiments |

Read the research plan and setup guide before changing Part2Action or PartInstruct.
The three maintained documents are research plan, setup and dataset; older
analysis and run journals are archived. This README provides repository orientation.

## Current experiment

The active stage adds **supplied GT grasp-pose conditioning to the original
visual Transformer policy**, retaining RGB, scene geometry, language, robot
state, learned task progression and the original diffusion action head. Two
matched arms fine-tune the same pretrained checkpoint for two epochs, with pose
guidance enabled versus disabled. Evaluation plans the same 149 Test1 scenarios
at three seeds per arm and measures both grasp and full-task success using policy
actions throughout. The full-data label audit and first exact-reset comparison
passed. GPU preflight also passed; training awaits GPU scheduling, and evaluation
is queued behind it. Learned-pose training comes afterward; its architecture is
now implemented. See the [active report](results/diagnostics/visual_oracle_pose_20261110_01/README.md)
and [job ledger](results/diagnostics/visual_oracle_pose_20261110_01/submissions.json).

The second supported path now implements shared visual/scene features, a
skill-conditioned soft part region, a grasp-pose estimator and pose conditioning
before the original diffusion action head. It passed implementation checks but
has not completed a training run or physical evaluation. See the [implementation](part2action/models/grounded_visual_pose_policy_v1.py)
and [two-architecture diagram](part2action/docs/figures/2026-11-11_grounded_visual_pose/visual_pose_architectures.pdf).
Fifteen CPU checks and
[GPU implementation smoke job 6865392](results/diagnostics/grounded_visual_pose_20261111_01/README.md)
passed. Full training of this new path has not been submitted.
The non-visual pose/controller studies and their eight result roots were removed
at the user's request; [cleanup record](part2action/docs/cleanup_20261111_nonvisual_pose.json).

The completed original visual control checkpoint initializes the pose study.
Its historical 64/447 task score is context; the newly fine-tuned control will be
the matched comparator. No new grasp/task success rate is available. Current
protocols, completed checks and queued jobs are in the
[research plan](part2action/docs/RESEARCH_PLAN.md).

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
