# Experiment ownership across the three projects

Audited November 8, 2026 (America/New_York). This index resolves the ambiguous
phrase "20k training" across nearby checkouts. Project job records and frozen
run outputs remain the sources for live status and performance.

## Project identities

| Project and short label | Robot / environment | What was trained in the runs discussed | Data and budget | Canonical evidence |
| --- | --- | --- | --- | --- |
| **Part2Action / P2A** — `research_projects/interactive-robotics` | Franka Panda / PartGym, PartInstruct | Original visual Transformer + GT pose (queued), shared visual grounding/pose architecture (implemented) | 4,053 training demos; planned 149 Test1 scenarios × three seeds per arm | [Active visual GT study](../../results/diagnostics/visual_oracle_pose_20261110_01/README.md) |
| **Part-Aware Pose / PAP** — `part-aware-policy-learning` | RB-Y1 / MolmoSpaces | Added DINOv2-based pose transformer; visual encoder and original MolmoBot remain frozen | 4,866 official Train early-grasp trajectories / 973 houses; 19,464 observation–pose pairs; 20,000 updates, effective batch four, seed 6198 | [PAP training record](/u/xna8aw/workspace/part-aware-policy-learning/pose_learning/TRAINING_RUN.md), feature job 6838321 and training job 6838322 |
| **Live Robotics Lab / LRL** — `live-robotics-lab/LRL_project` | RB-Y1 / MolmoSpaces | MolmoBot language and action components; vision/connector frozen. Previous and corrected recipes are distinct intended comparisons | Both house-split recipes: 432 Train trajectories / 51 houses; 169 validation trajectories / 12 houses; 20,000 updates each | [LRL research plan](/u/xna8aw/workspace/live-robotics-lab/LRL_project/docs/RESEARCH_PLAN.md), [checkpoint/data record](/u/xna8aw/workspace/live-robotics-lab/LRL_project/docs/DATASET.md) |

PAP's 20,000-update run is **not** either LRL 20,000-update run and is not a
Part2Action run. The same update budget, effective batch size or random seed
does not identify an experiment. An optimizer update adjusts trainable weights;
it is not an epoch, a demonstration or a simulator rollout.

The earlier Part2Action goal-pose actor was a learned neural component. PAP's
door-opening oracle pilot used frozen MolmoBot plus a programmed local servo.
Explanations of that servo must not be applied to the P2A neural actor.

## Distinct retained training outputs

| Unambiguous run label | Output |
| --- | --- |
| `P2A non-visual geometry/controller studies` | Removed November 11 at user request; [cleanup receipt](cleanup_20261111_nonvisual_pose.json) |
| `PAP pose predictor / 20k` | `/bigtemp/xna8aw/part-aware-policy-learning/training/pose_20k_20261107_01/train/checkpoints/step20000.pt` |
| `LRL previous policy / 20k` | `/bigtemp/xna8aw/nav_door/train/final/house/llm_20k_gb4_20261021/model/step20000` |
| `LRL corrected policy / 20k` | `/bigtemp/xna8aw/nav_door/train/final/house/grounded_20k_gb4_20261030/model/step20000` |

Verified PAP plan: 19,464 Train observations, frozen Train-index SHA-256
`b4c7ce5cfdbd312422e0876025ae1e5541acdf57917c17c312b6e2ea899343cf`.
The LRL corrected configuration instead names the generated
`nav_door/data/final/house` dataset, enables language/action fine-tuning and writes
a different policy checkpoint. Thus these are not repeated copies of the same
training experiment.

## Active evaluation streams at audit time

| Owner / stream | Jobs | Population and purpose |
| --- | --- | --- |
| PAP, existing official Val197 Test | 6843326–6843329 | Offline pose errors on 187 labeled trajectories; 197 matched starts for baseline versus pose-guided control, 394 policy rollouts. Offline evaluation has completed; staging/paired simulation remains a separate stage. |
| PAP, independent Benchmark400 collection | 6843418–6843426 | Stage assets, run native experts, review the first 20 recordings and export labels for 400 fixed starts. The first/pilot/rest jobs partition one population; they are not repeated full datasets. |
| LRL, opening-yaw robustness | 6826513–6826515 | Pretrained versus corrected LRL policy on 832 reviewed physical states across three action seeds, up to 4,992 opening-only rollouts. |

The PAP cohort-comparison manifest records zero original Val197 cases included
in the new 400-case cohort. Asset staging is serialized because the streams
share a simulator cache. Shared original MolmoBot weights, simulator assets or
source data are dependencies, not duplicate training runs to delete.

This audit does not certify that all physical houses or pretrained-model
exposures are disjoint across the two door projects. Their datasets, metrics,
checkpoints and evaluation questions differ; do not pool their success counts.
LRL's reported grasp measure uses cumulative target-handle contact by opening
success; PAP's current evaluation uses onset-based force-qualified retention.
These are not interchangeable grasp-success definitions.

## Cleanup performed

No exact duplicate training run or redundant active evaluation stream was found
among the records above. No jobs were cancelled and no checkpoints, data,
videos or dated slide history were removed.

The cleanup makes ownership explicit: this index is linked from all three
project READMEs, and the PAP training record names its pose predictor and frozen
MolmoBot explicitly. The PAP research plan's stale "not submitted" statement
was corrected against the completed training receipt and current evaluation
records. Use the labels above rather than "the 20k model". Before
reporting a status or result, identify the project, job/run directory, trainable
module, checkpoint and dataset. An IDE tab alone does not select the experiment.

Frozen training and evaluation sources were not changed. Historical slides
remain dated snapshots; current status belongs in each project's run record.

November 11 update: the non-visual P2A pose/controller studies and their eight result roots were removed at the user's request. The earlier audit above describes its date; it is not a current retention decision. PAP and LRL outputs were not changed.
