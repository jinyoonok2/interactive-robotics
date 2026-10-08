# Project Environment and Agent Guide

Last reconciled: 2026-10-08. Owner: Jinyoon (`xna8aw`), UVA.

Read this before changing Part2Action or PartInstruct. This guide is the current
reference for architecture, runs, Slurm, environments, and storage. Older slides
and setup recipes describe historical experiments. Do not commit or push unless
asked. Proposed experiments: [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Storage and entry points

All retained project files previously on liverobotics and bigtemp are consolidated
under the user's 500 GiB personal allocation, `/p/part2action`. Repository source,
configs, small reports, slides, and local diagnostic galleries remain in the repo.
Unrelated Molmo and Live Robotics Lab directories are outside this migration.

```text
/p/part2action/
  data/PartInstruct/               full released demos, assets, split metadata
  caches/partinstruct/             DP image and DP3 point-cloud zarr caches
  envs/
    part2action_train/             Part2Action training
    partgym_rollout/               Part2Action simulator rollout
    partinstruct/                  official DP / DP3
  results/
    part2action/runs/              four completed current model runs
    part2action/evaluations/       retained failed-setup report
    partinstruct/runs/             DP3 epoch1500, stopped DP epoch2500
    partinstruct/evaluations/      DP3 five-trial pilot and source videos
    partinstruct/archive/          earlier DP training logs, no checkpoints
    diagnostics/                  CPU contact-switch audit
  logs/
    part2action/slurm/
    partinstruct/slurm/
```

Repository: `/u/xna8aw/workspace/research_projects/interactive-robotics`.
`part2action/storage` is the single shortcut to `/p/part2action`.
Browse experiments through repository-root `results/`; its links point into the
structure above. See [the results guide](../../results/README.md).
`datasets/PartInstruct`, `PartInstruct/data`, and `results/data/PartInstruct`
point to the same primary dataset. Cache filenames in `demos/` link into
`caches/partinstruct/`. Project Slurm log directories link into personal storage.
Named micromamba environments link directly to `/p/part2action/envs/`.
Do not create new outputs under `part2action/results/` or on bigtemp/liverobotics.
Historical saved configs remain verbatim; override obsolete absolute paths when
reproducing them. Live launchers use the new storage paths.

## Current run status — observed 2026-10-08

No jobs were active when migration started. This is a dated snapshot; query Slurm
before reporting new status.

| Run | Job | Status / retained artifact |
| --- | --- | --- |
| Baseline | 6645296 | Completed 30 epochs; `last.pt` |
| Near-contact | 6645297 | Completed 30 epochs; `last.pt` |
| Unified control | 6687620 | Completed 30 epochs; `last.pt`; final action training loss 0.007299 |
| Unified contact | 6687621 | Completed 30 epochs; `last.pt`; final action training loss 0.007799 |
| Official DP3-S | 6613129 | Completed 1550 epochs; evaluated epoch1500 checkpoint |
| DP3 privileged-input pilot | 6619773 | 1/5 scissors trials succeeded; four reached 250 steps |
| Official DP-S | 6668506 | Cancelled at user's request; latest checkpoint epoch2500; do not resume without a request |

Training losses are not simulator success rates. Unified policies have not yet
been evaluated in PartGym. Baseline and near-contact evaluations produced no
scores: 6663209/6663210 failed on a stale framework path; 6687525/6687526 failed
on a broken editable Gym install; 6700956/6700957 failed at PyTorch3D CUDA
farthest-point sampling (`no kernel image is available for execution on the
device`). The framework path and Gym import were repaired. On 2026-10-08, farthest-point
sampling was moved to the deterministic PyTorch3D CPU implementation, preserving
the output index device; policy inference remains CUDA. H100 sampler preflight
6701224 passed, then exposed a stale pybullet_tools editable-package mapping.
The rollout environment's pybullet_tools and PartInstruct mappings now point to
the current root clone. Four-model H100 smoke6701227 COMPLETED (0:0) in53s: each checkpoint reset
the simulator and executed five actions. Full matched evaluations are now
RUNNING on H100 (observed 2026-10-08): unified control6701228, unified contact6701229, baseline6701230,
near-contact6701231. Each uses39 test1 trials, seed42, max120 steps, legacy
termination, execute1, CPU FPS, CUDA policy inference, and no video recording.
Submissions: `results/part2action/evaluations/submissions_20261008.json`.
Do not treat smoke tests as task-success results; full scores remain pending.
Error logs and the failure report remain under `results/part2action/evaluations/`.

DP3 pilot used ground-truth part-cloud input and one test1 trial each for task
types 1, 2, 5, 7, 9. The sole success was task2, grasp the scissors at its left
part, episode300471, 83 steps. Three failed trials had partial task completion.
This five-trial sample is separate from Part2Action's 39-trial protocol. Source
videos, success/failure GIFs, and provenance are retained. Official DP gets a
part mask; official DP3 gets a part cloud. Neither predicts the part. SAM-2 is
separate and its weights are not downloaded. GPT skill selection needs an API
key; local T5 encoding does not. Never print keys or embed them in jobs.

## What this project is

Part2Action is a robot policy for part-level manipulation. It trains on PartInstruct demonstrations and is scored in the PartGym simulator. The comparison policies are the official PartInstruct DP-S (image diffusion) and DP3-S (point-cloud diffusion) models, trained from the user's fork on branch `partinstruct-edited`.

The novelty to keep: the policy must not receive the dataset part mask, or a part-cloud part flag, as an input. Real cameras do not provide that. The ground-truth mask and the target-part cloud are training labels only. Deployment inputs are the RGB image, the full scene point cloud, the TCP pose, the task sentence, and the measured finger opening.

## Current Part2Action model

Two recipes, both trained from scratch on bottle, mug, pliers, and scissors. Joint angles stay out. The TCP pose already has position and rotation.

Shared stack:

- Frozen DINOv2 on the camera image. The part gate predicts the part and reweights visual tokens. The dataset mask is the training target for that gate.
- The contact head predicts one 3D point on the scene cloud. It is not the part-mask predictor.
- The action head writes world XYZ, wrist, and gripper. Finger opening is added only on this path.
- The hierarchy reads the task sentence and picks the skill. The current skill sentence is a training label, not an input.

Baseline, config `part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_no_residual.yaml`. One camera frame. The action head's XYZ is the command for the whole path. The contact point is trained and then unused in the command.

Near-contact, config `part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_near_contact.yaml`. Same stack, plus two camera frames through the temporal encoder. Farther than 3 cm, XYZ stays the action-head command. Inside 3 cm, XYZ becomes the predicted contact plus an offset of at most 3 cm. Wrist and gripper stay on the action head. During training the distance check uses the demo contact label. At rollout it uses the predicted point.

The old ±2 cm residual is off. It was `cloned XYZ + at most 2 cm` after the action head, not a feedback loop into earlier modules. Misses were 10–15 cm, so the nudge sat at the cap. The fair from-scratch pair was 2/39 without it and 2/39 with contact-plus-offset on the whole path. The older 8/39 number is a residual-trained checkpoint (`6529250`) and is not the comparison. Do not raise the 40 cm full-path offset and retrain. Do not give Part2Action the dataset mask or the part cloud.

## Training and validation data

Training files are the four HDF5 demos under `datasets/PartInstruct/demos/`. `val_hdf5` is empty, so there is no held-out validation set. Offline checks fall back to the training files. The 39 test episodes are different demonstrations: none of their IDs are in the training list. A validation split, if added, should be carved from the training demos. Do not train on `test1`.

The 39-trial protocol is 4 objects, not 13 objects. Task types 1–4 that exist for those objects:

- Bottle: 1, 2, 3, 4
- Mug: 1, 2, 3, 4
- Pliers: 1, 2, 3
- Scissors: 1, 2

Three trials each is 13 pairs × 3 = 39. Launcher: `part2action/slurm/rollout_bigtemp_full_eval_a40.sbatch`.


## Unified controlled experiment

Both variants keep two-frame temporal encoding, hierarchy, visual part gating,
and auxiliary part/contact supervision. Neither switches XYZ at a distance
threshold, adds the contact-relative bounded offset, or uses the old residual.
The same MLP action head operates throughout motion.

Unified contact feeds detached predicted geometry `(contact_world - TCP_world)
/ 0.1` through a learned projection into action features. Unified control supplies
zeros to the same projection. Ground-truth geometry supervises auxiliary heads;
it is not action conditioning. Nonfinite geometry is replaced by zero. Contact
confidence is not an implemented input. These matched seed42, 30-epoch runs use
only the original four object categories, despite the complete release now being
available. User explicitly chose train/test only, with no validation split.

## Do not undo

- Do not feed Part2Action the dataset part mask or the part cloud.
- Do not treat 8/39 as the baseline for the new runs.
- Do not turn the 3 cm offset into a whole-path offset.
- Do not delete `diagnostics/`.
- Do not download official DP or DP3 checkpoints, or SAM-2, unless the user asks.
- PartInstruct edits stay on `partinstruct-edited`. Do not mix them into Part2Action.
- Hydra changes the working directory, so demo and data paths in jobs must be absolute.
- DP3 cache must not reuse the DP image cache. The suffix is `+dataset.cache_suffix=dp3s`.


## Full released dataset

`/p/part2action/data/PartInstruct` contains all 11 released categories: bottle,
box, bucket, dispenser, display, kitchenpot, knife, mug, pliers, scissors,
stapler. Raw release contains 44,191 files, 69.09 GiB total; the 11 HDF5s are
67.26 GiB (72.23 GB). Expanded training coverage is a future experiment; do not
silently change the four-object matched comparison. Test episodes remain excluded
from training. Release download verification is in
`results/data/full_dataset_download/`; original dataset migration verification
is in `results/data/storage_migration_20261007/`.

## University GPU and environments

Reuse installed environments; do not recreate old `part2action310`, workstation,
Habitat, or ManiSkill environments. Environments were relocated with conda-pack
and embedded prefixes rewritten; editable package sources remain in the repo.

```bash
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX=/u/xna8aw/workspace/.micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action_train
# Rollout: micromamba activate partgym_rollout
# Official DP/DP3: micromamba activate partinstruct
```

Submit GPU work through Slurm; never run training on login/portal nodes. CUDA
may be unavailable there. Use your configured university SSH access; do not
invent an SSH hostname. Batch jobs survive disconnects; tmux is useful for long
interactive setup. Scripts should set `PYTHONNOUSERSITE=1` to isolate installed
environments from incompatible user-site packages.

```bash
squeue -u xna8aw -o "%.18i %.12P %.40j %.10T %.12M %.12l %R"
sacct -u xna8aw --starttime today --format=JobID,JobName,State,Elapsed,ExitCode
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

`Resources` means resources are unavailable; `Priority` means other jobs take
precedence. Once a job leaves squeue, inspect sacct, logs, and checkpoints. A
timeout can leave a usable checkpoint. Cancel only when authorized. If sandbox
scheduler sockets report `Operation not permitted`, use tool escalation.

Maintained Part2Action training/evaluation launchers request a generic GPU with
`a40|a100_40gb|a100_80gb|h100_94gb`, without node pinning. Filename suffixes
`h100`/`a40` are historical. Training: 48h, 8 CPUs, 40G; full rollout: 8h, 8 CPUs,
32G. Submit from the outer repo. Use
`part2action/slurm/train_world_geometry_unified_h100.sbatch` with `VARIANT=control`
or `contact`. Baseline and near-contact recipes remain for reproductions.
Official wrappers under `PartInstruct/scripts/slurm_scripts/` submit from the
inner PartInstruct repo and may have different resource requests. DP starts
fresh unless RESUME_FROM is explicitly supplied; stopped DP continuation remains
opt-in. DP3 cache suffix must be `dp3s` to avoid reusing image caches.

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
# Activate environment inside the allocation, inspect nvidia-smi, then exit.
```

Hydra changes working directories; use absolute dataset, framework, checkpoint,
and output paths. Register new outputs with `part2action/scripts/register_result.py`
in the appropriate root results section. Dataset terms/authentication are covered
in [DATA_SUBSETS.md](DATA_SUBSETS.md); tokens must never be logged. Reuse cached
models/offline mode where available. Removed `part2action/third_party/PartInstruct`
is obsolete; current framework is the root `PartInstruct/` clone.

## Evaluation setup repair — 2026-10-08

PartGym `vision_utils.downsample_pcd` defaults to CPU FPS.
`PARTGYM_FPS_DEVICE=cpu` is explicit in evaluation launchers; `input` opts into
the original input-device kernel for an environment with a compatible CUDA build.
The algorithm, start index0, sample count, policy inputs, checkpoints, and
39-trial protocol are retained. CPU sampling adds preprocessing latency; all
four comparisons use the same backend. Results record the sampling device.
Retargeted stale editable package finder mappings with backups under
`results/diagnostics/partgym_gpu_smoke_6701224/setup_repair/`. Package versions
are unchanged. Standalone BulletPlanner/pybullet_tools imports pass.

## Storage and cleanup history

2026-10-08: Repaired evaluation point-cloud sampling with deterministic CPU FPS,
retargeted stale pybullet_tools/PartInstruct editable mappings, and verified
five-step rollouts for all four checkpoints on H100 (smoke6701227). Submitted
matched full evaluations6701228–6701231 with flexible A40/A100/H100 placement
and unique source-run/job-ID output directories under personal storage.


2026-10-08: Consolidated project storage into `/p/part2action`, relocated three
environments, verified retained outputs/caches and dataset copies, replaced
storage/result/environment/log links, and updated future launcher defaults.
Removed failed empty rollout directories, superseded DP checkpoints, unused DP3
intermediate checkpoints, retired prototype launchers and queue scripts using absent runs,
stray `.q2`, and finished-download redirect. Kept completed checkpoints, training
histories, DP3 source videos/GIFs, contact diagnostics, historical residual run
6529250, source/configs, and failure evidence. Audit and verification manifests:
`results/data/storage_cleanup_20261008/`.

2026-10-07: Full raw release copied to personal allocation and SHA-256 verified;
assets checksum comparison clean. Gym 0.26.2 repaired stale editable Gym install;
rollout uses isolated NumPy1.26.4. GPU sampling remains the evaluation blocker.

2026-10-06: Submitted matched unified training6687620/6687621 with train/test only
at user's request. Broadened training and evaluation placement across A40/A100/H100.
2026-10-05: Cancelled DP6668506 at user's request. Added unified contact-conditioned
MLP and diagnostic gate GIFs; exported DP3 success and failures.
2026-10-04: Consolidated results at repository root; CPU contact audits and review
galleries added. 3/5/7/10cm thresholds are diagnostic gates, not physical contact.
2026-10-03: Consolidated environment docs; removed stale guide redirects and
superseded Part2Action runs6570908,6571099,6617401,6617402.

Historical 8/39 residual model6529250 and GT demo videos remain locally indexed.
Do not interpret it as the baseline for current comparisons. Source ManiSkill3
remains for later work; its environment, Habitat Lab, and IsaacLab were removed.
Do not touch unrelated Molmo or Live Robotics Lab storage.
