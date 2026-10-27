# Project Environment and Agent Guide

Last reconciled: 2026-10-15. Owner: Jinyoon (`xna8aw`), UVA.

Evaluation protocol reconciled: 2026-10-12; all 149 test1 scenarios across evaluation seeds 42, 43, 44.

Read this before changing Part2Action or PartInstruct. This guide is the current
reference for architecture, runs, Slurm, environments, and storage. Older slides
and setup recipes describe historical experiments. Do not commit or push unless
asked. Proposed experiments: [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Documentation roles

| Document | Purpose |
| --- | --- |
| [SETUP.md](SETUP.md) | Existing environment, dataset access, validation, training/evaluation commands, and review workflow |
| [PROJECT_ENVIRONMENT.md](PROJECT_ENVIRONMENT.md) | UVA CS Slurm, durable paths, data counts, architecture conventions, and current run records |
| [RESEARCH_PLAN.md](RESEARCH_PLAN.md) | Findings, active comparisons, limitations, bottlenecks, and future methods |
| [DATASET.md](DATASET.md) | Dataset contents, category counts, split definitions, and current evaluation coverage |

Supporting references: [DIFFUSION_ABLATION.md](DIFFUSION_ABLATION.md) contains
decoder implementation details; [CONTACT_LABEL_AUDIT.md](CONTACT_LABEL_AUDIT.md)
is the retained personal historical label analysis. `DATA_SUBSETS.md` was an older
download guide; its access instructions are in setup and dataset details are
now in `DATASET.md`. Slide history remains unchanged.

## Results and job status — October 19, 2026

Early-contact evaluation `6724354` and GIF export `6724355` completed successfully.
All 447 expert-left comparison GIFs are ready:
`results/part2action/evaluations/diffusion_early_contact_20261018/index.html`.
The 30-epoch, two-frame, stride-2, same-index comparison scored control 33/447
(7.4%), late contact 22/447 (4.9%), early contact 40/447 (8.9%). Early fusion
also changes the phase/contact grounding path, and only one training seed was
used, so this is not isolated causal evidence that contact conditioning helps.

The scratch 30-epoch stride-1 timing trainings `6719765`/`6719766` and their
evaluations `6719767`/`6719768` also completed: same-index 10/447 (2.2%),
next-index 16/447 (3.6%). The matched timing result favors next-index in this
run. It should not be conflated with the original stride-2 control's 33/447.
GIF export `6719769` was still running at this check.
Review: `results/part2action/evaluations/diffusion_timing_scratch_20261015/`.

New 10-epoch trainings `6724877`/`6724878` are running, both near the end of
epoch 3/10; real-data GPU preflights passed for both. Evaluations `6724879`/
`6724880` and export `6724881` are pending on their existing dependencies.
Every finished current training has already been evaluated; no duplicate
jobs were submitted in this status check. The cancelled superseded 30-epoch
Transformer pipeline remains cancelled.

## Current decoder screening — October 18, 2026

The working baseline is now previous diffusion with two observation frames,
next-index action targets, stride 1, and contact conditioning disabled. This is
an experiment-design choice, not a claim that the pending timing comparison
already established the better alignment. Both new policies train from scratch
for 10 epochs with seed 42, batch 16, LR 1e-4, and identical data and losses.
Only the diffusion denoiser differs.

| Stage | Previous diffusion baseline | Transformer diffusion |
| --- | --- | --- |
| Fresh 10-epoch training | `6724877` | `6724878` |
| Dependent 447-rollout evaluation | `6724879` | `6724880` |

Combined GIF export: `6724881`, after both evaluations. Panels: expert → baseline
→ Transformer, in one flat review folder. Each model is evaluated on all 149
four-object test1 scenarios at seeds 42/43/44. Both training jobs were pending
when checked after submission, with A40/A100/H100 eligibility.

Submission records/config snapshots:
`results/part2action/runs/diffusion_screening_10epoch_submission_20261018/submissions.json`.
Review: `results/part2action/evaluations/diffusion_screening_10epoch_next_action_20261018/`.
Launcher: `part2action/slurm/train_diffusion_screening.sbatch`.
Superseded pending 30-epoch Transformer pipeline `6724615`/`6724616`/`6724617`
was cancelled before starting; its records are under
`results/part2action/archive/superseded/diffusion_transformer_30epoch_20261018/`.
Other timing and early-contact experiments were not cancelled.

## Superseded 30-epoch Transformer submission — October 18, 2026

Fresh 30-epoch Transformer-denoiser training `6724615` is submitted, followed by
447-rollout evaluation `6724616` and expert-left GIF export `6724617`. It uses
two frames, stride 2, same-index targets, and no explicit contact conditioning,
matching the completed original diffusion control. The visual part-mask gate
remains enabled; the separate 3D point-part head remains disabled. The training
job checks a real-data GPU optimization step and sampling before full training.

Configuration snapshot, source hashes, protocol, and job commands:
`results/part2action/archive/superseded/diffusion_transformer_30epoch_20261018/submission/submissions.json`.
Review: `results/part2action/archive/superseded/diffusion_transformer_30epoch_20261018/review/`.
Panels: expert reference → original diffusion → Transformer diffusion. Original
control recordings are reused; no extra baseline training or evaluation is run.

## Latest evaluation submission — October 18, 2026

Early-contact training `6721407` completed all 30 epochs successfully. Its
447-rollout evaluation is submitted as `6724354`, with automatic comparison-GIF
export `6724355` depending on successful evaluation. The new review reuses the
completed diffusion control and late-contact recordings, with panels expert →
control → late contact → early contact. No MLP model is included or rerun.
Review and request/submission records:
`results/part2action/evaluations/diffusion_early_contact_20261018/`.
The standard 149 test1 scenarios, seeds 42/43/44, 120-action limit, video recording,
and contact diagnostics are retained. The timing experiments have their own
existing evaluation dependencies and were not resubmitted.

## Early-fusion submission — October 16, 2026

Fresh 30-epoch, two-frame early-contact diffusion training `6721407` is queued
with flexible A40/A100/H100 allocation. Early control `6721406` was cancelled
before starting at the user's request; compare early contact with existing models. They use stride 2 and same-index targets; the separate
timing experiment below retains stride 1 and tests offsets 0/1. No new
early-fusion evaluation jobs were submitted yet.

Submission record and immutable configuration snapshots:
`results/part2action/runs/diffusion_early_submission_20261016/submissions.json`.
At this check, timing runs `6719765` and `6719766` were both running on
`jaguar01`, on epoch 10 of 30 after roughly 22 hours. Their existing dependent
evaluations and GIF export remained pending.

## Timing experiment — October 15, 2026

The completed matched30-epoch evaluation scored MLP control3/447 (0.7%),
diffusion control33/447 (7.4%), and diffusion contact22/447 (4.9%). All447
expert-left comparison GIFs are ready in
`results/part2action/evaluations/diffusion_30epoch_20261013/`.
All three evaluated on A40; steady policy-forward means were10.5ms for MLP
and77.8/77.9ms for diffusion control/contact, excluding the first planning cycle.

New contact-disabled diffusion timing comparison **from scratch**:

| Stage | Same-index | Next-action |
| --- | --- | --- |
| Training |6719765|6719766|
| Dependent evaluation |6719767|6719768|

Both train for30 epochs with LR1e-4, batch16, two frames, stride1,
eight-action chunks, training seed42, identical losses, frozen pretrained
DINOv2/T5, and no validation split. Only action-target offset0/1 differs.
Neither uses trained policy weights or an old optimizer. The shared file
`runs/diffusion_timing_initialization_20261015/policy_seed42.pt` contains
newly randomized policy tensors with epoch-1 and zero training steps.
Both jobs verify its fingerprint against their loaded state before optimizing;
verification is written to each run's `initialization_verification.json`.

Flexible A40/A100/H100 allocation is retained. Each training job requests up
to96 hours (the partition maximum), because stride1 increases sample count.
Both were pending priority at submission; query Slurm for current status.

Automatic evaluations cover149 test1 scenarios ×seeds42,43,44,447 rollouts
per model. Export6719769 follows both evaluations. Panels: expert reference →
same-index from scratch → next-action from scratch.
Outputs: `results/part2action/evaluations/diffusion_timing_scratch_20261015/`.
Snapshots, initialization fingerprint, source hashes, and dependency records:
`results/part2action/runs/diffusion_timing_scratch_submissions_20261015.json`.
Launcher: `slurm/train_diffusion_timing.sbatch`.

Superseded continuations6719759/6719760 and dependent jobs6719761–6719763
were cancelled at the user's request. Their partial run folders, configs,
review directory, result links, and logs were removed. Completed original
models/evaluations and contact-analysis results were retained. A cancellation
record is retained inside the replacement review for provenance.

Existing-result contact investigation (no new GPU inference): paired outcomes
are both-success8, control-only25, contact-only14, both-fail400. By first skill,
grasp-starting tasks score29/339 for control versus13/339 for contact;
touch-starting tasks score4/108 versus9/108. Mug is the only category with more
contact-model successes (14 versus9). This is a comparison of separately
trained models, not evidence from a frozen-checkpoint causal intervention.
Geometry is scored against the first task part tracked with object motion.
Focused review: `results/part2action/evaluations/diffusion_30epoch_20261013/contact_impact_review.html`.
Summary/paired rows are beside it, generated by
`scripts/summarize_contact_impact.py`. Do not infer contact is uniformly useless
from the lower aggregate score, or infer causal benefits from these subgroups.

## Results and jobs — October 13, 2026

Full four-object test1 timing comparison and GIF export completed successfully:

| Model | Success across447 rollouts | Success per149-scenario seed |
| --- | ---: | ---: |
| Same-index continuation |9/447 (2.0%)|3/149 for each seed|
| Next-action continuation |24/447 (5.4%)|8/149 for each seed|

Both models succeeded on the same respective episodes at seeds42,43,44;
additional seeds did not add unique successful scenarios. Mean seed success
was2.01% versus5.37%, with evaluation-seed sample SD0 for both. These remain
single-training-seed continuation results. All447 expert-left comparison GIFs
are ready under `results/part2action/archive/mlp/evaluations/action_timing_full_test1_20261012/`.

Diffusion control6712229 and contact6712230 completed all30 epochs on H100,
with successful checkpoints and final noise-prediction losses0.03232/0.03293.
Contact's learned global scalar ended at0.12476; its size does not establish
that conditioning improves task success.

New evaluation comparison:

| Model | Job | Status when checked |
| --- | --- | --- |
| Original30-epoch two-frame MLP control |6717931|Running, jaguar06|
| Diffusion control |6717932|Pending|
| Diffusion contact |6717933|Pending|
| Expert-left combined GIF export |6717934|Pending, dependency|

All three use149 test1 scenarios ×seeds42,43,44, with matched30-epoch training
budgets, two frames, stride2, and same-index targets. Diffusion is not using
the next-action shift. The baseline isolates the decoder comparison more
cleanly than the40-epoch stride1 timing continuations. Action objectives and
normalization differ between MLP and diffusion, so compare rollout outcomes,
not raw training loss. Per-cycle synchronized policy-forward timing records
exclude simulation, observation conversion, and media export; steady timing
excludes each trial's first planning cycle. GPU model names are saved, since
latency on different allocated GPUs is not a pure decoder-speed comparison.

Output: `results/part2action/evaluations/diffusion_30epoch_20261013/`.
Panel order: expert reference → MLP control → diffusion control → diffusion contact.
Submission/config references and the scenario list are retained in that folder.
Query Slurm before reporting a later live status. No new training was submitted.

## Historical job snapshot — October 12, 2026

| Experiment | Job | State when checked |
| --- | --- | --- |
| Diffusion control training |6712229| Running on H100 |
| Diffusion contact training |6712230| Running on H100 |
| Full-test1 same-index evaluation |6714294| Pending, priority |
| Full-test1 next-action evaluation |6714295| Pending, priority |
| Combined GIF export |6714296| Pending, dependency on both evaluations |

These are dated observations; query Slurm before reporting a new live status.
The diffusion pair uses same-index targets and stride 2; the completed timing
MLPs use stride 1 and offsets0/1. The extra from-scratch timing pair is skipped.
The cancelled single-seed reruns6714288/6714289/6714290 never started; their
unused outputs and logs were removed. Diffusion training is unaffected.

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
    part2action/runs/              training runs and checkpoint histories
    part2action/evaluations/       full-set and historical simulator evaluations
    partinstruct/runs/             DP3 epoch1500, stopped DP epoch2500
    partinstruct/evaluations/      DP3 five-trial pilot and source videos
    partinstruct/archive/          earlier DP training logs, no checkpoints
    diagnostics/                  contact, timing, intervention, and replay reports
  logs/
    part2action/slurm/
    partinstruct/slurm/
```

Repository: `/u/xna8aw/workspace/research_projects/interactive-robotics`.
`part2action/storage` is the single shortcut to `/p/part2action`.
Browse experiments through repository-root `results/`; its links point into the
structure above. See [the results guide](../../../../results/README.md).
`datasets/PartInstruct`, `PartInstruct/data`, and `results/data/PartInstruct`
point to the same primary dataset. Cache filenames in `demos/` link into
`caches/partinstruct/`. Project Slurm log directories link into personal storage.
Named micromamba environments link directly to `/p/part2action/envs/`.
Do not create new outputs under `part2action/results/` or on bigtemp/liverobotics.
Historical saved configs remain verbatim; override obsolete absolute paths when
reproducing them. Live launchers use the new storage paths.

## Default evaluation review format — 2026-10-10

The user's standing preference is **one review folder of side-by-side GIFs**,
unless a different format is required. Each matched trial gets one combined
GIF with two or more labeled policy panels, synchronized by simulation frame.
Always prepend a **ground-truth expert reference on the far left**. Use an exact
episode demo when available; otherwise require the same complete skill chain
and parameters, prefer the same object instance, and explicitly identify its
different source episode. Dataset expert trajectories have different starting
scenes and are not phase-aligned test trajectories. Demo/episode IDs and matching
details are stored in the manifest and shown in the GIF. The shared
`scripts/ground_truth_reference.py` resolver exports these recorded HDF5 RGB
frames automatically; ground truth is review-only, never a policy input.
All GIFs, `index.html`, metrics, the manifest, and a portable ZIP are directly
in the evaluation directory. Supporting recordings and rollout JSON stay under
`_raw/`. Do not create separate visible policy galleries or nested GIF folders.
Single-policy evaluations export reference-plus-policy GIFs.

Current full-set review destination:
`results/part2action/archive/mlp/evaluations/action_timing_full_test1_20261012/`.
It will contain447 combined GIFs: expert reference, same-index, next-action.
Per-seed and aggregate success metrics are generated after the evaluations finish.

Historical timing continuation results:
`results/part2action/archive/mlp/evaluations/action_timing_20261011/index.html`.
Both evaluations and paired GIF export completed: same-index6712177 scored1/39,
next-action6711750 scored5/39, and paired export 6712178 produced39 GIFs.
Training6705831/6705832 completed ten additional epochs from the same30-epoch
control checkpoint6687620, with contact conditioning disabled, two frames,
stride 1, and offsets0/1. Final training action losses were0.002598/0.002838.
These are continuation results, not a from-scratch alignment comparison.
Submission record: `results/part2action/archive/mlp/runs/timing_submissions_20261010.json`.

Historical normalized MLP review:
`results/part2action/archive/mlp/evaluations/normalized_global_20261010/index.html`.
Control6705020 and contact 6705021 scored4/39 and0/39. Both received ten extra
epochs from the shared30-epoch control checkpoint. Do not mix these pilot
scores with the expanded task coverage of the current full-set protocol.

The recording-order audit scanned all 4,053 valid four-object demonstrations
and replayed eight initial prefixes on CPU. Next-action labels are supported
through `data.action_target_offset:1`, while historical/default configs use0.
The released PartInstruct loader also uses same-index labels; the shift is a
hypothesis. Native touch frame spacing/controller history remains unresolved.
Evidence: `results/diagnostics/observation_action_timing_20261010/`.

For future comparisons, use `scripts/submit_evaluation_comparison.py` with
repeated `--run LABEL=TRAINING_DIRECTORY` and a fresh `--out-dir`. It schedules
flexible GPU evaluations and an `afterok` CPU GIF export automatically. The
standalone full-rollout launcher also exports flat GIFs by default. Raw-only
jobs set `FLAT_REVIEW=0`; comparisons do this internally and export once after
all GPU jobs succeed. `RECORD=0` is an explicit opt-out. Commands and exporter
retry instructions are in [SETUP.md](SETUP.md#evaluate).

## Historical run status — observed 2026-10-08

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

Training losses are not simulator success rates. Earlier baseline and near-contact
evaluation attempts produced no scores: 6663209/6663210 failed on a stale framework path; 6687525/6687526 failed
on a broken editable Gym install; 6700956/6700957 failed at PyTorch3D CUDA
farthest-point sampling (`no kernel image is available for execution on the
device`). The framework path and Gym import were repaired. On 2026-10-08, farthest-point
sampling was moved to the deterministic PyTorch3D CPU implementation, preserving
the output index device; policy inference remains CUDA. H100 sampler preflight
6701224 passed, then exposed a stale pybullet_tools editable-package mapping.
The rollout environment's pybullet_tools and PartInstruct mappings now point to
the current root clone. Four-model H100 smoke 6701227 COMPLETED (0:0): each
checkpoint reset the simulator and executed five actions. Full matched
evaluations 6701228–6701231 then COMPLETED (0:0), with all 39 results saved per
model. Baseline and near-contact each succeeded in 1/39 trials; unified control
and unified contact each succeeded in 0/39. Each used test1, seed 42, max 120 steps,
legacy termination, execute 1, CPU FPS, CUDA policy inference, and no recording.
Submissions: `results/part2action/archive/mlp/evaluations/submissions_20261008.json`.
These results do not demonstrate improvement from contact conditioning.

Recorded reruns completed (0:0) on 2026-10-08 with the same checkpoints and
protocol: unified control 6701252, unified contact 6701253, baseline 6701254,
near-contact 6701255. `RECORD=1` saves a per-trial MP4 and GIF under each run's
`hierarchical/videos/` directory. Each run is registered under root
`results/part2action/evaluations/`; submission records are in
`recorded_submissions_20261008.json` in that section. The full-evaluation launcher
now defaults to recording; `RECORD=0` opts out. Query Slurm for live progress.
2026-10-08 cleanup: only the four recorded evaluation directories remain in
root `results/part2action/evaluations/`. Each has39 MP4s,39 GIFs, rollout metrics,
and `contact_localization/` with its initial ground-truth measurements. Earlier
unrecorded traces were verified identical and removed; four standalone contact
measurement directories were consolidated, and the setup-failure folder removed.
All156 MP4s and156 GIFs were SHA-256 verified unchanged. Submission manifests
and saved analysis source references now point to retained files. Scheduler logs
remain under project logs. Browse `results/part2action/evaluations/README.md`.

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

## Training and evaluation data

[DATASET.md](DATASET.md) is the reference for demonstration/scenario formats,
full-release and four-object counts, the five benchmark splits, and metadata
identity checks. Current training uses bottle, mug, pliers, and scissors with
no demo-count cap and no validation split, as requested.

The operational evaluation default is all four-object test1 scenarios for fixed
seeds 42, 43, 44, selected explicitly once per seed. Test2–test4 are separate
coverage extensions and are not in the currently queued comparison. Detailed
coverage and totals are in [DATASET.md](DATASET.md#current-evaluation-coverage);
submission commands are in [SETUP.md](SETUP.md#evaluate).

Dataset location, aliases, size, and verification paths are listed under
[Dataset storage](#dataset-storage). Do not train on test metadata or silently
expand training categories when reproducing a matched experiment.

## Unified controlled experiment

Both variants keep two-frame temporal encoding, hierarchy, visual part gating,
and auxiliary part/contact supervision. Neither switches XYZ at a distance
threshold, adds the contact-relative bounded offset, or uses the old residual.
The same MLP action head operates throughout motion.

Unified contact feeds detached predicted geometry `(contact_world - TCP_world)
/ 0.1` through a learned projection into action features. Unified control supplies
zeros to the same projection. Ground-truth geometry supervises auxiliary heads;
it is not action conditioning. Nonfinite geometry is replaced by zero. Contact
confidence is not an implemented input. These matched seed 42, 30-epoch runs use
only the original four object categories, despite the complete release now being
available. User explicitly chose train/test only, with no validation split.

## Initial contact localization check — 2026-10-08

Frozen baseline, near-contact, unified control, and unified contact checkpoints
were scored against simulator ground-truth target-part geometry on the
same 39 initial test1 observations. Jobs: control 6701538, contact 6701539,
baseline6701540, near-contact 6701541. The contact alignment measurement is taken
before the first action against ground-truth skill0, independent of predicted
phase. Each diagnostic trial executes only one action; its success count is not
comparable to the full rollout protocol. Ground-truth geometry is logging-only.
`CONTACT_DIAGNOSTICS=1` enables this logging without exhaustive predicates.
All four jobs COMPLETED (0:0), with supported measurements for all 39 trials.
Mean predicted-contact distance to the intended part at reset: baseline0.959cm,
near-contact 1.236cm, control 0.997cm, contact 1.272cm. Counts within1cm are30,26,27,24
respectively; within3cm are37,37,37,36. All models used identical episode/seed pairs.
This measures initial proximity to target-part surface points, not a uniquely
correct contact, grasp pose, or localization throughout motion.
Submission records are `results/part2action/archive/mlp/evaluations/contact_localization_submissions_20261008.json`.
Run `python3 part2action/scripts/summarize_contact_localization.py` after completion
for mean/median part distance and 1/3/5cm proximity counts, retaining unsupported
cases. Report: `results/part2action/archive/mlp/diagnostics/contact_motion_20261008/reference_reports/contact_localization_20261008/README.md`.

## Full-motion contact diagnostics — 2026-10-08

Completed (0:0): unified control 6701641, unified contact 6701642, baseline6701643,
and near-contact 6701644. Each runs all 39 test1 trials with max 120 steps, seed 42,
legacy termination, execute 1, recorded MP4/GIFs, and bounded gripper commands.
These are separate from the earlier unbounded recorded evaluations.

All outputs are grouped under `results/part2action/archive/mlp/diagnostics/contact_motion_20261008/`,
physically `/p/part2action/results/part2action/archive/mlp/diagnostics/contact_motion_20261008`.
Each model has a subfolder and automatically generated `contact_motion_report/`
with mean/start/end/last-third distances, per-trial CSV, and per-step CSV.
Measurements are before action: predicted contact, TCP, and commanded XYZ to
live ground-truth target-part points. Main comparisons track the first task part
throughout motion; separate traces follow selected skill index and explicitly
retain unsupported skills. Ground-truth data is logging-only. The unified pair
is the main matched comparison. Query Slurm for current progress.

## Unified comparison videos — 2026-10-08

`results/part2action/archive/mlp/diagnostics/unified_policy_comparison_20261008/index.html` contains39
synchronized control/contact comparison videos for the bounded-gripper trials.
Left is control, right is contact. Blue marks predicted contact; green marks the
nearest sampled point on the first true target part. These are geometry overlays,
not expert trajectories. Matching test-episode demonstrations were not found in
the four local HDF5 files. Playback is half speed, with terminal frames held
when one trial ends earlier. Source episode IDs, seeds, and paths are in manifest.json.
`comparison_bundle.zip` contains the standalone gallery and all 39 H264 MP4s.

## Diffusion decoder preparation

Diffusion decoder preparation (2026-10-08): unified conditioning now supports
monolithic MLP or diffusion heads. Matched diffusion control/contact configs use
100 cosine steps and fixed per-dimension action normalization saved in the head.
All training/inference builders accept these settings. Six configs and ten
focused checks passed. Final real-data GPU preflight6701391 COMPLETED (0:0)
on H100: both variants passed finite loss/gradients, one optimizer step, and bounded
8×7 action sampling with deployment inputs. Estimated clean samples are clipped
to ±2 normalized units during reverse sampling. Verification is under
`results/diagnostics/preflight_checks/unified_diffusion_smoke_6701391/verification.json`. Full
diffusion training was submitted on October 12: control `6712229`, contact
`6712230`. Both use fresh initialization, 30 epochs, two frames, stride 2,
same-index action targets, and normalized-global conditioning enabled only
for contact. Launcher: `slurm/train_unified_diffusion.sbatch`, using flexible
A40/A100/H100 allocation and the durable project environment. Submission-time
config snapshots and source hashes are recorded in
`results/part2action/runs/diffusion_submissions_20261012.json`.
Matched 39-trial evaluations are planned after training; they are not submitted.
The additional from-scratch timing pair is skipped at the user's request.
See [DIFFUSION_ABLATION.md](DIFFUSION_ABLATION.md).

## Gripper command repair

Gripper repair (2026-10-08): Part2Action rollout now bounds action[6] to Panda's
per-finger position range [0,0.04]m before env.step, independently of decoder type.
Arm pose and raw model output are preserved. New results include raw and executed
gripper traces, bounds, clipping count, and measured finger position diagnostics.
Two boundary checks and all 18593 previously recorded action commands passed;
GPU smoke6701594 COMPLETED (0:0) on H100; all four checkpoints executed
five actions with bounded commands, raw predictions, and measured finger logs. Existing
recorded evaluations used unbounded commands and are preserved as that protocol.
No retraining or new gripper loss is implied. Contact predictions have not been
changed: first check ground-truth localization throughout motion and action
response before selecting a contact-head or conditioning change.

## Do not undo

- Do not feed Part2Action the dataset part mask or the part cloud.
- Do not treat 8/39 as the baseline for the new runs.
- Do not turn the 3 cm offset into a whole-path offset.
- Do not delete `diagnostics/`.
- Do not download official DP or DP3 checkpoints, or SAM-2, unless the user asks.
- PartInstruct edits stay on `partinstruct-edited`. Do not mix them into Part2Action.
- Hydra changes the working directory, so demo and data paths in jobs must be absolute.
- DP3 cache must not reuse the DP image cache. The suffix is `+dataset.cache_suffix=dp3s`.


## Dataset storage

Primary dataset: `/p/part2action/data/PartInstruct`.
Repository shortcuts `datasets/PartInstruct`, `PartInstruct/data`, and
`results/data/PartInstruct` resolve to that same durable installation.
Training HDF5 files are under `demos/`; simulator assets are under `assets/`;
train/test and object metadata are at the dataset root. DP/DP3 generated caches
are stored separately under `/p/part2action/caches/partinstruct/`.

The verified raw download contains 44,191 files, 69.09 GiB total. Its 11 HDF5 files
occupy 67.26 GiB (72.23 GB). Download verification:
`results/data/full_dataset_download/`. Original migration verification:
`results/data/storage_migration_20261007/`.

Dataset contents, split meanings, and category counts are in
[DATASET.md](DATASET.md); access/download commands are in
[SETUP.md](SETUP.md#dataset-access-and-download-helper). The entire released
installation is retained, while training configs still select the four-object
scope. Avoid downloading a second copy onto bigtemp or liverobotics.

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
in [SETUP.md](SETUP.md#dataset-access-and-download-helper); tokens must never be logged. Reuse cached
models/offline mode where available. Removed `part2action/third_party/PartInstruct`
is obsolete; current framework is the root `PartInstruct/` clone.

## Evaluation setup repair — 2026-10-08

PartGym `vision_utils.downsample_pcd` defaults to CPU FPS.
`PARTGYM_FPS_DEVICE=cpu` is explicit in evaluation launchers; `input` opts into
the original input-device kernel for an environment with a compatible CUDA build.
The algorithm, start index0, sample count, policy inputs, checkpoints, and
39-trial protocol are retained. CPU sampling adds preprocessing latency; all
four comparisons use the same backend. Results record the sampling device.
Retargeted stale editable package finder mappings to the current root clones.
The superseded repair/preflight folder was removed during result cleanup;
the successful checks remain under `results/diagnostics/preflight_checks/`.
Package versions are unchanged. Standalone BulletPlanner/pybullet_tools imports pass.

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

2026-10-06: Submitted matched unified training 6687620/6687621 with train/test only
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

## Official released checkpoints — October 22, 2026

User authorized downloading and evaluating the released DP-S/DP3-S weights,
replacing the earlier self-trained versions for the reproduction objective.
Earlier entries calling our locally trained DP/DP3 runs "official" refer to the
framework implementation, not these released pretrained weights. Keep the
historical checkpoints as archives; do not resume their training.

Downloaded both files to `/p/part2action/data/official_checkpoints/PartInstruct/`.
Source dataset revision: `e519e93dc5882cdaadad45bd7c26be6986b0a436`.
DP-S: 5,504,403,026 bytes; DP3-S: 4,614,106,626 bytes.
The download manifest records their source and durable paths.

Evaluation launcher: `part2action/slurm/evaluate_official_partinstruct.sbatch`.
The isolated adapter in `part2action/scripts/official_partinstruct/` maps legacy
Hydra paths and recovers the DP encoder's five-channel input from its actual
weights, retaining the public RGB-plus-repeated-mask preprocessing. Strict EMA
weight loading is required; no weights are reshaped or omitted. Each GPU job
runs a native scissors preflight before a native full-test1 pass: 16 sampled
rollouts per each of 11 categories, 176 per model, with videos and summaries.
These use GT part observations and oracle skill progression. They are neither
the fixed 149 x three-seed Part2Action protocol nor an exact reproduction of
all settings and repetitions from the paper. No SAM or VLM calls are used.

Both released EMA checkpoints passed CPU strict loading. Submitted official
DP-S evaluation `6734186` and DP3-S evaluation `6734187`; each includes its own
GPU/simulator preflight. Ledger and outputs:
`results/partinstruct/evaluations/official_release_20261022/`.
