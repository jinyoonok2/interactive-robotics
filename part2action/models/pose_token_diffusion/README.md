# Pose-token Diffusion

Independent visual pose-to-action implementation. The existing
[Part-grounded Diffusion](../part_grounded_diffusion/README.md) sources, trainers,
configs and checkpoint identities remain unchanged. The new family uses
`architecture: pose_token_diffusion_v1`, model-config schema 2, and checkpoint
schema 2. Old composition-only scaffold configs/checkpoints are rejected.

## Implemented model

```text
Observed RGB history + scene PCD + instruction + measured TCP/opening
  -> frozen DINOv2 + Flan-T5 (computed once and shared)
  -> temporal features + PointNet geometry + calibrated RGB/3D association
  -> predicted skill plan/phase and separate scene/language/robot tokens
  -> task-conditioned region -> target TCP pose + predicted applicability
  -> separate relative-translation / relative-orientation pose tokens
  -> cross-attention Transformer diffusion decoder -> 8 x 7 action chunk
```

Both global image context and observed 3D scene context reach the action decoder
directly. Pose embeddings are separate memory tokens, rather than an additive
vector injected into pooled visual features. The actor cross-attends to those
memories while denoising future action tokens. Encoder features and attention
memory are assembled once per observation, not once per diffusion step.

The compact defaults use width 256, two decoder layers, four heads, 100 cosine
DDPM steps, at most 64 global image tokens, 64 sampled scene tokens, and a small
local-region memory. These budgets limit extra attention cost; actual latency
and task success still require measurement. A pose-aware progress head is
implemented but off by default; the initial recipe retains the existing learned
skill-selection and termination mechanism.

| File | Role |
| --- | --- |
| `blocks/encoder.py` | Shared frozen backbones, temporal/PointNet/part features, RGB/3D association, learned skill plan |
| `blocks/grounding.py` | Skill/state-conditioned soft region, center, covariance and local tokens |
| `blocks/pose_head.py` | Target gripper/TCP position, orientation and applicability |
| `blocks/pose_tokens.py` | Differentiable TCP-relative translation/orientation memory |
| `blocks/action_decoder.py` | Actual cross-attention Transformer and DDPM training/sampling |
| `blocks/progress.py` | Optional goal/scene/state-conditioned skill progress |
| `blocks/geometry.py` | Quaternion/rotation geometry and camera projection |
| `policy.py`, `factory.py`, `contracts.py` | Independent composition and explicit data interfaces |
| `dataset.py`, `losses.py`, `training.py` | Training-only supervision, losses and GPU training/smoke |
| `runtime.py`, `evaluation.py` | Native PartGym action/plan-cache adapter and evaluation |
| `checkpoint.py` | New-family save/load and explicit, validated warm-start mapping |

`build_policy(config)` now constructs the actual model. Tests may instead inject
small frozen backbones or custom blocks. No package/config import loads pretrained
weights automatically. Existing encoder, temporal, PointNet and hierarchy classes
are reused read-only; the new policy does not inherit the original policy class.

## Pose sources and supervision

- `provided`: explicit GT target position/orientation and validity; no silent fallback.
- `disabled`: zero/invalid pose memory, with all visual/state paths retained.
- `predicted`: the learned target-pose head supplies the goal; external targets are rejected.

The provided/disabled recipes construct the same blocks. The pose head receives
auxiliary supervision in both; only its use as an action input changes. Same-seed
initialization and batch-order digests are recorded for a matched comparison.

The current supervised target is the measured TCP immediately before the first
expert grasp closure. The provided target stays static through the episode,
matching the earlier GT study; tasks not starting with a grasp get an invalid
target. Pose regression is supervised only before that closure. Later samples
still train actions and phase/termination, while applicability learns to reduce
first-grasp pose guidance. Part masks and aligned part membership supervise
upstream grounding; there is no independent contact-coordinate action input.

This first version does **not** supply a sequence of new pose goals for later
skills. Generating those goals needs additional labels and validation. The pose
head predicts a desired gripper pose, not an object's 6D pose. No hidden room map,
reference clock, replayed actions, or analytic pose controller enters the policy.

All 3D target/scene coordinates use world meters. Measured TCP input is quaternion
xyzw followed by world XYZ. Relative pose error is translation in the current TCP
frame plus the row-interleaved first two relative rotation columns minus identity.
The actor outputs world XYZ, roll/pitch/yaw and per-finger opening; native rollout
bounds the gripper command exactly as for the original policy.

## Initialization and training

DINOv2 ViT-S/14 registers and Flan-T5-base stay frozen. An explicit legacy transfer
copies compatible temporal, PointNet, part-gate, skill-planner, phase and termination
weights, plus action self-attention/feedforward projections. New cross-attention,
RGB/3D fusion, pose and token-type parameters are newly initialized. The transfer
checks physical action normalization and the diffusion schedule, records every
mapping, and leaves the source checkpoint unchanged. It is not a full-model resume.

Run in a Slurm GPU allocation, with the training environment and
`PYTHONPATH=part2action:part2action/scripts`:

```bash
python part2action/scripts/train_pose_token_diffusion.py \
  --out-dir /p/part2action/results/part2action/runs/pose_token_provided_NEW \
  --pose-source provided
```

Use the same recipe/seed/budget with `--pose-source disabled` for the matched
control. `--smoke` performs one actual-data GPU update and a full diffusion sample.
The default recipe is `configs/train.yaml`; model settings are in `configs/model.yaml`.
Outputs must be fresh directories; checkpoints are saved per epoch without
replacing any existing file. `--init-checkpoint` may also explicitly transfer a
new-family checkpoint between pose-source modes when all structural settings match.

The launchers are `part2action/slurm/pose_token_diffusion.sbatch` and
`part2action/slurm/smoke_pose_token_diffusion.sbatch`. The smoke launcher checks
provided, disabled and predicted modes sequentially on one GPU; these one-step
artifacts are not trained policies or benchmark results.

This cluster has no default Slurm partition. Pass `--partition=gpu` to `sbatch`
and request BF16-capable GPU features, for example
`--constraint='a40|a100_40gb|a100_80gb|h100_94gb|a4000|rtx_4000_ada'`.

## Native evaluation and verification

Use the simulator environment for evaluation:

```bash
python part2action/scripts/evaluate_pose_token_diffusion.py \
  --ckpt /path/to/new/epoch_002.pt --out-dir /path/to/fresh/eval \
  --oracle-run-dir results/diagnostics/visual_oracle_pose_20261110_01
```

The oracle catalog is required for provided mode. It preserves original reference
indices and verifies the exact physical reset before giving the GT target. In
predicted/disabled modes, reference targets never enter the model. Missing GT
references are recorded as unknown coverage gaps; runtime failures fail fast.
`--limit` marks an explicit debug subset. Default coverage is four object classes,
all held-out Test1 scenarios and seeds 42/43/44; compare identical completed pairs.
The native grasp predicate permits either-finger contact and does not prove retention.

November 16 execution update: new studies use 250 executed actions and eight
actions per prediction, with native physics thresholds, from the
[shared protocol](../../configs/evaluation/partinstruct_native.json). Historical
120/1 plans/results keep their original settings and source snapshots. A new
study directory, protocol-matching smoke receipts and new source hashes are
required; never pool historical and new-protocol records. Training weights and
model computation are unchanged by this evaluator update.

```bash
PYTHONPATH=part2action python -m unittest \
  models.pose_token_diffusion.tests.test_composition \
  models.pose_token_diffusion.tests.test_model \
  models.pose_token_diffusion.tests.test_pipeline \
  models.part_grounded_diffusion.tests.test_namespaces -v
PYTHONPATH=part2action python3 -m models.part_grounded_diffusion.verify_sources
```

The [validation record](validation.json) distinguishes CPU tests with tiny frozen
backbones, real-HDF5 integration, and any actual pretrained-backbone GPU smoke.
Implementation checks and one-step smoke artifacts establish executable paths;
they do not establish grasp/task improvement. Full training and physical policy
performance evaluation remain separate experiment steps.
