#!/usr/bin/env bash
# Generic full PartGym eval for a Part2Action run directory.
# Matched protocol vs world_geometry_30epoch / skill-gate full evals.
#
# Required:
#   sbatch --export=ALL,RUN_DIR=/p/part2action/results/part2action/runs/<name>,OUT_DIR=/p/part2action/results/part2action/evaluations/<name>_full_eval \
#     part2action/slurm/rollout_bigtemp_full_eval_a40.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=p2a-pgym-eval
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail

ROOT="/u/xna8aw/workspace/research_projects/interactive-robotics"
PART="$ROOT/part2action"
PARTINSTRUCT_ROOT="${PARTINSTRUCT_ROOT:-$ROOT/PartInstruct}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/PartInstruct}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the training run directory containing last.pt}"
OUT_DIR="${OUT_DIR:-/p/part2action/results/part2action/evaluations/$(basename "$RUN_DIR")_eval_${SLURM_JOB_ID:-manual}}"
CONFIG="${CONFIG:-$RUN_DIR/config_source.yaml}"
CKPT="${CKPT:-$RUN_DIR/last.pt}"
TERMINATION_MODE="${TERMINATION_MODE:-legacy}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-3}"
NUM_EPISODES="${NUM_EPISODES:-16}"
MAX_STEPS="${MAX_STEPS:-120}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-20}"
RECORD="${RECORD:-0}"
PREDICATE_DIAGNOSTICS="${PREDICATE_DIAGNOSTICS:-0}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PARTGYM_FPS_DEVICE=cpu
export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX="/u/xna8aw/workspace/.micromamba"
cd "$ROOT"

mkdir -p "$PART/logs/slurm" "$OUT_DIR"

if [[ ! -s "$CKPT" ]]; then
  echo "[slurm] missing checkpoint: $CKPT" >&2
  exit 2
fi
if [[ ! -s "$CONFIG" ]]; then
  echo "[slurm] missing config: $CONFIG" >&2
  exit 2
fi

for REQUIRED in "$PARTINSTRUCT_ROOT/PartInstruct/PartGym/config/config_oracle.yaml" "$PARTINSTRUCT_ROOT/PartInstruct/PartGym/config/config.yaml" "$DATA_ROOT/episodes_meta_test.json"; do
  test -s "$REQUIRED" || { echo "[slurm] missing setup file: $REQUIRED" >&2; exit 2; }
done

set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate partgym_rollout
set -u

python part2action/scripts/register_result.py --section part2action/evaluations --target "$OUT_DIR"

echo "[slurm] job_id=${SLURM_JOB_ID:-manual}"
echo "[slurm] node=$(hostname)"
echo "[slurm] ckpt=$CKPT"
echo "[slurm] config=$CONFIG"
echo "[slurm] out_dir=$OUT_DIR"
echo "[slurm] termination_mode=$TERMINATION_MODE"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

ARGS=(
  --partinstruct-root "$PARTINSTRUCT_ROOT"
  --data-root "$DATA_ROOT"
  --model-key hierarchical
  --config "$CONFIG"
  --ckpt "$CKPT"
  --obj-classes bottle mug pliers scissors
  --task-types 1 2 3 4
  --num-episodes "$NUM_EPISODES"
  --trials-per-task "$TRIALS_PER_TASK"
  --max-steps "$MAX_STEPS"
  --execute-steps 1
  --device cuda
  --termination-mode "$TERMINATION_MODE"
  --diagnostic-every "$DIAGNOSTIC_EVERY"
  --out-dir "$OUT_DIR"
  --diagnostics
)
if [[ "$RECORD" == "1" ]]; then
  ARGS+=(--record)
fi
if [[ "$PREDICATE_DIAGNOSTICS" == "1" ]]; then
  ARGS+=(--predicate-diagnostics)
fi

echo "[slurm] python part2action/scripts/rollout_partgym.py ${ARGS[*]}"
PYTHONPATH=part2action:part2action/scripts \
python part2action/scripts/rollout_partgym.py "${ARGS[@]}"

if [[ -n "${LINK_NAME:-}" ]]; then
  python part2action/scripts/register_result.py --section part2action/evaluations --target "$OUT_DIR" --name "$LINK_NAME"
fi
echo "[slurm] done: $OUT_DIR"
