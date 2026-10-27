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
#SBATCH --time=1-00:00:00
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
EXPECTED_EPISODES="${EXPECTED_EPISODES:-149}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS"
MAX_STEPS="${MAX_STEPS:-120}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-20}"
# Save per-trial MP4/GIF media for failure review; RECORD=0 opts out.
RECORD="${RECORD:-1}"
# A standalone evaluation exports flat GIFs automatically. The comparison
# submitter sets this to 0 and exports all policies together after GPU jobs end.
FLAT_REVIEW="${FLAT_REVIEW:-1}"
RAW_OUT_DIR="$OUT_DIR"
if [[ "$FLAT_REVIEW" == "1" ]]; then
  RAW_OUT_DIR="$OUT_DIR/_raw/run"
fi
PREDICATE_DIAGNOSTICS="${PREDICATE_DIAGNOSTICS:-0}"
CONTACT_DIAGNOSTICS="${CONTACT_DIAGNOSTICS:-0}"
RESULT_SECTION="${RESULT_SECTION:-part2action/evaluations}"
REGISTER_RESULTS="${REGISTER_RESULTS:-1}"
CONTACT_REPORT="${CONTACT_REPORT:-0}"

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

if [[ "$REGISTER_RESULTS" == "1" ]]; then
  python part2action/scripts/register_result.py --section "$RESULT_SECTION" --target "$OUT_DIR"
fi

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
  --expected-episodes "$EXPECTED_EPISODES"
  --eval-seeds "${EVAL_SEED_ARGS[@]}"
  --max-steps "$MAX_STEPS"
  --execute-steps 1
  --device cuda
  --termination-mode "$TERMINATION_MODE"
  --diagnostic-every "$DIAGNOSTIC_EVERY"
  --out-dir "$RAW_OUT_DIR"
  --diagnostics
)
if [[ -n "${NUM_EPISODES:-}" ]]; then
  ARGS+=(--num-episodes "$NUM_EPISODES")
fi
if [[ "$RECORD" == "1" ]]; then
  ARGS+=(--record)
fi
if [[ "$PREDICATE_DIAGNOSTICS" == "1" ]]; then
  ARGS+=(--predicate-diagnostics)
fi
if [[ "$CONTACT_DIAGNOSTICS" == "1" ]]; then
  ARGS+=(--contact-diagnostics)
fi
if [[ -n "${CONTACT_STRENGTH:-}" ]]; then
  ARGS+=(--contact-strength "$CONTACT_STRENGTH")
fi
if [[ "${COLLISION_DIAGNOSTICS:-0}" == "1" ]]; then
  ARGS+=(--collision-diagnostics)
fi

echo "[slurm] python part2action/scripts/rollout_partgym.py ${ARGS[*]}"
PYTHONPATH=part2action:part2action/scripts \
python part2action/scripts/rollout_partgym.py "${ARGS[@]}"

if [[ "$CONTACT_REPORT" == "1" ]]; then
  python part2action/scripts/summarize_contact_motion.py --run-dir "$RAW_OUT_DIR"
fi

if [[ "$RECORD" == "1" && "$FLAT_REVIEW" == "1" ]]; then
  PYTHONPATH=part2action:part2action/scripts python part2action/scripts/export_evaluation_gifs.py \
    --source "${REVIEW_LABEL:-$(basename "$RUN_DIR")}=$RAW_OUT_DIR" --out-dir "$OUT_DIR" --data-root "$DATA_ROOT"
fi

if [[ -n "${LINK_NAME:-}" ]]; then
  python part2action/scripts/register_result.py --section "$RESULT_SECTION" --target "$OUT_DIR" --name "$LINK_NAME"
fi
echo "[slurm] done: $OUT_DIR"
