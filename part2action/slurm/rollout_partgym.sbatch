#!/usr/bin/env bash
# Submit after a trained action checkpoint exists:
#   sbatch part2action/slurm/rollout_partgym.sbatch
#
# Common overrides:
#   sbatch --export=ALL,CKPT=results/part_action_mlp_real/last.pt,OBJ_CLASSES="scissors pliers" part2action/slurm/rollout_partgym.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=p2a-rollout
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail

if [ -d "${SLURM_SUBMIT_DIR:-}/part2action" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR/part2action" && pwd)"
elif [ -f "${SLURM_SUBMIT_DIR:-}/scripts/rollout_partgym.py" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    ROOT_DIR="/u/xna8aw/workspace/live-robotics-lab/interactive-robotics/part2action"
fi
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/u/xna8aw/workspace}"

MODEL_KEY="${MODEL_KEY:-mlp}"
CONFIG="${CONFIG:-configs/part_action_mlp_real.yaml}"
CKPT="${CKPT:-results/part_action_mlp_real/last.pt}"
OBJ_CLASSES="${OBJ_CLASSES:-scissors}"
TASK_TYPES="${TASK_TYPES:-1}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
EXECUTE_STEPS="${EXECUTE_STEPS:-1}"
DEVICE="${DEVICE:-cuda}"
RECORD="${RECORD:-1}"

export PATH="$WORKSPACE_ROOT/.bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORKSPACE_ROOT/.micromamba}"

mkdir -p "$ROOT_DIR/logs/slurm"
cd "$ROOT_DIR"

if [ ! -s "$CKPT" ]; then
    echo "[slurm] Missing checkpoint: $ROOT_DIR/$CKPT" >&2
    echo "[slurm] Train first, or submit with --export=ALL,CKPT=/path/to/last.pt" >&2
    exit 2
fi

echo "[slurm] job_id=${SLURM_JOB_ID:-manual}"
echo "[slurm] node=$(hostname)"
echo "[slurm] root=$ROOT_DIR"
echo "[slurm] model_key=$MODEL_KEY"
echo "[slurm] ckpt=$CKPT"
nvidia-smi || true

eval "$(micromamba shell hook --shell bash)"
micromamba activate partinstruct

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

read -r -a OBJ_CLASS_ARGS <<< "$OBJ_CLASSES"
read -r -a TASK_TYPE_ARGS <<< "$TASK_TYPES"

cmd=(
    python scripts/rollout_partgym.py
    --model-key "$MODEL_KEY"
    --config "$CONFIG"
    --ckpt "$CKPT"
    --obj-classes "${OBJ_CLASS_ARGS[@]}"
    --task-types "${TASK_TYPE_ARGS[@]}"
    --num-episodes "$NUM_EPISODES"
    --max-steps "$MAX_STEPS"
    --execute-steps "$EXECUTE_STEPS"
    --device "$DEVICE"
)

if [ "$RECORD" = "1" ]; then
    cmd+=(--record)
fi

echo "[slurm] command: ${cmd[*]}"
"${cmd[@]}"
