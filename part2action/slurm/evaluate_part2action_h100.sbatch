#!/usr/bin/env bash
# Evaluate one trained part2action checkpoint on an H100 GPU.
#
# Required overrides:
#   sbatch --export=ALL,CONFIG=/path/to/config_resolved.json,CKPT=/path/to/last.pt part2action/slurm/evaluate_part2action_h100.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=p2a-eval-h100
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail

if [ -d "${SLURM_SUBMIT_DIR:-}/part2action" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR/part2action" && pwd)"
elif [ -f "${SLURM_SUBMIT_DIR:-}/scripts/evaluate.py" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    ROOT_DIR="/u/xna8aw/workspace/research_projects/interactive-robotics/part2action"
fi
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/u/xna8aw/workspace}"
TRAIN_ENV="${TRAIN_ENV:-part2action_train}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.1}"
CONFIG="${CONFIG:?Set CONFIG to config_resolved.json or a config yaml}"
CKPT="${CKPT:?Set CKPT to last.pt}"

export PATH="$WORKSPACE_ROOT/.bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORKSPACE_ROOT/.micromamba}"

mkdir -p "$ROOT_DIR/logs/slurm"
cd "$ROOT_DIR"

echo "[slurm] job_id=${SLURM_JOB_ID:-manual}"
echo "[slurm] node=$(hostname)"
echo "[slurm] root=$ROOT_DIR"
echo "[slurm] config=$CONFIG"
echo "[slurm] ckpt=$CKPT"
nvidia-smi || true

set +u
if command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$TRAIN_ENV"
set -u

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

PYTHONPATH="$ROOT_DIR" python scripts/evaluate.py --config "$CONFIG" --ckpt "$CKPT"
