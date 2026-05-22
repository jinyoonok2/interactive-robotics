#!/usr/bin/env bash
# Submit from anywhere:
#   sbatch part2action/slurm/train_part2action.sbatch
#
# Common overrides:
#   sbatch --export=ALL,TRACK=action-mlp,TIME_TAG=test1 part2action/slurm/train_part2action.sbatch
#   sbatch --export=ALL,TRACK=all part2action/slurm/train_part2action.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=p2a-train
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail

if [ -d "${SLURM_SUBMIT_DIR:-}/part2action" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR/part2action" && pwd)"
elif [ -f "${SLURM_SUBMIT_DIR:-}/train_tracks.sh" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    ROOT_DIR="/u/xna8aw/workspace/live-robotics-lab/interactive-robotics/part2action"
fi
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/u/xna8aw/workspace}"
TRACK="${TRACK:-action-mlp}"
TIME_TAG="${TIME_TAG:-${SLURM_JOB_ID:-manual}}"
RUN_ID="${RUN_ID:-slurm_${TIME_TAG}}"
RESULTS_ROOT="${RESULTS_ROOT:-part2action/results/runs}"

export PATH="$WORKSPACE_ROOT/.bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORKSPACE_ROOT/.micromamba}"

mkdir -p "$ROOT_DIR/logs/slurm"
cd "$ROOT_DIR"

echo "[slurm] job_id=${SLURM_JOB_ID:-manual}"
echo "[slurm] node=$(hostname)"
echo "[slurm] root=$ROOT_DIR"
echo "[slurm] track=$TRACK"
echo "[slurm] run_id=$RUN_ID"
nvidia-smi || true

eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action310

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

RUN_ID="$RUN_ID" RESULTS_ROOT="$RESULTS_ROOT" PYTHON_BIN=python bash train_tracks.sh "$TRACK"
