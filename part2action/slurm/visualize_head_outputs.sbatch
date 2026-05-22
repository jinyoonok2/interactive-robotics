#!/usr/bin/env bash
# Generate offline PNG/GIF visualizations for heatmap/contact/approach heads.
#
# Example:
#   sbatch --export=ALL,RUN_DIR=results/runs/slurm_6201976,MODELS="heatmap_real heatmap_contact_real heatmap_approach_real" part2action/slurm/visualize_head_outputs.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=p2a-viz
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail

if [ -d "${SLURM_SUBMIT_DIR:-}/part2action" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR/part2action" && pwd)"
elif [ -f "${SLURM_SUBMIT_DIR:-}/scripts/visualize_head_outputs.py" ]; then
    ROOT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    ROOT_DIR="/u/xna8aw/workspace/live-robotics-lab/interactive-robotics/part2action"
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/u/xna8aw/workspace}"
RUN_DIR="${RUN_DIR:-results/runs/slurm_6201976}"
MODELS="${MODELS:-heatmap_real heatmap_contact_real heatmap_approach_real}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-300}"
MAX_DEMOS_PER_FILE="${MAX_DEMOS_PER_FILE:-12}"
OUT_DIR="${OUT_DIR:-results/visualizations/head_outputs}"
DEVICE="${DEVICE:-cuda}"

export PATH="$WORKSPACE_ROOT/.bin:$PATH"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORKSPACE_ROOT/.micromamba}"

mkdir -p "$ROOT_DIR/logs/slurm"
cd "$ROOT_DIR"

echo "[slurm] job_id=${SLURM_JOB_ID:-manual}"
echo "[slurm] node=$(hostname)"
echo "[slurm] root=$ROOT_DIR"
echo "[slurm] run_dir=$RUN_DIR"
echo "[slurm] models=$MODELS"
nvidia-smi || true

eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action310

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY

read -r -a MODEL_ARGS <<< "$MODELS"

python scripts/visualize_head_outputs.py \
    --run-dir "$RUN_DIR" \
    --models "${MODEL_ARGS[@]}" \
    --num-samples "$NUM_SAMPLES" \
    --sample-stride "$SAMPLE_STRIDE" \
    --max-demos-per-file "$MAX_DEMOS_PER_FILE" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR"
