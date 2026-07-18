#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --job-name=p2a-hier-all
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/u/xna8aw/workspace/research_projects/interactive-robotics}"
PART_DIR="$ROOT_DIR/part2action"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/u/xna8aw/workspace}"
CONFIG="part2action/configs/architecture_update/hierarchical_gated_pcd_contact3d_all_demos.yaml"
RUN_NAME="hierarchical_all_demos_${SLURM_JOB_ID:-manual}"
OUT_DIR="$PART_DIR/results/runs/$RUN_NAME"

mkdir -p "$PART_DIR/logs/slurm" "$OUT_DIR"
cd "$ROOT_DIR"
export PATH="$WORKSPACE_ROOT/.bin:$PATH"
export MAMBA_ROOT_PREFIX="$WORKSPACE_ROOT/.micromamba"
set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action_train
set -u
python - <<'PY'
import torch
print("[preflight] torch", torch.__version__, "cuda", torch.cuda.is_available())
print("[preflight] device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
PYTHONPATH=part2action python part2action/scripts/train.py \
  --config "$CONFIG" --override-out "$OUT_DIR" 2>&1 | tee "$OUT_DIR/train.log"
