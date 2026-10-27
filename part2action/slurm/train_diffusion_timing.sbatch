#!/usr/bin/env bash
# From-scratch diffusion timing pair; same random policy initialization.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=p2a-diff-timing-fresh
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/u/xna8aw/workspace/research_projects/interactive-robotics}"
STORAGE_ROOT="${STORAGE_ROOT:-/p/part2action/results/part2action}"
PYTHON="${PYTHON:-/p/part2action/envs/part2action_train/bin/python}"
VARIANT="${VARIANT:-next_action}"
case "$VARIANT" in same_index|next_action) ;; *) echo "Invalid timing variant" >&2; exit 2;; esac
CONFIG="${CONFIG:-part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_diffusion_timing_${VARIANT}.yaml}"
OUT_DIR="$STORAGE_ROOT/runs/world_geometry_30epoch_diffusion_timing_${VARIANT}_${SLURM_JOB_ID:-manual}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT_DIR/part2action:$ROOT_DIR/part2action/scripts"
cd "$ROOT_DIR"
"$PYTHON" part2action/scripts/validate_configs.py "$CONFIG" --check-files
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU training requires CUDA')
print('[preflight]', torch.__version__, torch.cuda.get_device_name(0))
PY
if [[ -e "$OUT_DIR" ]]; then echo "Output already exists: $OUT_DIR" >&2; exit 2; fi
mkdir -p "$OUT_DIR"
"$PYTHON" part2action/scripts/register_result.py --section part2action/runs --target "$OUT_DIR"
"$PYTHON" part2action/scripts/train.py --config "$CONFIG" --override-out "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"
