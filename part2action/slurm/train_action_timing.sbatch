#!/usr/bin/env bash
# Matched timing comparison; both variants disable contact and use stride1.
# VARIANT=same_index (offset0) or next_action (offset1).
# MLP timing continuations; fresh optimizer, shared30-epoch control checkpoint.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=p2a-action-timing
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/u/xna8aw/workspace/research_projects/interactive-robotics}"
STORAGE_ROOT="${STORAGE_ROOT:-/p/part2action/results/part2action}"
PYTHON="${PYTHON:-/p/part2action/envs/part2action_train/bin/python}"
VARIANT="${VARIANT:-next_action}"
case "$VARIANT" in same_index|next_action) ;; *) echo "VARIANT must be same_index or next_action" >&2; exit 2;; esac
CONFIG="${CONFIG:-part2action/configs/architecture_update/hierarchical_world_geometry_10epoch_timing_${VARIANT}.yaml}"
RUN_NAME="world_geometry_10epoch_timing_${VARIANT}_${SLURM_JOB_ID:-manual}"
OUT_DIR="$STORAGE_ROOT/runs/$RUN_NAME"
RESUME="${RESUME:-}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT_DIR/part2action:$ROOT_DIR/part2action/scripts"
cd "$ROOT_DIR"

"$PYTHON" part2action/scripts/validate_configs.py "$CONFIG" --check-files
"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU training requires CUDA')
print('[preflight] torch', torch.__version__, 'device', torch.cuda.get_device_name(0))
PY

# Refuse accidental replacement of an earlier result, including manual runs.
if [[ -e "$OUT_DIR" ]]; then
  echo "Output already exists: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
"$PYTHON" part2action/scripts/register_result.py --section part2action/runs --target "$OUT_DIR"
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  test -f "$RESUME"
  RESUME_ARGS=(--resume "$RESUME")
fi
"$PYTHON" part2action/scripts/train.py --config "$CONFIG" --override-out "$OUT_DIR" \
  "${RESUME_ARGS[@]}" 2>&1 | tee "$OUT_DIR/train.log"
echo "[done] checkpoint: $OUT_DIR/last.pt"
