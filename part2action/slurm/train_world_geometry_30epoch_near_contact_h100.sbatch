#!/usr/bin/env bash
# 30-epoch world geometry with two frames, finger opening, and last-centimeter contact.
# Writes to /p/part2action. Does not init from the 8/39 residual checkpoint.
#
#   sbatch part2action/slurm/train_world_geometry_30epoch_near_contact_h100.sbatch

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=p2a-near-contact
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/u/xna8aw/workspace/research_projects/interactive-robotics}"
PART_DIR="$ROOT_DIR/part2action"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/u/xna8aw/workspace}"
STORAGE_ROOT="${STORAGE_ROOT:-/p/part2action/results/part2action}"
CONFIG="part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_near_contact.yaml"
RUN_NAME="world_geometry_30epoch_near_contact_${SLURM_JOB_ID:-manual}"
OUT_DIR="$STORAGE_ROOT/runs/$RUN_NAME"
# Set RESUME to a last.pt to continue that run. Empty starts from scratch.
RESUME="${RESUME:-}"

mkdir -p "$PART_DIR/logs/slurm" "$OUT_DIR" "$ROOT_DIR/results/part2action"
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

python part2action/scripts/register_result.py --section part2action/runs --target "$OUT_DIR"

PYTHONPATH=part2action python part2action/scripts/validate_configs.py "$CONFIG"
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  test -f "$RESUME"
  RESUME_ARGS=(--resume "$RESUME")
  echo "[run] resume=$RESUME out=$OUT_DIR"
fi
PYTHONPATH=part2action python part2action/scripts/train.py \
  --config "$CONFIG" --override-out "$OUT_DIR" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/train.log"

echo "[done] near-contact checkpoint: $OUT_DIR/last.pt"
