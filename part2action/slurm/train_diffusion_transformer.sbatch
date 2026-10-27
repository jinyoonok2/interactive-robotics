#!/usr/bin/env bash
# Fresh Transformer denoiser; same protocol as completed diffusion control.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=p2a-diff-transformer
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-/u/xna8aw/workspace/research_projects/interactive-robotics}"
PYTHON="${PYTHON:-/p/part2action/envs/part2action_train/bin/python}"
STORAGE_ROOT="${STORAGE_ROOT:-/p/part2action/results/part2action}"
CONFIG="${CONFIG:-part2action/configs/architecture_update/hierarchical_world_geometry_30epoch_diffusion_transformer_control.yaml}"
OUT_DIR="$STORAGE_ROOT/runs/world_geometry_30epoch_diffusion_transformer_control_${SLURM_JOB_ID:-manual}"
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
# This process checks one real-data AMP update and sampling. Its in-memory
# weights are discarded; the subsequent process starts fresh from seed 42.
"$PYTHON" part2action/scripts/smoke_unified_diffusion.py --config "$CONFIG" \
  --out-dir "$OUT_DIR/preflight" 2>&1 | tee "$OUT_DIR/preflight.log"
"$PYTHON" part2action/scripts/train.py --config "$CONFIG" --override-out "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"
