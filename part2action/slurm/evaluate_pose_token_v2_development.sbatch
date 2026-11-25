#!/usr/bin/env bash
# One GPU array task per checkpoint; the CPU report runs after both finish.
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts:PartInstruct
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PARTGYM_FPS_DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export DINOV2_HUB_DIR=/u/xna8aw/.cache/torch/hub/facebookresearch_dinov2_main
export DINOV2_CKPT=/u/xna8aw/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth
export MPLCONFIGDIR=/tmp/p2a_v2_development_mpl
case "${MODE:?Set evaluate or report}" in
  evaluate) TASK_PYTHON=/p/part2action/envs/partgym_rollout/bin/python ;;
  report) TASK_PYTHON=/p/part2action/envs/part2action_train/bin/python ;;
  *) exit 2 ;;
esac
"$TASK_PYTHON" -B part2action/scripts/run_pose_token_v2_development.py \
  --mode "$MODE" --out-dir "${P2A_V2_DEV_STUDY:?Set development study directory}" \
  --index "${SLURM_ARRAY_TASK_ID:-0}"
