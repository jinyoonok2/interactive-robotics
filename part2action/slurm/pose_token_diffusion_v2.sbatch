#!/usr/bin/env bash
# Submitted by run_pose_token_diffusion_v2.py; pinned smoke gates joint training.
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export DINOV2_HUB_DIR=/u/xna8aw/.cache/torch/hub/facebookresearch_dinov2_main
export DINOV2_CKPT=/u/xna8aw/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth
MODE=${MODE:?Set smoke or train}
STUDY=${STUDY:?Set the prepared v2 pilot directory}
case "$MODE" in smoke|train|prepare-development) ;; *) exit 2;; esac
/p/part2action/envs/part2action_train/bin/python -B \
  part2action/scripts/run_pose_token_diffusion_v2.py --mode "$MODE" --out-dir "$STUDY"
