#!/bin/bash
#SBATCH --job-name=p2a-grounded-pose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
CONFIG=${CONFIG:-part2action/configs/pose_learning/grounded_visual_pose_v1.yaml}
TASK_OUTPUT=${TASK_OUTPUT:?Set TASK_OUTPUT to a fresh /p/part2action result directory}
ARGS=(--config "$CONFIG" --override-out "$TASK_OUTPUT")
if [ "${DRY_RUN:-0}" = 1 ]; then ARGS+=(--dry-run); fi
if [ -n "${EXPECTED_HASHES:-}" ]; then ARGS+=(--expected-hashes "$EXPECTED_HASHES"); fi
/p/part2action/envs/part2action_train/bin/python part2action/scripts/train_grounded_visual_pose_v1.py "${ARGS[@]}"
