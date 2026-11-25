#!/bin/bash
#SBATCH --job-name=p2a-pose-token
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
TASK_OUTPUT=${TASK_OUTPUT:?Set TASK_OUTPUT to a fresh pose-token result directory}
POSE_SOURCE=${POSE_SOURCE:-provided}
ARGS=(--out-dir "$TASK_OUTPUT" --pose-source "$POSE_SOURCE")
if [ "${SMOKE:-0}" = 1 ]; then ARGS+=(--smoke); fi
if [ -n "${EXPECTED_HASHES:-}" ]; then ARGS+=(--expected-hashes "$EXPECTED_HASHES"); fi
/p/part2action/envs/part2action_train/bin/python part2action/scripts/train_pose_token_diffusion.py "${ARGS[@]}"
