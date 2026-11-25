#!/bin/bash
#SBATCH --job-name=p2a-pose-token-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
TASK_OUTPUT=${TASK_OUTPUT:?Set TASK_OUTPUT to a fresh isolated smoke root}
EXPECTED_HASHES=${EXPECTED_HASHES:?Set EXPECTED_HASHES to the frozen submission plan}
for TASK_POSE_SOURCE in provided disabled predicted; do
  /p/part2action/envs/part2action_train/bin/python part2action/scripts/train_pose_token_diffusion.py \
    --smoke --pose-source "$TASK_POSE_SOURCE" --out-dir "$TASK_OUTPUT/$TASK_POSE_SOURCE" \
    --expected-hashes "$EXPECTED_HASHES"
done
