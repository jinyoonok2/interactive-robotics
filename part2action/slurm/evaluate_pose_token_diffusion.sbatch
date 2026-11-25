#!/usr/bin/env bash
# Exact trained-checkpoint smoke gates the matched benchmark and paired report.
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts:PartInstruct
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PARTGYM_FPS_DEVICE=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/p2a_pose_token_evaluation_mpl
case "${MODE:?Set smoke, evaluate or report}" in
  smoke|evaluate) TASK_PYTHON=/p/part2action/envs/partgym_rollout/bin/python ;;
  report) TASK_PYTHON=/p/part2action/envs/part2action_train/bin/python ;;
  *) exit 2 ;;
esac
"$TASK_PYTHON" part2action/scripts/run_pose_token_evaluation.py \
  --mode "$MODE" --out-dir "${P2A_TOKEN_STUDY:?Set prepared study directory}" \
  --index "${SLURM_ARRAY_TASK_ID:-0}"
