#!/usr/bin/env bash
# Versioned correction to evaluation adapter only; V1 models/training retained.
#SBATCH --no-requeue
set -euo pipefail
TASK_ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
STUDY="${P2A_VISUAL_POSE_STUDY:?Set prepared study directory}"
export PYTHONPATH="$TASK_ROOT/part2action:$TASK_ROOT/part2action/scripts:$TASK_ROOT/PartInstruct"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PARTGYM_FPS_DEVICE=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/p2a_visual_pose_mpl
case "${MODE:?Set evaluate or report}" in
  evaluate) TASK_PYTHON=/p/part2action/envs/partgym_rollout/bin/python ;;
  report) TASK_PYTHON=/p/part2action/envs/part2action_train/bin/python ;;
  *) exit 2 ;;
esac
cd "$TASK_ROOT"
"$TASK_PYTHON" part2action/scripts/evaluate_visual_oracle_pose_v2.py \
  --mode "$MODE" --out-dir "$STUDY" --index "${SLURM_ARRAY_TASK_ID:-0}"
