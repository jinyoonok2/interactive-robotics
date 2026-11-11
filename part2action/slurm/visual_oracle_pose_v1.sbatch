#!/usr/bin/env bash
# Resources are explicit in the versioned submission ledger.
#SBATCH --no-requeue
set -euo pipefail
TASK_ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
STUDY="${P2A_VISUAL_POSE_STUDY:?Set the prepared study directory}"
MODE="${MODE:?Set stage}"
export PYTHONPATH="$TASK_ROOT/part2action:$TASK_ROOT/part2action/scripts:$TASK_ROOT/PartInstruct"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PARTGYM_FPS_DEVICE=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/p2a_visual_pose_mpl
case "$MODE" in
  collect_oracle|evaluate) TASK_PYTHON=/p/part2action/envs/partgym_rollout/bin/python ;;
  *) TASK_PYTHON=/p/part2action/envs/part2action_train/bin/python ;;
esac
cd "$TASK_ROOT"
"$TASK_PYTHON" part2action/scripts/run_visual_oracle_pose_v1.py \
  --mode "$MODE" --out-dir "$STUDY" --index "${SLURM_ARRAY_TASK_ID:-0}"
