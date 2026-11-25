#!/usr/bin/env bash
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts:PartInstruct
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PARTGYM_FPS_DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
source part2action/slurm/project_storage_env.sh
P2A_V3_PHASE_STUDY="$(p2a_require_project_results "${P2A_V3_PHASE_STUDY:?Set phase-diagnostic directory}")"
P2A_V3_TRAIN_STUDY="$(p2a_require_project_results "${P2A_V3_TRAIN_STUDY:?Set v3 training study}")"
MODE=${MODE:?Set prepare, evaluate, or report}
case "$MODE" in
  evaluate) TASK_PYTHON=/p/part2action/envs/partgym_rollout/bin/python ;;
  prepare|report) TASK_PYTHON=/p/part2action/envs/part2action_train/bin/python ;;
  *) exit 2 ;;
esac
extra=()
if [[ "$MODE" == prepare ]]; then
  if [[ -n "${P2A_V3_CHECKPOINT_EPOCH:-}" ]]; then
    extra+=(--checkpoint-epoch "$P2A_V3_CHECKPOINT_EPOCH")
  fi
  if [[ "${P2A_V3_INTERRUPTED_DIAGNOSTIC:-0}" == 1 ]]; then
    extra+=(--allow-interrupted-training)
  fi
  if [[ -n "${P2A_V3_SOURCE_CHANGE_REVIEW:-}" ]]; then
    extra+=(--source-change-review "$P2A_V3_SOURCE_CHANGE_REVIEW")
  fi
fi
"$TASK_PYTHON" -B part2action/scripts/run_pose_token_v3_phase_diagnostic.py \
  --mode "$MODE" --out-dir "${P2A_V3_PHASE_STUDY:?Set phase-diagnostic directory}" \
  --training-study "${P2A_V3_TRAIN_STUDY:?Set v3 training study}" \
  --reference-plan "${P2A_V3_REFERENCE_PLAN:?Set prior development plan}" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" "${extra[@]}"
