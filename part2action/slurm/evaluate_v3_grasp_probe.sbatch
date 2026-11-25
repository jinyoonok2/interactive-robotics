#!/usr/bin/env bash
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts:PartInstruct
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PARTGYM_FPS_DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
source part2action/slurm/project_storage_env.sh
PROBE_STUDY="$(p2a_require_project_results "${PROBE_STUDY:?Set prepared probe directory}")"
case "${MODE:?Set reference, policy or report}" in
  reference) PROBE_PYTHON=/p/part2action/envs/partgym_rollout/bin/python; export CUDA_VISIBLE_DEVICES="" ;;
  policy) PROBE_PYTHON=/p/part2action/envs/partgym_rollout/bin/python ;;
  report) PROBE_PYTHON=/p/part2action/envs/part2action_train/bin/python ;;
  *) exit 2 ;;
esac
"$PROBE_PYTHON" -B part2action/scripts/run_v3_grasp_probe.py \
  --mode "$MODE" --out-dir "$PROBE_STUDY" --index "${PROBE_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
