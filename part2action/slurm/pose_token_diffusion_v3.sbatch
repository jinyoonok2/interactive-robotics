#!/usr/bin/env bash
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action:part2action/scripts
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
source part2action/slurm/project_storage_env.sh
STUDY=${STUDY:?Set the prepared v3 study directory}
STUDY="$(p2a_require_project_results "$STUDY")"
MODE=${MODE:?Set smoke or train}
export P2A_JOB_HEALTH_DIR="$STUDY/health/${MODE}_${SLURM_JOB_ID:-manual}"
trap 'p2a_status=$?; /p/part2action/envs/part2action_train/bin/python -B -m diagnostics.run_health --shell-exit "$p2a_status" --directory "$P2A_JOB_HEALTH_DIR" || true; exit "$p2a_status"' EXIT
case "$MODE" in
  smoke) extra=(--smoke) ;;
  train) extra=(--smoke-receipt "$STUDY/smoke/smoke.json") ;;
  resume) extra=(--smoke-receipt "$STUDY/smoke/smoke.json" --resume "${RESUME_STATE:?Set the same-run training state}") ;;
  *) exit 2 ;;
esac
if [[ "$MODE" == resume ]]; then MODE=train; fi
/p/part2action/envs/part2action_train/bin/python -B part2action/scripts/train_pose_token_diffusion_v3.py \
  --config "$STUDY/pilot_resolved.yaml" --out-dir "$STUDY/$MODE" "${extra[@]}"
