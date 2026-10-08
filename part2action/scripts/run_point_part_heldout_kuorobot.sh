#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/workspace/research_projects/interactive-robotics"
DATA_ROOT="${HOME}/partinstruct_runtime"
RUN_DIR="${ROOT}/results/part2action/runs/world_geometry_point_part_pretrain_6554989"
OUT="${ROOT}/results/part2action/evaluations/point_part_heldout_kuorobot_corrected"
RESULT="${OUT}/hierarchical/rollout_results_test1_bottle-mug-pliers-scissors_1-2-3-4_legacy.json"

test -s "${DATA_ROOT}/episodes_meta_test.json"
test -s "${DATA_ROOT}/assets/urdfs/robots/franka_panda/panda.urdf"
test -s "${RUN_DIR}/config_source.yaml"
test -s "${RUN_DIR}/last.pt"

cd "${ROOT}"
export PATH="/u/xna8aw/workspace/.bin:${PATH}"
export MAMBA_ROOT_PREFIX="/u/xna8aw/workspace/.micromamba"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate partgym_rollout
set -u
mkdir -p "${OUT}"

PYTHONPATH=part2action:part2action/scripts \
python part2action/scripts/rollout_partgym.py \
  --model-key hierarchical \
  --config "${RUN_DIR}/config_source.yaml" \
  --ckpt "${RUN_DIR}/last.pt" \
  --data-root "${DATA_ROOT}" \
  --obj-classes bottle mug pliers scissors \
  --task-types 1 2 3 4 \
  --num-episodes 16 \
  --trials-per-task 3 \
  --max-steps 1 \
  --execute-steps 1 \
  --device cuda \
  --termination-mode legacy \
  --out-dir "${OUT}" \
  --diagnostics \
  2>&1 | tee "${OUT}.log"

PYTHONPATH=part2action \
python part2action/scripts/analyze_point_part_rollouts.py \
  --input "${RESULT}" \
  --out "${OUT}/point_part_analysis.json"
