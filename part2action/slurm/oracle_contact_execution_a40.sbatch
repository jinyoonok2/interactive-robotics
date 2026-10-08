#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=p2a-oracle-contact
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err

set -euo pipefail
ROOT="/u/xna8aw/workspace/research_projects/interactive-robotics"
OUT="${OUT:?Set OUT to the diagnostic JSON path}"
NUM_EPISODES="${NUM_EPISODES:-16}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-3}"
OBJ_CLASSES="${OBJ_CLASSES:-bottle mug pliers scissors}"
TASK_TYPES="${TASK_TYPES:-1 2 3 4}"

export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX="/u/xna8aw/workspace/.micromamba"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"
mkdir -p "part2action/logs/slurm" "$(dirname "$OUT")"

set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate partgym_rollout
set -u

read -r -a OBJ_ARGS <<< "$OBJ_CLASSES"
read -r -a TASK_ARGS <<< "$TASK_TYPES"
PYTHONPATH=part2action:part2action/scripts \
python part2action/scripts/oracle_contact_execution.py \
  --obj-classes "${OBJ_ARGS[@]}" \
  --task-types "${TASK_ARGS[@]}" \
  --num-episodes "$NUM_EPISODES" \
  --trials-per-task "$TRIALS_PER_TASK" \
  --out "$OUT"
