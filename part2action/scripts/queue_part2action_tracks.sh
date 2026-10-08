#!/usr/bin/env bash
# Queue one Slurm job per part2action training track.
#
# Examples:
#   bash scripts/queue_part2action_tracks.sh
#   bash scripts/queue_part2action_tracks.sh all-heads-mlp no-heatmap-mlp
#   DRY_RUN=1 bash scripts/queue_part2action_tracks.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
SBATCH_SCRIPT="$ROOT_DIR/slurm/train_part2action.sbatch"

DEFAULT_TRACKS=(
  heatmap
  all-heads-mlp
  all-heads-diffusion
  no-heatmap-mlp
  no-heatmap-diffusion
)

if [[ "$#" -gt 0 ]]; then
  TRACKS=("$@")
else
  TRACKS=("${DEFAULT_TRACKS[@]}")
fi

QUEUE_TAG="${QUEUE_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

echo "[queue] repo=$REPO_DIR"
echo "[queue] tag=$QUEUE_TAG"
echo "[queue] tracks=${TRACKS[*]}"

cd "$REPO_DIR"

for track in "${TRACKS[@]}"; do
  time_tag="${QUEUE_TAG}_${track//-/_}"
  job_name="p2a-${track}"
  cmd=(
    sbatch
    --job-name="$job_name"
    --export="ALL,TRACK=$track,TIME_TAG=$time_tag"
    "$SBATCH_SCRIPT"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[queue] dry-run: ${cmd[*]}"
  else
    echo "[queue] submitting job=$job_name track=$track time_tag=$time_tag"
    "${cmd[@]}"
  fi
done
