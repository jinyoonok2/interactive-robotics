#!/usr/bin/env bash
# Launch part2action training tracks with descriptive names.
#
# Examples:
#   bash train_tracks.sh heatmap
#   bash train_tracks.sh mlp
#   bash train_tracks.sh diffusion
#   bash train_tracks.sh temporal-mlp
#   bash train_tracks.sh temporal-diffusion
#   bash train_tracks.sh all

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage:
  bash train_tracks.sh <track>

Tracks:
  heatmap              Heatmap only, single frame
  mlp                  Heatmap + contact + approach + MLP action, single frame
  diffusion            Heatmap + contact + approach + diffusion action, single frame
  temporal-mlp         2-frame temporal + MLP action
  temporal-diffusion   2-frame temporal + diffusion action
  all                  Run all tracks sequentially

Optional:
  PYTHON_BIN=/path/to/python bash train_tracks.sh mlp
  RUN_ID=my_run bash train_tracks.sh all
  RESULTS_ROOT=part2action/results/runs bash train_tracks.sh all

Each run writes a terminal log to the track result folder:
  results/runs/<run_id>/<track_name>/train_<track>_<timestamp>.log
EOF
}

config_for_track() {
    case "$1" in
        heatmap) echo "configs/heatmap_real.yaml" ;;
        mlp) echo "configs/part_action_mlp_real.yaml" ;;
        diffusion) echo "configs/part_action_diffusion_real.yaml" ;;
        temporal-mlp) echo "configs/temporal_part_action_mlp_real.yaml" ;;
        temporal-diffusion) echo "configs/temporal_part_action_diffusion_real.yaml" ;;
        *)
            echo "[train_tracks] Unknown track: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

run_track() {
    local track="$1"
    local cfg
    local track_name
    local out_dir
    local log_path
    local timestamp
    cfg="$(config_for_track "$track")"
    track_name="$(
        cd "$REPO_DIR"
        "$PYTHON_BIN" - "part2action/$cfg" <<'PY'
import sys
from pathlib import Path

import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r") as f:
    cfg = yaml.safe_load(f)
out = Path(cfg["output_dir"])
print(out.name)
PY
    )"
    out_dir="$RESULTS_ROOT/$RUN_ID/$track_name"
    timestamp="$(date +%Y%m%d_%H%M%S)"
    log_path="$REPO_DIR/$out_dir/train_${track}_${timestamp}.log"
    mkdir -p "$REPO_DIR/$out_dir"
    echo ""
    echo "[train_tracks] Starting '$track' with $cfg"
    echo "[train_tracks] Log: $log_path"
    (
        cd "$REPO_DIR"
        PYTHONPATH=part2action "$PYTHON_BIN" part2action/scripts/train.py \
            --config "part2action/$cfg" \
            --override-out "$REPO_DIR/$out_dir" 2>&1 | tee "$log_path"
    )
    echo "[train_tracks] Finished '$track'"
}

main() {
    if [ "$#" -ne 1 ]; then
        usage
        exit 2
    fi

    RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
    RESULTS_ROOT="${RESULTS_ROOT:-part2action/results/runs}"
    echo "[train_tracks] RUN_ID=$RUN_ID"
    echo "[train_tracks] RESULTS_ROOT=$RESULTS_ROOT"

    case "$1" in
        all)
            run_track heatmap
            run_track mlp
            run_track diffusion
            run_track temporal-mlp
            run_track temporal-diffusion
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            run_track "$1"
            ;;
    esac
}

main "$@"
