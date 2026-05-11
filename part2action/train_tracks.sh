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
    cfg="$(config_for_track "$track")"
    echo ""
    echo "[train_tracks] Starting '$track' with $cfg"
    (
        cd "$REPO_DIR"
        PYTHONPATH=part2action "$PYTHON_BIN" part2action/scripts/train.py \
            --config "part2action/$cfg"
    )
    echo "[train_tracks] Finished '$track'"
}

main() {
    if [ "$#" -ne 1 ]; then
        usage
        exit 2
    fi

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
