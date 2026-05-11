#!/usr/bin/env bash
# Download scissors + pliers demo files from PartInstruct (smallest two categories,
# ~4 GB total) plus the metadata JSONs, using curl (no hf-cli required).
#
# Prerequisites:
#   1. Accept dataset terms at https://huggingface.co/datasets/SCAI-JHU/PartInstruct
#   2. Write your HF token to ~/.cache/huggingface/token  (or export HF_TOKEN)

set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/workspace/interactive-robotics/datasets/PartInstruct}"
mkdir -p "$DATA_DIR/demos"

# ── Resolve token ──────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    TOKEN_FILE="$HOME/.cache/huggingface/token"
    if [ -f "$TOKEN_FILE" ]; then
        HF_TOKEN="$(cat "$TOKEN_FILE")"
        export HF_TOKEN
        echo "[download_subset] Loaded HF token from $TOKEN_FILE"
    else
        echo "[download_subset] ERROR: No HF token found. Set HF_TOKEN or write token to $TOKEN_FILE"
        exit 1
    fi
fi

BASE_URL="https://huggingface.co/datasets/SCAI-JHU/PartInstruct/resolve/main"

hf_curl() {
    local url="$1"
    local out="$2"
    echo "[download_subset] Downloading $(basename "$out") ..."
    curl -L --connect-timeout 30 --retry 3 --retry-delay 5 \
        -H "Authorization: Bearer $HF_TOKEN" \
        "$url" -o "$out" \
        --progress-bar
}

# ── Metadata JSONs ─────────────────────────────────────────────────────────────
hf_curl "$BASE_URL/object_meta.json"          "$DATA_DIR/object_meta.json"
hf_curl "$BASE_URL/part_semantic_lexicon.json" "$DATA_DIR/part_semantic_lexicon.json"
hf_curl "$BASE_URL/episodes_meta_train.json"  "$DATA_DIR/episodes_meta_train.json"
hf_curl "$BASE_URL/episodes_meta_test.json"   "$DATA_DIR/episodes_meta_test.json"

# ── Demo HDF5 files (smallest two categories) ─────────────────────────────────
hf_curl "$BASE_URL/demos/scissors.hdf5" "$DATA_DIR/demos/scissors.hdf5"
hf_curl "$BASE_URL/demos/pliers.hdf5"   "$DATA_DIR/demos/pliers.hdf5"

echo ""
echo "[download_subset] Done. Data at: $DATA_DIR"
ls -lah "$DATA_DIR/demos/"
