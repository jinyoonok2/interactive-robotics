#!/usr/bin/env bash
# Download selected PartInstruct demo files plus metadata JSONs using curl
# (no hf-cli required).
#
# By default this downloads the original small subset:
#   OBJECTS="scissors pliers" bash download_subset.sh
#
# To download all demos for a custom object subset:
#   OBJECTS="mug bottle scissors" bash download_subset.sh
#
# Prerequisites:
#   1. Accept dataset terms at https://huggingface.co/datasets/SCAI-JHU/PartInstruct
#   2. Write your HF token to ~/.cache/huggingface/token  (or export HF_TOKEN)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_DIR/datasets/PartInstruct}"
OBJECTS="${OBJECTS:-scissors pliers}"
OBJECTS="${OBJECTS//,/ }"
FORCE="${FORCE:-0}"
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
    if [ "$FORCE" != "1" ] && [ -s "$out" ]; then
        echo "[download_subset] Exists, skipping: $out"
        return
    fi
    echo "[download_subset] Downloading $(basename "$out") ..."
    curl -4 -L --connect-timeout 30 --retry 3 --retry-delay 5 \
        -H "Authorization: Bearer $HF_TOKEN" \
        "$url" -o "$out.part" \
        --progress-bar
    mv "$out.part" "$out"
}

# ── Metadata JSONs ─────────────────────────────────────────────────────────────
hf_curl "$BASE_URL/object_meta.json"          "$DATA_DIR/object_meta.json"
hf_curl "$BASE_URL/part_semantic_lexicon.json" "$DATA_DIR/part_semantic_lexicon.json"
hf_curl "$BASE_URL/episodes_meta_train.json"  "$DATA_DIR/episodes_meta_train.json"
hf_curl "$BASE_URL/episodes_meta_test.json"   "$DATA_DIR/episodes_meta_test.json"

# ── Demo HDF5 files ───────────────────────────────────────────────────────────
echo "[download_subset] Objects: $OBJECTS"
for obj in $OBJECTS; do
    hf_curl "$BASE_URL/demos/${obj}.hdf5" "$DATA_DIR/demos/${obj}.hdf5"
done

echo ""
echo "[download_subset] Done. Data at: $DATA_DIR"
ls -lah "$DATA_DIR/demos/"
