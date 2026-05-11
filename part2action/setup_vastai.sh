#!/usr/bin/env bash
# Set up Part2Action on a fresh Vast.ai GPU instance.
#
# This installs only the lightweight Part2Action training stack, not the
# upstream PartInstruct/PartGym simulator environment.
#
# Typical use:
#   git clone https://github.com/jinyoonok2/interactive-robotics.git
#   cd interactive-robotics/part2action
#   HF_TOKEN=hf_... DOWNLOAD_DATA=1 bash setup_vastai.sh
#
# Options:
#   ENV_NAME=part2action          Conda/micromamba env name
#   DOWNLOAD_DATA=1              Download scissors + pliers PartInstruct subset
#   DATA_DIR=/workspace/datasets/PartInstruct
#   CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-part2action}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/datasets/PartInstruct}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"

echo "[setup_vastai] Repo: $REPO_DIR"
echo "[setup_vastai] Env:  $ENV_NAME"
echo "[setup_vastai] Data: $DATA_DIR"

create_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "[setup_vastai] Conda env '$ENV_NAME' already exists."
    else
        echo "[setup_vastai] Creating conda env '$ENV_NAME'..."
        conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi
    conda activate "$ENV_NAME"
}

create_micromamba_env() {
    eval "$(micromamba shell hook --shell bash)"
    if micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "[setup_vastai] Micromamba env '$ENV_NAME' already exists."
    else
        echo "[setup_vastai] Creating micromamba env '$ENV_NAME'..."
        micromamba create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" -c conda-forge
    fi
    micromamba activate "$ENV_NAME"
}

create_venv() {
    echo "[setup_vastai] Conda/micromamba not found; creating .venv instead."
    python3 -m venv "$REPO_DIR/.venv-part2action"
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv-part2action/bin/activate"
}

if command -v conda >/dev/null 2>&1; then
    create_conda_env
elif command -v micromamba >/dev/null 2>&1; then
    create_micromamba_env
else
    create_venv
fi

python -m pip install --upgrade pip

echo "[setup_vastai] Installing PyTorch CUDA wheels..."
python -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url "$CUDA_INDEX_URL"

echo "[setup_vastai] Installing Part2Action dependencies..."
python -m pip install \
    "numpy<2" \
    "h5py>=3.10" \
    "opencv-python>=4.9" \
    "pillow>=10.2" \
    "tqdm>=4.66" \
    "pyyaml>=6.0" \
    "scikit-image>=0.22" \
    "matplotlib>=3.8" \
    "transformers>=4.41" \
    "sentence-transformers>=2.7" \
    "huggingface_hub>=0.23" \
    "einops>=0.7"

mkdir -p "$DATA_DIR/demos" "$ROOT_DIR/results" "$HOME/.cache/huggingface" "$HOME/.cache/torch"

if [ -n "${HF_TOKEN:-}" ]; then
    printf "%s" "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
    chmod 600 "$HOME/.cache/huggingface/token"
    echo "[setup_vastai] Saved HF token to ~/.cache/huggingface/token"
fi

if [ "$DOWNLOAD_DATA" = "1" ]; then
    echo "[setup_vastai] Downloading PartInstruct subset..."
    DATA_DIR="$DATA_DIR" bash "$ROOT_DIR/download_subset.sh"
else
    echo "[setup_vastai] Skipping dataset download. Set DOWNLOAD_DATA=1 to download now."
fi

echo ""
echo "[setup_vastai] Setup complete."
echo ""
echo "Activate later with:"
if command -v conda >/dev/null 2>&1; then
    echo "  conda activate $ENV_NAME"
elif command -v micromamba >/dev/null 2>&1; then
    echo "  micromamba activate $ENV_NAME"
else
    echo "  source $REPO_DIR/.venv-part2action/bin/activate"
fi
echo ""
echo "Run first three experiments sequentially:"
echo "  cd $REPO_DIR"
echo "  PYTHONPATH=part2action python part2action/scripts/train.py --config part2action/configs/heatmap_real.yaml"
echo "  PYTHONPATH=part2action python part2action/scripts/train.py --config part2action/configs/part_action_mlp_real.yaml"
echo "  PYTHONPATH=part2action python part2action/scripts/train.py --config part2action/configs/part_action_diffusion_real.yaml"
echo ""
echo "Or use the helper:"
echo "  cd $ROOT_DIR"
echo "  bash train_tracks.sh heatmap"
echo "  bash train_tracks.sh mlp"
echo "  bash train_tracks.sh diffusion"
