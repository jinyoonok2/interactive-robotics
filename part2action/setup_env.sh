#!/usr/bin/env bash
# Create the minimal part2action conda env for training and offline evaluation.
# PartGym simulation rollouts require a separate upstream PartInstruct install.

set -euo pipefail

ENV_NAME="part2action"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup_env] Conda env '$ENV_NAME' already exists. Skipping create."
else
    echo "[setup_env] Creating conda env '$ENV_NAME' (python=3.10)..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[setup_env] Installing PyTorch + vision (CUDA 12.1 wheels)..."
pip install --upgrade pip
pip install "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121

echo "[setup_env] Installing core training deps..."
pip install \
    "numpy<2" \
    "h5py>=3.10" \
    "opencv-python>=4.9" \
    "pillow>=10.2" \
    "tqdm>=4.66" \
    "pyyaml>=6.0" \
    "scikit-image>=0.22" \
    "matplotlib>=3.8" \
    "transformers>=4.41,<5" \
    "sentence-transformers>=2.7" \
    "huggingface_hub>=0.23" \
    "einops>=0.7"

echo "[setup_env] Done. Activate with:  conda activate $ENV_NAME"
