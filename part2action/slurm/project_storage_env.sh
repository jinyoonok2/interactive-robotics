#!/usr/bin/env bash
# Source from maintained Part2Action jobs after setting PYTHONPATH.
# This does not change HOME or global caches used by other projects.
export PART2ACTION_STORAGE_ROOT="$(readlink -f "${PART2ACTION_STORAGE_ROOT:-/p/part2action}")"
/p/part2action/envs/part2action_train/bin/python -B -c \
  'from diagnostics.storage import project_results_root; project_results_root()'
export HF_HOME="$PART2ACTION_STORAGE_ROOT/caches/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE
export TORCH_HOME="$PART2ACTION_STORAGE_ROOT/caches/torch"
export XDG_CACHE_HOME="$PART2ACTION_STORAGE_ROOT/caches/xdg"
export DINOV2_HUB_DIR="$TORCH_HOME/hub/facebookresearch_dinov2_main"
export DINOV2_CKPT="$TORCH_HOME/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
export MPLCONFIGDIR="$PART2ACTION_STORAGE_ROOT/caches/matplotlib"
export CUDA_CACHE_PATH="$PART2ACTION_STORAGE_ROOT/caches/cuda"
export TRITON_CACHE_DIR="$PART2ACTION_STORAGE_ROOT/caches/triton"
export TORCHINDUCTOR_CACHE_DIR="$PART2ACTION_STORAGE_ROOT/caches/torchinductor"
export NUMBA_CACHE_DIR="$PART2ACTION_STORAGE_ROOT/caches/numba"
mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
  "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR"

p2a_require_project_results() {
  /p/part2action/envs/part2action_train/bin/python -B -m diagnostics.storage \
    --check "$1" --check-slurm-logs
}
