#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=pi-official-eval
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
MODEL="${MODEL:?Set MODEL to DP-S or DP3-S}"
case "$MODEL" in DP-S|DP3-S) ;; *) exit 2;; esac
CKPT="/p/part2action/data/official_checkpoints/PartInstruct/checkpoints/$MODEL/latest.ckpt"
OUT="${OUT:?Set OUT to the evaluation output directory}"
export PYTHONPATH="$ROOT/PartInstruct:$ROOT/PartInstruct/third_party/diffusion_policy:$ROOT/PartInstruct/third_party/3D-Diffusion-Policy"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PARTGYM_FPS_DEVICE=cpu HYDRA_FULL_ERROR=1
PYTHON=/p/part2action/envs/partinstruct/bin/python
cd "$ROOT"
test -s "$CKPT"
ARGS=(--model "$MODEL" --checkpoint "$CKPT" --data-root /p/part2action/data/PartInstruct --framework-root "$ROOT/PartInstruct")
# First run strict checkpoint loading and one native scissors rollout.
"$PYTHON" part2action/scripts/official_partinstruct/evaluate.py "${ARGS[@]}" --out-dir "$OUT/smoke" --smoke
# Continue only after the smoke completes successfully.
"$PYTHON" part2action/scripts/official_partinstruct/evaluate.py "${ARGS[@]}" --out-dir "$OUT/full"
