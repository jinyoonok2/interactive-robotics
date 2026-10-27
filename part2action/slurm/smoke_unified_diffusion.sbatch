#!/usr/bin/env bash
# Real-data training/backward/sampling preflight; does not launch full training.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=p2a-diffusion-smoke
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
OUT="/p/part2action/results/diagnostics/unified_diffusion_smoke_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/part2action:$ROOT/part2action/scripts"
PYTHON=/p/part2action/envs/part2action_train/bin/python
cd "$ROOT"
mkdir -p "$OUT"
"$PYTHON" part2action/scripts/register_result.py --section diagnostics --target "$OUT"
"$PYTHON" part2action/scripts/smoke_unified_diffusion.py --out-dir "$OUT"
