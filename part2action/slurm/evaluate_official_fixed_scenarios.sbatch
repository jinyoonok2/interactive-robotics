#!/usr/bin/env bash
# Scenario/budget-matched privileged reference; native skill scoring is retained.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --job-name=pi-fixed-eval
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
MODEL="${MODEL:?Set DP-S or DP3-S}"
case "$MODEL" in DP-S|DP3-S) ;; *) exit 2;; esac
OUT="${OUT:?Choose a new output directory}"
MANIFEST="${MANIFEST:?Set submission ledger with source fingerprints}"
SCENARIOS_MANIFEST="${SCENARIOS_MANIFEST:?Set original Part2Action scenario manifest}"
export PYTHONPATH="$ROOT/part2action:$ROOT/part2action/scripts/official_partinstruct:$ROOT/PartInstruct:$ROOT/PartInstruct/third_party/diffusion_policy:$ROOT/PartInstruct/third_party/3D-Diffusion-Policy"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PARTGYM_FPS_DEVICE=cpu
export MPLCONFIGDIR=/tmp/p2a_mpl_cache
PYTHON=/p/part2action/envs/partinstruct/bin/python
cd "$ROOT"
export MANIFEST SCENARIOS_MANIFEST
"$PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
record = json.loads(Path(os.environ['MANIFEST']).read_text())
for name, expected in record['source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Source changed after submission: '+name)
if hashlib.sha256(Path(os.environ['SCENARIOS_MANIFEST']).read_bytes()).hexdigest() != record['scenario_manifest_sha256']:
    raise RuntimeError('Scenario manifest changed after submission')
PY
ARGS=(--model "$MODEL"
  --checkpoint "/p/part2action/data/official_checkpoints/PartInstruct/checkpoints/$MODEL/latest.ckpt"
  --out-dir "$OUT" --data-root /p/part2action/data/PartInstruct
  --framework-root "$ROOT/PartInstruct" --scenarios-manifest "$SCENARIOS_MANIFEST"
  --max-steps 250 --execute-steps 8)
if [[ -n "${SCENARIO_INDICES:-}" ]]; then
  read -r -a INDICES <<< "$SCENARIO_INDICES"
  ARGS+=(--scenario-indices "${INDICES[@]}")
fi
if [[ -n "${EVAL_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "$EVAL_SEEDS"
  ARGS+=(--eval-seeds "${SEEDS[@]}")
fi
"$PYTHON" part2action/scripts/official_partinstruct/fixed_scenarios.py "${ARGS[@]}"
