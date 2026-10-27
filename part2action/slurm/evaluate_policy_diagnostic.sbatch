#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=p2a-frozen-diagnostic
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
MODE="${MODE:?Set diagnostic mode}"
RUN_DIR="${RUN_DIR:?Set run directory}"
OUT_DIR="${OUT_DIR:?Set fresh full evaluation directory}"
SMOKE_OUT="${SMOKE_OUT:?Set fresh preflight directory}"
MANIFEST="${MANIFEST:?Set source fingerprint ledger}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PARTGYM_FPS_DEVICE=cpu
export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX=/u/xna8aw/workspace/.micromamba
export MANIFEST
cd "$ROOT"
/p/part2action/envs/part2action_train/bin/python - <<'PY'
import hashlib,json,os
from pathlib import Path
record=json.loads(Path(os.environ['MANIFEST']).read_text())
for path,expected in record['source_sha256'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Diagnostic source changed after submission: '+path)
PY
set +u
eval "$(micromamba shell hook --shell bash)"
micromamba activate partgym_rollout
set -u
export PYTHONPATH="$ROOT/part2action:$ROOT/part2action/scripts"
test -s "$RUN_DIR/last.pt"
test ! -e "$SMOKE_OUT"
test ! -e "$OUT_DIR"
ARGS=(--diagnostic-mode "$MODE" --config "$RUN_DIR/config_source.yaml" --ckpt "$RUN_DIR/last.pt"
 --partinstruct-root "$ROOT/PartInstruct" --data-root /p/part2action/data/PartInstruct
 --model-key hierarchical --obj-classes bottle mug pliers scissors --execute-steps 1
 --device cuda --termination-mode legacy --diagnostics --contact-diagnostics --diagnostic-every 20)
python part2action/scripts/run_policy_diagnostic.py "${ARGS[@]}" \
 --num-episodes 1 --expected-episodes 1 --eval-seeds 42 --max-steps 5 --out-dir "$SMOKE_OUT"
python part2action/scripts/run_policy_diagnostic.py "${ARGS[@]}" \
 --expected-episodes 149 --eval-seeds 42 43 44 --max-steps 120 --record --out-dir "$OUT_DIR"
python part2action/scripts/summarize_contact_motion.py --run-dir "$OUT_DIR"
