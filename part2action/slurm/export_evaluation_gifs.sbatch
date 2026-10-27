#!/usr/bin/env bash
# CPU export after all matched GPU evaluations complete successfully.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=p2a-gif-review
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/part2action:$ROOT/part2action/scripts"
export COMPARISON_REQUEST="${COMPARISON_REQUEST:?Set COMPARISON_REQUEST to the comparison request JSON}"
/p/part2action/envs/partgym_rollout/bin/python - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path
request = json.loads(Path(os.environ['COMPARISON_REQUEST']).read_text())
command = [sys.executable, 'part2action/scripts/export_evaluation_gifs.py',
           '--out-dir', request['out_dir'], '--speed', str(request['speed']), '--fps', str(request['fps'])]
if request.get('data_root'):
    command.extend(['--data-root',request['data_root']])
for source in request['sources']:
    command.extend(['--source', source])
subprocess.run(command, check=True)
if request.get('quantitative_report', False):
    command = ['/p/part2action/envs/part2action_train/bin/python',
               'part2action/scripts/compare_rollout_metrics.py',
               '--out-dir', str(Path(request['out_dir']) / 'quantitative_report'),
               '--data-root', request['data_root']]
    for source in request['sources']:
        command.extend(['--source', source])
    subprocess.run(command, check=True)
PY
