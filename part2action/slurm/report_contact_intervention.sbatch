#!/usr/bin/env bash
# Submit with an afterok dependency on all three intervention evaluations.
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --job-name=p2a-contact-comparison
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT=/u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$ROOT/part2action:$ROOT/part2action/scripts"
cd "$ROOT"
/p/part2action/envs/partgym_rollout/bin/python "part2action/scripts/${REPORT_SCRIPT:-build_contact_intervention_report.py}" \
    --source-dir "${SOURCE_DIR:?Set SOURCE_DIR to the group containing normal/reduced/zero}"
