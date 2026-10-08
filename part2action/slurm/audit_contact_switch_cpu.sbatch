#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=p2a-contact-audit
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
cd /u/xna8aw/workspace/research_projects/interactive-robotics
export PYTHONPATH=part2action
export OMP_NUM_THREADS=2
OUT_DIR="/p/part2action/results/diagnostics/contact_switch_audit_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
python3 part2action/scripts/register_result.py --section diagnostics --target "$OUT_DIR"
/p/part2action/envs/part2action_train/bin/python part2action/scripts/audit_contact_switch.py \
 --data-root /u/xna8aw/workspace/research_projects/interactive-robotics/datasets/PartInstruct/demos \
 --out "$OUT_DIR" --max-demos 32
