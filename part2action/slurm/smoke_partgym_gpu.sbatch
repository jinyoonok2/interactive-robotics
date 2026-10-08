#!/usr/bin/env bash
# GPU setup/inference smoke check for all four retained Part2Action policies.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100_40gb|a100_80gb|h100_94gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=p2a-gpu-smoke
#SBATCH --output=part2action/logs/slurm/%x-%j.out
#SBATCH --error=part2action/logs/slurm/%x-%j.err
set -euo pipefail
ROOT="/u/xna8aw/workspace/research_projects/interactive-robotics"
OUT="/p/part2action/results/diagnostics/partgym_gpu_smoke_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1 PARTGYM_FPS_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/PartInstruct:$ROOT/part2action:$ROOT/part2action/scripts"
export OMP_NUM_THREADS=4
cd "$ROOT"
PYTHON=/p/part2action/envs/partgym_rollout/bin/python
mkdir -p "$OUT"
"$PYTHON" part2action/scripts/register_result.py --section diagnostics --target "$OUT"
"$PYTHON" - <<'PY'
import torch, time
from PartInstruct.PartGym.env.backend.utils.vision_utils import downsample_pcd
from pytorch3d.ops import sample_farthest_points
from PartInstruct.PartGym.env.backend.planner.bullet_planner import BulletPlanner
assert torch.cuda.is_available(), 'GPU allocation must provide CUDA'
print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0), flush=True)
p = torch.rand(1,20000,3,generator=torch.Generator().manual_seed(42))
t = time.perf_counter()
idx = downsample_pcd(p.cuda(), 1024)
_, expected = sample_farthest_points(p, K=1024, random_start_point=False)
assert idx.is_cuda and torch.equal(idx.cpu(), expected)
print('CUDA input -> CPU FPS -> CUDA indices matches reference; seconds',time.perf_counter()-t,flush=True)
PY
for RUN in world_geometry_30epoch_no_residual_6645296 world_geometry_30epoch_near_contact_6645297 world_geometry_30epoch_unified_control_6687620 world_geometry_30epoch_unified_contact_6687621; do
    RUN_DIR="/p/part2action/results/part2action/runs/$RUN"
    "$PYTHON" part2action/scripts/rollout_partgym.py \
      --partinstruct-root "$ROOT/PartInstruct" --data-root /p/part2action/data/PartInstruct \
      --model-key hierarchical --config "$RUN_DIR/config_source.yaml" --ckpt "$RUN_DIR/last.pt" \
      --obj-classes bottle --task-types 1 --num-episodes 1 --trials-per-task 1 \
      --max-steps 5 --execute-steps 1 --device cuda --termination-mode legacy \
      --diagnostics --diagnostic-every 1 --out-dir "$OUT/$RUN" \
      > "$OUT/${RUN}.log" 2>&1
    echo "SMOKE PASSED $RUN"
done
printf 'passed\n' > "$OUT/SMOKE_PASSED"
