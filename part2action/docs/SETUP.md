# part2action — Setup and Run Order

This is the end-to-end recipe to reproduce the concept-validation experiment.
It assumes a single workstation with a CUDA GPU (the workstation already
has the `habitat-grasp` env and DINOv2 ViT-S/14 reg4 weights cached at
`~/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth`).

## 1. Create the training env

```bash
cd /home/jinyoon/workspace/interactive-robotics/part2action
bash setup_env.sh
```

This creates a separate `part2action` conda env (Python 3.10, PyTorch
2.4.1+cu121, transformers, h5py, opencv). It does **not** touch your
existing `habitat-grasp` or `uad` envs.

## 2. Smoke test (no PartInstruct download required)

```bash
conda activate part2action
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/make_synthetic_demo.py --out results/synth/bottle.hdf5 --n_demos 4 --steps 24 --img 96
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/make_synthetic_demo.py --out results/synth/kettle.hdf5 --n_demos 4 --steps 24 --img 96 --seed 1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/train_heatmap.py     --config configs/heatmap_synth.yaml --override-out results/_smoke_a
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/train_part_action.py --config configs/part_action_mlp_synth.yaml --override-out results/_smoke_b
```

Expected: training loss decreases for both; checkpoints land in
`results/_smoke_a/last.pt` and `results/_smoke_b/last.pt`. Total wallclock
on CPU is ~10-20s per baseline.

## 3. Download the real PartInstruct subset

The dataset is gated. Accept the terms once at
<https://huggingface.co/datasets/SCAI-JHU/PartInstruct>, then:

```bash
huggingface-cli login         # one-time
bash download_subset.sh       # scissors.hdf5 + pliers.hdf5 + metadata jsons
```

Output goes to
`/home/jinyoon/workspace/interactive-robotics/datasets/PartInstruct/`.
This subset is much smaller than the full 83 GB dataset
(scissors + pliers only, about 3.8 GB).

## 4. Real-data configs

Use the real-data configs directly:

- `configs/heatmap_real.yaml`
- `configs/part_action_mlp_real.yaml`
- `configs/part_action_diffusion_real.yaml`
- `configs/temporal_part_action_mlp_real.yaml`
- `configs/temporal_part_action_diffusion_real.yaml`

## 5. Train tracks

```bash
conda activate part2action
python scripts/train.py --config configs/heatmap_real.yaml
python scripts/train.py --config configs/part_action_mlp_real.yaml
python scripts/train.py --config configs/part_action_diffusion_real.yaml
python scripts/train.py --config configs/temporal_part_action_mlp_real.yaml
python scripts/train.py --config configs/temporal_part_action_diffusion_real.yaml
```

## 6. Offline evaluation (always available)

```bash
python scripts/evaluate.py --config configs/heatmap_real.yaml --ckpt results/heatmap_real/last.pt
python scripts/evaluate.py --config configs/part_action_mlp_real.yaml --ckpt results/part_action_mlp_real/last.pt
```

Reports:
- `iou`           Part heatmap IoU.
- `contact_l1`    Normalized contact-point L1 (part-action tracks only).
- `approach_cos`  Approach-direction cosine similarity (part-action tracks only).
- `action_l1`     Action-chunk smooth-L1 (part-action tracks only).

The first decision rule is: **`part_action_mlp_real` should beat
`heatmap_real` on task success**, even if heatmap IoU is similar.

## 7. Optional: PartGym sim rollouts

Sim rollouts use the upstream PartInstruct package, which has heavy
dependencies (pytorch3d, sam2, custom diffusion-policy fork). It runs
in its own env to keep `part2action` lightweight. This repo no longer
ships a PartInstruct setup script; install upstream PartInstruct manually
only when you are ready to run simulator rollouts.

```bash
conda activate partinstruct
python /home/jinyoon/workspace/interactive-robotics/part2action/scripts/evaluate.py \
    --config configs/part_action_mlp_real.yaml \
    --ckpt   results/part_action_mlp_real/last.pt \
    --use_partgym \
    --rollout_episodes 20 \
    --rollout_splits test1
```

`scripts/evaluate.py::maybe_partgym_rollout` is a scaffold. Wiring the
trained head to PartGym's action runner requires:
1. Mapping our 7-DoF action chunk (pos delta + axis-angle + gripper) to
   PartGym's `bullet_env` step interface.
2. Selecting hard-subset tasks from PartInstruct's `episodes_meta_*.json`
   that exercise the lid/handle distinction.
3. Computing task success from `bullet_env.compute_reward(...)` per
   episode.

The structure is in place; the integration code is intentionally left
as the next concrete deliverable.

## 8. Decision step

Compare offline metrics first; if `part_action_mlp_real`'s task-success delta in
PartGym over `heatmap_real` is clearly positive (>=10 percentage points), proceed to the
Habitat integration phase. Otherwise revise [PROPOSAL.md](../../PROPOSAL.md).

## Troubleshooting

- **CUDA OOM**: drop `batch_size` to 4, set `model.img_size: 196`,
  or set `model.num_fusion_layers: 1`.
- **DINOv2 download attempt**: the backbone falls back to torch.hub if
  the local cache is missing. To force offline, set
  `DINOV2_CKPT=/home/jinyoon/.cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth`.
- **HF model download attempt**: set
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` if Flan-T5-base is already
  cached at `~/.cache/huggingface/hub/`.
- **HDF5 schema mismatch**: print the keys with
  `python -c "import h5py; f=h5py.File('demos/bottle.hdf5','r'); print(list(f['data/demo_0/obs'].keys()))"`
  and update `rgb_key` / `mask_key` in `configs/*.yaml` if they differ.
