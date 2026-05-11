# part2action — Concept Validation

Minimal experiment to test whether **part-to-action supervision beats heatmap-only supervision** on a controlled, single-GPU benchmark.

This is the implementation of the plan in [`PART_TO_ACTION_DIRECTION.md`](../PART_TO_ACTION_DIRECTION.md).

It is **not** a Habitat integration. The goal is fast concept validation on a small PartInstruct subset using scissors + pliers categories.

## Layout

```
part2action/
  README.md                           This file
  setup_env.sh                        Create the part2action conda env
  setup_vastai.sh                     Set up Part2Action on a Vast.ai GPU instance
  download_subset.sh                  Download scissors.hdf5 + pliers.hdf5 from HF
  configs/
    heatmap_synth.yaml                Synthetic heatmap-only config
    part_action_mlp_synth.yaml        Synthetic part-action MLP config
    heatmap_real.yaml                 Real-data heatmap-only config
    part_action_mlp_real.yaml         Real-data MLP action config
    part_action_diffusion_real.yaml   Real-data diffusion action config
    temporal_part_action_mlp_real.yaml        2-frame temporal + MLP action
    temporal_part_action_diffusion_real.yaml  2-frame temporal + diffusion action
  data/
    partinstruct_loader.py            HDF5 dataset adapter
    targets.py                        Target derivation: contact point, approach dir, action chunk
  models/
    backbone.py                       Frozen DINOv2 + frozen text encoder
    heads.py                          Heatmap / contact / approach / action heads
    part2action_model.py              Top-level model with shared backbone
  scripts/
    train_heatmap.py                  Trains heatmap-only config
    train_part_action.py              Trains part-action MLP config
    evaluate.py                       Computes offline metrics
  docs/
    SETUP.md                          End-to-end setup notes (env, dataset, run order)
  results/                            Experiment outputs (gitignored)
```

## Quick start

```bash
# 1. Create env
bash setup_env.sh

# 2. Download a small subset (HF login required)
bash download_subset.sh

# 3. Train smoke configs
conda activate part2action
python scripts/train_heatmap.py     --config configs/heatmap_synth.yaml
python scripts/train_part_action.py --config configs/part_action_mlp_synth.yaml

# 4. Evaluate offline metrics
python scripts/evaluate.py --config configs/heatmap_synth.yaml --ckpt results/heatmap_synth/last.pt
python scripts/evaluate.py --config configs/part_action_mlp_synth.yaml --ckpt results/part_action_mlp_synth/last.pt
```

See [`docs/SETUP.md`](docs/SETUP.md) for the full setup and install notes.

## Vast.ai setup

On a fresh Vast.ai instance:

```bash
git clone https://github.com/jinyoonok2/interactive-robotics.git
cd interactive-robotics/part2action
HF_TOKEN=hf_... DOWNLOAD_DATA=1 bash setup_vastai.sh
```

For newer GPUs that fail with `no kernel image is available`, use newer
PyTorch CUDA wheels:

```bash
TORCH_INSTALL=nightly-cu128 HF_TOKEN=hf_... DOWNLOAD_DATA=1 bash setup_vastai.sh
```

Then run the real-data tracks:

```bash
conda activate part2action310
bash train_tracks.sh heatmap
bash train_tracks.sh mlp
bash train_tracks.sh diffusion
```
