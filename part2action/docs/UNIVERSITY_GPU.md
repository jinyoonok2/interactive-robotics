# University GPU Workflow

This guide is for running `part2action` on a university GPU server over SSH.
It is written for a new reader or coding agent who did not follow the original
chat history.

## Goal

We are training a lightweight Part2Action model:

```text
RGB observation + part-level instruction -> future robot action chunk
```

The model uses frozen DINOv2 and Flan-T5 backbones, then trains only the fusion
layers and prediction heads. The main comparison is:

- `heatmap_real`: predicts the task-relevant part region.
- `part_action_mlp_real`: predicts the part-conditioned action chunk.

First validate offline training/evaluation. Only run PartGym simulator rollouts
after the offline tracks are working.

## Important Server Rule

Do not run training on a login/portal node unless your cluster explicitly allows
it. First get a GPU allocation using your university scheduler, then run setup
and training inside that allocation.

Examples vary by cluster:

```bash
# Example only. Use the command required by your university cluster.
srun --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 --pty bash
```

After you land on a GPU node, verify:

```bash
hostname
nvidia-smi
```

## Recommended SSH Session

Use `tmux` or `screen` so downloads/training survive SSH disconnects:

```bash
tmux new -s part2action
```

Inside the session:

```bash
cd ~/workspace/live-robotics-lab/part2action
```

If your clone uses a different directory name, use that path instead.

## Environment

Create or reuse the lightweight training environment:

```bash
bash setup_env.sh
conda activate part2action
```

Expected environment:

```text
conda env: part2action
Python: 3.10
PyTorch: 2.4.1+cu121
torchvision: 0.19.1+cu121
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Hugging Face Access

PartInstruct is gated. Accept the dataset terms in a browser:

```text
https://huggingface.co/datasets/SCAI-JHU/PartInstruct
```

Save a read token on the server:

```bash
mkdir -p ~/.cache/huggingface
printf '%s' 'YOUR_HF_TOKEN_HERE' > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

Do not commit or paste the token into tracked files. If the token is exposed,
revoke it and create a new one.

`download_subset.sh` uses `curl -4` internally to force IPv4, which avoids the
Hugging Face IPv6 hang observed on some machines.

## Data Subset

Start with a small selected-object subset, not the full 83 GB dataset:

```bash
OBJECTS="scissors pliers" bash download_subset.sh
```

For a broader subset:

```bash
OBJECTS="mug scissors pliers" bash download_subset.sh
python scripts/make_object_configs.py --objects mug scissors pliers --tag three_objects
```

Large objects such as `mug` can be many GB. If a download is interrupted, remove
the partial file before retrying:

```bash
rm -f ../datasets/PartInstruct/demos/mug.hdf5.part
```

See [`DATA_SUBSETS.md`](DATA_SUBSETS.md) for the full selected-object workflow.

## Smoke Test

Run the synthetic smoke test before real training:

```bash
conda activate part2action
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/make_synthetic_demo.py --out results/synth/bottle.hdf5 --n_demos 4 --steps 24 --img 96
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/make_synthetic_demo.py --out results/synth/kettle.hdf5 --n_demos 4 --steps 24 --img 96 --seed 1

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/train_heatmap.py --config configs/heatmap_synth.yaml --override-out results/_smoke_a
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/train_part_action.py --config configs/part_action_mlp_synth.yaml --override-out results/_smoke_b
```

Expected output:

```text
results/_smoke_a/last.pt
results/_smoke_b/last.pt
```

The synthetic files and smoke checkpoints are disposable.

## Train Selected Objects

If you generated configs with `--tag three_objects`, run:

```bash
python scripts/train.py --config configs/generated/heatmap_real_three_objects.yaml
python scripts/train.py --config configs/generated/part_action_mlp_real_three_objects.yaml
```

Outputs go under `results/`, namespaced by the generated config output path.

## Agent Notes

If a coding agent is operating on this SSH server:

- Read `part2action/README.md`, `docs/SETUP.md`, this file, and
  `docs/DATA_SUBSETS.md` before changing code.
- Never print, commit, or copy Hugging Face tokens.
- Keep large data/checkpoints under ignored directories such as
  `datasets/PartInstruct/` and `part2action/results/`.
- Prefer `tmux` for long downloads/training runs.
- Confirm `nvidia-smi` on a GPU node before starting training.
- Do not assume `/home/jinyoon/...` paths on the server; use the actual clone
  path, for example `~/workspace/live-robotics-lab/part2action`.
