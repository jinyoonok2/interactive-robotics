# UVA Slurm Setup

This guide is the repeatable setup for running `part2action` on the UVA Slurm
cluster. It assumes the repo is cloned at:

```text
/u/xna8aw/workspace/live-robotics-lab/interactive-robotics
```

If your clone is elsewhere, replace that path in the commands below. Do not run
training directly on `portal*` login nodes; submit GPU work through Slurm.

## 1. Enter The Project

```bash
cd /u/xna8aw/workspace/live-robotics-lab/interactive-robotics/part2action
```

## 2. Bootstrap Micromamba

The UVA login environment may not have `conda`, and system Python may not have
`venv`. Use a local micromamba install under the workspace:

```bash
mkdir -p /u/xna8aw/workspace/.bin /u/xna8aw/workspace/.micromamba /u/xna8aw/workspace/.tmp-micromamba
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /u/xna8aw/workspace/.tmp-micromamba bin/micromamba
cp /u/xna8aw/workspace/.tmp-micromamba/bin/micromamba /u/xna8aw/workspace/.bin/micromamba
chmod +x /u/xna8aw/workspace/.bin/micromamba

export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX="/u/xna8aw/workspace/.micromamba"
```

For interactive shell activation:

```bash
eval "$(micromamba shell hook --shell bash)"
```

## 3. Offline Training Env

Create the lightweight `part2action310` env for HDF5 training and offline
metrics:

```bash
DOWNLOAD_DATA=0 bash setup_training_env.sh
```

Verify it:

```bash
micromamba run -n part2action310 python - <<'PY'
import torch, torchvision, transformers, h5py
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda build", torch.version.cuda)
print("transformers", transformers.__version__)
print("h5py", h5py.__version__)
PY
```

CUDA will be `False` on login nodes. It should become `True` inside a Slurm GPU
job.

## 4. Hugging Face Access

Accept the gated dataset terms in a browser:

```text
https://huggingface.co/datasets/SCAI-JHU/PartInstruct
```

Then authenticate on the server. Either use:

```bash
micromamba run -n part2action310 huggingface-cli login
```

or save a read token directly:

```bash
mkdir -p ~/.cache/huggingface
printf '%s' 'YOUR_HF_TOKEN_HERE' > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
```

Do not paste tokens into chat, terminal transcripts, or tracked files.

## 5. HDF5 Demo Subset

Download the default scissors/pliers subset:

```bash
OBJECTS="scissors pliers" bash download_subset.sh
```

The script now defaults to the clone-relative dataset path:

```text
../datasets/PartInstruct/
```

Expected files:

```text
../datasets/PartInstruct/
  object_meta.json
  part_semantic_lexicon.json
  episodes_meta_train.json
  episodes_meta_test.json
  demos/
    scissors.hdf5
    pliers.hdf5
```

For a custom object subset:

```bash
OBJECTS="mug bottle scissors" bash download_subset.sh
micromamba run -n part2action310 python scripts/make_object_configs.py \
  --objects mug bottle scissors \
  --tag mug_bottle_scissors
```

## 6. PartGym Env And Assets

PartGym uses upstream `PartInstruct` and a separate env named `partinstruct`.

```bash
mkdir -p third_party
git clone --recurse-submodules https://github.com/SCAI-JHU/PartInstruct.git third_party/PartInstruct

micromamba create -y -n partinstruct -c conda-forge \
  python=3.9 cmake=3.24.3 open3d ninja gcc_linux-64=12 gxx_linux-64=12

micromamba run -n partinstruct python -m pip install --upgrade pip
micromamba run -n partinstruct python -m pip install torch torchvision torchaudio

cd third_party/PartInstruct
micromamba run -n partinstruct python -m pip install -r requirements.txt
micromamba run -n partinstruct python -m pip install omegaconf
micromamba run -n partinstruct python -m pip install -e .
micromamba run -n partinstruct python -m pip install -e ./third_party/pybullet_planning/
micromamba run -n partinstruct python -m pip install -e ./third_party/gym-0.21.0/
micromamba run -n partinstruct python -m pip install --no-build-isolation ./third_party/pytorch3d/
cd ../..
```

Download and unpack PartGym assets:

```bash
cd third_party/PartInstruct
micromamba run -n partinstruct huggingface-cli download SCAI-JHU/PartInstruct \
  --repo-type dataset \
  --local-dir ./data \
  --include "*.json" "assets.zip"

unzip ./data/assets.zip -d ./data/
rm ./data/assets.zip
cd ../..
```

Expected paths:

```text
third_party/PartInstruct/data/episodes_meta_test.json
third_party/PartInstruct/data/assets/urdfs/robots/franka_panda/panda.urdf
third_party/PartInstruct/data/assets/partnet-grasping/
```

Verify imports:

```bash
micromamba run -n partinstruct python - <<'PY'
import gym, pybullet, cv2, open3d, omegaconf, pytorch3d
import PartInstruct.PartGym.env.bullet_env
print("PartGym imports OK")
PY
```

## 7. Slurm Batch Training

Use batch jobs for real training instead of holding an interactive allocation.
Submit from the repo root (`interactive-robotics/`):

```bash
cd /u/xna8aw/workspace/live-robotics-lab/interactive-robotics
sbatch part2action/slurm/train_part2action.sbatch
```

The provided batch scripts request one `a100_80gb` GPU on the `gpu` partition.
The default training track is `action-mlp`. Override with:

```bash
sbatch --export=ALL,TRACK=heatmap part2action/slurm/train_part2action.sbatch
sbatch --export=ALL,TRACK=heatmap-contact part2action/slurm/train_part2action.sbatch
sbatch --export=ALL,TRACK=heatmap-approach part2action/slurm/train_part2action.sbatch
sbatch --export=ALL,TRACK=heatmap-contact-approach part2action/slurm/train_part2action.sbatch
sbatch --export=ALL,TRACK=action-mlp part2action/slurm/train_part2action.sbatch
sbatch --export=ALL,TRACK=action-diffusion part2action/slurm/train_part2action.sbatch
sbatch --export=ALL,TRACK=all part2action/slurm/train_part2action.sbatch
```

Monitor jobs and logs:

```bash
squeue -u "$USER"
ls -lah part2action/logs/slurm/
```

Training outputs go under:

```text
part2action/results/runs/slurm_<job_id>/
```

## 8. PartGym Rollouts

Run rollouts after training produces an action checkpoint. For example:

```bash
cd /u/xna8aw/workspace/live-robotics-lab/interactive-robotics
sbatch --export=ALL,CKPT=results/part_action_mlp_real/last.pt \
  part2action/slurm/rollout_partgym.sbatch
```

If the checkpoint came from `train_tracks.sh`, pass the actual path under
`part2action/results/runs/...`.

Useful overrides:

```bash
sbatch --export=ALL,CKPT=path/to/last.pt,OBJ_CLASSES="scissors pliers",TASK_TYPES="1",NUM_EPISODES=2 \
  part2action/slurm/rollout_partgym.sbatch
```

## 9. Quick Interactive GPU Check

Use this only for debugging. It allocates a GPU shell and releases it when you
exit:

```bash
srun -p gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

Inside the allocation:

```bash
hostname
nvidia-smi
export PATH="/u/xna8aw/workspace/.bin:$PATH"
export MAMBA_ROOT_PREFIX="/u/xna8aw/workspace/.micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate part2action310
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

When done:

```bash
exit
```
