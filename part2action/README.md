# part2action

`part2action` is a small concept-validation project for learning:

```text
RGB observation + part-level instruction -> robot action chunk
```

It uses PartInstruct data, but it is still separate from the larger Habitat
integration. The current workflow has two parts:

- **Offline training** from PartInstruct HDF5 demonstrations.
- **Closed-loop simulation** in PartGym using the trained action checkpoints.

The first successful simulator smoke result so far is the MLP action model on a
PartGym scissors grasp task. Pliers still fails in the current prototype.

## Directory Layout

```text
part2action/
  configs/                         Training configs
  data/                            HDF5 dataset loader and target derivation
  docs/                            Extra setup notes
  models/                          DINOv2/T5 backbone, fusion, heads
  scripts/
    train.py                       Unified offline trainer
    evaluate.py                    Offline metrics
    probe_action_models.py         Offline action/visualization probe
    rollout_partgym.py             Closed-loop PartGym rollout runner
  third_party/
    PartInstruct/                  Local upstream PartInstruct clone, gitignored
  results/                         Checkpoints, rollouts, videos, GIFs, gitignored
```

Large files are intentionally ignored by git:

- `third_party/PartInstruct/`
- `results/`
- `*.pt`, `*.ckpt`, `*.mp4`, `*.zip`

## What The Model Trains

The action-capable models train on PartInstruct HDF5 demonstrations. Each sample
contains:

- `agentview_rgb`: camera image
- `skill_instructions`: language instruction
- `actions`: 7D robot actions, shaped `(T, 7)`
- `agentview_part_mask`: target part mask
- `gripper_state`, `joint_states`

The action models predict an `8 x 7` future action chunk. The 7D action is the
same format used by PartGym:

```text
x, y, z, roll, pitch, yaw, gripper
```

Extra heads predict heatmap/contact/approach targets for auxiliary supervision.
The action labels come directly from demos; contact/approach are heuristic.

## Fresh Machine Setup

From a clean machine:

```bash
git clone https://github.com/jinyoonok2/interactive-robotics.git
cd interactive-robotics/part2action
```

### 1. Training Environment

Create the lightweight offline training env:

```bash
bash setup_env.sh
conda activate part2action
```

This env is used for HDF5 training and offline metrics. It does not install
PartGym.

### 2. Download The HDF5 Demo Subset

PartInstruct is gated on Hugging Face. First accept the dataset terms at:

```text
https://huggingface.co/datasets/SCAI-JHU/PartInstruct
```

Then log in and download the small scissors/pliers subset:

```bash
huggingface-cli login
bash download_subset.sh
```

Expected output:

```text
../datasets/PartInstruct/
  demos/
    scissors.hdf5
    pliers.hdf5
  episodes_meta_test.json
  episodes_meta_train.json
  object_meta.json
  part_semantic_lexicon.json
```

The subset is about `3.8 GB`. The full PartInstruct demos are much larger.

### 3. Train Offline Models

Run individual tracks:

```bash
conda activate part2action

python scripts/train.py --config configs/heatmap_real.yaml
python scripts/train.py --config configs/part_action_mlp_real.yaml
python scripts/train.py --config configs/part_action_diffusion_real.yaml
python scripts/train.py --config configs/temporal_part_action_mlp_real.yaml
python scripts/train.py --config configs/temporal_part_action_diffusion_real.yaml
```

Or use the launcher:

```bash
bash train_tracks.sh all
```

Outputs go under `results/`. Each run typically contains:

- `last.pt`
- `history.json`
- `config_source.yaml`
- `config_resolved.json`

For action rollouts, start with:

```text
results/prototype/part_action_mlp_real/last.pt
```

or your newly trained:

```text
results/part_action_mlp_real/last.pt
```

## PartGym Simulator Setup

PartGym uses the upstream PartInstruct repo and heavier simulator dependencies,
so it uses a **separate conda env** named `partinstruct`.

### 1. Clone PartInstruct Locally Under part2action

We avoid git submodules. Clone upstream directly into `third_party/`:

```bash
cd /home/jinyoon/workspace/interactive-robotics/part2action
mkdir -p third_party
git clone --recurse-submodules https://github.com/SCAI-JHU/PartInstruct.git third_party/PartInstruct
```

### 2. Create The Simulator Env

Follow upstream dependency versions:

```bash
conda create -n partinstruct -c conda-forge \
  python=3.9 cmake=3.24.3 open3d ninja gcc_linux-64=12 gxx_linux-64=12 -y

conda activate partinstruct
pip install torch torchvision torchaudio

cd /home/jinyoon/workspace/interactive-robotics/part2action/third_party/PartInstruct
pip install -r requirements.txt
pip install omegaconf

pip install -e .
pip install -e ./third_party/pybullet_planning/
pip install -e ./third_party/diffusion_policy/
pip install -e ./third_party/3D-Diffusion-Policy/
pip install -e ./third_party/gym-0.21.0/
pip install -e ./third_party/pytorch3d/
```

`sam_2` is only needed for SAM-based PartGym variants. The current rollout
script uses non-SAM `PartInstruct.PartGym.env.bullet_env`, so SAM2 is not
required for first tests.

### 3. Download PartGym Assets

The HDF5 demos are enough for offline training, but **not** for simulator
rollouts. PartGym also needs robot/object/scene assets.

```bash
cd /home/jinyoon/workspace/interactive-robotics/part2action/third_party/PartInstruct
huggingface-cli download SCAI-JHU/PartInstruct \
  --repo-type dataset \
  --local-dir ./data \
  --include "*.json" "assets.zip"

unzip ./data/assets.zip -d ./data/
```

Expected paths:

```text
third_party/PartInstruct/data/episodes_meta_test.json
third_party/PartInstruct/data/assets/urdfs/robots/franka_panda/panda.urdf
third_party/PartInstruct/data/assets/partnet-grasping/
```

You can remove the zip after unzipping:

```bash
rm ./data/assets.zip
```

## PartGym Rollouts

Run from `part2action/` using the `partinstruct` env:

```bash
cd /home/jinyoon/workspace/interactive-robotics/part2action
```

Scissors, 200 max steps, record video:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n partinstruct \
python scripts/rollout_partgym.py \
  --model-key mlp \
  --obj-classes scissors \
  --task-types 1 \
  --num-episodes 1 \
  --max-steps 200 \
  --execute-steps 1 \
  --device cpu \
  --record
```

Pliers:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n partinstruct \
python scripts/rollout_partgym.py \
  --model-key mlp \
  --obj-classes pliers \
  --task-types 1 \
  --num-episodes 1 \
  --max-steps 200 \
  --execute-steps 1 \
  --device cpu \
  --record
```

`rollout_partgym.py` defaults to:

```text
third_party/PartInstruct
```

so you do not need `--partinstruct-root` if the local clone is in the expected
location.

### Rollout Outputs

Results are saved under:

```text
results/prototype/partgym_rollouts/mlp/
```

Important files:

```text
rollout_results_test1_scissors_1.json
rollout_results_test1_pliers_1.json
videos/scissors_1_test1/rollout.mp4
videos/pliers_1_test1/rollout.mp4
```

Current prototype result:

- `scissors`, task `1`: succeeded with the MLP model.
- `pliers`, task `1`: failed with the MLP model.

## GIFs For Slides

Convert rollout MP4s into smaller GIFs:

```bash
conda run -n partinstruct python -c "from pathlib import Path; import cv2, imageio.v2 as imageio
base=Path('results/prototype/partgym_rollouts/mlp/videos')
for name in ['scissors_1_test1','pliers_1_test1']:
    mp4=base/name/'rollout.mp4'; gif=base/name/'rollout_slide.gif'
    cap=cv2.VideoCapture(str(mp4)); fps=cap.get(cv2.CAP_PROP_FPS) or 20
    frames=[]; idx=0; stride=max(1, round(fps/10))
    while True:
        ok, frame=cap.read()
        if not ok: break
        if idx % stride == 0:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame=cv2.resize(frame, (300,300), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    imageio.mimsave(str(gif), frames, duration=0.1, loop=0)
    print(gif)"
```

## Offline Evaluation

Offline metrics are still useful before simulation:

```bash
conda activate part2action
python scripts/evaluate.py --config configs/part_action_mlp_real.yaml --ckpt results/part_action_mlp_real/last.pt
```

`scripts/probe_action_models.py` creates qualitative overlays and action plots:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n part2action \
python scripts/probe_action_models.py \
  --models part_action_mlp_real temporal_part_action_mlp_real \
  --num-samples 12 \
  --text-device cpu
```

Outputs:

```text
results/prototype/action_model_probe/
```

## Notes And Troubleshooting

- `part2action` env is for offline training.
- `partinstruct` env is for PartGym simulation.
- `third_party/PartInstruct` is intentionally gitignored. Re-clone it on a
  fresh machine.
- If Hugging Face returns `401 GatedRepo`, accept the dataset terms and run
  `huggingface-cli login`.
- If VLC reports GPU/VDPAU errors, play videos with CPU decoding:

```bash
vlc --avcodec-hw=none results/prototype/partgym_rollouts/mlp/videos/scissors_1_test1/rollout.mp4
```

- If CUDA is unavailable in PyTorch, fall back to `--device cpu` for rollouts.
- DINOv2/Flan-T5 may need internet the first time unless cached locally.
