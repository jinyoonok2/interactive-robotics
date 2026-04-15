# Interactive Robotics

Language-instructed robotic manipulation in Habitat-Sim. A user gives a natural language task instruction (e.g., *"pour coffee,"* *"cut the bread"*), and a Fetch robot must understand **which part** of **which object** to interact with — then plan and execute a grasp accordingly.

The core challenge is **part-aware affordance detection conditioned on language**: the same object has different interaction regions depending on the task. *"Pour coffee"* → grip the mug **by the handle**. *"Cut the bread"* → grip the knife **by the handle**, not the blade. The affordance model must ground the instruction to the correct object part.

**Status:** The original affordance module (UAD / FiLM) was evaluated and found insufficient for part-level localization (SIM=0.407). A replacement using supervised cross-attention (DINOv2 ViT-L/14 + Flan-T5 encoder) trained on AGD20K with LLM-enriched descriptions is in development. The architecture produces reusable Task-Grounded Features designed to transfer from affordance detection to RL-based manipulation policy learning. See [PROPOSAL.md](PROPOSAL.md) for the full approach, [RESEARCH_JUSTIFICATION.md](RESEARCH_JUSTIFICATION.md) for positioning against related work, and [UAD_EVALUATION.md](UAD_EVALUATION.md) for the UAD evaluation.

---

## Architecture

```
Input:   "pour coffee"                          ← Natural language task instruction
           │
Stage 1: SceneCapture      → Habitat-Sim loads HSSD scene, captures RGB + Depth + Semantic
Stage 2: AffordanceDetector → Language-conditioned model → part-level heatmap (e.g., mug handle)
         GraspPlanner       → GraspNet + affordance weighting → 6-DoF grasp on the correct part
Stage 3: RobotExecutor     → Fetch robot IK + motion planning → execute grasp in Habitat-Sim
```

The instruction drives everything: the affordance model takes both the image and the text, and produces a heatmap highlighting the task-relevant object part. The grasp planner then targets that region specifically.

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| `SceneCapture` | `affordance-pipeline/core/scene_capture.py` | Habitat-Sim scene/sensor config, agent positioning, YCB spawning, RGB/D/Semantic capture |
| `AffordanceDetector` | `affordance-pipeline/core/affordance_detector.py` | Language-conditioned affordance heatmap — takes image + task instruction, outputs part-level heatmap (pending replacement) |
| `GraspPlanner` | `affordance-pipeline/core/grasp_planner.py` | Geometric + GraspNet neural grasp planning with affordance-weighted selection |
| `RobotExecutor` | `affordance-pipeline/core/robot_executor.py` | FetchRobot wrapper, PyBullet IK, motion planning, magic grasp |
| `Pipeline` | `affordance-pipeline/pipeline.py` | Main orchestrator: instruction → scene capture → affordance → grasp → execute |

### Robot

FetchRobot (7-DoF arm) via Habitat's articulated agent framework:
- Arm joints: `[15,16,17,18,19,20,21]`, Gripper: `[23,24]`, EE link: `22`
- IK via PyBullet DIRECT mode using `hab_fetch_arm.urdf`

---

## Project Structure

```
interactive-robotics/
├── README.md                  # This file
├── PROPOSAL.md                # Architecture, phased plan (1–4), model choices
├── RESEARCH_JUSTIFICATION.md  # Why supervised + positioning vs LOCATE, UAD, VLMs
├── UAD_EVALUATION.md          # UAD setup, evaluation results, and failure analysis
├── affordance-pipeline/       # Main pipeline code
│   ├── pipeline.py            # 3-stage orchestrator
│   ├── core/                  # SceneCapture, AffordanceDetector, GraspPlanner, RobotExecutor
│   ├── config/                # objects.json, prompts.json
│   └── RESEARCH_LOG.md        # DINOv2 / CLIPSeg / UAD investigation notes
├── habitat-lab/               # Habitat-Lab v0.3.3 (submodule)
│   └── data/                  # Datasets (~13GB)
├── graspnet-baseline/         # GraspNet 6-DoF grasp prediction (submodule)
│   └── checkpoints/           # checkpoint-rs.tar
├── unsup-affordance/          # UAD repo (submodule, evaluated & rejected)
│   ├── checkpoints/           # st_emb.pth, oai_emb.pth, eval_agd.pth
│   └── AGD20K_EXPERIMENT_LOG.md     # Detailed experiment record with metrics
├── datasets/                  # Downloaded datasets (AGD20K, etc.)
├── tests/                     # Environment & install verification
└── vposer-tools/              # VPoser motion generation tools
```

---

## Prerequisites

- **Ubuntu/Linux** with X11 display
- **NVIDIA GPU** with drivers (tested: RTX 4060 Max-Q, driver 590.48.01)
- **Miniconda/Conda**
- **git-lfs** (`sudo apt install git-lfs && git lfs install`)
- **OpenGL** (`sudo apt install libopengl0 libgl1-mesa-dev libglx0 libglx-mesa0`)

```bash
nvidia-smi  # Verify GPU + driver
```

---

## Environment Setup: `habitat-grasp`

This is the primary conda environment for simulation, grasp planning, and robot execution.

| Setting | Value |
|---------|-------|
| Python | 3.9 |
| PyTorch | 2.8.0+cu128 |
| numpy | 1.26.4 (pinned by habitat-lab) |
| habitat-sim | 0.3.3 (display + bullet) |

### Install

```bash
# 1. Create environment
conda create -n habitat-grasp python=3.9 cmake=3.14.0 -y
conda activate habitat-grasp

# 2. Install habitat-sim
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat -y

# 3. Install habitat packages (editable)
cd habitat-lab
pip install -e habitat-lab/
pip install -e habitat-baselines/
pip install -e habitat-hitl/
cd ..

# 4. Install CUDA toolkit (for compiling GraspNet extensions)
conda install -c "nvidia/label/cuda-12.8.0" cuda-nvcc cuda-cudart-dev -y

# 5. Install pipeline dependencies
pip install open3d opencv-python scipy pygame
pip install hydra-core websockets aiohttp

# 6. Install GraspNet dependencies
pip install graspnetAPI
pip install numpy==1.26.4    # graspnetAPI downgrades numpy — fix it back
pip install pillow==10.4.0   # pin to habitat-sim compatible version

# 7. Patch transforms3d (np.float removed in numpy 1.24+)
TRANSFORMS3D_DIR=$(python -c "import transforms3d; import os; print(os.path.dirname(transforms3d.__file__))")
find "$TRANSFORMS3D_DIR" -name "*.py" -exec sed -i \
  -e 's/dtype=np\.float)/dtype=np.float64)/g' \
  -e 's/np\.finfo(np\.float)/np.finfo(np.float64)/g' \
  -e 's/np\.maximum_sctype(np\.float)/np.float64/g' {} +

# 8. Build GraspNet CUDA extensions
export CUDA_HOME=$CONDA_PREFIX
cd graspnet-baseline/knn && python setup.py install && cd ../..
cd graspnet-baseline/pointnet2 && python setup.py install && cd ../..

# 9. Download GraspNet checkpoint
mkdir -p graspnet-baseline/checkpoints
pip install gdown
python -c "import gdown; gdown.download(id='1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk', output='graspnet-baseline/checkpoints/checkpoint-rs.tar')"

# 10. Download Habitat datasets (~13GB, requires git-lfs)
cd habitat-lab
python -m habitat_sim.utils.datasets_download --uids hssd-hab --data-path data/
python -m habitat_sim.utils.datasets_download --uids hab3-episodes habitat_humanoids hab_spot_arm hab_fetch ycb --data-path data/
cd ..
```

### Verify

```bash
conda activate habitat-grasp
python -c "
import habitat_sim; print(f'habitat-sim {habitat_sim.__version__}')
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy; print(f'numpy {numpy.__version__}')
import knn_pytorch; print('knn_pytorch OK')
import pointnet2; print('pointnet2 OK')
print('ALL OK')
"
```

### Data Assets

| Dataset | Location | Size |
|---------|----------|------|
| HSSD scenes | `habitat-lab/data/scene_datasets/hssd-hab/` | ~12GB |
| YCB objects | `habitat-lab/data/objects/ycb/` | ~479MB |
| hab3-episodes | `habitat-lab/data/datasets/hssd/rearrange/` | — |
| Fetch robot | `habitat-lab/data/robots/hab_fetch/` | — |
| Spot robot | `habitat-lab/data/robots/hab_spot_arm/` | — |
| Humanoids | `habitat-lab/data/humanoids/` | — |
| GraspNet checkpoint | `graspnet-baseline/checkpoints/checkpoint-rs.tar` | 12MB |

---

## Verification Suite

```bash
bash tests/check_all.sh              # Everything
bash tests/check_all.sh system       # GPU, driver, CUDA, conda, OpenGL
bash tests/check_all.sh data         # Datasets, checkpoints, config files
bash tests/check_all.sh habitat      # habitat-grasp packages + CUDA extensions
bash tests/check_all.sh uad          # uad env packages + DINOv2
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `libOpenGL.so.0` not found | `sudo apt install libopengl0 libgl1-mesa-dev libglx0 libglx-mesa0` |
| `git lfs` not found during dataset download | `sudo apt install git-lfs && git lfs install` |
| numpy downgraded after `pip install graspnetAPI` | `pip install numpy==1.26.4` |
| `np.float` error from transforms3d | Run the patch command in step 7 |
| pillow version mismatch | `pip install pillow==10.4.0` |

---

## Next Steps

1. **Phase 1: Train affordance model** — DINOv2 ViT-L/14 (visual) + Flan-T5 encoder (per-token language) + cross-attention fusion + affordance head, trained on AGD20K GT heatmaps (see [PROPOSAL.md](PROPOSAL.md))
2. **Phase 1.5: Validate DINOv2 correspondence** — verify patch-level feature matching between AGD20K objects and HSSD objects across 5 categories
3. **Phase 2: Habitat dataset augmentation** — render HSSD objects from multiple viewpoints, transfer AGD20K labels via DINOv2 correspondence
4. **Phase 3: Retrain with augmented data** — zero domain gap between training and Habitat deployment
5. **Phase 4 (future): RL policy learning** — attach Policy/Value heads to frozen Task-Grounded Features, train with RL in Habitat

---

## Resources

- [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) — Embodied AI simulation
- [GraspNet-baseline](https://github.com/graspnet/graspnet-baseline) — 6-DoF grasp detection
- [unsup-affordance (UAD)](https://github.com/TangYihe/unsup-affordance) — Evaluated & rejected, see [UAD_EVALUATION.md](UAD_EVALUATION.md)
- [AGD20K](https://github.com/lhc1224/Cross-View-AG) — Affordance Grounding Dataset (Luo et al.)
