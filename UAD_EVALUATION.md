# UAD (Unsupervised Affordance Distillation) — Setup & Evaluation

**Verdict: Evaluated and rejected.** UAD's FiLM architecture does not achieve part-level affordance localization. Best model scored only 41% overlap (SIM) with ground truth on the authors' own AGD20K benchmark. For the full experiment — configurations, metrics, qualitative analysis, and root cause diagnosis — see the [AGD20K Experiment Log](unsup-affordance/AGD20K_EXPERIMENT_LOG.md).

---

## What is UAD?

**Repository:** [TangYihe/unsup-affordance](https://github.com/TangYihe/unsup-affordance)  
**Paper:** Unsupervised Affordance Distillation (Stanford — Yihe Tang, Wenlong Huang, Jiajun Wu, Li Fei-Fei)

UAD uses DINOv2 visual features + unsupervised clustering to discover affordance regions without manual labels. The model architecture is:

```
RGB Image + Text Query → DINOv2 (frozen) → patch features
                       → SentenceTransformer or OpenAI → text embedding
                       → Conv2DFiLMNet (FiLM conditioning) → affordance heatmap
```

FiLM (Feature-wise Linear Modulation) performs **channel-wise** scaling/shifting of visual features conditioned on text. This can select *which feature channels* matter but cannot perform fine **spatial** selection — it has no mechanism to attend to specific image regions.

---

## Environment Setup: `uad`

The UAD model requires a separate conda environment due to hard numpy/PyTorch conflicts with `habitat-grasp`.

| Setting | Value |
|---------|-------|
| Python | 3.10 (required — DINOv2 uses `float \| None` syntax) |
| PyTorch | 2.7.1+cu126 |
| numpy | 2.0.2 |

### Install

```bash
# 1. Create environment
conda create -n uad python=3.10 -y
conda activate uad

# 2. Install from requirements
cd unsup-affordance
pip install -r requirements.txt

# 3. Install additional dependency
pip install sentence-transformers

# 4. Install package in editable mode
pip install -e .
cd ..
```

### Verify

```bash
conda activate uad
python -c "
import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import numpy; print(f'numpy {numpy.__version__}')
from model.network import Conv2DFiLMNet; print('UAD network OK')
print('ALL OK')
"
```

### Models (auto-download on first run)

| Model | Source | Size | Purpose |
|-------|--------|------|---------|
| DINOv2 ViT-S/14 (with registers) | torch.hub → `~/.cache/torch/hub/checkpoints/` | 85MB | Visual backbone for local model |
| DINOv2 ViT-L/14 | torch.hub → `~/.cache/torch/hub/checkpoints/` | 1.2GB | Visual backbone for API model |
| SentenceTransformer all-MiniLM-L6-v2 | HuggingFace → `~/.cache/huggingface/hub/` | ~80MB | Text embedding (local model) |

### Checkpoints (included in repo)

| File | Size | Purpose |
|------|------|---------|
| `unsup-affordance/checkpoints/st_emb.pth` | 15MB | Local model (ViT-Small + Sentence-Transformer) |
| `unsup-affordance/checkpoints/oai_emb.pth` | 20MB | OpenAI embedding model |
| `unsup-affordance/checkpoints/eval_agd.pth` | 13MB | Best model (ViT-Large + OpenAI, for AGD20K eval) |

### OpenAI API Key (for best model only)

```bash
# Create .env file (already in .gitignore)
echo "OPENAI_API_KEY=sk-..." > unsup-affordance/.env
```

---

## How UAD Integrates with the Pipeline

Communication between `habitat-grasp` and `uad` environments is via files:

```
habitat-grasp env                    uad env
┌──────────────┐                     ┌──────────────┐
│ SceneCapture │──► RGB image ──────►│ DINOv2 + UAD │
│              │    (saved to disk)  │              │
│ GraspPlanner │◄── affordance.npy ◄─│ inference.py │
│              │    (H×W float map)  │              │
│ RobotExecutor│                     └──────────────┘
└──────────────┘
```

Key bridge files in `affordance-pipeline/`:
- `uad_bridge.py` — subprocess bridge calling UAD inference, communicates via temp `.npy` files
- `_uad_worker.py` — worker script running inside the `uad` conda env

---

## Evaluation Summary

We evaluated UAD on **AGD20K** (Luo et al.), a dataset of ~20K real egocentric photos with pixel-level affordance ground truth — the same benchmark used in the UAD paper. Every configuration was verified against the authors' exact settings (14-item compliance checklist in the experiment log).

**Best result (ViT-Large + OpenAI, Unseen split, 540 images):** SIM = 0.407 (41% overlap with GT)

This means the model's predicted affordance region overlaps less than half of the actual ground truth, even with the authors' best checkpoint and recommended settings. Heatmaps consistently spread across the entire object rather than localizing to specific affordance parts.

For complete results (both models, both splits, qualitative analysis, visualizations, and root cause analysis), see the **[AGD20K Experiment Log](unsup-affordance/AGD20K_EXPERIMENT_LOG.md)**.

---

## What Replaces UAD

**Planned approach:** Supervised cross-attention model trained on AGD20K + Habitat-rendered augmentation.

| Aspect | UAD (rejected) | Replacement (planned) |
|--------|---------------|----------------------|
| Architecture | FiLM (channel-wise, no spatial attention) | Cross-attention (DINOv2 visual × CLIP text, spatial) |
| Training labels | Unsupervised clustering + GPT-4o (noisy) | AGD20K human-annotated GT (20K images) |
| Training data augmentation | None | Habitat multi-view renders with DINOv2-transferred labels |
| Domain match | Synthetic only | AGD20K (real) + Habitat renders (deployment domain) |
