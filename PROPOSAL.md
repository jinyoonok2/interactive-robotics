# Proposal: Task-Grounded Visual Representation for Language-Instructed Manipulation

> **Status:** Proposal — not yet started
> **Date:** April 2026
> **Goal:** Build a task-grounded visual representation that (1) enables part-aware affordance detection from language instructions, and (2) transfers to RL-based robot policy learning — replacing UAD (SIM=0.407).
> **Justification:** See [RESEARCH_JUSTIFICATION.md](RESEARCH_JUSTIFICATION.md) for detailed positioning against unsupervised methods and related work.

---

## Core Insight

UAD intentionally avoided supervised training on AGD20K because their paper's contribution is "unsupervised." We have no such constraint — we just need the best task-grounded representation for our Fetch robot.

The key design principle: **don't build a heatmap predictor — build a reusable task-grounded representation** that supports affordance detection now and RL-based manipulation policy learning later.

---

## Architecture

```
                         FULL MODEL — designed for current + future use
                         ═══════════════════════════════════════════════

  "pour coffee"  ──→  [Flan-T5 Encoder]  ──→  Token Embeddings (T×768)
                       (frozen, CPU)                │
  RGB Image      ──→  [DINOv2 ViT-L/14]  ──→  Patch Tokens (N×1024)
                       (frozen, GPU)                │
                                                    ▼
                                         ┌─────────────────────┐
                                         │   Cross-Attention    │
                                         │   Fusion Module      │  ← TRAINABLE CORE
                                         │   (multiple layers)  │
                                         └──────────┬──────────┘
                                                    │
                                       Task-Grounded Features (N×D)
                         Per-patch D-dim vectors encoding functional ROLES
                         (handle → GRIP role, rim → POUR role, etc.)
                                                    │
                           ┌────────────────────────┼──────────────────────────┐
                           ▼                        ▼                          ▼
                    ┌─────────────┐       ┌───────────────┐          ┌──────────────┐
                    │  Affordance │       │  Policy Head   │          │  Value Head   │
                    │    Head     │       │  + Robot State  │          │  + Robot State │
                    └──────┬──────┘       └───────┬───────┘          └──────┬───────┘
                           ▼                      ▼                        ▼
                      Heatmap (H×W)          Action (7-DoF)           V(s) scalar
                      ← PHASE 1              ← PHASE 4 (RL)          ← PHASE 4 (RL)
```

### Components

**Frozen Encoders (pretrained, not trained):**
- **DINOv2 ViT-L/14** — visual encoder. Outputs N patch tokens × 1024-dim. Strong part-level features. Already cached locally (1.2GB).
- **Flan-T5-base Encoder** — task/language encoder. Outputs **per-token embeddings** (T × 768-dim), not a single vector. Instruction-tuned on 1,800+ tasks — understands affordance instructions like "grip the handle and tilt to pour" natively. Runs on CPU (zero VRAM cost). ~500MB.

**Why Flan-T5 over MiniLM/CLIP:**
- **Per-token output:** Each word gets its own 768-dim embedding. In cross-attention, "handle" tokens attend to handle patches, "pour" tokens attend to rim patches. This creates **role-specific** features per patch — not possible with single-vector encoders (MiniLM) that apply the same modulation to all patches.
- **Instruction-tuned:** Understands task language ("grip," "tilt," "press") better than BERT (masked LM) or CLIP (caption-trained).
- **RL-ready:** Rich per-token embeddings preserve action semantics ("how to interact") that a future policy head can exploit.

**Trainable Fusion Module:**
- Multi-layer cross-attention: per-token text embeddings attend to DINOv2 patch tokens
- Produces **Task-Grounded Features** — a spatial feature map that encodes "for this task, these parts of the scene matter in this way"
- This is the reusable representation that transfers across heads

**Swappable Heads:**
- **Affordance Head** (Phase 1): upsample Task-Grounded Features → per-pixel heatmap. Trained with GT heatmap supervision.
- **Policy Head** (Phase 4, future): takes Task-Grounded Features + robot joint state → 7-DoF arm action. Trained with RL.
- **Value Head** (Phase 4, future): takes Task-Grounded Features + robot joint state → scalar value. Trained with RL.

### Why This Transfers to RL

Goal-conditioned RL needs: "given the scene and a goal, what action should I take?"

Our task embedding IS the goal. Our task-grounded features ARE the goal-conditioned scene representation. Phase 1 trains the fusion module to understand task-relevant parts with supervised affordance labels. Phase 4 reuses that understanding — RL only needs to learn motor control, not visual-linguistic grounding.

| Phase | What Trains | What's Frozen | Supervision |
|-------|------------|---------------|-------------|
| Phase 1 | Fusion Module + Affordance Head | DINOv2, Task Encoder | AGD20K heatmap GT |
| Phase 4 (future) | Policy Head + Value Head | DINOv2, Task Encoder, Fusion Module | RL reward in Habitat |

### LLM-Enriched Task Descriptions

Instead of encoding bare phrases like "grasp the mug," we use an LLM (offline, one-time) to generate rich descriptions:

```
Input:   "grasp the mug"
LLM →    "To grasp a mug, grip the handle — the curved protrusion on the
          side. Hold it firmly with fingers wrapped around the handle loop."
```

This injects part-level and motion-intent knowledge into the task embedding. The LLM runs once for all 36 actions × 50 objects = 1,800 pairs. Cached as text files — no LLM needed at train or inference time.

**Why this matters for RL too:** The enriched description contains *how* to interact ("tilt," "press," "grip firmly"), not just *where*. This motion-intent signal, embedded in the task features, gives the future policy head a head start.

---

## Phase 1: Train Affordance Model on AGD20K

**Objective:** Validate the architecture with supervised training.

**Data:**
- **Train:** Seen testset GT — 1,675 images with grayscale heatmap labels (the only labeled split)
- **Eval:** Unseen testset GT — 540 images (different object instances, true generalization test)
- **GT format:** Grayscale heatmaps (uint8, 0-255), same resolution as input images
- **Note:** AGD20K trainsets (26K+ images) have NO GT labels — designed for UAD's unsupervised approach

**Steps:**
1. Build AGD20K dataloader (image + GT heatmap + action-object text)
2. Generate LLM-enriched descriptions for all action×object pairs
3. Implement model (DINOv2 + Task Encoder + Fusion Module + Affordance Head)
4. Train on Seen GT, evaluate on Unseen GT
5. Compare against UAD baseline: SIM/KLD/NSS metrics

**VRAM budget:** ~1.2GB (DINOv2-L on GPU) + ~0 (Flan-T5 on CPU) + ~0.5GB (fusion + head + gradients) ≈ 1.7GB — well within 8GB RTX 4060.

**Deliverable:** Trained model + metrics beating UAD's SIM=0.407 on Unseen testset.

---

## Phase 1.5: DINOv2 Correspondence Validation (Early Checkpoint)

> **When:** After Phase 1 training, before building the full Phase 2 pipeline.

**Objective:** Empirically verify that DINOv2 feature correspondence works across AGD20K → HSSD object pairs. This answers the key Phase 2 risk: *"What happens when the target object has different geometry than the source?"*

**Steps:**
1. Pick 5 representative AGD20K categories (mug, knife, chair, bottle, bowl)
2. Find their closest HSSD matches
3. Extract DINOv2 patch features from both, compute cosine similarity matrix
4. Visualize correspondence: do handle→handle, blade→blade, rim→rim mappings hold?
5. Set confidence threshold — pairs below threshold won't get labels in Phase 2

**Decision gate:** If correspondence is clean for ≥3/5 categories → Phase 2 proceeds. If correspondence broadly fails → rethink label transfer strategy.

---

## Phase 2: Habitat Dataset Augmentation

> **When:** After Phase 1.5 validates DINOv2 correspondence.

**Objective:** Generate labeled training data in the deployment domain using DINOv2 feature correspondence.

**Pipeline:**
1. **Category Mapping:** Map AGD20K's 50 object categories → HSSD object instances
   - HSSD has 18K+ objects — many overlap (mugs, knives, bottles, chairs, etc.)
   - Determine overlap percentage to estimate augmentation scale
   - Only include pairs that passed Phase 1.5 correspondence validation
2. **Multi-View Rendering:** For each matching HSSD object, render 100+ viewpoints in Habitat-Sim
   - Vary camera elevation, azimuth, distance, lighting
3. **Label Transfer via DINOv2 Feature Correspondence:**
   - Extract DINOv2 patch features from AGD20K source (with GT heatmap)
   - Extract DINOv2 patch features from Habitat render (same category, different instance)
   - Match patches by cosine similarity — DINOv2 features activate consistently on the same part across instances (e.g., "handle" on mug A matches "handle" on mug B)
   - Transfer affordance heatmap from source → target via the correspondence map
4. **Quality Filtering:** Discard transfers with low correspondence confidence

**Result:** Thousands of labeled Habitat renders per category, in the exact deployment visual domain.

---

## Phase 3: Retrain with Augmented Data

> **When:** After Phase 2 produces the Habitat-rendered dataset.

**Objective:** Zero domain gap between training and deployment.

- Retrain the Fusion Module + Affordance Head on AGD20K + Habitat augmented data
- The model now sees the exact visual distribution it encounters at deployment
- Evaluate on Unseen testset + Habitat-specific test scenes

---

## Phase 4: RL Policy Learning (Future)

> **When:** After Phase 3 achieves strong affordance results in Habitat.

**Objective:** Go from "where to interact" to "how to interact."

- Freeze DINOv2 + Task Encoder + Fusion Module (all pretrained in Phase 1-3)
- Attach Policy Head (Task-Grounded Features + robot joint state → 7-DoF action)
- Attach Value Head (Task-Grounded Features + robot joint state → V(s))
- Train with RL in Habitat-Sim — Fetch robot executes tasks end-to-end
- The fused representation already knows which parts matter — RL only learns motor control

---

## Summary: What Gets Built When

| Phase | Build | Train | Data |
|-------|-------|-------|------|
| **1** | Dataloader, Model (DINOv2 + Flan-T5 + Fusion + Affordance Head) | Fusion Module + Affordance Head | AGD20K GT (1,675 train / 540 eval) |
| **1.5** | Correspondence validation script | — (analysis only) | 5 categories × AGD20K↔HSSD pairs |
| **2** | Rendering pipeline, Label transfer | — (data generation only) | HSSD objects × 100+ views |
| **3** | — | Retrain Fusion + Affordance Head | AGD20K + Habitat augmented |
| **4** | Policy Head, Value Head, RL loop | Policy + Value Heads | RL reward in Habitat |

---

## Next Steps (Immediate)

- [ ] Build AGD20K dataloader (image + GT heatmap + action-object text)
- [ ] Generate LLM-enriched descriptions (36 actions × 50 objects, cached as text)
- [ ] Implement model architecture (DINOv2 + Flan-T5 + Cross-Attention Fusion + Affordance Head)
- [ ] Train Phase 1 on Seen GT, evaluate on Unseen GT (beat UAD SIM=0.407)
- [ ] Phase 1.5: DINOv2 correspondence validation on 5 categories
- [ ] Map AGD20K categories ↔ HSSD objects (overlap check for Phase 2)

---

## References

**Dataset & Evaluation:**
- **AGD20K** (Luo et al.) — ~20K images, 50 object categories, 36 actions, human-annotated affordance heatmaps
- **HSSD** — Habitat Synthetic Scenes Dataset, 18K+ object instances across 200+ scenes
- **UAD evaluation:** See [UAD_EVALUATION.md](UAD_EVALUATION.md) for setup and verdict (SIM=0.407, rejected)
- **Experiment log:** See [unsup-affordance/AGD20K_EXPERIMENT_LOG.md](unsup-affordance/AGD20K_EXPERIMENT_LOG.md) for detailed UAD metrics

**Foundation Models (used in our architecture):**
- **DINOv2** (Oquab et al., 2023) — Self-supervised ViT, strong part-level correspondence. We use ViT-L/14.
- **Flan-T5** (Chung et al., 2022) — Instruction-tuned encoder-decoder. We use the encoder only for per-token text embeddings.

**Related Work (affordance grounding):**
- **UAD** (Tang et al., 2024) — Unsupervised affordance discovery, DINOv2 + MeanShift + FiLM. Our predecessor, evaluated and rejected.
- **LOCATE** (CVPR 2024) — Weakly supervised exocentric→egocentric affordance via visual prototypes + k-means. Same dataset, different philosophy (visual transfer vs our language conditioning).
- **AffordanceLLM** (2024) — VLM-based affordance grounding using LLM world knowledge.
- **LoopTrans** (2025) — Iterative weakly supervised affordance grounding.
- **VideoAfford** (2026) — Acknowledges unsupervised methods produce coarse masks insufficient for robotic manipulation.

**Related Work (robot learning):**
- **R3M, VIP** — Pretrained visual representations for RL. Related to our Phase 4 transfer.
- **RT-2, OpenVLA, π₀** — End-to-end VLM robot policies. Alternative approach requiring much more compute/data.
- **SayCan** — Language-conditioned robot planning with pretrained representations.

**Research justification:** See [RESEARCH_JUSTIFICATION.md](RESEARCH_JUSTIFICATION.md) for detailed positioning.
