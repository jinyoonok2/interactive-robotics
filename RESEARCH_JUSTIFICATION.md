# Research Justification: Why Supervised Affordance Grounding for Embodied Robotics

> **Purpose:** Document why our supervised, language-conditioned approach is a valid and defensible research direction despite the field's push toward unsupervised methods.

---

## 1. The Field's Current Direction: Unsupervised for Scaling

The affordance grounding community (2024–2026) is heavily focused on bypassing manual annotation:

| Paper | Year | Approach | Key Idea |
|-------|------|----------|----------|
| **UAD** | 2024 | Unsupervised (DINOv2 + MeanShift clustering) | Discover affordances from unlabeled egocentric/exocentric image pairs |
| **LOCATE** | CVPR 2024 | Weakly supervised (DINO + CAM + k-means) | Transfer affordance from exocentric human interaction images to egocentric views using visual prototypes |
| **AffordanceLLM** | 2024 | VLM-based | Ground affordances using world knowledge from large vision-language models |
| **LoopTrans** | 2025 | Weakly supervised | Image-level labels + iterative refinement |
| **VideoAfford** | 2026 | Video-based | Leverage video sequences for temporal affordance understanding |

**Why they avoid supervised training:** Creating pixel-perfect affordance heatmaps is expensive. AGD20K (Luo et al.) has ~20K images but GT labels only exist for test sets (2,215 images). Researchers either avoid using GT entirely (for the unsupervised contribution) or work with weak/noisy supervision.

**The driving goal:** Open-vocabulary generalization — a model that can predict affordances for any object "in the wild," even ones never seen during training.

---

## 2. Why Unsupervised Methods Fail at Robotic Manipulation

The core problem, acknowledged by recent papers themselves:

> *"Recent unsupervised egocentric methods often produce scene-level coarse-grained masks [that fall] short of the object-centric fine-grained affordance segmentation required for precise robotic manipulation and grasping."* — VideoAfford, 2026

### What "coarse" means in practice

```
Unsupervised output (UAD, SIM=0.407):
  "pour + mug" → blurry blob covering top half of mug
  → Robot tries to grasp… somewhere in the blob
  → Grabs the rim instead of the handle → fails to pour

Supervised output (our goal):
  "pour + mug" → sharp activation on handle, secondary on rim
  → Robot targets the handle precisely
  → Successful grasp → successful pour
```

A fuzzy blob is acceptable for:
- Computer vision benchmarks (mIoU, KLD on images)
- Generating visualizations for papers

A fuzzy blob is **not** acceptable for:
- 7-DoF robotic arm needing exact grasp points
- RL policy learning that requires consistent, precise part localization
- Safety-critical manipulation (knife by handle, not blade)

---

## 3. Why Supervised Is the Right Choice for Our Pipeline

### 3.1 Precision is non-negotiable

Our end goal is not a 2D heatmap on a benchmark image — it's a Fetch robot executing "pour coffee" by gripping the correct part of the correct object in Habitat-Sim. The robot needs:
- **Part-level spatial precision** (handle, not rim)
- **Role-aware features** (this part is for gripping, that part defines pour direction)
- **Consistency** (same object + same instruction → same affordance, every time)

Unsupervised methods optimize for coverage and recall ("activate somewhere relevant"). We need precision and spatial fidelity.

### 3.2 The scaling problem doesn't exist in our context

The unsupervised obsession is driven by a data bottleneck that **doesn't apply to us**:

| Concern | Open-Vocabulary Research | Our Pipeline |
|---------|------------------------|-------------|
| Object diversity | Infinite (any internet image) | Fixed (HSSD objects, ~50 relevant categories) |
| Annotation cost | Prohibitive at scale | Already done (AGD20K GT exists for test sets) |
| Domain | Unconstrained real world | Controlled simulator (Habitat-Sim) |
| Scaling path | Must avoid labels → unsupervised | Automate label transfer → supervised at scale |

We operate in a closed environment with known objects. We don't need open-vocabulary generalization to 10,000 unseen categories.

### 3.3 Our scaling path: simulation, not weaker supervision

```
Traditional supervised scaling:   More human annotators → more labels  (expensive, slow)
Unsupervised scaling:             Weaker supervision → more data       (cheap, imprecise)
Our scaling (Phase 2):            Habitat renders + DINOv2 correspondence → automated label transfer
                                  (cheap, preserves precision)
```

Phase 2 of our proposal generates unlimited labeled training data by:
1. Rendering HSSD objects from 100+ viewpoints in Habitat-Sim
2. Transferring AGD20K GT labels to renders via DINOv2 feature correspondence
3. Filtering by correspondence confidence

This gives us the scale of unsupervised methods with the precision of supervised labels.

---

## 4. How We Differ From Existing Work

### vs. UAD (2024)

| | UAD | Ours |
|--|-----|------|
| Supervision | Unsupervised (zero labels) | Fully supervised (GT heatmaps) |
| Text encoding | Single-vector (MiniLM, FiLM conditioning) | Per-token (Flan-T5 encoder, cross-attention) |
| Part understanding | No — same modulation for all patches | Yes — different tokens attend to different parts |
| Output | 2D heatmap (dead-end) | Task-Grounded Features (reusable for RL) |
| Spatial precision | SIM=0.407 (coarse, blobby) | Target: significantly higher |

### vs. LOCATE (CVPR 2024)

| | LOCATE | Ours |
|--|--------|------|
| Supervision | Weakly supervised (image-level labels + CAM + clustering) | Fully supervised (pixel-level GT) |
| Input modality | Exocentric images (humans interacting with objects) | Language instructions (text) |
| Transfer mechanism | Visual: exocentric → egocentric via prototypes | Linguistic: text instruction → visual patches via cross-attention |
| Architecture | k-means clustering (PartSelect) to find relevant regions | Trainable cross-attention (no clustering needed) |
| End goal | 2D heatmap on novel images (CV task) | Task-grounded representation for RL policy (robotics task) |

### vs. VLM Chains (GPT-4V + Grounding-DINO + SAM)

| | VLM Chain | Ours |
|--|-----------|------|
| Part-aware reasoning | Yes (at language level) | Yes (at feature level) |
| Pixel precision | Weak (bounding box → mask) | Strong (per-patch cross-attention) |
| End-to-end trainable | No (API calls, separate models) | Yes (single differentiable model) |
| RL-transferable | No (no shared representation) | Yes (Task-Grounded Features → Policy Head) |
| Latency | High (API calls) | Low (local inference) |
| Internet required | Yes | No |

### vs. End-to-End VLMs (RT-2, OpenVLA, π₀)

| | End-to-End VLM | Ours |
|--|----------------|------|
| Approach | Giant VLM (7B+) fine-tuned on robot demonstrations | Modular: frozen encoders + trainable fusion |
| Data requirement | Thousands of real robot demonstrations | 1,675 labeled images + simulation |
| Compute | A100+ GPUs | RTX 4060 (8GB) |
| Part-aware | Implicit (learned internally) | Explicit (visualizable heatmaps + role features) |
| Interpretable | Black box | Heatmap output shows what model attends to |

---

## 5. Our Novelty Claims

1. **Supervised precision + automated simulation scaling:** Do the hard supervised work once on AGD20K, then automate label transfer to Habitat via DINOv2 feature correspondence. Combines the precision of supervised methods with the scale of unsupervised approaches.

2. **Language-conditioned part-role features (not just heatmaps):** Per-token cross-attention between Flan-T5 instruction embeddings and DINOv2 visual patches produces D-dimensional Task-Grounded Features where each patch encodes its functional role in the task — not just an importance score. Different from LOCATE's visual transfer and UAD's FiLM conditioning.

3. **Affordance pretraining → RL transfer:** The Task-Grounded Features are designed as a shared representation. Affordance Head (Phase 1) validates the features with spatial supervision. Policy Head (Phase 4) reuses the frozen features for RL — the robot inherits part-level language-grounded understanding without learning it from scratch.

4. **Sim-to-real label transfer via DINOv2 correspondence (Phase 2):** Automatic affordance annotation for simulator objects by matching DINOv2 patch features across object instances. Novel application of foundation model features for bridging real-image datasets to simulation.

---

## 6. Known Risks and Mitigations

### Risk 1: DINOv2 correspondence fails on geometrically dissimilar objects

**Example:** AGD20K office chair → HSSD beanbag chair. No structural correspondence.

**Mitigation:** Correspondence quality filtering. Before Phase 2 implementation, validate on 5 categories:
1. Compute DINOv2 patch features for AGD20K objects and HSSD objects
2. Build similarity matrix
3. Only transfer labels between pairs with >0.7 cosine similarity
4. Accept that some HSSD objects won't get labels — partial coverage is fine

### Risk 2: 1,675 training images may not be enough for Phase 1

**Mitigation:** Frozen DINOv2-L features are already powerful — we only train a lightweight fusion module (~2-5M params). Data efficiency is much higher when the visual backbone is frozen. If needed, augment with standard transforms (flip, crop, color jitter) before Phase 2.

### Risk 3: RL transfer (Phase 4) may not work

**Mitigation:** Phase 4 is explicitly marked as future work. Phases 1-3 stand alone as a contribution regardless. The architecture supports RL transfer by design, but proving it requires separate experiments.

---

## 7. Related Work to Cite

- **UAD** (Tang et al., 2024) — Unsupervised affordance discovery. Our direct predecessor, evaluated and rejected (SIM=0.407).
- **LOCATE** (CVPR 2024) — Weakly supervised exocentric→egocentric affordance transfer. Same dataset, different philosophy.
- **AGD20K** (Luo et al.) — The dataset we train on. 50 objects, 36 actions, GT heatmaps.
- **AffordanceLLM** (2024) — VLM-based affordance grounding. Shows LLM world knowledge is useful for affordance.
- **LoopTrans** (2025) — Iterative weakly supervised affordance. Representative of the "avoid GT" trend.
- **VideoAfford** (2026) — Acknowledges unsupervised methods produce coarse masks insufficient for robotics.
- **R3M, VIP** — Pretrained visual representations for RL. Related to our Phase 4 transfer idea.
- **RT-2, OpenVLA** — End-to-end VLM policies. Alternative approach requiring much more compute/data.
