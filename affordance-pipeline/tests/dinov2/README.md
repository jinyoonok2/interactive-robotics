# DINOv2 Part-Awareness Test

## Purpose

Test whether DINOv2's visual features can separate different parts of an object
(e.g., handle, body, rim of a mug). This matters because UAD uses DINOv2 as its
visual backbone — if DINOv2 can't tell parts apart, UAD can never work.

## How to run

```bash
cd /home/jinyoon/workspace/interactive-robotics/habitat-lab
/home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/dinov2/test_part_awareness.py
```

Requires: scene capture already exists in `affordance-pipeline/output/` (rgb.png + depth_raw.npy).

## Output images

Results are saved to `results/diagnosis/dinov2/`.

---

### 1. `kmeans_clustering.png`

**What it shows**: The mug pixels clustered into k groups (k=2,3,4,5) based on
DINOv2 feature similarity. Each color is a different cluster.

**Top row** — Cluster overlay on the mug:
- Each pixel is colored by which cluster it belongs to (red, blue, green, purple, orange)
- These clusters are computed purely from DINOv2's 384-dimensional feature vectors
- **What to look for**: Do the colors align with real parts (handle, body, rim)?
  Or are they just random spatial splits (top half vs bottom half)?

**Bottom row** — Silhouette plots:
- Each horizontal bar is one pixel; its length = how well it fits its assigned cluster
- Bars grouped and colored by cluster
- Red dashed line = average silhouette score
- **Longer bars = better separation.** If bars extend far right, that cluster is well-defined
- **Negative bars (left of 0)** = pixels assigned to the wrong cluster

**Silhouette score interpretation**:
| Score | Meaning |
|-------|---------|
| > 0.5 | Strong separation — clusters clearly distinct |
| 0.3 – 0.5 | Moderate — some structure but overlap |
| 0.15 – 0.3 | Weak — barely distinguishable clusters |
| < 0.15 | No structure — random assignment would be similar |

**Title shows** `k=N (silhouette=X.XX)` for each value of k.

---

### 2. `feature_vs_spatial.png`

**What it shows**: A critical control test comparing three clustering approaches,
all at k=3 (expecting handle/body/rim).

**Image 1 — RGB Input**: The original scene capture.

**Image 2 — DINOv2 Features**: Clusters based on DINOv2 feature similarity only.
Pixels are grouped because their 384-dim feature vectors are similar, regardless
of where they are on the image.

**Image 3 — Spatial Position Only**: Clusters based purely on pixel (x, y)
coordinates. No DINOv2 features — just "nearby pixels go together." This is the
**baseline/control**.

**Image 4 — Features + Spatial**: Combined clustering using both DINOv2 features
and spatial position (features weighted higher, spatial at 30%).

**How to read the title**: Shows whether feature clustering beats spatial clustering.
- **Feature > Spatial (positive difference)**: DINOv2 adds real semantic value —
  it groups pixels by what they ARE (handle vs body), not just where they are.
- **Feature <= Spatial (negative difference)**: DINOv2 is not adding useful part
  information — it's just encoding pixel position, which a simple spatial clustering
  does equally well or better.

---

## Key metrics

**Silhouette score**: Measures how similar each pixel is to its own cluster vs
the nearest other cluster. Ranges from -1 (bad) to +1 (perfect separation).

**Separation ratio** (inter/intra): Ratio of average distance between cluster
centers vs average distance of pixels to their own center. Higher = more separated.
- \> 2.0 = well-separated clusters
- 1.0 – 2.0 = some overlap
- < 1.0 = clusters heavily overlap

## Our results (mug scene)

| Metric | Value |
|--------|-------|
| Best k | 3 (silhouette=0.371) |
| Feature-only silhouette | 0.371 |
| Spatial-only silhouette | 0.395 |
| Feature vs spatial advantage | -0.024 |

**Conclusion**: DINOv2 feature clustering does NOT outperform simple spatial
clustering. The features have moderate structure (silhouette=0.37 is non-trivial),
but it's primarily driven by spatial position, not semantic part identity. This
means **DINOv2 is also part of the problem** in UAD's failure — the backbone
doesn't cleanly separate parts for our Habitat-rendered objects.
