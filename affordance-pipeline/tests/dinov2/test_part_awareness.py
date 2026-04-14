#!/usr/bin/env python3
"""
DINOv2 Part-Awareness Test: K-Means Clustering

Definitive test: can DINOv2 features separate mug handle / body / rim?

Method:
  1. Extract DINOv2 patch features for the object
  2. K-means cluster them (k=2,3,4,5)
  3. Overlay cluster labels on the RGB image
  4. Compute silhouette score (how well-separated are the clusters?)
  5. Compute inter-cluster vs intra-cluster distance ratio
  6. Control test: compare feature clustering vs spatial-only clustering

Usage (from habitat-lab/):
  /home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/dinov2/test_part_awareness.py --object mug
  /home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/dinov2/test_part_awareness.py --object hammer
  /home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/dinov2/test_part_awareness.py --object drill

Outputs to: affordance-pipeline/results/diagnosis/dinov2/{object_name}/
See README.md in this folder for detailed explanation of the outputs.
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# ── Paths ──
PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
UAD_DIR = PIPELINE_DIR.parent / "unsup-affordance"
OUTPUT_DIR = PIPELINE_DIR / "output"
DIAG_BASE = PIPELINE_DIR / "results" / "diagnosis" / "dinov2"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(UAD_DIR))
sys.path.insert(0, str(UAD_DIR / "src"))


def get_object_config(obj_name):
    """Load object info from objects.json."""
    cfg_path = PIPELINE_DIR / "config" / "objects.json"
    with open(cfg_path) as f:
        catalog = json.load(f)
    if obj_name not in catalog:
        raise ValueError(f"Object '{obj_name}' not in {cfg_path}. Available: {[k for k in catalog if not k.startswith('_')]}")
    return catalog[obj_name]


def load_rgb():
    rgb = np.array(Image.open(OUTPUT_DIR / "rgb.png"))
    print(f"  RGB loaded: {rgb.shape}")
    return rgb


def load_depth_mask():
    dep = np.load(OUTPUT_DIR / "depth_raw.npy")
    valid = dep[dep > 0]
    near = float(np.percentile(valid, 2))
    cutoff = near * 1.3
    mask = (dep > 0) & (dep <= cutoff)
    print(f"  Depth mask: {int(mask.sum())} pixels")
    return mask


def extract_dinov2_features(rgb, obj_mask):
    """Extract DINOv2 features and return (features_map, obj_mask_resized)."""
    from utils.img_utils import load_pretrained_dino, get_dino_features

    dinov2 = load_pretrained_dino('dinov2_vits14', use_registers=True)
    print(f"  Model: dinov2_vits14_reg (384-dim)")

    features_full = get_dino_features(dinov2, rgb, blur=False, repeat_to_orig_size=True)
    features_full_np = features_full[0]  # (H', W', 384)

    fH, fW = features_full_np.shape[:2]
    iH, iW = obj_mask.shape
    if (fH, fW) != (iH, iW):
        print(f"  Resizing mask {iH}x{iW} -> {fH}x{fW}")
        obj_mask_r = cv2.resize(obj_mask.astype(np.uint8), (fW, fH),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        obj_mask_r = obj_mask

    print(f"  Feature map: {features_full_np.shape}, object pixels: {int(obj_mask_r.sum())}")
    return features_full_np, obj_mask_r


def test_kmeans_clustering(rgb, features_map, obj_mask, obj_name="object", num_parts=3, diag_dir=None):
    """
    K-means clustering on DINOv2 features.
    Tests k=2..5 and visualizes cluster assignments on RGB.
    """
    print("\n" + "=" * 60)
    print("TEST: K-Means Clustering on DINOv2 Features")
    print("=" * 60)

    obj_feats = features_map[obj_mask]  # (N, 384)
    N = len(obj_feats)
    print(f"  Object feature vectors: {N}")

    # L2-normalize features (cosine distance via k-means on unit sphere)
    obj_feats_norm = obj_feats / (np.linalg.norm(obj_feats, axis=1, keepdims=True) + 1e-8)

    fH, fW = features_map.shape[:2]
    iH, iW = rgb.shape[:2]

    # Distinct colors for cluster visualization
    cluster_colors = [
        [228, 26, 28],    # red
        [55, 126, 184],   # blue
        [77, 175, 74],    # green
        [152, 78, 163],   # purple
        [255, 127, 0],    # orange
    ]

    k_values = [2, 3, 4, 5]
    fig, axes = plt.subplots(2, len(k_values), figsize=(6 * len(k_values), 12))

    results = {}

    for ki, k in enumerate(k_values):
        print(f"\n  k={k}:")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(obj_feats_norm)

        # Silhouette score
        if k > 1:
            sil = silhouette_score(obj_feats_norm, labels)
            sil_samples = silhouette_samples(obj_feats_norm, labels)
        else:
            sil = 0.0
            sil_samples = np.zeros(N)

        # Intra-cluster vs inter-cluster distance
        centers = kmeans.cluster_centers_
        intra_dists = []
        for c in range(k):
            mask_c = labels == c
            if mask_c.sum() > 0:
                dists = np.linalg.norm(obj_feats_norm[mask_c] - centers[c], axis=1)
                intra_dists.append(dists.mean())

        inter_dists = []
        for i in range(k):
            for j in range(i + 1, k):
                inter_dists.append(np.linalg.norm(centers[i] - centers[j]))

        avg_intra = np.mean(intra_dists)
        avg_inter = np.mean(inter_dists) if inter_dists else 0
        separation_ratio = avg_inter / (avg_intra + 1e-8)

        print(f"    Silhouette score: {sil:.3f}  (>0.3 = reasonable, >0.5 = strong)")
        print(f"    Avg intra-cluster dist: {avg_intra:.4f}")
        print(f"    Avg inter-cluster dist: {avg_inter:.4f}")
        print(f"    Separation ratio (inter/intra): {separation_ratio:.2f}  (>2 = well-separated)")

        for c in range(k):
            count = (labels == c).sum()
            pct = 100.0 * count / N
            avg_sil = sil_samples[labels == c].mean() if k > 1 else 0
            print(f"    Cluster {c}: {count} pixels ({pct:.1f}%), avg silhouette={avg_sil:.3f}")

        results[k] = {
            'silhouette': sil,
            'separation_ratio': separation_ratio,
            'labels': labels,
        }

        # ── Row 0: Cluster overlay on RGB ──
        ax = axes[0, ki]
        # Create cluster image at feature resolution, then resize to RGB
        cluster_img = np.zeros((fH, fW, 3), dtype=np.uint8)
        ys, xs = np.where(obj_mask)
        for idx in range(N):
            c = labels[idx]
            cluster_img[ys[idx], xs[idx]] = cluster_colors[c]

        # Resize to original RGB size
        cluster_img_full = cv2.resize(cluster_img, (iW, iH), interpolation=cv2.INTER_NEAREST)
        obj_mask_full = cv2.resize(obj_mask.astype(np.uint8), (iW, iH),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        # Blend with RGB
        blended = rgb.copy()
        alpha = 0.55
        blended[obj_mask_full] = (
            (1 - alpha) * rgb[obj_mask_full] + alpha * cluster_img_full[obj_mask_full]
        ).astype(np.uint8)

        ax.imshow(blended)
        ax.set_title(f"k={k}  (silhouette={sil:.2f})", fontsize=13, fontweight='bold')
        ax.axis('off')

        # ── Row 1: Silhouette plot per cluster ──
        ax2 = axes[1, ki]
        if k > 1:
            y_lower = 0
            for c in range(k):
                c_sil = sil_samples[labels == c]
                c_sil.sort()
                size = len(c_sil)
                y_upper = y_lower + size
                color = np.array(cluster_colors[c]) / 255.0
                ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil,
                                  facecolor=color, alpha=0.7)
                ax2.text(-0.05, y_lower + 0.5 * size, str(c), fontsize=11, fontweight='bold')
                y_lower = y_upper

            ax2.axvline(x=sil, color='red', linestyle='--', linewidth=1.5, label=f'avg={sil:.2f}')
            ax2.set_xlabel("Silhouette coefficient", fontsize=11)
            ax2.set_ylabel("Pixels (sorted)", fontsize=11)
            ax2.legend(fontsize=10)
            ax2.set_title(f"Silhouette plot (k={k})", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "N/A for k=1", ha='center', va='center', fontsize=14)
            ax2.axis('off')

    plt.suptitle(
        f"DINOv2 K-Means Clustering — {obj_name} (expected {num_parts} parts)",
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    out_path = diag_dir / "kmeans_clustering.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {out_path}")

    # ── Summary ──
    print("\n  SUMMARY:")
    print("  " + "-" * 50)
    for k in k_values:
        r = results[k]
        print(f"  k={k}: silhouette={r['silhouette']:.3f}, separation={r['separation_ratio']:.2f}")

    best_k = max(k_values, key=lambda k: results[k]['silhouette'])
    best_sil = results[best_k]['silhouette']
    print(f"\n  Best k={best_k} (silhouette={best_sil:.3f})")

    if best_sil > 0.5:
        print("  → STRONG part separation. DINOv2 clearly sees distinct parts.")
    elif best_sil > 0.3:
        print("  → MODERATE part separation. DINOv2 distinguishes some parts.")
    elif best_sil > 0.15:
        print("  → WEAK part separation. Features have some structure but parts overlap.")
    else:
        print("  → POOR separation. DINOv2 does NOT clearly separate parts.")

    return results


def test_spatial_vs_feature_clustering(rgb, features_map, obj_mask, obj_name="object", num_parts=3, diag_dir=None):
    """
    Control test: compare feature-based clustering vs purely spatial clustering.
    If DINOv2 is part-aware, feature clustering should be BETTER than spatial clustering.
    If they're the same, DINOv2 is just encoding position, not semantics.
    """
    print("\n" + "=" * 60)
    print("CONTROL: Feature clustering vs Spatial-only clustering")
    print("=" * 60)

    obj_feats = features_map[obj_mask]
    obj_feats_norm = obj_feats / (np.linalg.norm(obj_feats, axis=1, keepdims=True) + 1e-8)

    # Get spatial coordinates of object pixels
    ys, xs = np.where(obj_mask)
    spatial = np.stack([xs, ys], axis=1).astype(np.float32)
    spatial_norm = (spatial - spatial.mean(axis=0)) / (spatial.std(axis=0) + 1e-8)

    fH, fW = features_map.shape[:2]
    iH, iW = rgb.shape[:2]

    cluster_colors = [
        [228, 26, 28],    # red
        [55, 126, 184],   # blue
        [77, 175, 74],    # green
        [152, 78, 163],   # purple
    ]

    k = num_parts

    # Feature-based clustering
    km_feat = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_feat = km_feat.fit_predict(obj_feats_norm)
    sil_feat = silhouette_score(obj_feats_norm, labels_feat)

    # Spatial-only clustering
    km_spat = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_spat = km_spat.fit_predict(spatial_norm)
    sil_spat = silhouette_score(spatial_norm, labels_spat)

    # Feature + spatial combined (30% spatial weight)
    combined = np.hstack([obj_feats_norm, 0.3 * spatial_norm])
    km_comb = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_comb = km_comb.fit_predict(combined)
    sil_comb = silhouette_score(combined, labels_comb)

    print(f"  k={k}")
    print(f"  Feature-only  silhouette: {sil_feat:.3f}")
    print(f"  Spatial-only  silhouette: {sil_spat:.3f}")
    print(f"  Combined      silhouette: {sil_comb:.3f}")

    # Visualize all 3
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Input", fontsize=14)
    axes[0].axis('off')

    obj_mask_full = cv2.resize(obj_mask.astype(np.uint8), (iW, iH),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

    for ax_idx, (name, labels, sil_val) in enumerate([
        ("DINOv2 Features", labels_feat, sil_feat),
        ("Spatial Position Only", labels_spat, sil_spat),
        ("Features + Spatial", labels_comb, sil_comb),
    ], start=1):
        cluster_img = np.zeros((fH, fW, 3), dtype=np.uint8)
        for idx in range(len(labels)):
            cluster_img[ys[idx], xs[idx]] = cluster_colors[labels[idx]]
        cluster_img_full = cv2.resize(cluster_img, (iW, iH), interpolation=cv2.INTER_NEAREST)

        blended = rgb.copy()
        blended[obj_mask_full] = (
            0.45 * rgb[obj_mask_full] + 0.55 * cluster_img_full[obj_mask_full]
        ).astype(np.uint8)

        axes[ax_idx].imshow(blended)
        axes[ax_idx].set_title(f"{name}\nsilhouette={sil_val:.3f}", fontsize=13, fontweight='bold')
        axes[ax_idx].axis('off')

    improvement = sil_feat - sil_spat
    pct = 100 * improvement / (abs(sil_spat) + 1e-8)

    fig.suptitle(
        f"Feature vs Spatial Clustering (k={k}) — "
        f"Feature silhouette {'>' if sil_feat > sil_spat else '<='} Spatial "
        f"({'+' if improvement > 0 else ''}{improvement:.3f}, {'+' if pct > 0 else ''}{pct:.0f}%)",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = diag_dir / "feature_vs_spatial.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    if sil_feat > sil_spat:
        print(f"\n  → DINOv2 features produce BETTER clusters than spatial position alone (+{improvement:.3f})")
        print(f"    This means DINOv2 encodes SEMANTIC part information, not just pixel location.")
    else:
        print(f"\n  → DINOv2 features do NOT produce better clusters than spatial position ({improvement:.3f})")
        print(f"    DINOv2 may be encoding position rather than semantic parts.")

    return {
        'sil_feature': sil_feat,
        'sil_spatial': sil_spat,
        'sil_combined': sil_comb,
    }


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 Part-Awareness Clustering Test")
    parser.add_argument("--object", required=True, help="Object name (must exist in objects.json)")
    args = parser.parse_args()

    obj_name = args.object
    obj_cfg = get_object_config(obj_name)
    num_parts = len(obj_cfg["parts"])
    part_names = list(obj_cfg["parts"].keys())

    diag_dir = DIAG_BASE / obj_name
    diag_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"DINOv2 Part-Awareness: {obj_name} ({num_parts} parts: {', '.join(part_names)})")
    print("=" * 60)

    rgb = load_rgb()
    obj_mask = load_depth_mask()

    features_map, obj_mask_r = extract_dinov2_features(rgb, obj_mask)

    # Test 1: K-means at multiple k values
    kmeans_results = test_kmeans_clustering(rgb, features_map, obj_mask_r,
                                            obj_name=obj_name, num_parts=num_parts,
                                            diag_dir=diag_dir)

    # Test 2: Feature clustering vs spatial-only clustering (control)
    control_results = test_spatial_vs_feature_clustering(rgb, features_map, obj_mask_r,
                                                         obj_name=obj_name, num_parts=num_parts,
                                                         diag_dir=diag_dir)

    print("\n" + "=" * 60)
    print(f"FINAL VERDICT — {obj_name}")
    print("=" * 60)

    best_k = max([2, 3, 4, 5], key=lambda k: kmeans_results[k]['silhouette'])
    best_sil = kmeans_results[best_k]['silhouette']
    feat_vs_spat = control_results['sil_feature'] - control_results['sil_spatial']

    print(f"  Object: {obj_name} (expected {num_parts} parts: {', '.join(part_names)})")
    print(f"  Best clustering: k={best_k}, silhouette={best_sil:.3f}")
    print(f"  Feature vs spatial advantage: {feat_vs_spat:+.3f}")

    if best_sil > 0.3 and feat_vs_spat > 0:
        print("\n  CONCLUSION: DINOv2 IS part-aware.")
        print("     Features encode semantic part structure beyond spatial position.")
        print("     The problem is NOT DINOv2. UAD's FiLM network is the failure point.")
    elif best_sil > 0.15 and feat_vs_spat > 0:
        print("\n  CONCLUSION: DINOv2 has SOME part structure, but it's weak.")
        print("     May contribute partially to UAD's failure.")
    else:
        print("\n  CONCLUSION: DINOv2 does NOT separate parts well.")
        print("     DINOv2 features may be part of the problem.")

    print("\n  Output images:")
    print(f"    {diag_dir / 'kmeans_clustering.png'}")
    print(f"    {diag_dir / 'feature_vs_spatial.png'}")
