"""
AffordanceDetector — UAD-based affordance detection.

Uses the Unsupervised Affordance Detection (UAD) model to identify
affordance regions (object parts) in RGB+depth scenes.  Replaces the
previous Grounding DINO + SAM approach.

Pipeline:
  1. UAD produces an affordance heatmap from RGB + text query
  2. Heatmap is thresholded to a binary mask
  3. Mask is optionally intersected with semantic ground truth
  4. Mask is lifted to 3D world coordinates via depth map

Usage:
    detector = AffordanceDetector()
    result = detector.detect("mug", "handle", rgb, depth, metadata)
    # result contains: mask, world_points, detection_confidence, ...
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict

import cv2

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

PIPELINE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PIPELINE_DIR))
from objects import get_object, validate_part, get_part_info


class AffordanceDetector:
    """Detects affordance regions using UAD (unsupervised affordance detection)."""

    # Part colors (BGR for OpenCV)
    PART_COLORS = {
        "handle": (100, 255, 150),
        "body":   (255, 200, 100),
        "rim":    (100, 200, 255),
        "chuck":  (150, 100, 255),
        "head":   (100, 255, 255),
        "spout":  (255, 150, 200),
    }

    def __init__(self, threshold: float = 0.5, obj_name: str = None):
        """
        Initialize affordance detector.

        Args:
            threshold: UAD affordance binarization threshold (0-1).
                       Higher = more selective, lower = more inclusive.
            obj_name:  Object name used for per-object output directory.
                       If None, falls back to flat results/affordance/.
        """
        self.threshold = threshold

        # Paths
        self.input_dir = PIPELINE_DIR / "output"
        if obj_name:
            self.results_dir = PIPELINE_DIR / "results" / obj_name / "uad"
        else:
            self.results_dir = PIPELINE_DIR / "results" / "uad"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # TEXT QUERY GENERATION
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def build_query(obj_name: str, part_name: str) -> str:
        """
        Build a natural language affordance query for UAD.

        Reads the per-part ``query`` field from objects.json.  The query
        may contain ``{obj}`` which is replaced with the object name.
        If no query is configured, falls back to a generic template.

        Args:
            obj_name:  Object name (e.g. "mug")
            part_name: Part name (e.g. "handle")

        Returns:
            Text query string
        """
        part_info = get_part_info(obj_name, part_name)
        q = part_info.get("query")
        if q:
            return q.replace("{obj}", obj_name)
        # Fallback: generic action-oriented query
        return f"grasp the {part_name} of the {obj_name}"

    # ════════════════════════════════════════════════════════════════
    # GEOMETRIC PART PRIORS
    # ════════════════════════════════════════════════════════════════
    #
    # Each prior produces a soft (H, W) float32 map in [0, 1] from the
    # object’s binary semantic mask.  Higher values mean “more likely this
    # part”.  The prior is later multiplied into the UAD heatmap so that
    # irrelevant regions (e.g. the mug body) get suppressed.
    #
    # Prior types (matching the "prior" field in objects.json):
    #   protrusion – thin appendages that stick out (handles, grips)
    #   extremity  – pixels far from the centroid (chuck, head)
    #   boundary   – pixels near the mask border (rim, edge)
    #   none       – no geometric prior applied
    # ────────────────────────────────────────────────────────────────

    PRIOR_DISPATCH = {}          # filled at end of prior section
    BODY_SCALE     = 0.3         # how much to suppress non-prior pixels
    PRIOR_THRESH   = 0.2         # binary threshold when a prior is active

    def compute_prior(self, prior_type: str, obj_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Dispatch to the right prior by name.

        Returns:
            Prior map (H, W) float32 in [0, 1], or None if type is "none".
        """
        if prior_type == "none" or prior_type not in self.PRIOR_DISPATCH:
            return None
        fn = self.PRIOR_DISPATCH[prior_type]
        return fn(obj_mask)

    # ──── erosion thickness (shared helper) ───────────────────────────

    @staticmethod
    def _erosion_thickness(mask_u8: np.ndarray, max_iter: int = 60) -> np.ndarray:
        """
        Compute how many erosion iterations each pixel survives.

        Thin regions disappear first (→ low values), thick body regions
        survive longest (→ high values).

        Returns:
            thickness (H, W) float32 — raw erosion count per pixel.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thickness = np.zeros_like(mask_u8, dtype=np.float32)
        remaining = mask_u8.copy()
        last_i = 0
        for i in range(1, max_iter + 1):
            eroded = cv2.erode(remaining, kernel)
            removed = (remaining > 0) & (eroded == 0)
            thickness[removed] = i
            remaining = eroded
            last_i = i
            if remaining.sum() == 0:
                break
        thickness[remaining > 0] = last_i + 1
        return thickness

    # ──── protrusion prior ─────────────────────────────────────────

    @staticmethod
    def _prior_protrusion(obj_mask: np.ndarray) -> np.ndarray:
        """
        Highlight thin protrusions that stick out of the main body.

        Combines:
          • inverse erosion thickness (thin → high)
          • low cross-section occupancy along *both* axes (direction-agnostic)
        """
        if obj_mask.sum() == 0:
            return np.zeros_like(obj_mask, dtype=np.float32)

        mask_u8 = obj_mask.astype(np.uint8)

        # Erosion thinness
        thickness = AffordanceDetector._erosion_thickness(mask_u8)
        max_t = thickness.max() or 1.0
        thin_score = 1.0 - (thickness / max_t)
        thin_score[~obj_mask] = 0

        # Cross-section occupancy — direction-agnostic
        # Take per-column and per-row counts, keep the smaller one per pixel.
        col_occ = obj_mask.astype(np.float32).sum(axis=0)   # (W,)
        row_occ = obj_mask.astype(np.float32).sum(axis=1)   # (H,)
        max_col = col_occ.max() or 1.0
        max_row = row_occ.max() or 1.0
        col_thin = 1.0 - (col_occ / max_col)                # 0–1
        row_thin = 1.0 - (row_occ / max_row)                # 0–1

        # Broadcast both to (H, W) and take element-wise max
        section_score = np.maximum(
            np.broadcast_to(col_thin[None, :], obj_mask.shape),
            np.broadcast_to(row_thin[:, None], obj_mask.shape),
        ).copy()
        section_score[~obj_mask] = 0

        # Combine
        prior = 0.5 * thin_score + 0.5 * section_score
        pmax = prior.max()
        if pmax > 0:
            prior /= pmax

        return prior

    # ──── extremity prior ─────────────────────────────────────────

    @staticmethod
    def _prior_extremity(obj_mask: np.ndarray) -> np.ndarray:
        """
        Highlight pixels far from the object centroid.

        Useful for tips / ends like the hammer head.
        """
        if obj_mask.sum() == 0:
            return np.zeros_like(obj_mask, dtype=np.float32)

        ys, xs = np.where(obj_mask)
        cy, cx = ys.mean(), xs.mean()

        # Euclidean distance from centroid (only within mask)
        yy, xx = np.mgrid[:obj_mask.shape[0], :obj_mask.shape[1]]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
        dist[~obj_mask] = 0

        dmax = dist.max() or 1.0
        prior = dist / dmax
        return prior

    # ──── boundary prior ──────────────────────────────────────────

    @staticmethod
    def _prior_boundary(obj_mask: np.ndarray) -> np.ndarray:
        """
        Highlight pixels near the mask boundary (outline).

        Useful for rims, edges, and spouts.
        """
        if obj_mask.sum() == 0:
            return np.zeros_like(obj_mask, dtype=np.float32)

        mask_u8 = obj_mask.astype(np.uint8)

        # Distance to the nearest background pixel (inverse = near edge)
        dist_interior = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        dmax = dist_interior.max() or 1.0
        # Invert: edge → high, interior → low
        prior = 1.0 - (dist_interior / dmax)
        prior[~obj_mask] = 0
        return prior.astype(np.float32)

    # Register all priors
    PRIOR_DISPATCH["protrusion"] = _prior_protrusion.__func__
    PRIOR_DISPATCH["extremity"]  = _prior_extremity.__func__
    PRIOR_DISPATCH["boundary"]   = _prior_boundary.__func__

    # ════════════════════════════════════════════════════════════════
    # MAIN DETECTION
    # ════════════════════════════════════════════════════════════════

    def detect(
        self,
        obj_name: str,
        part_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        metadata: dict,
        query: str = None,
    ) -> Optional[Dict]:
        """
        Detect an affordance region for a specific object part.

        Args:
            obj_name:  Object name (e.g., "mug")
            part_name: Part name (e.g., "handle")
            rgb:       RGB image (H, W, 3) uint8
            depth:     Depth image (H, W) float32
            metadata:  Stage 1 metadata dict
            query:     Custom text query (auto-generated if None)

        Returns:
            Dict with keys: mask, world_points, pixel_indices,
            detection_confidence, text_query, num_3d_points, raw_heatmap
            Or None if detection fails.
        """
        validate_part(obj_name, part_name)

        # Build text query
        if query is None:
            query = self.build_query(obj_name, part_name)

        print(f"  UAD query: \"{query}\"")

        # ── Step 1: Build depth-based object mask ────────────────────────────
        # Dynamically spawned objects are not in Habitat's semantic map, so we
        # use depth: the YCB object is the nearest thing in the scene.
        obj_mask = None
        dep_path = self.input_dir / "depth_raw.npy"
        if dep_path.exists():
            dep = np.load(dep_path)
            valid = dep[dep > 0]
            if len(valid) > 0:
                near = float(np.percentile(valid, 2))
                cutoff = near * 1.3
                obj_mask = (dep > 0) & (dep <= cutoff)
                print(f"  Depth mask: near={near:.2f}m  cutoff={cutoff:.2f}m  "
                      f"→ {int(obj_mask.sum())} pixels")

        # ── Step 2: Crop object region and upscale to UAD input size ─────────
        # UAD (DINOv2) was evaluated on tight object crops (~224×224).
        # Passing the full 512×512 scene gives the object only ~11×11 patches,
        # which is too coarse for part-level discrimination.
        # We crop the depth-mask bounding box (+ 10 % padding) and upscale to
        # 224×224 so the object fills the frame as in the paper's conditions.
        UAD_SIZE = 224
        crop_rgb   = rgb          # fallback: use full image
        crop_bbox  = None         # (y0, x0, y1, x1) in full-image coords
        if obj_mask is not None and obj_mask.sum() > 0:
            ys, xs = np.where(obj_mask)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            # Add 10 % padding on each side, clamped to image bounds
            H, W = rgb.shape[:2]
            pad_y = max(1, int((y1 - y0) * 0.10))
            pad_x = max(1, int((x1 - x0) * 0.10))
            y0c = max(0, y0 - pad_y)
            y1c = min(H, y1 + pad_y)
            x0c = max(0, x0 - pad_x)
            x1c = min(W, x1 + pad_x)
            crop_bbox = (y0c, x0c, y1c, x1c)
            crop_patch = rgb[y0c:y1c, x0c:x1c]
            # Upscale to UAD_SIZE × UAD_SIZE
            from PIL import Image as _PILImage
            crop_rgb = np.array(
                _PILImage.fromarray(crop_patch).resize(
                    (UAD_SIZE, UAD_SIZE), _PILImage.BILINEAR
                )
            )
            print(f"  Crop: ({y0c},{x0c})→({y1c},{x1c})  "
                  f"patch={crop_patch.shape[1]}×{crop_patch.shape[0]}px  "
                  f"→ upscaled to {UAD_SIZE}×{UAD_SIZE}")

        # ── Step 3: Run UAD on the (possibly cropped+upscaled) image ─────────
        from uad_bridge import predict_affordance_raw

        try:
            heatmap_uad = predict_affordance_raw(crop_rgb, query)
        except Exception as e:
            print(f"  ERROR: UAD inference failed: {e}")
            return None

        # ── Step 4: Map cropped heatmap back to full-image coordinates ────────
        if crop_bbox is not None:
            y0c, x0c, y1c, x1c = crop_bbox
            patch_h = y1c - y0c
            patch_w = x1c - x0c
            from PIL import Image as _PILImage
            # Resize heatmap back to the original crop patch size
            hmap_img = _PILImage.fromarray((heatmap_uad * 255).astype(np.uint8))
            hmap_patch = np.array(
                hmap_img.resize((patch_w, patch_h), _PILImage.BILINEAR)
            ).astype(np.float32) / 255.0
            # Place into full-image canvas (background = 0)
            raw_heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
            raw_heatmap[y0c:y1c, x0c:x1c] = hmap_patch
        else:
            raw_heatmap = heatmap_uad

        # Apply geometric prior from config (protrusion / extremity / boundary).
        # The prior suppresses regions that don’t match the target part shape,
        # so the thresholded mask concentrates on the correct sub-region.
        part_info = get_part_info(obj_name, part_name)
        prior_type = part_info.get("prior", "none")
        effective_heatmap = raw_heatmap
        prior_active = False

        if prior_type != "none" and obj_mask is not None:
            prior_map = self.compute_prior(prior_type, obj_mask)
            if prior_map is not None:
                prior_pixels = int((prior_map > 0.5).sum())
                if prior_pixels > 0:
                    blend = self.BODY_SCALE + (1.0 - self.BODY_SCALE) * prior_map
                    effective_heatmap = raw_heatmap * blend
                    prior_active = True
                    print(f"  {prior_type} prior applied "
                          f"(prior pixels={prior_pixels})")

        # Use a lower threshold when a geometric prior is active, because
        # the multiplicative suppression pulls values below the default
        # threshold.
        active_threshold = self.PRIOR_THRESH if prior_active else self.threshold

        # Threshold to binary mask
        mask = effective_heatmap > active_threshold

        # Intersect with object mask (depth-based)
        if obj_mask is not None:
            n_before = int(mask.sum())
            mask = mask & obj_mask
            n_after = int(mask.sum())
            print(f"  Object filter: {n_before} -> {n_after} pixels")

        n_pixels = int(mask.sum())
        if n_pixels == 0:
            print(f"  WARNING: No affordance pixels at threshold={self.threshold}")
            # Try progressively lower thresholds (using effective_heatmap
            # so the handle prior is still active)
            for t in [0.3, 0.2, 0.1]:
                fallback_mask = effective_heatmap > t
                if obj_mask is not None:
                    fallback_mask = fallback_mask & obj_mask
                n_pixels = int(fallback_mask.sum())
                if n_pixels > 0:
                    mask = fallback_mask
                    print(f"  Recovered {n_pixels} pixels at threshold={t}")
                    break

        if n_pixels == 0:
            print(f"  ERROR: No affordance pixels detected for '{query}'")
            return None

        # Lift to 3D
        from segment_utils import lift_mask_to_3d

        fx = metadata["focal_length_px"]
        fy = fx
        cx, cy = metadata["principal_point"]
        sensor_R = metadata["sensor_rotation_matrix"]
        sensor_t = metadata["sensor_position"]
        max_depth = metadata.get("max_depth_m", 10.0)

        world_points, pixel_indices = lift_mask_to_3d(
            mask, depth, fx, fy, cx, cy,
            sensor_R, sensor_t, max_depth,
        )

        if len(world_points) == 0:
            print(f"  ERROR: No valid 3D points after depth lifting")
            return None

        # Compute confidence as mean heatmap value in masked region
        confidence = float(raw_heatmap[mask].mean()) if n_pixels > 0 else 0.0

        print(f"  Detected {len(world_points)} 3D points for '{part_name}' "
              f"(confidence={confidence:.3f})")

        return {
            "mask": mask,
            "world_points": world_points,
            "pixel_indices": pixel_indices,
            "detection_confidence": confidence,
            "detection_label": f"{part_name} (UAD)",
            "text_query": query,
            "mask_pixels": n_pixels,
            "num_3d_points": len(world_points),
            "raw_heatmap": raw_heatmap,
        }

    # ════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ════════════════════════════════════════════════════════════════

    def visualize(
        self,
        rgb: np.ndarray,
        seg_result: dict,
        obj_name: str,
        part_name: str,
        grasp=None,
        metadata: dict = None,
    ) -> np.ndarray:
        """
        Draw affordance visualization on RGB image.

        Args:
            rgb:        Original RGB image
            seg_result: Result from detect()
            obj_name:   Object name
            part_name:  Part name
            grasp:      Optional GraspPose for crosshair overlay
            metadata:   Stage 1 metadata (needed for 3D->2D projection of grasp)

        Returns:
            Annotated image as numpy array
        """
        from segment_utils import draw_segmentation_overlay

        part_color = self.PART_COLORS.get(part_name, (100, 255, 150))

        img_annotated = draw_segmentation_overlay(
            rgb, seg_result["mask"], part_name, color=part_color,
        )

        # Overlay grasp crosshair if provided
        if grasp is not None and metadata is not None:
            self._draw_grasp_overlay(
                img_annotated, grasp, metadata,
                part_color, rgb.shape[1], rgb.shape[0],
            )

        return img_annotated

    def _draw_grasp_overlay(self, img, grasp, metadata, color, width, height):
        """Draw grasp crosshair and approach arrow on image."""
        sensor_R = metadata["sensor_rotation_matrix"]
        sensor_t = metadata["sensor_position"]
        fx = metadata["focal_length_px"]
        cx, cy = metadata["principal_point"]

        grasp_pos = np.array(grasp.position)
        grasp_px = _project_3d_to_2d(grasp_pos, sensor_R, sensor_t, fx, cx, cy, width, height)

        if grasp_px is not None:
            cv2.drawMarker(img, grasp_px, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            approach_end = grasp_pos + np.array(grasp.approach_dir) * 0.15
            end_px = _project_3d_to_2d(approach_end, sensor_R, sensor_t, fx, cx, cy, width, height)
            if end_px is not None:
                cv2.arrowedLine(img, grasp_px, end_px, (255, 255, 255), 2, tipLength=0.3)

        # Label
        if hasattr(grasp, 'grasp_type'):
            label = f"{grasp.object_name}/{grasp.part_name}: {grasp.grasp_type} (conf={grasp.confidence:.0%})"
        else:
            label = f"{grasp.part_name}"

        lx = grasp_px[0] + 15 if grasp_px else 10
        ly = grasp_px[1] - 15 if grasp_px else height - 40

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), (0, 0, 0), -1)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ════════════════════════════════════════════════════════════════

    def save_results(
        self,
        rgb: np.ndarray,
        img_annotated: np.ndarray,
        seg_result: dict,
        grasp,
        obj_name: str,
        part_name: str,
        metadata: dict,
    ):
        """Save visualization images and grasp JSON."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
        from dataclasses import asdict

        text_query = seg_result["text_query"]

        # Single annotated image
        out_path = self.results_dir / f"affordance_{obj_name}_{part_name}.png"
        PILImage.fromarray(img_annotated).save(out_path)
        print(f"  Saved: {out_path}")

        # Comparison figure: original | raw heatmap | binary mask | overlay
        fig, axes = plt.subplots(1, 4, figsize=(32, 8))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Scene: {obj_name}", fontsize=16)
        axes[0].axis('off')

        if "raw_heatmap" in seg_result:
            axes[1].imshow(seg_result["raw_heatmap"], cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f"UAD Heatmap: \"{text_query}\"", fontsize=13)
        else:
            axes[1].imshow(np.zeros_like(rgb))
            axes[1].set_title("(no heatmap)", fontsize=13)
        axes[1].axis('off')

        axes[2].imshow(seg_result["mask"], cmap='gray')
        axes[2].set_title(f"Binary Mask (thresh={self.threshold})", fontsize=14)
        axes[2].axis('off')

        part_3d_count = seg_result["num_3d_points"]
        axes[3].imshow(img_annotated)
        axes[3].set_title(
            f"Affordance: {part_name} ({part_3d_count} 3D pts)\n"
            f"Grasp: {grasp.grasp_type} @ {grasp.confidence:.0%}",
            fontsize=13,
        )
        axes[3].axis('off')

        plt.tight_layout()
        compare_path = self.results_dir / f"comparison_{obj_name}_{part_name}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {compare_path}")

        # Save raw heatmap as npy for debugging
        if "raw_heatmap" in seg_result:
            heatmap_path = self.results_dir / f"heatmap_{obj_name}_{part_name}.npy"
            np.save(heatmap_path, seg_result["raw_heatmap"])
            print(f"  Saved: {heatmap_path}")

        # Grasp JSON
        grasp_json = {
            "object_name": obj_name,
            "part_name": part_name,
            "method": "uad",
            "text_query": text_query,
            "detection_confidence": seg_result["detection_confidence"],
            "detection_label": seg_result["detection_label"],
            "mask_pixels": seg_result["mask_pixels"],
            "part_3d_points": part_3d_count,
            "threshold": self.threshold,
            "grasp": asdict(grasp),
        }
        json_path = self.results_dir / "grasp_poses.json"
        with open(json_path, "w") as f:
            json.dump(grasp_json, f, indent=2)
        print(f"  Saved: {json_path}")


# ════════════════════════════════════════════════════════════════════════
# UTILITIES (module-level for reuse)
# ════════════════════════════════════════════════════════════════════════

def _project_3d_to_2d(point_3d, sensor_R, sensor_t, fx, cx, cy, width, height):
    """Project a world 3D point to 2D image coordinates."""
    R_w2c = np.array(sensor_R).T
    p_cam = R_w2c @ (np.array(point_3d) - np.array(sensor_t))
    x_c, y_c, z_c = p_cam

    if z_c > -0.01:
        return None

    u = fx * (x_c / (-z_c)) + cx
    v = fx * (-y_c / (-z_c)) + cy

    if 0 <= u < width and 0 <= v < height:
        return (int(u), int(v))
    return None
