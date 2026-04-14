"""
CLIPSegDetector — CLIPSeg-based affordance detection.

Uses the CLIPSeg model (CLIP + lightweight decoder) for text-guided
part segmentation on RGB images.  Drop-in alternative to AffordanceDetector.

Pipeline:
  1. Depth mask isolates the object
  2. Crop to object bounding box for higher resolution
  3. CLIPSeg(crop, text_query) → probability heatmap
  4. Threshold → binary mask
  5. Lift to 3D via depth map

Usage:
    detector = CLIPSegDetector(obj_name="hammer")
    result = detector.detect("hammer", "handle", rgb, depth, metadata)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict

import cv2
from PIL import Image as PILImage

PIPELINE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PIPELINE_DIR))
from objects import get_object, validate_part, get_part_info


class CLIPSegDetector:
    """Detects affordance regions using CLIPSeg (text-guided segmentation)."""

    PART_COLORS = {
        "handle": (100, 255, 150),
        "body":   (255, 200, 100),
        "rim":    (100, 200, 255),
        "head":   (100, 255, 255),
        "spout":  (255, 150, 200),
    }

    MODEL_NAME = "CIDAS/clipseg-rd64-refined"

    def __init__(self, threshold: float = 0.4, obj_name: str = None):
        self.threshold = threshold
        self.input_dir = PIPELINE_DIR / "output"

        if obj_name:
            self.results_dir = PIPELINE_DIR / "results" / obj_name / "clipseg"
        else:
            self.results_dir = PIPELINE_DIR / "results" / "clipseg"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._processor = None
        self._model = None
        self._device = None

    # ════════════════════════════════════════════════════════════════
    # MODEL LOADING (lazy)
    # ════════════════════════════════════════════════════════════════

    def _ensure_model(self):
        """Load CLIPSeg model on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        print(f"  Loading CLIPSeg model ({self.MODEL_NAME})...")
        self._processor = CLIPSegProcessor.from_pretrained(self.MODEL_NAME)
        self._model = CLIPSegForImageSegmentation.from_pretrained(self.MODEL_NAME)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()
        print(f"  CLIPSeg ready on {self._device}")

    # ════════════════════════════════════════════════════════════════
    # INFERENCE
    # ════════════════════════════════════════════════════════════════

    def _predict_heatmap(self, img_pil: PILImage.Image, query: str) -> np.ndarray:
        """Run CLIPSeg on a PIL image with a text query.

        Returns:
            Probability heatmap (H, W) float32 in [0, 1] at original image size.
        """
        import torch

        self._ensure_model()

        inputs = self._processor(
            text=[query], images=[img_pil],
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()

        # Resize to original image dimensions
        w, h = img_pil.size
        prob_resized = np.array(
            PILImage.fromarray(probs).resize((w, h), PILImage.BILINEAR)
        )
        return prob_resized

    # ════════════════════════════════════════════════════════════════
    # QUERY
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def build_query(obj_name: str, part_name: str) -> str:
        """Build text query — uses the simple part name from objects.json."""
        part_info = get_part_info(obj_name, part_name)
        q = part_info.get("query")
        if q:
            return q.replace("{obj}", obj_name)
        return part_name

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
        Detect an affordance region using CLIPSeg.

        Same interface as AffordanceDetector.detect() for drop-in use.

        Returns:
            Dict with: mask, world_points, pixel_indices,
            detection_confidence, text_query, num_3d_points, raw_heatmap
        """
        validate_part(obj_name, part_name)

        if query is None:
            query = self.build_query(obj_name, part_name)
        print(f"  CLIPSeg query: \"{query}\"")

        # ── Step 1: Depth-based object mask ─────────────────────────
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

        # ── Step 2: Crop to object bbox and run CLIPSeg ─────────────
        crop_bbox = None
        if obj_mask is not None and obj_mask.sum() > 0:
            ys, xs = np.where(obj_mask)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            H, W = rgb.shape[:2]
            pad_y = max(1, int((y1 - y0) * 0.15))
            pad_x = max(1, int((x1 - x0) * 0.15))
            y0c = max(0, y0 - pad_y)
            y1c = min(H, y1 + pad_y)
            x0c = max(0, x0 - pad_x)
            x1c = min(W, x1 + pad_x)
            crop_bbox = (y0c, x0c, y1c, x1c)
            crop_patch = rgb[y0c:y1c, x0c:x1c]
            crop_pil = PILImage.fromarray(crop_patch)
            print(f"  Crop: ({y0c},{x0c})→({y1c},{x1c})  "
                  f"{crop_patch.shape[1]}×{crop_patch.shape[0]}px")
        else:
            crop_pil = PILImage.fromarray(rgb)

        # Run CLIPSeg on the crop
        heatmap_crop = self._predict_heatmap(crop_pil, query)

        # ── Step 3: Map crop heatmap back to full image ─────────────
        if crop_bbox is not None:
            y0c, x0c, y1c, x1c = crop_bbox
            raw_heatmap = np.zeros(rgb.shape[:2], dtype=np.float32)
            raw_heatmap[y0c:y1c, x0c:x1c] = heatmap_crop
        else:
            raw_heatmap = heatmap_crop

        # ── Step 4: Threshold to binary mask ────────────────────────
        mask = raw_heatmap > self.threshold

        # Intersect with object depth mask
        if obj_mask is not None:
            n_before = int(mask.sum())
            mask = mask & obj_mask
            n_after = int(mask.sum())
            print(f"  Object filter: {n_before} → {n_after} pixels")

        n_pixels = int(mask.sum())
        if n_pixels == 0:
            # Fallback: try lower thresholds
            for t in [0.3, 0.2, 0.1, 0.05]:
                fallback = raw_heatmap > t
                if obj_mask is not None:
                    fallback = fallback & obj_mask
                n_pixels = int(fallback.sum())
                if n_pixels > 0:
                    mask = fallback
                    print(f"  Recovered {n_pixels} pixels at threshold={t}")
                    break

        if n_pixels == 0:
            print(f"  ERROR: No affordance pixels for '{query}'")
            return None

        # ── Step 5: Lift to 3D ──────────────────────────────────────
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

        confidence = float(raw_heatmap[mask].mean()) if n_pixels > 0 else 0.0
        print(f"  Detected {len(world_points)} 3D points for '{part_name}' "
              f"(confidence={confidence:.3f})")

        return {
            "mask": mask,
            "world_points": world_points,
            "pixel_indices": pixel_indices,
            "detection_confidence": confidence,
            "detection_label": f"{part_name} (CLIPSeg)",
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
        """Draw affordance visualization — same interface as AffordanceDetector."""
        from segment_utils import draw_segmentation_overlay

        part_color = self.PART_COLORS.get(part_name, (100, 255, 150))
        img_annotated = draw_segmentation_overlay(
            rgb, seg_result["mask"], part_name, color=part_color,
        )

        if grasp is not None and metadata is not None:
            self._draw_grasp_overlay(
                img_annotated, grasp, metadata,
                part_color, rgb.shape[1], rgb.shape[0],
            )
        return img_annotated

    def _draw_grasp_overlay(self, img, grasp, metadata, color, width, height):
        """Draw grasp crosshair and approach arrow."""
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
        from dataclasses import asdict

        text_query = seg_result["text_query"]

        # Annotated image
        out_path = self.results_dir / f"affordance_{obj_name}_{part_name}.png"
        PILImage.fromarray(img_annotated).save(out_path)
        print(f"  Saved: {out_path}")

        # 4-panel comparison
        fig, axes = plt.subplots(1, 4, figsize=(32, 8))

        axes[0].imshow(rgb)
        axes[0].set_title(f"Scene: {obj_name}", fontsize=16)
        axes[0].axis('off')

        if "raw_heatmap" in seg_result:
            axes[1].imshow(seg_result["raw_heatmap"], cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f'CLIPSeg: "{text_query}"', fontsize=13)
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

        # Raw heatmap
        if "raw_heatmap" in seg_result:
            heatmap_path = self.results_dir / f"heatmap_{obj_name}_{part_name}.npy"
            np.save(heatmap_path, seg_result["raw_heatmap"])
            print(f"  Saved: {heatmap_path}")

        # Grasp JSON
        grasp_json = {
            "object_name": obj_name,
            "part_name": part_name,
            "method": "clipseg",
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
# UTILITIES
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
