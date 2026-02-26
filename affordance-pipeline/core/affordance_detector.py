"""
AffordanceDetector — Language-guided part segmentation using DINO + SAM.

Wraps the language_segment module into a class interface for detecting
affordance regions (object parts) in RGB+depth scenes.

Handles:
  - Loading and caching DINO + SAM models
  - Detecting object part bounding boxes (Grounding DINO)
  - Generating precise segmentation masks (SAM)
  - Lifting 2D masks to 3D point clouds
  - PCA-based geometric part segmentation (fallback)

Usage:
    detector = AffordanceDetector()
    result = detector.detect("mug", "handle", rgb, depth, metadata)
    # result contains: mask, bbox, world_points, detection_confidence
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
from objects import get_object, validate_part


class AffordanceDetector:
    """Detects affordance regions using Grounding DINO + SAM."""

    # Part colors (BGR for OpenCV)
    PART_COLORS = {
        "handle": (100, 255, 150),
        "body":   (255, 200, 100),
        "rim":    (100, 200, 255),
        "chuck":  (150, 100, 255),
        "head":   (100, 255, 255),
        "spout":  (255, 150, 200),
    }

    def __init__(self, box_threshold: float = 0.25, text_threshold: float = 0.25):
        """
        Initialize affordance detector.

        Args:
            box_threshold:  Grounding DINO detection confidence threshold
            text_threshold: Grounding DINO text matching threshold
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Paths
        self.input_dir = PIPELINE_DIR / "output"
        self.results_dir = PIPELINE_DIR / "results" / "language"
        self.results_dir.mkdir(parents=True, exist_ok=True)

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
        prompt: str = None,
    ) -> Optional[Dict]:
        """
        Detect an affordance region for a specific object part.

        Args:
            obj_name:  Object name (e.g., "mug")
            part_name: Part name (e.g., "handle")
            rgb:       RGB image (H, W, 3) uint8
            depth:     Depth image (H, W) float32
            metadata:  Stage 1 metadata dict
            prompt:    Custom text prompt (auto-generated if None)

        Returns:
            Dict with keys: mask, bbox, world_points, detection_confidence,
            detection_label, sam_mask_pixels
            Or None if detection fails.
        """
        validate_part(obj_name, part_name)

        # Build text prompt
        if prompt is None:
            obj_cfg = get_object(obj_name)
            prompt = f"{part_name} of the {obj_name}"

        print(f"  Text prompt: \"{prompt}\"")

        # Load semantic mask if available
        semantic_raw = None
        semantic_id = None
        sem_path = self.input_dir / "semantic_raw.npy"
        if sem_path.exists():
            semantic_raw = np.load(sem_path)
            if metadata.get("spawned_objects"):
                semantic_id = metadata["spawned_objects"][0].get("semantic_id")
            print(f"  Semantic mask: loaded (object id={semantic_id})")

        # Run language-guided segmentation
        from language_segment import segment_part_by_language

        seg_result = segment_part_by_language(
            rgb, depth, prompt, metadata,
            part_name=part_name,
            obj_name=obj_name,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            semantic_raw=semantic_raw,
            semantic_id=semantic_id,
        )

        if seg_result is None:
            print(f"  ERROR: Language segmentation failed for '{prompt}'")
            return None

        print(f"  Segmented {len(seg_result['world_points'])} 3D points for '{part_name}'")
        return seg_result

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
            metadata:   Stage 1 metadata (needed for 3D→2D projection of grasp)

        Returns:
            Annotated image as numpy array
        """
        from language_segment import draw_segmentation_overlay

        part_color = self.PART_COLORS.get(part_name, (100, 255, 150))

        img_annotated = draw_segmentation_overlay(
            rgb, seg_result["mask"], seg_result["bbox"],
            part_name, color=part_color,
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
        prompt: str,
    ):
        """Save visualization images and grasp JSON."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
        from dataclasses import asdict

        # Single annotated image
        out_path = self.results_dir / f"affordance_{obj_name}_{part_name}.png"
        PILImage.fromarray(img_annotated).save(out_path)
        print(f"  Saved: {out_path}")

        # Comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        axes[0].imshow(rgb)
        axes[0].set_title(f"Scene: {obj_name}", fontsize=16)
        axes[0].axis('off')

        axes[1].imshow(seg_result["mask"], cmap='gray')
        axes[1].set_title(f"SAM Mask: \"{prompt}\"", fontsize=14)
        axes[1].axis('off')

        obj_cfg = get_object(obj_name)
        part_3d_count = len(seg_result["world_points"])
        axes[2].imshow(img_annotated)
        axes[2].set_title(
            f"Affordance: {part_name} ({part_3d_count} 3D pts)\n"
            f"Grasp: {grasp.grasp_type} @ {grasp.confidence:.0%}",
            fontsize=13,
        )
        axes[2].axis('off')

        plt.tight_layout()
        compare_path = self.results_dir / f"comparison_{obj_name}_{part_name}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {compare_path}")

        # Grasp JSON
        grasp_json = {
            "object_name": obj_name,
            "part_name": part_name,
            "method": "language",
            "text_prompt": prompt,
            "detection_confidence": seg_result["detection_confidence"],
            "detection_label": seg_result["detection_label"],
            "mask_pixels": seg_result["sam_mask_pixels"],
            "part_3d_points": part_3d_count,
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
