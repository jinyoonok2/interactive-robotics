"""
Segment Utilities — Model-agnostic helpers for mask lifting and visualization.

Functions retained from the original language_segment.py that are
independent of any particular affordance/detection model:

  - lift_mask_to_3d:  2D binary mask + depth → 3D world point cloud
  - draw_segmentation_overlay: annotate an RGB image with mask + bbox

These are used by the AffordanceDetector regardless of whether the mask
comes from UAD, SAM, or any other source.
"""

import numpy as np
import cv2
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
# DEPTH LIFTING: 2D Mask → 3D Point Cloud
# ═══════════════════════════════════════════════════════════════════════════

def lift_mask_to_3d(
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sensor_R: np.ndarray,
    sensor_t: np.ndarray,
    max_depth: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lift a 2D binary mask to 3D world coordinates using depth.

    Args:
        mask: (H, W) binary mask
        depth: (H, W) depth map in meters
        fx, fy: focal lengths in pixels
        cx, cy: principal point
        sensor_R: 3x3 camera-to-world rotation
        sensor_t: (3,) camera position in world
        max_depth: maximum valid depth

    Returns:
        world_points: (N, 3) array of 3D world coordinates
        pixel_indices: (N, 2) array of (v, u) pixel coordinates
    """
    H, W = mask.shape

    # Get masked pixel coordinates
    vs, us = np.where(mask)

    # Filter by valid depth
    depths = depth[vs, us]
    valid = (depths > 0) & (depths < max_depth)
    vs, us, depths = vs[valid], us[valid], depths[valid]

    if len(depths) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)

    # Back-project to camera frame (Habitat convention)
    # Habitat camera: X-right, Y-up, Z-backward (looking along -Z)
    x_cam = (us.astype(np.float64) - cx) * depths / fx
    y_cam = -(vs.astype(np.float64) - cy) * depths / fy  # flip Y
    z_cam = -depths  # negative Z (camera looks along -Z)

    cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Transform to world frame
    R = np.array(sensor_R, dtype=np.float64)
    t = np.array(sensor_t, dtype=np.float64)
    world_points = (R @ cam_points.T).T + t

    pixel_indices = np.stack([vs, us], axis=-1)

    print(f"  Lifted {len(world_points)} mask pixels to 3D world coordinates")

    return world_points, pixel_indices


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_segmentation_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    part_name: str,
    color: Tuple[int, int, int] = (100, 255, 150),
    alpha: float = 0.5,
    bbox: list = None,
) -> np.ndarray:
    """
    Draw the segmentation mask overlay on the RGB image.

    Args:
        rgb: RGB image (H, W, 3)
        mask: binary mask (H, W)
        part_name: label text
        color: RGB color for the overlay
        alpha: transparency
        bbox: optional [x1, y1, x2, y2] bounding box to draw

    Returns:
        Annotated RGB image
    """
    img = rgb.copy()

    # Create colored overlay
    overlay = img.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw mask contours
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 2)

    # Draw bounding box if provided
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Compute label position from mask if no bbox
    if bbox is not None:
        lx, ly = int(bbox[0]), int(bbox[1])
    else:
        ys, xs = np.where(mask)
        if len(ys) > 0:
            lx, ly = int(xs.min()), int(ys.min())
        else:
            lx, ly = 10, 30

    # Label
    label = f"{part_name}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (lx, ly - th - 10), (lx + tw + 10, ly), (0, 0, 0), -1)
    cv2.putText(img, label, (lx + 5, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return img
