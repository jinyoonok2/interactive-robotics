"""
Language-Guided Part Segmentation using Grounding DINO + SAM
=============================================================
Replaces hand-coded geometric segmentation with vision-language models.

Given an RGB image and a text prompt (e.g., "handle of the mug"),
produces a pixel-precise mask, then lifts to 3D using the depth map.

Models used:
  - Grounding DINO (IDEA Research): open-vocabulary object detection
  - SAM (Meta): precise segmentation from bounding box prompts

Usage (standalone test):
    cd habitat-lab
    python ../affordance-pipeline/language_segment.py --prompt "handle" --visualize
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import torch
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
INPUT_DIR    = PIPELINE_DIR / "output"
RESULTS_BASE = PIPELINE_DIR / "results"

# ── Model IDs ────────────────────────────────────────────────────────────
GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

# Local cache for SAM checkpoint
SAM_CACHE_DIR = PIPELINE_DIR / "models"
SAM_CHECKPOINT_PATH = SAM_CACHE_DIR / "sam_vit_h_4b8939.pth"


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (lazy, cached)
# ═══════════════════════════════════════════════════════════════════════════

_grounding_dino_model = None
_grounding_dino_processor = None
_sam_predictor = None


def _get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_grounding_dino():
    """Load Grounding DINO model (cached after first call)."""
    global _grounding_dino_model, _grounding_dino_processor

    if _grounding_dino_model is not None:
        return _grounding_dino_model, _grounding_dino_processor

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print("  Loading Grounding DINO model...")
    device = _get_device()

    _grounding_dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
    _grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_ID
    ).to(device)
    _grounding_dino_model.eval()

    print(f"  Grounding DINO loaded on {device}")
    return _grounding_dino_model, _grounding_dino_processor


def _download_sam_checkpoint():
    """Download SAM checkpoint if not cached locally."""
    if SAM_CHECKPOINT_PATH.exists():
        return SAM_CHECKPOINT_PATH

    SAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading SAM checkpoint (~2.4GB)...")
    print(f"  URL: {SAM_CHECKPOINT_URL}")
    print(f"  Saving to: {SAM_CHECKPOINT_PATH}")

    import urllib.request
    urllib.request.urlretrieve(SAM_CHECKPOINT_URL, str(SAM_CHECKPOINT_PATH))
    print(f"  Download complete!")
    return SAM_CHECKPOINT_PATH


def _load_sam():
    """Load SAM model (cached after first call)."""
    global _sam_predictor

    if _sam_predictor is not None:
        return _sam_predictor

    from segment_anything import sam_model_registry, SamPredictor

    checkpoint_path = _download_sam_checkpoint()

    print("  Loading SAM model...")
    device = _get_device()

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint_path))
    sam.to(device)
    sam.eval()

    _sam_predictor = SamPredictor(sam)
    print(f"  SAM loaded on {device}")
    return _sam_predictor


# ═══════════════════════════════════════════════════════════════════════════
# GROUNDING DINO: Text → Bounding Box
# ═══════════════════════════════════════════════════════════════════════════

def detect_part_bbox(
    image: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
) -> Optional[Tuple[np.ndarray, float, str]]:
    """
    Detect an object/part in the image using a text prompt.

    Args:
        image: RGB numpy array (H, W, 3)
        text_prompt: e.g., "handle", "handle of the mug", "rim"
        box_threshold: confidence threshold for detection
        text_threshold: text matching threshold

    Returns:
        (bbox_xyxy, confidence, label) or None if nothing detected
        bbox_xyxy: [x1, y1, x2, y2] in pixel coordinates
    """
    model, processor = _load_grounding_dino()
    device = _get_device()

    # Grounding DINO expects the prompt to end with a period
    prompt = text_prompt.strip()
    if not prompt.endswith("."):
        prompt += "."

    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process  (transformers ≥4.57 uses 'threshold' not 'box_threshold')
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[pil_image.size[::-1]],  # (H, W)
    )[0]

    boxes = results["boxes"].cpu().numpy()     # (N, 4) in xyxy format
    scores = results["scores"].cpu().numpy()   # (N,)
    labels = results["labels"]                 # list of strings

    if len(boxes) == 0:
        print(f"  No detection for prompt '{text_prompt}' "
              f"(box_thresh={box_threshold}, text_thresh={text_threshold})")
        return None

    # Pick the highest confidence detection
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]
    best_score = float(scores[best_idx])
    best_label = labels[best_idx]

    print(f"  Detected '{best_label}' with confidence {best_score:.3f}")
    print(f"  Bounding box: [{best_box[0]:.0f}, {best_box[1]:.0f}, "
          f"{best_box[2]:.0f}, {best_box[3]:.0f}]")

    if len(boxes) > 1:
        print(f"  ({len(boxes)} total detections, using best)")

    return best_box, best_score, best_label


# ═══════════════════════════════════════════════════════════════════════════
# SAM: Bounding Box → Precise Mask
# ═══════════════════════════════════════════════════════════════════════════

def segment_from_bbox(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
    bbox_padding: float = 0.10,
    crop_context: float = 3.0,
) -> np.ndarray:
    """
    Create a precise segmentation mask from a bounding box using SAM.

    **Crop-and-zoom strategy**: the image is first cropped to a region
    ``crop_context`` × the bbox size so SAM sees the part at much higher
    resolution (critical when the object is small). A centre-point prompt
    guides SAM toward the part, and the smallest qualifying mask is chosen
    to avoid whole-object leakage. Finally, the mask is clipped to the
    bbox (with a small padding) and mapped back to the original image
    coordinates.

    Args:
        image: RGB numpy array (H, W, 3)
        bbox_xyxy: [x1, y1, x2, y2] bounding box in original coords
        bbox_padding: fraction of bbox size to pad when clipping (0.10 = 10%)
        crop_context: how many bbox-widths/heights the crop region covers
                      (3.0 = crop is 3× the bbox size)

    Returns:
        Binary mask (H, W) as bool array
    """
    predictor = _load_sam()
    H, W = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1

    # ── 1. Compute crop region (crop_context × bbox, centred on bbox) ──
    bbox_cx, bbox_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_half_w = bw * crop_context / 2.0
    crop_half_h = bh * crop_context / 2.0
    # Make it square for best SAM utilisation
    crop_half = max(crop_half_w, crop_half_h)

    crop_x1 = max(int(bbox_cx - crop_half), 0)
    crop_y1 = max(int(bbox_cy - crop_half), 0)
    crop_x2 = min(int(bbox_cx + crop_half), W)
    crop_y2 = min(int(bbox_cy + crop_half), H)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_h, crop_w = crop.shape[:2]
    print(f"  Crop region: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}] "
          f"({crop_w}×{crop_h}px, {crop_context:.0f}× bbox)")

    # ── 2. Map bbox coords into the crop ──
    local_x1 = x1 - crop_x1
    local_y1 = y1 - crop_y1
    local_x2 = x2 - crop_x1
    local_y2 = y2 - crop_y1

    # ── 3. Run SAM on the cropped image ──
    predictor.set_image(crop)

    input_box = np.array([local_x1, local_y1, local_x2, local_y2]).reshape(1, 4)
    cx_local = (local_x1 + local_x2) / 2.0
    cy_local = (local_y1 + local_y2) / 2.0
    point_coords = np.array([[cx_local, cy_local]])
    point_labels = np.array([1])  # foreground

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=input_box,
        multimask_output=True,
    )

    # ── 4. Choose the smallest mask that covers the centre point ──
    ci, cj = int(round(cy_local)), int(round(cx_local))
    ci = min(ci, crop_h - 1)
    cj = min(cj, crop_w - 1)

    candidate_indices = []
    for i in range(len(masks)):
        if masks[i][ci, cj]:
            candidate_indices.append(i)

    if candidate_indices:
        best_idx = min(candidate_indices, key=lambda i: masks[i].sum())
    else:
        best_idx = int(np.argmax(scores))

    local_mask = masks[best_idx]
    mask_sizes = [int(masks[i].sum()) for i in range(len(masks))]
    print(f"  SAM 3 masks: {mask_sizes} px  (chose idx {best_idx}={mask_sizes[best_idx]})")
    print(f"  SAM confidence: {scores[best_idx]:.3f}")

    # ── 5. Clip to padded bbox (in crop coords) ──
    pad_x, pad_y = bw * bbox_padding, bh * bbox_padding
    clip_lx1 = max(int(local_x1 - pad_x), 0)
    clip_ly1 = max(int(local_y1 - pad_y), 0)
    clip_lx2 = min(int(local_x2 + pad_x), crop_w)
    clip_ly2 = min(int(local_y2 + pad_y), crop_h)

    clipped_local = np.zeros_like(local_mask)
    clipped_local[clip_ly1:clip_ly2, clip_lx1:clip_lx2] = \
        local_mask[clip_ly1:clip_ly2, clip_lx1:clip_lx2]

    raw_pixels = int(local_mask.sum())
    n_pixels = int(clipped_local.sum())
    print(f"  Raw {raw_pixels} → clipped to bbox: {n_pixels} px "
          f"({100*n_pixels/(H*W):.2f}% of full image)")

    # ── 6. Map the clipped mask back to original image coordinates ──
    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = clipped_local

    return full_mask


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
# MULTI-PROMPT DETECTION: Try several prompts, pick tightest bbox
# ═══════════════════════════════════════════════════════════════════════════

def _load_prompt_variants() -> Dict[str, list]:
    """Load part prompt variants from config/prompts.json."""
    config_path = Path(__file__).resolve().parent / "config" / "prompts.json"
    if not config_path.exists():
        print(f"  WARNING: {config_path} not found, using fallback prompts")
        return {}
    with open(config_path, "r") as f:
        data = json.load(f)
    # Strip JSON comment keys (starting with _)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# Loaded once at import time
_PART_PROMPT_VARIANTS = _load_prompt_variants()


def detect_part_bbox_multi(
    image: np.ndarray,
    part_name: str,
    obj_name: str,
    custom_prompt: Optional[str] = None,
    min_confidence: float = 0.10,
) -> Optional[Tuple[np.ndarray, float, str, str]]:
    """
    Try multiple text prompts and return the detection with the
    tightest (smallest-area) bounding box.

    Args:
        image: RGB numpy array (H, W, 3)
        part_name: e.g. "handle", "rim"
        obj_name: e.g. "mug", "power_drill"
        custom_prompt: if provided, used as the only prompt
        min_confidence: minimum detection confidence

    Returns:
        (bbox_xyxy, confidence, label, winning_prompt) or None
    """
    if custom_prompt:
        prompts = [custom_prompt]
    else:
        variants = _PART_PROMPT_VARIANTS.get(part_name, ["{part} of the {obj}"])
        prompts = [v.format(part=part_name, obj=obj_name) for v in variants]

    print(f"  Trying {len(prompts)} prompt variants for '{part_name}'...")
    best = None  # (bbox, confidence, label, prompt, area)

    for prompt in prompts:
        det = detect_part_bbox(image, prompt,
                               box_threshold=min_confidence,
                               text_threshold=min_confidence)
        if det is None:
            continue
        bbox, conf, label = det
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        print(f"    '{prompt}': conf={conf:.3f}  area={area:.0f}px²  "
              f"bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")
        if best is None or area < best[4]:
            best = (bbox, conf, label, prompt, area)

    if best is None:
        return None

    bbox, conf, label, prompt, area = best
    print(f"  → Best: '{prompt}' (area={area:.0f}px², conf={conf:.3f})")
    return bbox, conf, label, prompt


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE PIPELINE: Text Prompt → 3D Part Point Cloud
# ═══════════════════════════════════════════════════════════════════════════

def segment_part_by_language(
    rgb: np.ndarray,
    depth: np.ndarray,
    text_prompt: str,
    metadata: dict,
    part_name: str = "",
    obj_name: str = "",
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    semantic_raw: Optional[np.ndarray] = None,
    semantic_id: Optional[int] = None,
    use_sam: bool = True,
) -> Optional[Dict]:
    """
    Full pipeline: text prompt → 2D mask → 3D part point cloud.

    **Hybrid strategy** (when ``semantic_raw`` is provided):

    1. Grounding DINO locates the part bbox in the image (multi-prompt:
       tries several prompt variants and picks the tightest bbox).
    2. A **semantic object mask** (from simulator ground-truth) isolates
       which pixels belong to the target object.
    3. The part mask = object mask ∩ DINO bbox  (precise, no leakage).
    4. Optionally, SAM refines the mask within the bbox, and the result
       is further intersected with the object mask.

    If ``semantic_raw`` is *not* provided, falls back to SAM-only (less
    reliable for small objects).

    Args:
        rgb: RGB image (H, W, 3)
        depth: depth map (H, W) in meters
        text_prompt: what part to segment (e.g., "handle of the mug")
        metadata: scene metadata from Stage 1
        part_name: part name (e.g. "handle") for multi-prompt lookup
        obj_name: object name (e.g. "mug") for multi-prompt lookup
        box_threshold: Grounding DINO detection confidence
        text_threshold: Grounding DINO text matching threshold
        semantic_raw: (H, W) int32 semantic ID map (optional, from semantic_raw.npy)
        semantic_id: the integer semantic ID of the target object (e.g., 1000)
        use_sam: whether to use SAM for mask refinement (default True)

    Returns:
        dict with mask, world_points, pixel_indices, bbox, etc.
    """
    print(f"\n  ── Language-guided segmentation ──")
    print(f"  Prompt: \"{text_prompt}\"")

    have_semantic = semantic_raw is not None and semantic_id is not None
    if have_semantic:
        obj_mask = (semantic_raw == semantic_id)
        n_obj = int(obj_mask.sum())
        print(f"  Semantic object mask: {n_obj} pixels (id={semantic_id})")

    # ── Step 1: Multi-prompt DINO detection (picks tightest bbox) ──
    # If part_name is known, try multiple prompt variants; otherwise
    # fall back to single-prompt detection with the user's text.
    custom_prompt = text_prompt if not part_name else None
    detection = detect_part_bbox_multi(
        rgb, part_name or text_prompt, obj_name,
        custom_prompt=custom_prompt,
        min_confidence=min(box_threshold, 0.10),
    )

    if detection is None:
        print(f"  ERROR: Could not detect '{text_prompt}' in the image")
        return None

    bbox, det_confidence, det_label, winning_prompt = detection
    x1, y1, x2, y2 = bbox

    # ── Step 2: Build the part mask ──
    H, W = rgb.shape[:2]

    if have_semantic:
        # Hybrid approach: object mask ∩ DINO bbox
        bbox_mask = np.zeros((H, W), dtype=bool)
        ix1, iy1 = max(int(x1), 0), max(int(y1), 0)
        ix2, iy2 = min(int(x2), W), min(int(y2), H)
        bbox_mask[iy1:iy2, ix1:ix2] = True

        if use_sam:
            # SAM refinement, then intersect with object mask
            sam_mask = segment_from_bbox(rgb, bbox)
            mask = sam_mask & obj_mask & bbox_mask
            sam_pixels = int(sam_mask.sum())
            print(f"  SAM∩object∩bbox: {int(mask.sum())} px "
                  f"(SAM={sam_pixels}, obj∩bbox={int((obj_mask & bbox_mask).sum())})")
        else:
            mask = obj_mask & bbox_mask
            print(f"  Object∩bbox: {int(mask.sum())} px")
    else:
        # SAM-only fallback (no semantic ground truth)
        mask = segment_from_bbox(rgb, bbox)

    # ── Step 3: Lift to 3D ──
    fx = metadata["focal_length_px"]
    fy = fx
    cx, cy_px = metadata["principal_point"]
    sensor_R = metadata["sensor_rotation_matrix"]
    sensor_t = metadata["sensor_position"]
    max_depth = metadata.get("max_depth_m", 10.0)

    world_points, pixel_indices = lift_mask_to_3d(
        mask, depth, fx, fy, cx, cy_px,
        sensor_R, sensor_t, max_depth,
    )

    if len(world_points) == 0:
        print(f"  ERROR: No valid 3D points after depth lifting")
        return None

    return {
        "mask": mask,
        "world_points": world_points,
        "pixel_indices": pixel_indices,
        "bbox": bbox.tolist(),
        "detection_confidence": det_confidence,
        "detection_label": det_label,
        "winning_prompt": winning_prompt,
        "sam_mask_pixels": int(mask.sum()),
        "prompt": text_prompt,
        "num_3d_points": len(world_points),
    }


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_segmentation_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox: list,
    part_name: str,
    color: Tuple[int, int, int] = (100, 255, 150),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Draw the segmentation mask overlay on the RGB image.

    Args:
        rgb: RGB image (H, W, 3)
        mask: binary mask (H, W)
        bbox: [x1, y1, x2, y2]
        part_name: label text
        color: RGB color for the overlay
        alpha: transparency

    Returns:
        Annotated RGB image
    """
    img = rgb.copy()

    # Create colored overlay
    overlay = img.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw mask contours
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 2)

    # Draw bounding box
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # Label
    label = f"{part_name}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return img


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Test the language segmentation pipeline standalone."""
    parser = argparse.ArgumentParser(
        description="Test language-guided part segmentation",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt (e.g., 'handle')")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--visualize", action="store_true", help="Save visualization")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load Stage 1 data
    meta_path = INPUT_DIR / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run Stage 1 first.")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)

    rgb = np.array(Image.open(INPUT_DIR / "rgb.png"))
    depth = np.load(INPUT_DIR / "depth_raw.npy")

    print(f"Image: {rgb.shape}, Depth: {depth.shape}")
    print(f"Object: {metadata.get('object_name', 'unknown')}")

    # Run pipeline
    result = segment_part_by_language(
        rgb, depth, args.prompt, metadata,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if result is None:
        print("\nSegmentation failed!")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Results:")
    print(f"  Prompt:      {result['prompt']}")
    print(f"  Detection:   {result['detection_label']} ({result['detection_confidence']:.3f})")
    print(f"  Mask pixels: {result['sam_mask_pixels']}")
    print(f"  3D points:   {result['num_3d_points']}")
    print(f"{'='*50}")

    if args.visualize:
        RESULTS_DIR = RESULTS_BASE / "language"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save mask overlay
        annotated = draw_segmentation_overlay(
            rgb, result["mask"], result["bbox"], args.prompt,
        )
        out_path = RESULTS_DIR / f"segment_{args.prompt.replace(' ', '_')}.png"
        Image.fromarray(annotated).save(out_path)
        print(f"  Saved: {out_path}")

        # Save comparison
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        axes[0].imshow(rgb)
        axes[0].set_title("Original", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(result["mask"], cmap="gray")
        axes[1].set_title(f"SAM Mask: \"{args.prompt}\"", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(annotated)
        axes[2].set_title(f"Overlay ({result['num_3d_points']} 3D pts)", fontsize=14)
        axes[2].axis("off")

        plt.tight_layout()
        compare_path = RESULTS_DIR / f"comparison_{args.prompt.replace(' ', '_')}.png"
        plt.savefig(compare_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {compare_path}")

        # Save mask as numpy
        mask_path = RESULTS_DIR / f"mask_{args.prompt.replace(' ', '_')}.npy"
        np.save(mask_path, result["mask"])
        print(f"  Saved: {mask_path}")


if __name__ == "__main__":
    main()
