#!/usr/bin/env python3
"""
UAD Worker — Runs inside the `uad` conda environment.

Called by uad_bridge.py via subprocess. Reads a request JSON, runs
AffordanceInference, and writes the result as .npy files.

This script should NOT be called directly by users — use uad_bridge.predict_affordance().
"""

import sys
import json
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: _uad_worker.py <request.json>", file=sys.stderr)
        sys.exit(1)

    # Load request
    with open(sys.argv[1]) as f:
        request = json.load(f)

    image_path = request["image_path"]
    text_query = request["text_query"]
    threshold = request["threshold"]
    config_path = request["config_path"]
    checkpoint_path = request["checkpoint_path"]
    output_path = request["output_path"]
    output_raw_path = request["output_raw_path"]

    # Load image
    rgb = np.load(image_path)
    print(f"Image shape: {rgb.shape}")
    print(f"Text query: \"{text_query}\"")
    print(f"Threshold: {threshold}")

    # Import UAD inference (this file runs in the uad env with correct paths)
    from inference import AffordanceInference
    from utils.vlm_utils import get_text_embedding_options
    from utils.file_utils import load_config

    # Load config to get text embedding setting
    cfg = load_config(config_path)
    text_embedding_option = cfg.get("text_embedding", "embeddings_st")
    print(f"Text embedding: {text_embedding_option}")

    text_embedding_func = get_text_embedding_options(text_embedding_option)

    # Initialize model
    print("Loading UAD model...")
    inference = AffordanceInference(config_path, checkpoint_path, text_embedding_func)

    # Run prediction — get raw (continuous) heatmap first
    raw_heatmap = inference.predict(rgb, text_query, thresh=None)
    print(f"Raw heatmap: shape={raw_heatmap.shape}, range=[{raw_heatmap.min():.3f}, {raw_heatmap.max():.3f}]")

    # Save raw heatmap
    np.save(output_raw_path, raw_heatmap.astype(np.float32))

    # Apply threshold if requested
    if threshold is not None:
        mask = (raw_heatmap > threshold).astype(np.uint8)
        print(f"Binary mask: {int(mask.sum())} pixels at threshold={threshold}")
    else:
        mask = raw_heatmap.astype(np.float32)

    np.save(output_path, mask)
    print("Done.")


if __name__ == "__main__":
    main()
