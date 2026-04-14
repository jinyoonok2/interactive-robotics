# CLIPSeg Pure Part Detection Test

Tests whether CLIPSeg can localize object parts from text queries **without any depth mask or cropping assistance**.

## Why This Test?

The pipeline's `core/clipseg_detector.py` uses depth-based object masking (crop to bounding box + intersection with depth mask) before running CLIPSeg. This artificially boosts performance. This test evaluates CLIPSeg's **raw** ability.

## Output Files

### `part_heatmaps.png`
- **Row 1**: Raw CLIPSeg activation heatmaps overlaid on RGB for each part query (e.g. "handle", "rim", "body")
- **Row 2**: Binary masks at threshold=0.5

### `threshold_sweep.png`
- Shows binary masks at thresholds 0.3, 0.5, 0.7 for each query
- Reveals how sensitive the detections are to threshold choice

### `discrimination.png`
- **Winner-take-all map**: For each pixel, which query has the highest activation? (analogous to DINOv2 k-means cluster map)
- **Confident winner map**: Same but only pixels with max activation > 0.5
- **Bar chart**: Mean and max activation per query

## Usage

```bash
# From habitat-lab/ directory:
/home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/clipseg/test_part_detection.py --object mug
/home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/clipseg/test_part_detection.py --object hammer
/home/jinyoon/miniconda3/envs/uad/bin/python -u ../affordance-pipeline/tests/clipseg/test_part_detection.py --object drill
```

## Comparison with DINOv2

| Aspect | DINOv2 (tests/dinov2/) | CLIPSeg (this test) |
|--------|----------------------|---------------------|
| Input | RGB image + depth mask (object pixels only) | RGB image (full scene, no mask) |
| Method | K-means on patch features | Text-guided segmentation |
| Queries | None (unsupervised) | Part names ("handle", "rim", etc.) |
| Output | Cluster assignments | Per-query heatmaps |
| Metric | Silhouette score | Activation stats + discrimination map |
