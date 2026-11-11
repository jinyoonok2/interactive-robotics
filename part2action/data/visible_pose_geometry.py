"""Depth-only geometry with explicit GT-derived object/part membership labels.

Annotation clouds assign membership within a fixed distance; their coordinates
are never returned as observed shape. Every output point is backprojected from
the current depth image. These are approximate privileged region labels, not a
learned detector or an exact simulator segmentation mask.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.spatial import cKDTree

from .pose_samples import PoseSample


def depth_world_points(depth_m, intrinsic, world_to_camera_opengl, max_depth_m=3.):
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    view = np.asarray(world_to_camera_opengl, dtype=np.float64)
    if depth.ndim != 2 or intrinsic.shape != (3, 3) or view.shape != (4, 4):
        raise ValueError('Expected depth HxW, camera K 3x3 and world-to-OpenGL view 4x4')
    if not np.isfinite(intrinsic).all() or not np.isfinite(view).all() or max_depth_m <= 0:
        raise ValueError('Invalid camera calibration or depth limit')
    if not np.allclose(view[3], [0, 0, 0, 1]) or abs(np.linalg.det(view[:3, :3])) < 1e-8:
        raise ValueError('Invalid camera transform')
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0) & (depth < max_depth_m))
    if not len(x):
        raise ValueError('No finite visible depth points')
    pixels = np.column_stack((x, y, np.ones(len(x))))
    camera = (pixels @ np.linalg.inv(intrinsic).T) * depth[y, x, None]
    opengl = camera.copy(); opengl[:, 1:3] *= -1
    world = (np.c_[opengl, np.ones(len(opengl))] @ np.linalg.inv(view).T)[:, :3]
    return world.astype(np.float32), np.column_stack((y, x))


def visible_geometry(sample, *, match_distance_m=.005, min_points=16, max_depth_m=3.):
    if match_distance_m <= 0 or min_points < 16:
        raise ValueError('Version 1 requires a positive mask tolerance and at least 16 points')
    # Model labels, reference goals, points and event clocks are not requested.
    inputs = sample.model_inputs(allow_privileged=('target_part_cloud_world_m', 'object_cloud_world_m'),
                                 require=('depth_m', 'camera_intrinsic', 'world_to_camera_opengl'))
    points, pixels = depth_world_points(inputs['depth_m'], inputs['camera_intrinsic'],
                                        inputs['world_to_camera_opengl'], max_depth_m)
    masks, clouds, counts = {}, {}, {}
    for name, key in [('object', 'object_cloud_world_m'), ('part', 'target_part_cloud_world_m')]:
        annotation = np.asarray(inputs[key])
        if annotation.ndim != 2 or annotation.shape[1] != 3 or not len(annotation) or not np.isfinite(annotation).all():
            raise ValueError('Invalid annotation cloud: ' + key)
        distances = cKDTree(annotation).query(points, workers=1,
            distance_upper_bound=np.nextafter(match_distance_m, np.inf))[0]
        masks[name] = distances <= match_distance_m
    # A part label cannot turn a non-object depth point into an object point.
    masks['part'] &= masks['object']
    for name in ('part', 'object'):
        clouds[name] = points[masks[name]].copy()
        counts[name] = len(clouds[name])
    part_mask = np.zeros(np.asarray(inputs['depth_m']).shape, dtype=np.uint8)
    part_pixels = pixels[masks['part']]
    part_mask[part_pixels[:, 0], part_pixels[:, 1]] = 1
    return {'part_cloud': clouds['part'], 'object_cloud': clouds['object'],
            'part_mask': part_mask, 'part_pixel_indices': part_pixels,
            'object_pixel_indices': pixels[masks['object']],
            'counts': counts, 'valid': min(counts.values()) >= min_points,
            'reason': None if min(counts.values()) >= min_points else 'insufficient_visible_geometry',
            'match_distance_m': match_distance_m, 'min_points': min_points,
            'label_source': 'GT asset surface proximity; approximate region membership only',
            'geometry_source': 'current depth pixels, backprojected with recorded calibration'}


def visible_pose_sample(sample, result):
    if not result['valid']:
        raise ValueError('Insufficient visible part/object geometry; no GT geometry fallback')
    observations, labels = copy.deepcopy(sample.observations), copy.deepcopy(sample.labels)
    labels['visible_part_mask_gt_proximity'] = result['part_mask']
    privileged = {key: copy.deepcopy(value) for key, value in sample.privileged.items()
                  if key != 'reference_interaction_point_world_m'}
    privileged['target_part_cloud_world_m'] = result['part_cloud'].copy()
    privileged['object_cloud_world_m'] = result['object_cloud'].copy()
    provenance = copy.deepcopy(sample.provenance)
    provenance.update(geometry_source=result['geometry_source'],
                      region_label_source=result['label_source'],
                      mask_match_distance_m=result['match_distance_m'],
                      geometry_point_counts=result['counts'], interaction_point='not supplied or used',
                      parent_sample_id=sample.provenance['sample_id'])
    return PoseSample(observations, labels, privileged, provenance).validate()
