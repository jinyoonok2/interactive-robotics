"""Fit sparse observed geometry to the existing statistics-head input contract.

The old head requires at least 16 cloud rows, although two distinct observed
points already define finite sample statistics. Repeat every observed point an
equal number of times when needed. This adds no surface coordinates or new
information; receipts retain the actual number of observed pixels. Empty or
single-point regions remain unavailable, with no GT shape substitution.
"""
from __future__ import annotations

import numpy as np

from .visible_pose_geometry import visible_geometry


def sparse_visible_geometry(sample, *, match_distance_m=.005, min_points=2, max_depth_m=3.):
    if min_points != 2:
        raise ValueError('Version 2 fixes the actual observed-point minimum at two')
    result = visible_geometry(sample, match_distance_m=match_distance_m, min_points=16,
                              max_depth_m=max_depth_m)
    result['min_points'] = min_points
    result['valid'] = min(result['counts'].values()) >= min_points
    result['reason'] = None if result['valid'] else 'insufficient_visible_geometry'
    result['stored_point_counts'] = dict(result['counts'])
    result['geometry_source'] = 'current depth pixels; balanced repetition only when fewer than 16 rows'
    if result['valid']:
        for name in ('part', 'object'):
            count = result['counts'][name]
            if count < 16:
                result[name + '_cloud'] = np.repeat(result[name + '_cloud'], int(np.ceil(16 / count)), axis=0)
            result['stored_point_counts'][name] = len(result[name + '_cloud'])
    return result
