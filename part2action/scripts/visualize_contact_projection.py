"""Projection shared by recorded-data review tools (fixed PartGym camera)."""
import numpy as np
from data.geometry import _AGENTVIEW_TO_WORLD
_WORLD_TO_CAMERA = np.linalg.inv(_AGENTVIEW_TO_WORLD)

def project(point, width, height):
    gl = _WORLD_TO_CAMERA @ np.r_[point, 1.0]
    x, y, z = gl[:3] * np.array([1, -1, -1])
    if z <= 0 or not np.isfinite([x,y,z]).all():
        return None
    focal = height / (2 * np.tan(np.pi / 6))
    return (float(focal*x/z+width/2), float(focal*y/z+height/2))
