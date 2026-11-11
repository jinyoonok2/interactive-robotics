"""Controller reconstruction hypotheses and unit-aware replay diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .interaction_tolerances import controller_gains

PROFILES = ('legacy', 'closure_001', 'closure_001_pregrasp015')


def gain_hypothesis(actions, window, profile):
    if profile not in PROFILES:
        raise ValueError(profile)
    rows = controller_gains(actions, window)
    if profile == 'legacy':
        return rows
    # Generation can exit switch_gripper as soon as grasp succeeds, before its
    # stronger 20-step settling branch. HDF5 contains no gain records. This tests
    # that source-supported hypothesis instead of guessing from final length.
    rows = [(arm, .01, near) for arm, finger, near in rows]
    if profile == 'closure_001_pregrasp015':
        # Explicit calibration hypothesis: 15 cm instead of 20 cm in the legacy
        # positional argument's threshold slot. Inferred from approach residuals,
        # NOT known release metadata and NOT a change to any eligibility limit.
        rows = [(arm, finger, .15 if near == .2 else near) for arm, finger, near in rows]
    return rows


def repeat_metrics(first_tcp, other_tcp, first_finger=None, other_finger=None):
    """Do not mix quaternion components and metres in a maximum difference."""
    a, b = np.asarray(first_tcp), np.asarray(other_tcp)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7:
        raise ValueError('Need matching (T,7) quaternion-then-XYZ trajectories')
    xyz = np.linalg.norm(a[:, 4:] - b[:, 4:], axis=-1)
    angular = np.rad2deg((Rotation.from_quat(a[:, :4]).inv() * Rotation.from_quat(b[:, :4])).magnitude())
    result = {'xyz_max_m': float(xyz.max()), 'orientation_max_deg': float(angular.max()),
              'first_xyz_divergence_frame': int(np.flatnonzero(xyz > 1e-8)[0]) if np.any(xyz > 1e-8) else None}
    if first_finger is not None:
        result['finger_max_m'] = float(np.max(np.abs(np.asarray(first_finger)-other_finger)))
    return result
