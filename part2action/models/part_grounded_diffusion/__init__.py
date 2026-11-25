"""Meaningful entry points for the preserved Part-grounded Diffusion family.

Exports are aliases to existing classes, loaded lazily. Original class identity,
state-dictionary names, import paths and training entry points remain intact.
"""
from importlib import import_module

__all__ = ['PartGroundedDiffusionPolicy', 'PartGroundedDiffusionWithPose',
           'PartGroundedDiffusionPosePrototype']

_EXPORTS = {
    'PartGroundedDiffusionPolicy': ('.baseline', 'PartGroundedDiffusionPolicy'),
    'PartGroundedDiffusionWithPose': ('.gt_pose', 'PartGroundedDiffusionWithPose'),
    'PartGroundedDiffusionPosePrototype': ('.learned_pose_prototype', 'PartGroundedDiffusionPosePrototype'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
