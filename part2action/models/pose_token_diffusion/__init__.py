"""Independent Pose-token Diffusion model with lazy public entry points."""

from importlib import import_module

ARCHITECTURE_ID = 'pose_token_diffusion_v1'

__all__ = ['ARCHITECTURE_ID', 'PoseTokenDiffusionConfig', 'PoseTokenDiffusionPolicy',
           'PoseTokenDiffusionBlocks', 'build_policy', 'load_config']

_EXPORTS = {
    'PoseTokenDiffusionConfig': ('.config', 'PoseTokenDiffusionConfig'),
    'PoseTokenDiffusionPolicy': ('.policy', 'PoseTokenDiffusionPolicy'),
    'PoseTokenDiffusionBlocks': ('.policy', 'PoseTokenDiffusionBlocks'),
    'build_policy': ('.policy', 'build_policy'),
    'load_config': ('.config', 'load_config'),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
