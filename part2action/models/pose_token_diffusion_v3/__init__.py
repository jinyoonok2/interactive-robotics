"""Explicit semantic planning and part-aware pose control, isolated from v1/v2."""

ARCHITECTURE_ID = 'pose_token_diffusion_v3'


def __getattr__(name):
    if name == 'PoseTokenDiffusionV3Config':
        from .config import PoseTokenDiffusionV3Config
        return PoseTokenDiffusionV3Config
    if name in {'PoseTokenDiffusionV3Policy', 'build_policy'}:
        from .policy import PoseTokenDiffusionV3Policy, build_policy
        return {'PoseTokenDiffusionV3Policy': PoseTokenDiffusionV3Policy,
                'build_policy': build_policy}[name]
    raise AttributeError(name)
