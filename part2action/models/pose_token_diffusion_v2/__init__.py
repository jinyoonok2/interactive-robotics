"""Skill-specific pose goals; separate identity from the first-grasp v1 policy."""

ARCHITECTURE_ID = 'pose_token_diffusion_v2'


def __getattr__(name):
    if name == 'PoseTokenDiffusionV2Config':
        from .config import PoseTokenDiffusionV2Config
        return PoseTokenDiffusionV2Config
    if name in {'PoseTokenDiffusionV2Policy', 'build_policy'}:
        from .policy import PoseTokenDiffusionV2Policy, build_policy
        return {'PoseTokenDiffusionV2Policy': PoseTokenDiffusionV2Policy, 'build_policy': build_policy}[name]
    raise AttributeError(name)
