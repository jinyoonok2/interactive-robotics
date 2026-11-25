"""Construct actual new-family blocks, optionally injecting test backbones."""
from .backbones import load_frozen_backbones
from .blocks.encoder import SharedObservationEncoder
from .blocks.grounding import TaskRegionGrounding
from .blocks.pose_head import TargetPoseHead
from .blocks.pose_tokens import RelativePoseTokenizer
from .blocks.action_decoder import PoseTokenDiffusionDecoder
from .blocks.progress import PoseAwareProgress


def build_default_blocks(config, *, backbones=None):
    from .policy import PoseTokenDiffusionBlocks
    if backbones is None:
        backbones = load_frozen_backbones(config, img_size=config.img_size, text_device=config.text_device)
    return PoseTokenDiffusionBlocks(SharedObservationEncoder(config, backbones), TaskRegionGrounding(config),
        RelativePoseTokenizer(config), PoseTokenDiffusionDecoder(config), TargetPoseHead(config),
        PoseAwareProgress(config) if config.progress_enabled else None)
