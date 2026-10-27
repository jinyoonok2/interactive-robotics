"""Map released legacy names to the public PartInstruct implementation.

The saved DP config names its mask agentview_mask. The public constructor also
expects the equivalent agentview_part_mask alias. The released first convolution
has five channels (RGB + repeated mask), despite a stale four-channel config.
The caller reads this dimension from the weights; learned tensors are untouched.
"""
import copy
from torch import nn
from PartInstruct.baselines.utils.encoders import VisLangObsImageEncoder


class ReleasedImageEncoder(VisLangObsImageEncoder):
    def __init__(self, shape_meta, **kwargs):
        shape_meta = copy.deepcopy(shape_meta)
        obs = shape_meta['obs']
        if 'agentview_part_mask' not in obs:
            obs['agentview_part_mask'] = copy.deepcopy(obs['agentview_mask'])
        super().__init__(shape_meta=shape_meta, **kwargs)


def released_resnet(name, weights=None, input_channels=5, **kwargs):
    from PartInstruct.baselines.utils.encoders import get_resnet
    model = get_resnet(name, weights=weights, input_channels=3, **kwargs)
    conv = model.conv1
    model.conv1 = nn.Conv2d(input_channels, conv.out_channels, conv.kernel_size,
                           conv.stride, conv.padding, bias=False)
    return model


def resolve_legacy_targets(value):
    if isinstance(value,list):
        return [resolve_legacy_targets(v) for v in value]
    if not isinstance(value,dict):
        return value
    targets={
        'lgm_bc.policy.robodiff_unet_image_policy.RobodiffUnetImagePolicy':
            'PartInstruct.baselines.policy.diffusion_policy.RobodiffUnetImagePolicy',
        'lgm_bc.encoders.VisLangObsImageEncoder':'compat.ReleasedImageEncoder',
        'lgm_bc.encoders.get_resnet':'compat.released_resnet',
        'lgm_bc.encoders.T5Encoder':'PartInstruct.baselines.utils.encoders.T5Encoder',
        'lgm_bc.policy.dp3_policy.DP3':'PartInstruct.baselines.policy.diffusion_policy_3d.DP3',
    }
    result={k:resolve_legacy_targets(v) for k,v in value.items()}
    if '_target_' in result:
        result['_target_']=targets.get(result['_target_'],result['_target_'])
        if result['_target_'].startswith('lgm_bc.'):
            raise ValueError('Unmapped release class: '+result['_target_'])
    return result
