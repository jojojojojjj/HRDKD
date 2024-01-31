from mmengine.config import read_base
from seg.models.backbones import ShuffleNetV2
with read_base():
    from ..resnet.fcn_r18_d8_40k_flare22 import *  # noqa
    from .._base_.schedules.schedule_20k import * # noqa
    from .._base_.datasets.flare22 import *  # noqa

model['backbone'] = dict(
        type=ShuffleNetV2,
        out_indices=(0, 1, 2,3),
        widen_factor=1.)

model.update(dict(
    pretrained=None,
    decode_head=dict(in_channels=1024)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='shufflenet-v2-d8_fcn-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')