from mmengine.config import read_base
from seg.models.backbones import MobileNetV2

with read_base():
    from ..resnet.fcn_r18_d8_40k_synapse import *  # noqa

model['backbone'] = dict(
            type=MobileNetV2,
            widen_factor=1.,
            strides=(1, 2, 2, 1, 1, 1, 1),
            dilations=(1, 1, 1, 2, 2, 4, 4),
            out_indices=(1, 2, 4, 6),
            norm_cfg=dict(type='SyncBN', requires_grad=True))

# model.merge(
#     dict(
#         backbone=dict(
#             _delete_=True,
#             type=MobileNetV2,
#             widen_factor=1.,
#             strides=(1, 2, 2, 1, 1, 1, 1),
#             dilations=(1, 1, 1, 2, 2, 4, 4),
#             out_indices=(1, 2, 4, 6),
#             norm_cfg=dict(type='SyncBN', requires_grad=True))))

model.update(
    dict(
        pretrained=None,
        decode_head=dict(in_channels=320)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='mv2-d8_fcn-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')