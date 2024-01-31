from mmengine.config import read_base

with read_base():
    from .fcn_r50_d8_40k_flare22 import *  # noqa

model.update(
    dict(
        pretrained=None,
        backbone=dict(
            depth=18,
            init_cfg=None),
        decode_head=dict(
            in_channels=512,
            channels=128),
        # auxiliary_head=dict(
        #     in_channels=256,
        #     channels=64)
    ))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='flare22', name='fcn-r18-40k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
