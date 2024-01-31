from mmengine.config import read_base
with read_base():
    from .fcn_r50_d8_80k_synapse import * # noqa

model.update(dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='open-mmlab://resnet101_v1c'))))
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r101-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer.update(
    dict(vis_backends=vis_backends,
         name='visualizer'))
