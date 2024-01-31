from mmengine.config import read_base
from seg.engine.hooks import MyCheckpointHook

with read_base():
    from .._base_.models.fcn_r50_d8 import *  # noqa
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_80k import *  # noqa
    from .._base_.default_runtime import *  # noqa
crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(
    dict(data_preprocessor=data_preprocessor,
         pretrained=None,
         decode_head=dict(num_classes=9),
         auxiliary_head=None,
         test_cfg=dict(mode='whole')))

train_dataloader.update(dict(batch_size=2, num_workers=2))
val_dataloader.update(dict(batch_size=1, num_workers=4))
test_dataloader = val_dataloader

default_hooks.update(dict(
    checkpoint=dict(
        type=MyCheckpointHook,
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=1,
        save_best=['mDice'], rule='greater')))

# custom_hooks = [dict(type='DeterministicHook',
#                      deterministic=False,
#                      warn_only=True,
#                      cublas_cfg=':4096:8')]

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='fcn-r50-80k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
