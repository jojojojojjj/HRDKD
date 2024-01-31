from torch.optim import SGD
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import LinearLR, MultiStepLR
from mmengine.runner.loops import EpochBasedTrainLoop, ValLoop, TestLoop, IterBasedTrainLoop
from mmengine.hooks import IterTimerHook, LoggerHook, ParamSchedulerHook, DistSamplerSeedHook, \
    CheckpointHook
from mmdet.engine.hooks import DetVisualizationHook
# from seg.engine.hooks import MyCheckpointHook
# training schedule for 1x
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=12, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook,
                    interval=1,
                    # max_keep_ckpts=1,
                    # save_best=['bbox_mAP'], rule='greater'
                    ),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook))
