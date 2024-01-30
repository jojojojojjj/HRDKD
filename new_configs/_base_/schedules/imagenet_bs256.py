from torch.optim import SGD
from mmengine.optim.scheduler import MultiStepLR
from mmengine.hooks import IterTimerHook, LoggerHook, ParamSchedulerHook, DistSamplerSeedHook
from mmpretrain.engine.hooks import VisualizationHook
from seg.engine.hooks import MyCheckpointHook
# optimizer
optim_wrapper = dict(
    optimizer=dict(type=SGD, lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    type=MultiStepLR, by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(enable=True, base_batch_size=256)

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=MyCheckpointHook,
                    interval=1,
                    max_keep_ckpts=1,
                    save_best=['accuracy/top1'], rule='greater'),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
    # validation results visualization, set True to enable it.
    visualization=dict(type=VisualizationHook, enable=False),
)
