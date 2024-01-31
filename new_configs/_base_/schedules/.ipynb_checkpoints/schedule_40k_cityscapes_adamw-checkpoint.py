from torch.optim import AdamW
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import PolyLR
from mmengine.runner.loops import IterBasedTrainLoop
from mmengine.runner.loops import ValLoop, TestLoop
from mmengine.hooks import IterTimerHook, LoggerHook, ParamSchedulerHook, DistSamplerSeedHook, CheckpointHook
from seg.engine.hooks import MyCheckpointHook
from mmseg.engine.hooks import SegVisualizationHook
# optimizer
optimizer = dict(
    type=AdamW, lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper=dict(
    type=OptimWrapper,
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
# training schedule for 40k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=40000, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=MyCheckpointHook,
                    by_epoch=False,
                    interval=4000,
                    max_keep_ckpts=1,
                    save_best='mIoU', rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
