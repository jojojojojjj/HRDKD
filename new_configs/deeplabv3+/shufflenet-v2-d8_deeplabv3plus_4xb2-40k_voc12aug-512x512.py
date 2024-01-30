from torch.nn import SyncBatchNorm, InstanceNorm2d, LeakyReLU, ReLU
from mmseg.models import SegDataPreProcessor
from mmseg.models.segmentors import EncoderDecoder
from mmpretrain.models.backbones import ShuffleNetV2
from mmseg.models.decode_heads import FCNHead
from mmseg.models.decode_heads import DepthwiseSeparableASPPHead
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
from mmengine.model.weight_init import PretrainedInit
from mmengine.config import read_base


# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    #pretrained='https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth',
    backbone=dict(
        type=ShuffleNetV2,
        out_indices=(0, 1, 2, 3),
        widen_factor=1.,
        norm_cfg=dict(type=SyncBatchNorm, requires_grad=True),
        #init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth'),
    ),
    decode_head=dict(
        type=DepthwiseSeparableASPPHead,
        in_channels=1024,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=116,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type=FCNHead,
        in_channels=464,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth'),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

with read_base():
    from .pascal_voc12_aug import * # noqa
    from .default_runtime import * # noqa
    from .schedule_40k_voc12aug import * # noqa

crop_size = (512, 512)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(
    data_preprocessor=data_preprocessor))
#load_from='https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth'
work_dir='your_path/work_dirs/shufflenet-v2-d8_deeplabv3plus_4xb2-40k_voc12aug-512x1024'