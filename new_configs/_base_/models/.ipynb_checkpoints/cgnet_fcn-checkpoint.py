# import modules
from torch.nn import SyncBatchNorm, InstanceNorm2d, LeakyReLU, ReLU
from mmseg.models import SegDataPreProcessor
from seg.models.segmentors import EncoderDecoder
from mmseg.models.backbones import CGNet
from seg.models.decode_heads import FCNHead
from mmseg.models.losses import CrossEntropyLoss, DiceLoss
from seg.models.losses.dice import MemoryEfficientSoftDiceLoss
from mmengine.model.weight_init import PretrainedInit

# model settings
norm_cfg = dict(type=SyncBatchNorm, requires_grad=True)
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=None,
    std=None,
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type=EncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=CGNet,
        in_channels=1),
    decode_head=dict(
        type=FCNHead,
        in_channels=256,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type=CrossEntropyLoss, use_sigmoid=False, loss_weight=0.5),
            dict(
                type=MemoryEfficientSoftDiceLoss, loss_weight=0.5)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))