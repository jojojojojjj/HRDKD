from mmengine.config import read_base
from mmseg.models.backbones import ResNet
with read_base():
    from .deeplabv3plus_r50_d8_4xb2_40k_cityscapes_512x1024 import * # noqa


model.update(dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type=ResNet, depth=18),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64)))