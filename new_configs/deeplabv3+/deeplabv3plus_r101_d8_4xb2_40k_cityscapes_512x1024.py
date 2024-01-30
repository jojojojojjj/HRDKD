from mmengine.config import read_base
with read_base():
    from .deeplabv3plus_r50_d8_4xb2_40k_cityscapes_512x1024 import * # noqa

model.update(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))