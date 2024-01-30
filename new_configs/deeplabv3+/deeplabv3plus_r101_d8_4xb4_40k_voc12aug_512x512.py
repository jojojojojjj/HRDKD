from mmengine.config import read_base
with read_base():
    from .deeplabv3plus_r50_d8_4xb4_40k_voc12aug_512x512 import * # noqa

model.update(dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101)))
work_dir='your_path/work_dirs/deeplabv3plus_r101_d8_4xb4_40k_voc12aug_512x512'