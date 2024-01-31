# Copyright (c) OpenMMLab. All rights reserved.

from .resnet import ResNet, ResNetV1c
from .mobilenet_v2 import MobileNetV2
from .shufflenet_v2 import ShuffleNetV2
__all__ = [
    'ResNet', 'ResNetV1c', 'MobileNetV2','ShuffleNetV2'
]
