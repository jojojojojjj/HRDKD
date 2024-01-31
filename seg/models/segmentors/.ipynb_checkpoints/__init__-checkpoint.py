# Copyright (c) OpenMMLab. All rights reserved.

from .teacher_ddim4 import TeacherEncoderDecoder
from .teacher_en_de_test import TeacherEncoderDecoderTest
from .encoder_decoder import EncoderDecoder

__all__ = [
    'EncoderDecoder','TeacherEncoderDecoderTest','TeacherEncoderDecoder'
]
