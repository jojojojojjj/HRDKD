# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model, show_result_pyplot
from .mmseg_inferencer import MMSegInferencer
from .inference_kd import init_model_kd,inference_model_kd

__all__ = [
    'init_model', 'inference_model', 'show_result_pyplot', 'MMSegInferencer','init_model_kd','inference_model_kd'
]
