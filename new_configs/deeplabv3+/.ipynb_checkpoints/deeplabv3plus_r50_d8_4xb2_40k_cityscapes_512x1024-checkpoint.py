from mmengine.config import read_base
with read_base():
    from .deeplabv3plus_r50_d8 import * # noqa
    from .cityscapes import * # noqa
    from .default_runtime import * # noqa
    from .schedule_40k_cityscapes import * # noqa

crop_size = (512, 1024)
data_preprocessor.update(dict(size=crop_size))
model.update(dict(data_preprocessor=data_preprocessor))