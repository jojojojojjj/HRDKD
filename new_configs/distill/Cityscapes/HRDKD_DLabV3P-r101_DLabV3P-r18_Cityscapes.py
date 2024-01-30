from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses import HRDKDLoss
with read_base():
    from ..._base_.default_runtime import * # noqa
    from ..._base_.datasets.cityscapes import * # noqa
    from ..._base_.schedules.schedule_40k_cityscapes import * # noqa

teacher_cfg_path = "your_path/new_configs/deeplabv3+/deeplabv3plus_r101_d8_4xb2_80k_cityscapes_512x1024.py"  # noqa: E501
student_cfg_path = 'your_path//new_configs/deeplabv3+/deeplabv3plus_r18b-d8_4xb2-40k_cityscapes-512x1024.py'  # noqa: E501
teacher_ckpt = "your_path/vis_ckpts/deeplabv3+/deeplabv3plus_r101-d8_512x1024_40k_cityscapes_20200605_094614-3769eecf.pth"

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt = teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        distill_losses=dict(
            loss_hrdkd=dict(
                type=HRDKDLoss,
                tau=4,
                kernelsize=16,
                loss_weight=1,),
        ),
        loss_forward_mappings=dict(
            loss_hrdkd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
        ),
        ),
        ))
work_dir='your_path/work_dir/HRDKD_DLabV3P-r101_PSPNet-r18_Cityscapes'