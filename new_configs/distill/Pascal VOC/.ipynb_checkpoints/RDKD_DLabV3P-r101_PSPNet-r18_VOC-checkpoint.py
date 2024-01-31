from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses import RDKDLoss
with read_base():
    from ..._base_.default_runtime import * # noqa
    from ..._base_.datasets.pascal_voc12_aug import * # noqa
    from ..._base_.schedules.schedule_40k_voc12aug import * # noqa


teacher_cfg_path = "your_path/new_configs/deeplabv3+/deeplabv3plus_r101_d8_4xb4_80k_voc12aug_512x512.py"  # noqa: E501
student_cfg_path = 'your_path/new_configs/pspnet/pspnet_r18_d8_4xb4_40k_voc12aug_512x512.py'  # noqa: E501
teacher_ckpt = "your_path/vis_ckpts/deeplabv3+/deeplabv3plus_r101-d8_512x512_40k_voc12aug_20200613_205333-faf03387.pth"

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
            loss_rdkd=dict(
                type=RDKDLoss,
                tau=4,
                kernelsize=16,
                loss_weight=1,),
        ),
        loss_forward_mappings=dict(
            loss_rdkd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits'),
        ),
        ),
        ))
work_dir='your_path/work_dir/RDKD_DLabV3P-r101_PSPNet-r18_VOC'