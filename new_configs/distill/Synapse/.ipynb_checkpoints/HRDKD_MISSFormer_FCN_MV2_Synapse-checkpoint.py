from mmengine.config import read_base
from razor.models.algorithms import SingleTeacherDistill
from razor.models.distillers import ConfigurableDistiller
from mmrazor.models.task_modules.recorder import ModuleOutputsRecorder,ModuleInputsRecorder
from mmrazor.models.architectures.connectors import ConvModuleConnector
from razor.models.losses import HRDKDLoss
with read_base():
    from ..._base_.default_runtime import * # noqa
    from ..._base_.datasets.synapse import * # noqa
    from ..._base_.schedules.schedule_20k import * # noqa


teacher_ckpt = "your_path/vis_ckpts/missformer_40k_flare22/best_mDice_85-15_iter_40000.pth"
teacher_cfg_path = "your_path/new_configs/medical_seg/missformer_40k_flare22.py"  # noqa: E501
student_cfg_path = 'your_path/new_configs/mobilenet_v2/mobilenet_v2_fcn_synapse.py'  # noqa: E501

model = dict(
    type=SingleTeacherDistill,
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type=ConfigurableDistiller,
        student_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='decode_head.conv_seg')),
        teacher_recorders=dict(
            logits=dict(type=ModuleOutputsRecorder, source='backbone')),
        distill_losses=dict(
            loss_hrdkd=dict(
                type=HRDKDLoss,
                tau=4,
                kernelsize=16,
                loss_weight=0.2,)),
        loss_forward_mappings=dict(
            loss_hrdkd=dict(
                preds_S=dict(from_student=True, recorder='logits'),
                preds_T=dict(from_student=False, recorder='logits')))))


find_unused_parameters = True

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='synapse', name='kd_transunet_fcn_mv2-20k'),
        define_metric_cfg=dict(mDice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

default_hooks = dict(
    checkpoint=dict(type='MyCheckpointHook',
                    by_epoch=False,
                    interval=2000,
                    max_keep_ckpts=1,
                    save_best=['mDice'], rule='greater'))

work_dir='your_path/work_dirs/HRDKD_MISSFormer_FCN_MV2_Synapse'