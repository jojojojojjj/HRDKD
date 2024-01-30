from seg.datasets import KiTS19
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from seg.datasets.transforms.transforms import Normalize
from mmcv.transforms.processing import Resize
from mmseg.datasets.transforms.transforms import RandomRotFlip
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from seg.evaluation.metrics import IoUMetric
# from seg.evaluation.metrics.case_metric_legacy import CaseMetric
from seg.evaluation.metrics.case_metric_v2 import IoUMetric as CaseMetric

dataset_type = KiTS19
data_root = '/home/jz207/workspace/data/kits19/'
img_scale = (512, 512)
train_pipeline = [
    # dict(type=LoadImageFromFile),
    # dict(type=Normalize),
    dict(type=LoadImageFromFile, color_type='grayscale'),
    dict(type=Normalize, grayscale=True),
    dict(type=LoadAnnotations),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=RandomRotFlip, rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type=PackSegInputs)
]
test_pipeline = [
    # dict(type=LoadImageFromFile),
    # dict(type=Normalize),
    dict(type=LoadImageFromFile, color_type='grayscale'),
    dict(type=Normalize, grayscale=True),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# val_evaluator = dict(type=IoUMetric, iou_metrics=['mDice', 'mIoU'])
val_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard'])
# test_evaluator = val_evaluator
test_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard'])
#test_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard', 'HD95', 'ASD'])
# test_evaluator = dict(type=CaseMetric, iou_metrics=['mDice'], hd_metric=True, ignore_index=0)

