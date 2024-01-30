# train_id = ['case0045', 'case0003', 'case0017', 'case0002', 'case0032', 'case0050', 'case0038', 'case0027', 'case0036',
#             'case0006', 'case0026', 'case0029', 'case0039', 'case0037', 'case0022', 'case0013', 'case0047', 'case0005',
#             'case0007', 'case0042', 'case0033', 'case0035', 'case0041', 'case0031', 'case0048', 'case0016', 'case0044',
#             'case0019', 'case0009', 'case0043', 'case0011', 'case0025', 'case0021', 'case0034', 'case0001', 'case0023',
#             'case0008', 'case0004', 'case0030', 'case0012']
# test_id = ['case0040', 'case0018', 'case0028', 'case0024', 'case0015', 'case0049', 'case0020', 'case0014', 'case0046',
#            'case0010']

from seg.datasets.flare22 import FLARE22Dataset
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmcv.transforms.processing import Resize
from seg.datasets.transforms.transforms import Normalize
from mmseg.datasets.transforms.transforms import RandomRotFlip
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from seg.evaluation.metrics import IoUMetric
# from seg.evaluation.metrics.case_metric_legacy import CaseMetric
from seg.evaluation.metrics.case_metric_v3 import IoUMetric as CaseMetric

dataset_type = FLARE22Dataset
data_root = '/home/jz207/workspace/data/FLARE22/'
img_scale = (512, 512)
train_pipeline = [
    dict(type=LoadImageFromFile, color_type='grayscale'),
    # dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(type=Normalize, grayscale=True),
    dict(type=Resize, scale=img_scale, keep_ratio=True),
    dict(type=RandomRotFlip, rotate_prob=0.5, flip_prob=0.5, degree=20),
    dict(type=PackSegInputs)
]
test_pipeline = [
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

# val_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU', 'mDice'], ignore_index=0)
val_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard'])
#test_evaluator = val_evaluator
test_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard', 'HD95'])