from mmseg.datasets import PascalVOCDataset
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmcv.transforms.processing import Resize, RandomResize, RandomFlip, TestTimeAug,Pad
from mmseg.datasets.transforms.transforms import RandomCrop, PhotoMetricDistortion
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.dataset_wrapper import ConcatDataset
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from mmseg.evaluation.metrics import IoUMetric

# dataset settings
dataset_type = PascalVOCDataset
data_root = '/home/jz207/workspace/data/voc_aug/'
crop_size = (512, 512)
train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=LoadAnnotations),
    dict(
        type=RandomResize,
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type=RandomCrop, crop_size=crop_size, cat_max_ratio=0.75),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PhotoMetricDistortion),
    dict(type=Pad, size=crop_size),
    dict(type=PackSegInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=Resize, scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type=LoadAnnotations),
    dict(type=PackSegInputs)
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(
        type=TestTimeAug,
        transforms=[
            [
                dict(type=Resize, scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type=RandomFlip, prob=0., direction='horizontal'),
                dict(type=RandomFlip, prob=1., direction='horizontal')
            ], [dict(type=LoadAnnotations)], [dict(type='PackSegInputs')]
        ])
]
dataset_train = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='pascal/VOC2012/JPEGImages', seg_map_path='pascal/VOC2012/SegmentationClass'),
    ann_file='pascal/VOC2012/ImageSets/Segmentation/train.txt',
    pipeline=train_pipeline)

dataset_train_aug = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(
        img_path='sbd/dataset/img', seg_map_path='sbd/dataset/cls_png'),
    ann_file='sbd/dataset/train_val.txt',
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    dataset=dict(type=ConcatDataset, datasets=[dataset_train, dataset_train_aug]))

dataset_val = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='pascal/VOC2012/JPEGImages', seg_map_path='pascal/VOC2012/SegmentationClass'),
    ann_file='pascal/VOC2012/ImageSets/Segmentation/val.txt',
    pipeline=test_pipeline)


val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dataset_val)

test_dataloader = val_dataloader

val_evaluator = dict(type=IoUMetric, iou_metrics=['mIoU'])
test_evaluator = val_evaluator