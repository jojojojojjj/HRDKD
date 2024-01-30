from seg.datasets import LiTS17
from mmseg.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from mmcv.transforms.processing import Resize
from mmseg.datasets.transforms.transforms import RandomRotFlip, \
    BioMedicalGaussianNoise, BioMedicalGaussianBlur
from seg.datasets.transforms.transforms import Normalize
from mmseg.datasets.transforms.formatting import PackSegInputs
from mmengine.dataset.sampler import InfiniteSampler, DefaultSampler
from seg.evaluation.metrics import IoUMetric
# from seg.evaluation.metrics.case_metric_legacy import CaseMetric
from seg.evaluation.metrics.case_metric_v2 import IoUMetric as CaseMetric


dataset_type = LiTS17
data_root = '/home/jz207/workspace/data/lits17/'
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

# train/val list from https://github.com/MrGiovanni/SyntheticTumors/blob/main/datafolds/lits.json
LiTS17_train_list = ['case0014', 'case0069', 'case0077', 'case0120', 'case0018',
                     'case0065', 'case0030', 'case0116', 'case0108', 'case0053',
                     'case0022', 'case0104', 'case0041', 'case0096', 'case0088',
                     'case0084', 'case0001', 'case0098', 'case0086', 'case0094',
                     'case0003', 'case0067', 'case0122', 'case0079', 'case0075',
                     'case0016', 'case0130', 'case0118', 'case0043', 'case0106',
                     'case0020', 'case0114', 'case0032', 'case0007', 'case0090',
                     'case0082', 'case0055', 'case0028', 'case0110', 'case0036',
                     'case0047', 'case0059', 'case0102', 'case0024', 'case0071',
                     'case0012', 'case0063', 'case0126', 'case0026', 'case0100',
                     'case0045', 'case0038', 'case0034', 'case0112', 'case0057',
                     'case0124', 'case0061', 'case0010', 'case0073', 'case0128',
                     'case0009', 'case0005', 'case0080', 'case0092', 'case0000',
                     'case0089', 'case0097', 'case0085', 'case0109', 'case0031',
                     'case0117', 'case0040', 'case0023', 'case0105', 'case0076',
                     'case0037', 'case0029', 'case0054', 'case0103', 'case0025',
                     'case0058', 'case0046', 'case0091', 'case0083', 'case0006',
                     'case0081', 'case0093', 'case0008', 'case0004', 'case0060',
                     'case0125', 'case0129', 'case0072', 'case0011', 'case0039',
                     'case0044', 'case0027', 'case0101', 'case0056', 'case0035',
                     'case0113']

LiTS17_val_list = ['case0068', 'case0015', 'case0064', 'case0033', 'case0078',
                   'case0123', 'case0066', 'case0017', 'case0074', 'case0099',
                   'case0095', 'case0013', 'case0070', 'case0127', 'case0062',
                   'case0111', 'case0019', 'case0121', 'case0107', 'case0021',
                   'case0042', 'case0002']

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        case_list=LiTS17_train_list,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        case_list=LiTS17_val_list,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# val_evaluator = dict(type=IoUMetric, iou_metrics=['mDice', 'mIoU'])
val_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard'])
# test_evaluator = val_evaluator
test_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard'])
#test_evaluator = dict(type=CaseMetric, case_metrics=['Dice', 'Jaccard', 'HD95', 'ASD'])
# test_evaluator = dict(type=CaseMetric, iou_metrics=['mDice'], hd_metric=True, ignore_index=0)
