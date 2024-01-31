# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import json
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
from functools import partial
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist, track_parallel_progress, track_progress
from PIL import Image
from prettytable import PrettyTable
from .confusion_matrix import *
from seg.registry import METRICS
import nibabel as nib

def calculate_metric_percase(cls_index, pred, gt, metrics):
    confusion_matrix = ConfusionMatrix(test=pred == cls_index, reference=gt == cls_index)
    rets = []
    for metric in metrics:
        ret = ALL_METRICS[metric](pred, gt, confusion_matrix)
        rets.append(ret)
    return rets


def solve_case(start_slice, end_slice, preds, labels, num_classes, metrics):
    case_pred = torch.concat(preds[start_slice:end_slice], 0)
    case_label = torch.concat(labels[start_slice:end_slice], 0)

    ret = track_progress(
        partial(calculate_metric_percase, pred=case_pred, gt=case_label, metrics=metrics),
        [i for i in range(1, num_classes)])

    results = list(zip(*ret))

    ret_metrics = OrderedDict({
        metric: np.array(results[i])
        for i, metric in enumerate(metrics)})
    return ret_metrics

class IoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 case_metrics: List[str] = ['Dice', 'Jaccard', 'HD95', 'ASD'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = './preds',
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 save_outputs: bool = False,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = case_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.save_outputs = save_outputs
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        num_classes = len(self.dataset_meta['classes'])

        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].cpu().numpy()
            slice_name = osp.splitext(osp.basename(data_sample['img_path']))[0]
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].cpu().numpy()
                self.results.append(
                    (slice_name, pred_label, label))
            # format_result
            # if self.output_dir is not None:
            #     basename = osp.splitext(osp.basename(
            #         data_sample['img_path']))[0]
            #     png_filename = osp.abspath(
            #         osp.join(self.output_dir, f'{basename}.png'))
            #     output_mask = pred_label.cpu().numpy()
            #     # The index range of official ADE20k dataset is from 0 to 150.
            #     # But the index range of output is from 0 to 149.
            #     # That is because we set reduce_zero_label=True.
            #     if data_sample.get('reduce_zero_label', False):
            #         output_mask = output_mask + 1
            #     output = Image.fromarray(output_mask.astype(np.uint8))
            #     output.save(png_filename)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # The start index of the case
        start_slice = 0
        case_nums = self.dataset_meta['case_nums']
        class_names = self.dataset_meta['classes']
        num_classes = len(class_names)
        _results = tuple(zip(*results))
        case_metrics = []
        for i, (case_name, slice_nums) in enumerate(case_nums.items()):
            # The end index of the case equals to slice_nums + start_slice
            end_slice = slice_nums + start_slice
            logger.info(
                f'----------- Testing on {case_name}: [{i + 1}/{len(case_nums)}] ----------- ')
            # the range of the case
            case_pred = np.concatenate(_results[1][start_slice:end_slice], 0)
            case_label = np.concatenate(_results[2][start_slice:end_slice], 0)

            if self.save_outputs:
                self.save_predictions(case_pred, case_name, self.output_dir)
                
            ret_metrics = self.label_to_metrics(case_pred,
                                                case_label,
                                                num_classes,
                                                self.metrics)
            ret_metrics = self.format_metrics(ret_metrics)
            case_metrics.append(ret_metrics)
            # The start index of the case equals to end index now
            start_slice = end_slice
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        metrics = dict()
        for key in case_metrics[0].keys():
            metrics[key] = np.round(np.nanmean([case_metric[key] for case_metric in case_metrics]), 2)

        return metrics

    def format_metrics(self, ret_metrics):
        logger: MMLogger = MMLogger.get_current_instance()
        class_names = self.dataset_meta['classes']

        ret_metrics_summary = OrderedDict({
            metric: np.round(np.nanmean(ret_metrics[metric]), 2)
            for metric in self.metrics})

        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics_class = OrderedDict({
            metric: np.round(ret_metrics[metric], 2)
            for metric in self.metrics})

        for metric in self.metrics:  # ['Dice', 'Jaccard', 'HD95']
            for class_key, metric_value in zip(class_names[1:], ret_metrics_class[metric]):
                metrics[f'{metric} ({class_key})'] = np.round(metric_value, 4)

        ret_metrics_class.update({'Class': class_names[1:]})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics
    @staticmethod
    def label_to_metrics(prediction: np.ndarray, target: np.ndarray,
                         num_classes: int, metrics: List[str]):

        ret = track_parallel_progress(
            partial(calculate_metric_percase, pred=prediction, gt=target, metrics=metrics),
            [i for i in range(1, num_classes)],
            nproc=num_classes - 1)

        results = list(zip(*ret))

        ret_metrics = OrderedDict({
            metric: np.array(results[i])
            for i, metric in enumerate(metrics)})


        return ret_metrics
        
    @staticmethod
    def save_predictions(outputs: torch.Tensor, img_path: str, output_dir: str) -> None:
        with open('/home/jz207/workspace/data/synapse9/affine.json') as json_file:
            affine = json.load(json_file)
        affine_ = np.array(affine[img_path.replace('case', 'img')+'.nii.gz'])
        '''
        with open('/home/jz207/workspace/data/synapse9/affine.json') as json_file:
            affine = json.load(json_file)
        affine_ = np.array(affine[img_path.replace('case', 'FLARE22_Tr_')+'_0000.nii.gz'])
        '''
        outputs = outputs.astype(np.uint8)
        outputs = np.transpose(outputs, (1, 2, 0))

        save_dirs = os.path.join(
            output_dir, img_path + '.nii.gz')
        nib.save(
            nib.Nifti1Image(outputs.astype(np.uint8), affine_),
            save_dirs)
        logger: MMLogger = MMLogger.get_current_instance()
        print_log('Prediction is saved at:', logger=logger)
        print_log(save_dirs, logger=logger)