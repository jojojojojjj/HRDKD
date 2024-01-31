# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
import os
import os.path as osp
import mmengine
from seg.registry import DATASETS
# from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets import SynapseDataset

# @DATASETS.register_module()
# class SynapseDataset(BaseSegDataset):
#     """Synapse dataset.
#
#     In segmentation map annotation for Synapse, 0 stands for background, which
#     is not include in 13 categories. The ``img_suffix`` is fixed to '.jpg' and
#     ``seg_map_suffix`` is fixed to '.png'.
#     """
#     METAINFO = dict(
#         classes=('background', 'spleen', 'right_kidney', 'left_kidney',
#                  'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
#                  'inferior_vena_cava', 'portal_vein_and_splenic_vein',
#                  'pancreas', 'right_adrenal_gland', 'left_adrenal_gland'),
#         palette=[[0, 0, 0], [255, 127, 127], [224, 231, 161], [138, 204, 132],
#                  [64, 172, 136], [126, 152, 187], [140, 110, 160],
#                  [247, 88, 240], [202, 172, 161], [237, 213, 149],
#                  [139, 182, 139], [111, 192, 185], [82, 107, 163],
#                  [89, 54, 156]])
#
#     def __init__(self, **kwargs) -> None:
#         super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
#
#     def load_data_list(self) -> List[dict]:
#         """Load annotation from directory or annotation file.
#
#         Returns:
#             list[dict]: All data info of dataset.
#         """
#         data_list = []
#         img_dir = self.data_prefix.get('img_path', None)
#         ann_dir = self.data_prefix.get('seg_map_path', None)
#         if hasattr(self, 'case_list'):
#             lines = self.case_list
#         else:
#             if osp.isfile(self.ann_file):
#                 lines = mmengine.list_from_file(
#                     self.ann_file, backend_args=self.backend_args)
#             else:
#                 lines = os.listdir(img_dir)
#
#         case_nums = dict()
#
#         lines.sort()
#
#         for line in lines:
#             case_name = line.strip()
#             imgs = os.listdir(osp.join(img_dir, case_name))
#             imgs.sort()
#             case_nums[case_name] = len(imgs)
#             for img in imgs:
#                 data_info = dict(img_path=osp.join(img_dir, case_name, img))
#                 if ann_dir is not None:
#                     seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
#                     data_info['seg_map_path'] = osp.join(ann_dir, case_name, seg_map)
#                 data_info['label_map'] = self.label_map
#                 data_info['reduce_zero_label'] = self.reduce_zero_label
#                 data_info['seg_fields'] = []
#
#                 data_info['case_name'] = case_name
#                 data_list.append(data_info)
#
#         self._metainfo.update(case_nums=case_nums)
#         if self._indices is not None and self._indices > 0:
#             return data_list[:self._indices]
#         else:
#             return data_list


@DATASETS.register_module()
class SynapseDataset(SynapseDataset):
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if hasattr(self, 'case_list'):
            lines = self.case_list
        else:
            if osp.isfile(self.ann_file):
                lines = mmengine.list_from_file(
                    self.ann_file, backend_args=self.backend_args)
            else:
                lines = os.listdir(img_dir)

        case_nums = dict()

        lines.sort()

        for line in lines:
            case_name = line.strip()
            imgs = os.listdir(osp.join(img_dir, case_name))
            imgs.sort()
            case_nums[case_name] = len(imgs)
            for img in imgs:
                data_info = dict(img_path=osp.join(img_dir, case_name, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, case_name, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []

                data_info['case_name'] = case_name
                data_list.append(data_info)

        self._metainfo.update(case_nums=case_nums)
        if self._indices is not None and self._indices > 0:
            return data_list[:self._indices]
        else:
            return data_list