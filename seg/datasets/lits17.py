# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List
import os.path as osp
import mmengine
import mmengine.fileio as fileio
from mmseg.datasets.basesegdataset import BaseSegDataset
from seg.registry import DATASETS

@DATASETS.register_module()
class LiTS17(BaseSegDataset):
    """Synapse dataset.

    In segmentation map annotation for Synapse, 0 stands for background, which
    is not include in 13 categories. The ``img_suffix`` is fixed to '.jpg' and
    ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background', 'liver', 'tumor'),
        palette=[[0, 0, 0], [255, 127, 127], [224, 231, 161]])

    def __init__(self,
                 case_list=None,
                 **kwargs) -> None:
        self.case_list = case_list
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if self.case_list is None:
            assert osp.isfile(self.ann_file), f'{self.ann_file} is not a file!'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
        else:
            lines = self.case_list

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
