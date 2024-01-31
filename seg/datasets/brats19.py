# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.basesegdataset import BaseSegDataset
import os.path as osp
from typing import List
import mmengine
import mmengine.fileio as fileio
from seg.registry import DATASETS

@DATASETS.register_module()
class BraTS19(BaseSegDataset):
    """Synapse dataset.

    In segmentation map annotation for Synapse, 0 stands for background, which
    is not include in 13 categories. The ``img_suffix`` is fixed to '.jpg' and
    ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background',
                 'necrotic_and_non-enhancing_tumor',
                 'peritumoral_edema',
                 'enhancing_tumor'),
        palette=[[0, 0, 0], [255, 127, 127], [224, 231, 161], [138, 204, 132]])

    def __init__(self,
                 with_seg=False,
                 modalities=('flair', 't1ce', 't1', 't2'),
                 **kwargs) -> None:
        self.with_seg = with_seg
        self.modalities = modalities
        super().__init__(img_suffix='.nii.gz', seg_map_suffix='_seg.nii.gz', **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        assert osp.isfile(self.ann_file), f'{self.ann_file} is not a file!'
        lines = mmengine.list_from_file(
            self.ann_file, backend_args=self.backend_args)
        for line in lines:
            img_name = line.strip()
            data_info = dict(
                img_name=img_name.split('/')[-1],
                img_path=osp.join(img_dir, img_name))
            if self.with_seg:
                data_info['seg_map_path'] = osp.join(img_dir, img_name, img_name.split('/')[-1] + self.seg_map_suffix)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        if self._indices is not None and self._indices > 0:
            return data_list[:self._indices]
        else:
            return data_list

@DATASETS.register_module()
class BraTS19_2D(BaseSegDataset):
    """Synapse dataset.

    In segmentation map annotation for Synapse, 0 stands for background, which
    is not include in 13 categories. The ``img_suffix`` is fixed to '.jpg' and
    ``seg_map_suffix`` is fixed to '.png'.
    """
    METAINFO = dict(
        classes=('background',
                 'necrotic_and_non-enhancing_tumor',
                 'peritumoral_edema',
                 'enhancing_tumor'),
        palette=[[0, 0, 0], [255, 127, 127], [224, 231, 161], [138, 204, 132]])

    def __init__(self,
                 with_seg=False,
                 **kwargs) -> None:
        self.with_seg = with_seg
        super().__init__(img_suffix='.pkl', **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        if self.case_list is None:
            assert osp.isfile(self.ann_file), f'{self.ann_file} is not a file!'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
        else:
            lines = self.case_list

        for line in lines:
            img_name = line.strip()

            for img in fileio.list_dir_or_file(
                    dir_path=osp.join(img_dir, img_name),
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img_name, img))
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)

        if self._indices is not None and self._indices > 0:
            return data_list[:self._indices]
        else:
            return data_list
