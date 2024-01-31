# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional

import os.path as osp
import mmengine.fileio as fileio
from mmengine.utils import mkdir_or_exist
import numpy as np
from mmcv.transforms import BaseTransform

from seg.registry import TRANSFORMS
from mmseg.utils import datafrombytes

import tarfile

@TRANSFORMS.register_module()
class LoadMultiModalitiesData(BaseTransform):
    """Load multi modalities data and annotation from file for BraTS.
    """

    def __init__(self,
                 modalities=('flair', 't1ce', 't1', 't2'),
                 with_seg=False,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.modalities = modalities
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        modalities_data = []

        for modal in self.modalities:
            path = osp.join(results['img_path'], results['img_name'] + '_' + modal + '.nii.gz')
            data_bytes = fileio.get(path, self.backend_args)
            modalities_data.append(datafrombytes(data_bytes, backend=self.decode_backend))
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = np.stack(modalities_data, -1)
        img = img.transpose(3, 0, 1, 2)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]

        if self.with_seg:
            data_bytes = fileio.get(results['seg_map_path'], self.backend_args)
            gt_seg_map = datafrombytes(data_bytes, backend=self.decode_backend)
            results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str

@TRANSFORMS.register_module()
class LoadBraTSData2D(BaseTransform):
    """Load multi modalities data and annotation from file for BraTS.
    """

    def __init__(self,
                 modalities=('flair', 't1ce', 't1', 't2'),
                 with_seg=False,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.modalities = modalities
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['img_path'], self.backend_args)
        data = datafrombytes(data_bytes, backend='pickle')
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = data[0].transpose(2, 1, 0)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        if self.with_seg:
            results['gt_seg_map'] = data[1]
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str