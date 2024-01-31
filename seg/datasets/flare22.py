# Copyright (c) OpenMMLab. All rights reserved.
from seg.registry import DATASETS
# from mmseg.datasets.synapse import SynapseDataset
from .synapse import SynapseDataset
@DATASETS.register_module()
class FLARE22Dataset(SynapseDataset):
    METAINFO = dict(
        # classes=('background', 'liver', 'spleen', 'right_kidney',
        #          'pancreas', 'aorta', 'inferior_vena_cava', 'right_adrenal_gland',
        #          'left_adrenal_gland', 'gallbladder', 'esophagus', 'stomach',
        #          'duodenum', 'left_kidney'),
        classes=('background', 'liver', 'right_kidney', 'spleen',
                 'pancreas', 'aorta', 'inferior_vena_cava', 'right_adrenal_gland',
                 'left_adrenal_gland', 'gallbladder', 'esophagus', 'stomach',
                 'duodenum', 'left_kidney'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                 [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 239, 213],
                 [0, 0, 205], [205, 133, 63], [210, 180, 140], [102, 205, 170],
                 [0, 0, 128], [0, 139, 139]])

    def prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        return self.pipeline(data_info)
