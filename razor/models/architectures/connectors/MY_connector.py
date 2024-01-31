# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from razor.registry import MODELS
from mmrazor.models.architectures.connectors.base_connector import BaseConnector
from razor.models.architectures.connectors.abf_connector import ABFConnector

@MODELS.register_module()
class MyConnector(BaseConnector):
    def __init__(
        self,
        flag=False,
        T=1000,
        init_cfg=None
    ) -> None:
        super().__init__(init_cfg)
        self.T = T
        self.flag = flag
    def forward_train(self, x):
        # transform student features
        return x[self.T]

    def forward(self, x) -> torch.Tensor:

        return self.forward_train(x)
