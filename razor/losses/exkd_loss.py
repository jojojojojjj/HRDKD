# Copyright (c) OpenMMLab. All rights reserved.
import torch

from razor.registry import MODELS
from mmrazor.models.losses import L2Loss

@MODELS.register_module()
class EXKD_Loss(L2Loss):

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        t_feature = t_feature.permute(0, 3, 1, 2)
        if self.normalize:
            s_feature = self.normalize_feature(s_feature)
            t_feature = self.normalize_feature(t_feature)

        loss = torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))

        # Calculate l2_loss as dist.
        if self.dist:
            loss = torch.sqrt(loss)
        else:
            if self.div_element:
                loss = loss / s_feature.numel()
            else:
                loss = loss / s_feature.size(0)

        return self.loss_weight * loss
