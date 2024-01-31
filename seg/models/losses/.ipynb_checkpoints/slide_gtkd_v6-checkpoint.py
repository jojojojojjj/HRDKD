# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmrazor.registry import MODELS
import unfoldNd


class SlideGTKDLoss(nn.Module):
    """Decoupled Knowledge Distillation, CVPR2022.

    link: https://arxiv.org/abs/2203.08679
    reformulate the classical KD loss into two parts:
        1. target class knowledge distillation (TCKD)
        2. non-target class knowledge distillation (NCKD).
    Args:
    tau (float): Temperature coefficient. Defaults to 1.0.
    alpha (float): Weight of TCKD loss. Defaults to 1.0.
    beta (float): Weight of NCKD loss. Defaults to 1.0.
    reduction (str): Specifies the reduction to apply to the loss:
        ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
        ``'none'``: no reduction will be applied,
        ``'batchmean'``: the sum of the output will be divided by
            the batchsize,
        ``'sum'``: the output will be summed,
        ``'mean'``: the output will be divided by the number of
            elements in the output.
        Default: ``'batchmean'``
    loss_weight (float): Weight of loss. Defaults to 1.0.
    """
    def __init__(
        self,
        tau: float = 1.0,
        kernelsize: int=8,
        stride:int=8,
        loss_weight: float = 1.0,
        num_class = 9,
        loss_name='loss_gtkd',
    ) -> None:
        super(SlideGTKDLoss, self).__init__()
        self.tau = tau
        self.kernelsize=kernelsize
        self.stride=stride
        self.loss_weight = loss_weight
        self.num_class = num_class
        self._loss_name = loss_name

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwards
    ) -> torch.Tensor:
        """DKDLoss forward function.

        Args:
            preds_S (torch.Tensor): The student model prediction, shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction, shape (N, C).
            gt_labels (torch.Tensor): The gt label tensor, shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        target=self._get_gt_mask(pred,target)
        unfold = unfoldNd.UnfoldNd(kernel_size=(self.kernelsize, self.kernelsize),stride=self.stride)
        b,c,h,w= pred.shape
        pred = pred.reshape(b*c,1,h,w)
        target = (target.reshape(b*c,1,h,w)*self.num_class).float()
        #shape:b*c,kernelsize*kernelsize,h/kernelsize*w/kernelsize
        pred = unfold(pred)
        target = unfold(target)     
        softmax_pred_T = F.softmax(target / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logsoftmax_pred_T=logsoftmax(target / self.tau)
        logsoftmax_pred_S=logsoftmax(pred / self.tau)
        loss = torch.sum(softmax_pred_T *logsoftmax_pred_T -
                         softmax_pred_T *logsoftmax_pred_S) * (
                             self.tau**2)
        return self.loss_weight * loss/(b*c)

        return loss
    def _get_gt_mask(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name