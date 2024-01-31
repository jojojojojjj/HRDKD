# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmrazor.registry import MODELS


class RDKDLoss(nn.Module):
    """
    Args:
    tau (float): Temperature coefficient. Defaults to 1.0.
    kernelsize(int):Size of kernel. Defaults to 16.
    loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau: float = 1.0,
        kernelsize:int=16,
        loss_weight: float = 1.0,
    ) -> None:
        super(RDKDLoss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.kernelsize=kernelsize

    def forward(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """RDKDLoss forward function.

        Args:
            preds_S (torch.Tensor): The student model prediction, shape (B, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction, shape (B, C, H, W).
            
        Return:
            torch.Tensor: The calculated loss value.
        """  
        if len(preds_T.shape)==4:
            preds_S = resize(
                input=preds_S,
                size=preds_T.shape[2:],
                mode='bilinear',
                align_corners=False)
            
        unfold = torch.nn.Unfold(kernel_size=(self.kernelsize, self.kernelsize),stride=self.kernelsize)

        b,c,h,w= preds_T.shape
        preds_S = preds_S.reshape(b*c,1,h,w)
        preds_T = preds_T.reshape(b*c,1,h,w)

        #preds_S shape:b*c,kernelsize*kernelsize,h/kernelsize*w/kernelsize
        preds_S = unfold(preds_S)
        #preds_S shape:b*c,kernelsize*kernelsize,h/kernelsize*w/kernelsize
        preds_T = unfold(preds_T)
        
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logsoftmax_pred_T=logsoftmax(preds_T / self.tau)
        logsoftmax_pred_S=logsoftmax(preds_S / self.tau)
        loss = torch.sum(softmax_pred_T *logsoftmax_pred_T -
                         softmax_pred_T *logsoftmax_pred_S) * (
                             self.tau**2)
        return self.loss_weight * loss/(b*c)
