# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmrazor.registry import MODELS

class HRDKDLoss(nn.Module):
    """
    Args:
    tau (float): Temperature coefficient. Defaults to 1.0.
    kernelsize(int):Size of kernel. Defaults to 8.
    loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau: float = 1.0,
        kernelsize:int=8,
        loss_weight: float = 1.0,
    ) -> None:
        super(HRDKDLoss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.kernelsize=kernelsize

    def forward(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """HRDKDLoss forward function.

        Args:
            preds_S (torch.Tensor): The student model prediction, shape (B, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction, shape (B, C, H, W).
            
        Return:
            torch.Tensor: The calculated loss value.
        """    
        preds_S = resize(
                input=preds_S,
                size=preds_T.shape[2:],
                mode='bilinear',
                align_corners=False)
        b,c,h,w=preds_T.shape
        preds_S_avg=preds_S
        preds_T_avg=preds_T
        loss,preds_S_avg,preds_T_avg=self.rdkd(preds_S_avg,preds_T_avg)
        h=h/self.kernelsize
        losses = loss
        while h%self.kernelsize==0 and h>=self.kernelsize:
            loss,preds_S_avg,preds_T_avg=self.rdkd(preds_S_avg,preds_T_avg)
            losses+=loss
            h=h/self.kernelsize
        return self.loss_weight * losses/(b*c)
        
    def rdkd(self,preds_S,preds_T):
        b,c,h,w= preds_T.shape
        preds_S = preds_S.reshape(b*c,1,h,w)
        preds_T = preds_T.reshape(b*c,1,h,w)
            #shape:b*c,kernelsize*kernelsize,h/kernelsize*w/kernelsize
        unfold = torch.nn.Unfold(kernel_size=(self.kernelsize, self.kernelsize),stride=self.kernelsize)
        preds_S = unfold(preds_S)
        preds_T = unfold(preds_T)
        H=h//self.kernelsize
        W=w//self.kernelsize
        preds_S_avg=torch.mean(preds_S, dim=1, keepdim=True).transpose(1,2)
        preds_T_avg=torch.mean(preds_T, dim=1, keepdim=True).transpose(1,2) 
        preds_S_avg= preds_S_avg.view(b,c,H,W)
        preds_T_avg= preds_T_avg.view(b,c,H,W)
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logsoftmax_pred_T=logsoftmax(preds_T / self.tau)
        logsoftmax_pred_S=logsoftmax(preds_S / self.tau)
        loss = torch.sum(softmax_pred_T *logsoftmax_pred_T -
                             softmax_pred_T *logsoftmax_pred_S) * (
                                 self.tau**2)
        return loss,preds_S_avg,preds_T_avg