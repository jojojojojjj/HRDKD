# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import resize
from mmrazor.registry import MODELS


class CWDKDLoss(nn.Module):
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
            reduction: str = 'batchmean',
            loss_weight: float = 1.0,
            num_class=9,
    ) -> None:
        super(CWDKDLoss, self).__init__()
        self.tau = tau
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_class = num_class
        self.h = 255
        self.w = 255

    def forward(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            gt_labels: torch.Tensor,
    ) -> torch.Tensor:
        """DKDLoss forward function.

        Args:
            preds_S (torch.Tensor): The student model prediction, shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction, shape (N, C).
            gt_labels (torch.Tensor): The gt label tensor, shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """

        preds_S = resize(
            input=preds_S,
            size=gt_labels.shape[1:],
            mode='bilinear',
            align_corners=False)

        preds_T = resize(
            input=preds_T,
            size=gt_labels.shape[1:],
            mode='bilinear',
            align_corners=False)

        b,  h, w = gt_labels.shape
        self.h = h
        self.w = w

        preds_S = preds_S.reshape(-1, self.h*self.w)
        preds_T = preds_T.reshape(-1, self.h*self.w)

        gt_mask = self._get_gt_mask(preds_S, gt_labels)
        non_gt_mask = self._get_non_gt_mask(preds_S, gt_labels)
        
        tckd_loss = self._get_tckd_loss(preds_S, preds_T, non_gt_mask, gt_mask)
        bckd_loss = self._get_bckd_loss(preds_S, preds_T, non_gt_mask, gt_mask)
        nckd_loss = self._get_nckd_loss(preds_S, preds_T, non_gt_mask, gt_mask)
        loss = nckd_loss
        return self.loss_weight * loss

    def _get_nckd_loss(
        self,
        preds_S: torch.Tensor,
        preds_T: torch.Tensor,
        non_gt_mask: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-target class knowledge distillation."""
        # implementation to mask out gt_mask, faster than index
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        p_t = (t_tckd * non_gt_mask).sum(dim=1, keepdims=True)
        s_nckd = F.log_softmax(preds_S / self.tau - self.num_class * gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - self.num_class * gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

            
    def _get_tckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            non_gt_mask: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        t_tckd = F.softmax(preds_T / self.tau, dim=1)
        p_t = (t_tckd * gt_mask).sum(dim=1, keepdims=True)
        s_nckd = F.log_softmax(preds_S / self.tau - self.num_class * non_gt_mask, dim=1)
        t_nckd = F.softmax(preds_T / self.tau - self.num_class * non_gt_mask, dim=1)
        return self._kl_loss(s_nckd, t_nckd)

    def _get_bckd_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
            non_gt_mask: torch.Tensor,
            gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate target class knowledge distillation."""
        s_bckd = F.softmax(preds_S / self.tau, dim=1)
        t_bckd = F.softmax(preds_T / self.tau, dim=1)
        mask_student = torch.log(self._cat_mask(s_bckd, gt_mask, non_gt_mask))
        mask_teacher = self._cat_mask(t_bckd, gt_mask, non_gt_mask)
        return self._kl_loss(mask_student, mask_teacher)
    

    def _kl_loss(
            self,
            preds_S: torch.Tensor,
            preds_T: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the KL Divergence."""
        kl_loss = F.kl_div(
            preds_S, preds_T, size_average=False,
            reduction=self.reduction) * self.tau ** 2
        return kl_loss

    def _cat_mask(
            self,
            tckd: torch.Tensor,
            gt_mask: torch.Tensor,
            non_gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate preds of target (pt) & preds of non-target (pnt)."""
        t1 = (tckd * gt_mask).sum(dim=1, keepdims=True)
        t2 = (tckd * non_gt_mask).sum(dim=1, keepdims=True)
        return torch.cat([t1, t2], dim=1)

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
        logits = logits.reshape(-1, self.num_class)
        target = target.reshape(-1)
        gmask=torch.zeros_like(logits).scatter_(1, target.unsqueeze(1),
                                                 1).bool()
        return gmask.reshape(-1,self.h*self.w)

    def _get_non_gt_mask(
            self,
            logits: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate non-groundtruth mask on logits with target class tensor.

        Args:
            logits (torch.Tensor): The prediction logits with shape (N, C).
            target (torch.Tensor): The gt_label target with shape (N, C).

        Return:
            torch.Tensor: The masked logits.
        """
        logits=logits.reshape(-1,self.num_class)
        target = target.reshape(-1)
        ngmask=torch.ones_like(logits).scatter_(1, target.unsqueeze(1),
                                                0).bool()
        return ngmask.reshape(-1,self.h*self.w)
