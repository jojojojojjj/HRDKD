# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS
from mmseg.models.utils import resize


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1, keepdim=True),
                             b - b.mean(1, keepdim=True), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))



class DISTLoss(nn.Module):

    def __init__(
        self,
        inter_loss_weight=1.0,
        intra_loss_weight=1.0,
        tau=1.0,
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(DISTLoss, self).__init__()
        self.inter_loss_weight = inter_loss_weight
        self.intra_loss_weight = intra_loss_weight
        self.tau = tau

        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach

    def forward(self, preds_S, preds_T: torch.Tensor):
        preds_S = resize(
            input=preds_S,
            size=preds_T.shape[2:],
            mode='bilinear',
            align_corners=False)
        assert preds_S.ndim in (2, 4)
        if preds_S.ndim == 4:
            num_classes = preds_S.shape[1]
            preds_S = preds_S.transpose(1, 3).reshape(-1, num_classes)
            preds_T = preds_T.transpose(1, 3).reshape(-1, num_classes)
            
        if self.teacher_detach:
            preds_T = preds_T.detach()
        
        y_s = (preds_S / self.tau).softmax(dim=1)
        y_t = (preds_T / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.inter_loss_weight * inter_loss + self.intra_loss_weight * intra_loss  # noqa
        return kd_loss * self.loss_weight