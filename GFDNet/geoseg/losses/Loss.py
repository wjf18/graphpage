import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from .soft_ce import SoftCrossEntropyLoss
from .joint_loss import JointLoss
from .dice import DiceLoss
from torch.nn.modules.loss import _Loss
from .functional import focal_loss_with_logits
from functools import partial
from .lovasz import _lovasz_softmax
import cv2

class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.main_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)


    def forward(self, logits, labels):

        loss = self.main_loss(logits, labels)

        return loss

class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None, reduction="mean", normalized=False, reduced_threshold=None):
        """
        Focal loss for multi-class problem.

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.focal_loss_fn(cls_label_input, cls_label_target)
        return loss

class EdgeLoss(nn.Module):
    def __init__(self, n_classes=19, radius=1, alpha=0.01):
        super(EdgeLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha


    def forward(self, logits, label):
        prediction = F.softmax(logits, dim=1)
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        filt2 = torch.ones(self.n_classes, 1, ks, ks)
        filt2[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt2.requires_grad = False
        filt2 = filt2.cuda()
        prededge = F.conv2d(prediction.float(), filt2, bias=None,
                            stride=1, padding=self.radius, groups=self.n_classes)

        norm = torch.sum(torch.pow(prededge,2), 1).unsqueeze(1)
        prededge = norm/(norm + self.alpha)

        return BinaryDiceLoss()(prededge.float(),lbedge.float())


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den
        return loss.sum()


class BoundaryLoss(nn.Module):
    def __init__(self, n_classes=1, radius=1, alpha=0.01):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes
        self.radius = radius
        self.alpha = alpha


    def forward(self, logits, label):
        ks = 2 * self.radius + 1
        filt1 = torch.ones(1, 1, ks, ks)
        filt1[:, :, self.radius:2*self.radius, self.radius:2*self.radius] = -8
        filt1.requires_grad = False
        filt1 = filt1.cuda()
        label = label.unsqueeze(1)
        lbedge = F.conv2d(label.float(), filt1, bias=None, stride=1, padding=self.radius)
        lbedge = 1 - torch.eq(lbedge, 0).float()

        prededge = logits

        return BinaryDiceLoss()(prededge.float(),lbedge.float())



class ModelLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(ModelLoss, self).__init__()
        self.main_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

        self.aux_loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                                   DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)


    def forward(self, logits, labels):
        if self.training and len(logits) == 4:
            logit_main,aux2,aux3,aux4 = logits
            loss = self.main_loss(logit_main, labels)\
                   + (self.aux_loss(aux2, labels) + self.aux_loss(aux3, labels) + self.aux_loss(aux4, labels)) * 1.0

        else:
            loss = self.main_loss(logits, labels)

        return loss


if __name__ == '__main__':
    targets = torch.randint(low=0, high=2, size=(2, 16, 16))
    logits = torch.randn((2, 2, 16, 16))
    # print(targets)
    model = EdgeLoss()
    loss = model.compute_edge_loss(logits, targets)

    print(loss)

