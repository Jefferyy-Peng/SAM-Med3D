import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import GeneralizedDiceFocalLoss, FocalLoss
from monai.losses.dice import DiceLoss

# class FocalLossWithWeights(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
#         self.loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
#         self.weight = weight
#
#     def forward(self, input, target):
#         loss = self.loss(input[:, 1].unsqueeze(1), target[:, 1].unsqueeze(1))
#         weights = torch.zeros_like(target)
#         weights[target == 1] = self.weight[1]
#         weights[target == 0] = self.weight[0]
#         loss *= weights
#         return loss

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
#         """
#         Initializes the Focal Loss class.
#
#         Args:
#             alpha (float, optional): Balancing factor. Defaults to 0.25.
#             gamma (float, optional): Focusing parameter. Defaults to 2.0.
#             reduction (str, optional): Specifies the reduction to apply to the output:
#                                         'none' | 'mean' | 'sum'. Defaults to 'mean'.
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.weight = weight
#
#     def forward(self, inputs, targets):
#         """
#         Forward pass for the focal loss calculation.
#
#         Args:
#             inputs (torch.Tensor): Predictions from the model (logits, before softmax).
#             targets (torch.Tensor): Ground truth labels, same shape as inputs.
#
#         Returns:
#             torch.Tensor: Computed focal loss value.
#         """
#
#
#         # Compute the cross entropy loss
#         ce_loss = F.cross_entropy(inputs, targets, weight=torch.tensor(self.weight).to(inputs.device), reduction="none")
#
#         # get normalized nput
#         p = inputs
#         # Get the probability of the true class
#         pt = p * targets + (1 - p) * (1 - targets)
#
#         # Compute the focal loss
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#
#         if self.alpha is not None:
#             focal_loss *= self.alpha * targets + (1 - self.alpha) * (1 - targets)
#
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
class MultiDiceLoss():
    def __init__(self):
        self.dice_fn = DiceLoss()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=None, weight=None):
        super().__init__()
        self.alpha = alpha
        self.weight = weight
        self.focal_loss = FocalLoss(alpha=alpha, weight=weight, reduction='none')
        self.dice_loss = DiceLoss(weight=weight, sigmoid=True)

    def forward(self, input, target):
        focal_loss = self.focal_loss(input[:, 1].unsqueeze(1), target[:, 1].unsqueeze(1))
        weights = torch.zeros_like(target)
        weights[target == 1] = 0.9
        weights[target == 0] = 0.1
        focal_loss *= weights
        dice_loss = self.dice_loss(input, target)
        return focal_loss + dice_loss