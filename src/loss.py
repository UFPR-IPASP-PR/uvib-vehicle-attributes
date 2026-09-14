"""
Loss function implementations for the UVIB benchmark experiments.

This module provides the Focal Loss formulation to address class imbalance
and focus learning on hard negative samples during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Computes the Focal Loss between input logits and target labels.

    Focal Loss addresses class imbalance by down-weighting well-classified 
    (easy) examples and focusing on hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float or torch.Tensor, optional): Weighting factor for classes. Defaults to 1.0.
        gamma (float, optional): Focusing parameter for hard samples. Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to output ('mean', 'sum', 'none'). 
                                   Defaults to 'mean'.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Focal Loss computation.

        Args:
            inputs (torch.Tensor): Predictions from model logits of shape (N, C).
            targets (torch.Tensor): Ground truth labels of shape (N,) or (N, 1).

        Returns:
            torch.Tensor: Computed Focal Loss scalar or tensor.
        """
        # Ensure targets are 1D tensor of shape (N,) without flattening batch dimension if N=1
        if targets.dim() > 1:
            targets = targets.view(-1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if isinstance(self.alpha, (float, int)):
            focal_loss = self.alpha * focal_loss
        elif isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss