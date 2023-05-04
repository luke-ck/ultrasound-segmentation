import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    iou = iou_coeff(input, target, smooth=epsilon)
    return (2 * iou) / (iou + 1)


def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target)


def iou_coeff(y_true: Tensor, y_pred: Tensor, smooth=1e-6):
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (y_true * y_pred).sum()
    total = (y_true + y_pred).sum()
    union = total - intersection

    return (intersection + smooth) / (union + smooth)


def binary_focal_jaccard_loss(y_true, y_pred, gamma=2.0, alpha=0.991, smooth=1e-6):
    """
    Computes the binary focal Jaccard loss between the true and predicted binary tensors.

    Arguments:
    y_true -- true binary tensor of shape (batch_size, num_channels, height, width)
    y_pred -- predicted binary tensor of shape (batch_size, num_channels, height, width)
    gamma -- focusing parameter that adjusts the rate at which easy examples are down-weighted (default 2.0)
    alpha -- balancing parameter that adjusts the contribution of positive and negative examples (default 0.95)
    smooth -- smoothing parameter to avoid division by zero in the Jaccard similarity calculation (default 1e-7)

    Returns:
    Binary focal Jaccard loss value as a PyTorch scalar tensor.
    """

    # # Compute the intersection and union between the masks
    # intersection = torch.sum(y_true * y_pred, )
    # union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    #
    # # Compute the Jaccard index
    # jaccard = intersection / union

    jaccard = iou_coeff(y_true, torch.nn.functional.sigmoid(y_pred), smooth=smooth)

    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    # Compute BCE
    bce = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')

    bce_exp = torch.exp(-bce)
    bce = alpha * (1 - bce_exp) ** gamma * bce

    # Compute loss
    loss = bce + (1 - alpha) * (1 - jaccard)

    return loss


def power_jaccard_loss(y_true, y_pred, p=2, smooth=1e-6):
    """
    From https://www.scitepress.org/Papers/2021/103040/103040.pdf
    """
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)

    intersection = (y_true * y_pred).sum()
    total = (torch.pow(y_true, p) + torch.pow(y_pred, p)).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def jaccard_distance_loss(y_true, y_pred, smooth=1e-6):
    return 1 - iou_coeff(y_true, y_pred, smooth=smooth)
