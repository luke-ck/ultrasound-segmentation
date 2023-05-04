import numpy as np
import torch


def jaccard_distance_loss(y_true, y_pred, smooth=1e-6):
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (y_true * y_pred).sum()
    total = (y_true + y_pred).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def power_jaccard_loss(y_true, y_pred, p=2, smooth=1e-6):
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)

    intersection = (y_true * y_pred).sum()
    total = (torch.pow(y_true, p) + torch.pow(y_pred, p)).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return 1 - IoU


def binary_focal_jaccard_loss(y_true, y_pred, gamma=1.0, alpha=0.5, smooth=1e-6):

    jaccard = 1 + power_jaccard_loss(y_true, y_pred, p=2, smooth=smooth)

    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)
    # Compute BCE
    bce = torch.nn.functional.binary_cross_entropy(y_pred, y_true, reduction='mean')
    bce_exp = torch.exp(-bce)
    bce = alpha * (1 - bce_exp) ** gamma * bce

    # Compute loss
    loss = bce + (1 - alpha) * (1 - jaccard)

    return loss


batch_size = 4
h = 256
w = 256

# Ground truth segmentation mask
gt_mask = np.zeros((batch_size, 1, h, w), dtype=np.int32)
gt_mask[:2, :, 64:192, 64:192] = 1

# Almost right segmentation mask
almost_right_mask = np.zeros((batch_size, 1, h, w), dtype=np.int32)
almost_right_mask[:2, :, 64:192, 70:198] = 1

# Half right segmentation mask
half_right_mask = np.zeros((batch_size, 1, h, w), dtype=np.int32)
half_right_mask[:2, :, 64:192, 64:128] = 1
half_right_mask[:2, :, 64:192, 192:256] = 1

# All wrong segmentation mask
all_wrong_mask = np.ones((batch_size, 1, h, w), dtype=np.int32)
all_wrong_mask[:2, :, 64:192, 64:192] = 0

# Convert to torch tensors
gt_mask = torch.from_numpy(gt_mask)
almost_right_mask = torch.from_numpy(almost_right_mask)
half_right_mask = torch.from_numpy(half_right_mask)
all_wrong_mask = torch.from_numpy(all_wrong_mask)

print("JACCARD DISTANCE LOSS")
gt_almost_loss = jaccard_distance_loss(gt_mask, almost_right_mask)
gt_half_loss = jaccard_distance_loss(gt_mask, half_right_mask)
gt_all_loss = jaccard_distance_loss(gt_mask, all_wrong_mask)

print("gt_almost_loss: ", gt_almost_loss)
print("gt_half_loss: ", gt_half_loss)
print("gt_all_loss: ", gt_all_loss)

print("POWER JACCARD LOSS")
assert torch.all(gt_almost_loss == power_jaccard_loss(gt_mask, almost_right_mask, p=1))
assert torch.all(gt_half_loss == power_jaccard_loss(gt_mask, half_right_mask, p=1))
assert torch.all(gt_all_loss == power_jaccard_loss(gt_mask, all_wrong_mask, p=1))

gt_almost_loss = power_jaccard_loss(gt_mask, almost_right_mask, p=5)
gt_half_loss = power_jaccard_loss(gt_mask, half_right_mask, p=5)
gt_all_loss = power_jaccard_loss(gt_mask, all_wrong_mask, p=5)

print("gt_almost_loss: ", gt_almost_loss)
print("gt_half_loss: ", gt_half_loss)
print("gt_all_loss: ", gt_all_loss)


print("BINARY FOCAL JACCARD LOSS")
gt_almost_loss = binary_focal_jaccard_loss(gt_mask, almost_right_mask)
gt_half_loss = binary_focal_jaccard_loss(gt_mask, half_right_mask)
gt_all_loss = binary_focal_jaccard_loss(gt_mask, all_wrong_mask)

print("gt_almost_loss: ", gt_almost_loss)
print("gt_half_loss: ", gt_half_loss)
print("gt_all_loss: ", gt_all_loss)
