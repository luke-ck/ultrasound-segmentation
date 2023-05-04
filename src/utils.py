import re

import torch
import cv2
import numpy as np
import os

from matplotlib import pyplot as plt


def list_images(directory, ext='jpg|jpeg|bmp|png|tif'):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


def stratified_bootstrap(y_train, frame_indexes):
    num_bootstrap_samples = y_train.shape[0] - len(frame_indexes)

    # perform bootstrapping
    bootstrap_mask_indexes = np.random.choice(frame_indexes, size=num_bootstrap_samples, replace=True)
    bootstrap_masks = y_train[bootstrap_mask_indexes]

    return bootstrap_masks, bootstrap_mask_indexes


def plot_segmentation_overlay(BASE, epoch, iter, image, mask_pred, mask_true):

    image = cv2.resize(image, (256, 256))
    mask_pred = cv2.resize(mask_pred.astype(np.float32), (256, 256))
    mask_true = cv2.resize(mask_true.astype(np.float32), (256, 256))

    plt.figure(figsize=(20, 4))
    plt.axis('off')
    plt.subplot(1, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.subplot(1, 5, 2)
    plt.imshow(mask_true, cmap='gray')
    plt.title('True mask')
    plt.subplot(1, 5, 3)
    plt.imshow(mask_pred, cmap='gray')
    plt.title('Predicted mask')
    plt.subplot(1, 5, 4)
    plt.imshow(image, cmap='gray')
    plt.imshow(mask_true, alpha=0.5, cmap='gray')
    plt.title('True mask overlay')
    plt.subplot(1, 5, 5)
    plt.imshow(image, cmap='gray')
    plt.imshow(mask_pred, alpha=0.5, cmap='gray')
    plt.title('Predicted mask overlay')
    # make dir if not exists
    if not os.path.exists(BASE + '/figures/' + str(epoch)):
        os.makedirs(BASE + '/figures/' + str(epoch))
    plt.savefig(BASE + '/figures/' + str(epoch) + '/val_' + str(iter) + '.png')
    plt.close()

def heal_image(img: np.ndarray):
    assert len(img.shape) == 2, "Image must be grayscale"
    height, width = img.shape
    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a black image to draw the contours on
    contour_img = np.zeros((height, width, 3), np.uint8)
    # Define the minimum patch size
    min_patch_size = 30

    # Loop over the contours and fill in the patches with area less than min_patch_size
    for contour in contours:
        # Calculate the area of the contour
        contour_area = cv2.contourArea(contour)

        # Check if the contour area is less than the minimum patch size
        if contour_area < min_patch_size:
            cv2.drawContours(contour_img, [contour], 0, (0, 0, 255), 2)
            # Create a mask with the same shape as the image
            mask = np.zeros((height, width), np.uint8)

            # Draw the contour on the mask
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Apply the healing tool to the mask using the cv2.inpaint function
            img = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
    return img


# Functions below are from https://github.com/cszn/DPIR/tree/master/utils
# convert single (HxWxn_channels) to 4-dimensional torch tensor
def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def uint2single(img):
    return np.float32(img / 255.)


def single2uint(img):
    return np.uint8((img.clip(0, 1) * 255.).round())


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def test_onesplit(model, L, refield=32, sf=1):
    '''
    model:
    L: input Low-quality image
    refield: effective receptive filed of the network, 32 is enough
    sf: scale factor for super-resolution, otherwise 1
    modulo: 1 if split
    '''
    h, w = L.size()[-2:]

    top = slice(0, (h // 2 // refield + 1) * refield)
    bottom = slice(h - (h // 2 // refield + 1) * refield, h)
    left = slice(0, (w // 2 // refield + 1) * refield)
    right = slice(w - (w // 2 // refield + 1) * refield, w)
    Ls = [L[..., top, left], L[..., top, right], L[..., bottom, left], L[..., bottom, right]]
    Es = [model(Ls[i]) for i in range(4)]
    b, c = Es[0].size()[:2]
    E = torch.zeros(b, c, sf * h, sf * w).type_as(L)
    E[..., :h // 2 * sf, :w // 2 * sf] = Es[0][..., :h // 2 * sf, :w // 2 * sf]
    E[..., :h // 2 * sf, w // 2 * sf:w * sf] = Es[1][..., :h // 2 * sf, (-w + w // 2) * sf:]
    E[..., h // 2 * sf:h * sf, :w // 2 * sf] = Es[2][..., (-h + h // 2) * sf:, :w // 2 * sf]
    E[..., h // 2 * sf:h * sf, w // 2 * sf:w * sf] = Es[3][..., (-h + h // 2) * sf:, (-w + w // 2) * sf:]
    return E