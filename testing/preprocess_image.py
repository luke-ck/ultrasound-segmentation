import cv2
import numpy as np
from skimage import filters
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import sys
sys.path.insert(0, '../src')

from src.data_manager import DataManager
from src.drunet import UNetRes as net
from src.utils import heal_image, single2tensor4, tensor2uint, test_onesplit

n_channels = 1
noise_level_model = 5
# Load the pre-trained DRUNet model
model_path = "./checkpoints/drunet/drunet_gray.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


model = net(in_nc=n_channels + 1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R',
            downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(torch.load(model_path), strict=True)

model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)

# Load binary image
imgs = DataManager.load_zipped_pickle('./data/test.pkl')
img = imgs[0]['video'][..., 0]

del imgs
# img = single2uint(img)

h, w = img.shape[:2]
pad_w = max(h - w, 0)
pad_h = max(w - h, 0)

target_size = (224, 224)

# Find the contours in the binary image
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Create a black image to draw the contours on
contour_img = np.zeros((h, w, 3), np.uint8)
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
        mask = np.zeros((h, w), np.uint8)

        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], 0, 255, -1)

target_size = (max(h, w), max(h, w))

padded_image = cv2.copyMakeBorder(img, 0, target_size[0] - h, 0, target_size[1] - w, cv2.BORDER_CONSTANT, value=0)

# Apply adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))

equalized = clahe.apply(padded_image)
equalized_and_denoised = cv2.fastNlMeansDenoising(equalized, h=5)
equalized_and_filtered = cv2.medianBlur(equalized, 3)

kernel = np.ones((2, 4), np.uint8)
morphed_img = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel)

# Get the image shape
height, width = padded_image.shape
healed_img = heal_image(padded_image)

resized = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

img_L = np.float32(np.expand_dims(resized, axis=2) / 255.)  # HxWx1
img_L = single2tensor4(img_L)
img_L = torch.cat((img_L, torch.FloatTensor([noise_level_model / 255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])),
                  dim=1)
img_L = img_L.to(device)

denoised_img = test_onesplit(model, img_L, refield=32)
denoised_img = tensor2uint(denoised_img)
denoised_img = cv2.resize(denoised_img, (h, h), interpolation=cv2.INTER_LINEAR)

denoised_img = clahe.apply(denoised_img)
denoised_img = heal_image(denoised_img)
# Show the result
# cv2.imshow('Smoothed Image', filtered)
cv2.imshow('Original Image', img)
cv2.imshow('Denoised Image', denoised_img)
cv2.imshow('Contours', contour_img)
# cv2.imshow('Healed Image', healed_img)
# cv2.imshow('Morphed Image', morphed_img)
# cv2.imshow('Equalized and Filtered Image', equalized_and_filtered)
# cv2.imshow('Equalized and Denoised Image', equalized_and_denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()
