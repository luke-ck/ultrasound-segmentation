import cv2
import numpy as np
from skimage import filters

# load numpy image
mask = np.load('./data/visualised_images/mask_true_1.npy')

# np to cv2
mask = (mask * 255).astype(np.uint8)

# print(mask.shape)
# mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
# print(mask.shape)
# print(mask.dtype)

# Define kernel for morphology operations
kernel = np.ones((2, 2), np.uint8)

# Apply bilateral filter
# filtered = (mask / 255).astype(np.float32)
filtered = cv2.bilateralFilter(mask, d=9, sigmaColor=1, sigmaSpace=1)
# filtered = (filtered * 255).astype(np.uint8)

# Threshold the filtered image
threshold_value = filters.threshold_otsu(filtered)
filtered = (filtered > threshold_value * 2).astype(np.uint8) * 255 # Threshold

# Dilate the mask
dilated = cv2.dilate(mask, kernel, iterations=1)
dilated = (dilated > 0).astype(np.uint8) * 255

# Dilate the filtered image
dilated_and_filtered = cv2.dilate(filtered, kernel, iterations=1)

# Apply Gaussian blur to the mask
mask_blurred = cv2.GaussianBlur(mask, (3, 3), 2)
mask_blurred = (mask_blurred > threshold_value).astype(np.uint8) * 255 # Threshold

# Apply Gaussian blur to the dilated and filtered image
blurred_dilated_and_filtered = cv2.GaussianBlur(dilated_and_filtered, (3, 3), 2)

blurred_dilated_and_filtered = (blurred_dilated_and_filtered > threshold_value).astype(np.uint8) * 255

mask = cv2.resize(mask, (512, 512))
blurred_dilated_and_filtered = cv2.resize(blurred_dilated_and_filtered, (512, 512))
# Display the results
cv2.imshow('Mask', mask)
# cv2.imshow('Dilated', dilated)
# cv2.imshow('Bilateral Filter', filtered)
# cv2.imshow('Dilated and Filtered', dilated_and_filtered)
# cv2.imshow('Blurred', mask_blurred)
cv2.imshow('Blurred Dilated and Filtered', blurred_dilated_and_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
