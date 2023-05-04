import cv2
import numpy as np
import matplotlib.pyplot as plt

# Generate a segmentation mask with a sphere in the center
mask_size = (512, 512)
center = (mask_size[0]//2, mask_size[1]//2)
radius = mask_size[0]//4
mask = np.zeros(mask_size, dtype=np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# Apply different levels of blur to the mask
k_sizes = [3, 5, 7, 9]
blurs = []
for k_size in k_sizes:
    blur = cv2.GaussianBlur(mask, (k_size, k_size), 4)
    blurs.append(blur)

# Add Gaussian noise to the sphere
noise = np.random.normal(0, 20, mask_size).astype(np.int16)
mask = np.clip(mask + noise, 0, 255).astype(np.uint8)

# Visualize the original mask and the blurred masks
fig, axs = plt.subplots(1, len(blurs)+1, figsize=(12, 6))
axs[0].imshow(mask, cmap='gray')
axs[0].set_title('Original')
for i, blur in enumerate(blurs):
    axs[i+1].imshow(blur, cmap='gray')
    axs[i+1].set_title('Blur kernel size: {}'.format(k_sizes[i]))
plt.savefig('blur.png')