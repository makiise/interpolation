from random import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def corrupt_image(image_path, percentage):
    img = Image.open(image_path)
    img_array = np.array(img)
    height, width, channels = img_array.shape
    total_pixels = height * width
    pixels_to_remove = int(total_pixels * (percentage / 100))
    indices = [(i, j) for i in range(height) for j in range(width)]
    random_indices = random.sample(indices, pixels_to_remove)
    corrupted_img_array = img_array.copy()
    for i, j in random_indices:
        corrupted_img_array[i, j] = [0, 0, 0]
    return img_array, corrupted_img_array, random_indices


def rbf_interpolation(original_img, corrupted_img, missing_indices, rbf_function, epsilon):
    height, width, channels = corrupted_img.shape
    known_indices = [(i, j) for i in range(height) for j in range(width) if not (i, j) in missing_indices]
    known_pixels = [original_img[i, j] for i, j in known_indices]

    known_x = np.array([i for i, j in known_indices])
    known_y = np.array([j for i, j in known_indices])

    restored_img = corrupted_img.copy()

    for channel in range(channels):
        known_values = np.array([pixel[channel] for pixel in known_pixels])
        rbf = Rbf(known_x, known_y, known_values, function=rbf_function, epsilon=epsilon)
        missing_x = np.array([i for i, j in missing_indices])
        missing_y = np.array([j for i, j in missing_indices])
        restored_values = rbf(missing_x, missing_y)
        for (i, j), value in zip(missing_indices, restored_values):
            restored_img[i, j, channel] = value

    return restored_img


# Define parameters
image_path = 'utagawa.jpg'
percentage = 25
epsilon_values = [0.1, 1, 10]

# Corrupt the image
original_img, corrupted_img, missing_indices = corrupt_image(image_path, percentage)

# Plot the original and corrupted images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(original_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(corrupted_img)
axes[1].set_title('Corrupted Image')
axes[1].axis('off')

# Apply and plot RBF interpolation with different epsilons
for i, epsilon in enumerate(epsilon_values):
    restored_img = rbf_interpolation(original_img, corrupted_img, missing_indices, 'multiquadric', epsilon)
    axes[i + 2].imshow(restored_img)
    axes[i + 2].set_title(f'Restored (ε={epsilon})')
    axes[i + 2].axis('off')

plt.tight_layout()
plt.show()
