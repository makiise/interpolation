import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf


def remove_random_pixels(image_path, percentage):
    # Open an image file
    img = Image.open(image_path)
    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Get the dimensions of the image
    height, width, channels = img_array.shape

    # Calculate the number of pixels to remove
    total_pixels = height * width
    pixels_to_remove = int(total_pixels * (percentage / 100))

    # Generate random indices
    indices = [(i, j) for i in range(height) for j in range(width)]
    random_indices = random.sample(indices, pixels_to_remove)

    # Remove the pixels
    for i, j in random_indices:
        img_array[i, j] = [0, 0, 0]  # Setting pixel to black (0, 0, 0)
        # To set to white, use [255, 255, 255]
        # To set to transparent (if RGBA), use [0, 0, 0, 0]

    # Convert the array back to an image
    modified_img = Image.fromarray(img_array)
    return modified_img


# Test the function with different percentages
percentages = [10, 25, 50, 75]
image_path = 'utagawa.jpg'

# Create a list to store the modified images
modified_images = []

for perc in percentages:
    modified_image = remove_random_pixels(image_path, perc)
    modified_images.append((perc, modified_image))

# Plot the results
fig, axes = plt.subplots(1, len(percentages) + 1, figsize=(20, 5))

# Original image
original_image = Image.open(image_path)
axes[0].imshow(original_image)
axes[0].set_title('Original')
axes[0].axis('off')

# Modified images
for ax, (perc, img) in zip(axes[1:], modified_images):
    ax.imshow(img)
    ax.set_title(f'{perc}% Pixels Removed')
    ax.axis('off')

plt.tight_layout()
plt.show()


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

# Plot the original, corrupted, and restored images
fig, axes = plt.subplots(1, len(epsilon_values) + 2, figsize=(20, 5))
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