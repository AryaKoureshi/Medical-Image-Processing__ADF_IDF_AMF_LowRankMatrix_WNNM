# %% part a

from skimage import io, util, color, img_as_float
import numpy as np
import cv2
from skimage import io, util, color
import matplotlib.pyplot as plt
from skimage import img_as_float


def adaptive_median_filter(image_noisy, Smax=39):
    image = image_noisy.copy()
    # Get image dimensions
    m, n = image.shape
    # Initialize output image
    output_image = np.zeros((m, n))

    # Function to compute median, min, and max in the window
    def get_window_stats(img, x, y, w):
        window = img[max(0, x-w):min(m, x+w+1), max(0, y-w):min(n, y+w+1)]
        return np.median(window), np.min(window), np.max(window)

    # Loop through each pixel in the image
    for i in range(m):
        for j in range(n):
            w = 1
            while True:
                median, min_val, max_val = get_window_stats(image, i, j, w)

                if min_val < median < max_val:
                    if min_val < image[i, j] < max_val:
                        output_image[i, j] = image[i, j]
                    else:
                        output_image[i, j] = median
                    break
                else:
                    w += 1
                    if w > Smax:
                        output_image[i, j] = median
                        break

    return output_image


def add_salt_and_pepper_noise(image, amount=0.05):
    noisy_image = util.random_noise(image, mode='s&p', amount=amount)
    return noisy_image


# % Load the retina image
image = cv2.imread('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/retina.png', cv2.IMREAD_GRAYSCALE)
image = img_as_float(image)  # Normalized image

# % Add salt and pepper noise
noisy_image = add_salt_and_pepper_noise(image)

# % Apply the adaptive median filter
filtered_image = adaptive_median_filter(noisy_image)

# % Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Filtered Image')
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% part b


def adaptive_weighted_mean_filter(image_noisy, Smax=39):
    image_temp = image_noisy.copy()
    m, n = image_temp.shape
    output_image = np.zeros((m, n))

    def get_window_stats(img, x, y, w):
        window = img[max(0, x-w):min(m, x+w+1), max(0, y-w):min(n, y+w+1)]
        min_val = np.min(window)
        max_val = np.max(window)
        valid_pixels = window[(window > min_val) & (window < max_val)]
        if len(valid_pixels) > 0:
            mean_val = np.mean(valid_pixels)
        else:
            mean_val = -1
        return mean_val, min_val, max_val

    for i in range(m):
        for j in range(n):
            w = 1
            while True:
                mean, min_val, max_val = get_window_stats(image_temp, i, j, w)
                if mean != -1 and min_val < mean < max_val:
                    if min_val < image_temp[i, j] < max_val:
                        output_image[i, j] = image_temp[i, j]
                    else:
                        output_image[i, j] = mean
                    break
                else:
                    w += 1
                    if w > Smax:
                        output_image[i, j] = mean
                        break

    return output_image


def adaptive_median_filter(image_noisy, Smax=39):
    image_temp = image_noisy.copy()
    m, n = image_temp.shape
    output_image = np.zeros((m, n))

    def get_window_stats(img, x, y, w):
        window = img[max(0, x-w):min(m, x+w+1), max(0, y-w):min(n, y+w+1)]
        return np.median(window), np.min(window), np.max(window)

    for i in range(m):
        for j in range(n):
            w = 1
            while True:
                median, min_val, max_val = get_window_stats(image_temp, i, j, w)
                if min_val < median < max_val:
                    if min_val < image_temp[i, j] < max_val:
                        output_image[i, j] = image_temp[i, j]
                    else:
                        output_image[i, j] = median
                    break
                else:
                    w += 1
                    if w > Smax:
                        output_image[i, j] = median
                        break

    return output_image


def add_salt_and_pepper_noise(image, amount=0.05):
    noisy_image = util.random_noise(image, mode='s&p', amount=amount)
    return noisy_image


def calculate_mse(original, filtered):
    return np.mean((original - filtered) ** 2)


def calculate_psnr(original, filtered):
    mse = calculate_mse(original, filtered)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# Load the retina image
image_path = 'C:/Users/aryak/OneDrive/Desktop/MAM/HW03/retina.png'  # Update this path to the location of your retina image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = img_as_float(image)  # Normalize image to the range [0, 1]

# SNR values
snr_values = [0.05, 0.1, 0.2, 0.5, 0.8]

# Results storage
results = []

for snr in snr_values:
    # Add salt and pepper noise
    noisy_image = add_salt_and_pepper_noise(image, amount=snr)

    # Apply the adaptive median filter
    amf_filtered_image = adaptive_median_filter(noisy_image)

    # Apply the adaptive weighted mean filter
    awmf_filtered_image = adaptive_weighted_mean_filter(noisy_image)

    # Calculate MSE and PSNR for both filters
    amf_mse = calculate_mse(image, amf_filtered_image)
    awmf_mse = calculate_mse(image, awmf_filtered_image)
    amf_psnr = calculate_psnr(image, amf_filtered_image)
    awmf_psnr = calculate_psnr(image, awmf_filtered_image)

    # Store results
    results.append((snr, amf_mse, awmf_mse, amf_psnr, awmf_psnr))

# Print results
print(f"{'SNR':<10}{'AMF MSE':<10}{'AWMF MSE':<10}{'AMF PSNR':<10}{'AWMF PSNR':<10}")
for res in results:
    print(f"{res[0]:<10}{res[1]:<10.4f}{res[2]:<10.4f}{res[3]:<10.4f}{res[4]:<10.4f}")

# Display the results for the last SNR value as an example
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title(f'Noisy Image (SNR={snr_values[-1]})')
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('AMF Filtered Image')
plt.imshow(amf_filtered_image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('AWMF Filtered Image')
plt.imshow(awmf_filtered_image, cmap='gray', vmin=0, vmax=1)
plt.axis('off')

plt.tight_layout()
plt.show()
