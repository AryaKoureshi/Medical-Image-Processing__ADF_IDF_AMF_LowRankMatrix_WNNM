# %% part 1: ADF
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0.05, var=0.01):
    sigma = var**0.5
    gaussian = np.zeros_like(image)
    gaussian = cv2.randn(gaussian, mean, sigma)
    noisy_gray_image = cv2.add(image, gaussian)
    return noisy_gray_image


image = cv2.imread('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/image_anisotropic.png', cv2.IMREAD_GRAYSCALE)
image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)  # normalized image

noisy_image = add_gaussian_noise(image, mean=0.05, var=0.01)

cv2.imwrite('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/noisyimage_anisotropic.png', noisy_image * 255)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.tight_layout()
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image')
plt.axis("off")
plt.tight_layout()
plt.show()

# %%


def calculate_derivatives(image):
    h, w = image.shape

    padded_image = np.pad(image, 1, mode='reflect')

    # Calculate the derivatives
    derivative_top = padded_image[:-2, 1:-1] - image
    derivative_bottom = padded_image[2:, 1:-1] - image
    derivative_left = padded_image[1:-1, :-2] - image
    derivative_right = padded_image[1:-1, 2:] - image

    return derivative_top, derivative_bottom, derivative_left, derivative_right


derivative_top, derivative_bottom, derivative_left, derivative_right = calculate_derivatives(noisy_image)


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(derivative_top, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Derivative Top')
plt.axis("off")
axes[0, 1].imshow(derivative_bottom, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('Derivative Bottom')
plt.axis("off")
axes[1, 0].imshow(derivative_left, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title('Derivative Left')
plt.axis("off")
axes[1, 1].imshow(derivative_right, cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title('Derivative Right')
plt.axis("off")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
plt.suptitle("Lighter Version")
axes[0, 0].imshow(derivative_top, cmap='gray')
axes[0, 0].set_title('Derivative Top')
plt.axis("off")
axes[0, 1].imshow(derivative_bottom, cmap='gray')
axes[0, 1].set_title('Derivative Bottom')
plt.axis("off")
axes[1, 0].imshow(derivative_left, cmap='gray')
axes[1, 0].set_title('Derivative Left')
plt.axis("off")
axes[1, 1].imshow(derivative_right, cmap='gray')
axes[1, 1].set_title('Derivative Right')
plt.axis("off")
plt.tight_layout()
plt.show()


# %%


def anisotropic_diffusion(noisy_image, iterations=10, lambda_=0.1, k=0.05, method='exponential'):
    image = noisy_image.copy()
    for _ in range(iterations):
        derivative_top, derivative_bottom, derivative_left, derivative_right = calculate_derivatives(image)

        if method == 'exponential':
            c_top = np.exp(-(derivative_top / k) ** 2)
            c_bottom = np.exp(-(derivative_bottom / k) ** 2)
            c_left = np.exp(-(derivative_left / k) ** 2)
            c_right = np.exp(-(derivative_right / k) ** 2)
        elif method == 'inverse_quadratic':
            c_top = 1 / (1 + (derivative_top / k) ** 2)
            c_bottom = 1 / (1 + (derivative_bottom / k) ** 2)
            c_left = 1 / (1 + (derivative_left / k) ** 2)
            c_right = 1 / (1 + (derivative_right / k) ** 2)
        else:
            raise ValueError("Method must be 'exponential' or 'inverse_quadratic'")

        image += lambda_ * (
            c_top * derivative_top +
            c_bottom * derivative_bottom +
            c_left * derivative_left +
            c_right * derivative_right
        )

    return image


filtered_images = {}
MSEs = {}
best_result = None
best_params = None
best_image = None

itr = 100
# Test the ADF with different methods and parameters
for mth in ['exponential', 'inverse_quadratic']:
    for lam in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1]:
        for k in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1]:
            filtered_image = anisotropic_diffusion(noisy_image.copy(), iterations=itr, lambda_=lam, k=k, method=mth)
            filtered_images[f'lambda={lam}, k={k}, method={mth}'] = filtered_image

            mse = np.mean((filtered_image - image) ** 2)  # Using MSE as a simple performance metric
            MSEs[f'lambda={lam}, k={k}, method={mth}'] = mse

            if best_result is None or mse < best_result:
                best_result = mse
                best_params = {'iterations': itr, 'lambda_': lam, 'k': k, 'method': mth}
                best_image = filtered_image
        print(f"{int(lam/2*100)}%\r")

plt.figure(figsize=(12, 8))

plt.imshow(best_image, cmap='gray', vmin=0, vmax=1)
plt.title(f'Best Result with\n {best_params}')
plt.tight_layout()
plt.axis("off")
plt.show()

cv2.imwrite('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/anisotropic_best.png', best_image * 255)

print(f'Best parameters: {best_params}')

# %% plot best denoised for each method
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.tight_layout()
plt.axis("off")
plt.show()

plt.subplot(2, 2, 2)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image')
plt.axis("off")
plt.tight_layout()
plt.show()

plt.subplot(2, 2, 3)
plt.imshow(filtered_images[f'lambda=0.01, k=0.2, method=inverse_quadratic'], cmap='gray', vmin=0, vmax=1)
plt.title('Denoised Image - Inverse Quadratic')
plt.axis("off")
plt.tight_layout()
plt.show()

plt.subplot(2, 2, 4)
plt.imshow(filtered_images[f'lambda=0.01, k=0.4, method=exponential'], cmap='gray', vmin=0, vmax=1)
plt.title('Denoised Image - Exponential')
plt.axis("off")
plt.tight_layout()
plt.show()
