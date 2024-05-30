import cupy as cp
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.util import random_noise, view_as_windows
from scipy.spatial.distance import cdist
import cv2


# Function to implement WNNM
def WNNM(Y, tau, weights):
    U, sigma, VT = cp.linalg.svd(Y, full_matrices=False)  # Perform SVD on the CPU
    sigma = cp.asarray(sigma)  # Convert results back to CuPy
    U = cp.asarray(U)
    VT = cp.asarray(VT)
    sigma_w = cp.maximum(sigma - tau * weights, 0)
    return U @ cp.diag(sigma_w) @ VT


image = cv2.imread('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/Low Matrix Approximation question2.jpg', cv2.IMREAD_GRAYSCALE)
image = img_as_float(image)  # Normalized image

noisy_image1 = random_noise(image, mode='gaussian', var=0.01)
noisy_image = noisy_image1.copy()

h, w = image.shape

patch_size = 7
stride = 4

patches = view_as_windows(noisy_image, (patch_size, patch_size), step=stride)
denoised_patches = cp.zeros_like(patches)

patches = cp.asarray(patches.reshape(-1, patch_size * patch_size))

# %%  Implementing WNNM
tau = 0.1  # Regularization parameter
max_patches = 50  # Limit for the number of similar patches
weights = cp.linspace(1, 0.1, patch_size)  # Example weight values


patch_counts = cp.zeros(noisy_image.shape)
denoised_image = noisy_image.copy()

c = 0
for i in range(patches.shape[0]):
    patch = patches[i].reshape((patch_size, patch_size))

    distances = cdist([patch.flatten().get()], patches.get(), 'euclidean')[0]
    similar_patch_indices = cp.argsort(distances)[:max_patches]
    similar_patches = patches[similar_patch_indices].reshape(-1, patch_size, patch_size)

    Y = cp.stack(similar_patches, axis=-1)

    denoised_Y = cp.array([WNNM(Y[..., j], tau, weights) for j in range(Y.shape[-1])])

    denoised_patch = cp.mean(denoised_Y, axis=0)

    denoised_patches[c, i-(c*(denoised_patches.shape[1])), :patch_size, :patch_size] += denoised_patch

    if (i+1) % denoised_patches.shape[1] == 0:
        c += 1
    print(i/patches.shape[0]*100)

# %% Extract denoised image from denoised patches
for h in range(denoised_patches.shape[0]):
    for w in range(denoised_patches.shape[1]):
        denoised_image[h*stride:h*stride + patch_size, w*stride:w*stride + patch_size] = denoised_patches[h, w].get()

# %% Plots
plt.figure(figsize=(15, 5))
plt.subplot(3, 1, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.title("Noisy Image")
plt.imshow(noisy_image1, cmap='gray')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

cv2.imwrite('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/Low Matrix Approximation question222 denoised.jpg', denoised_image.get()*255)
