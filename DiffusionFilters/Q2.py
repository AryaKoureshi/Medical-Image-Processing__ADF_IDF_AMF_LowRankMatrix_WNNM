# %% part2: IDF, ADF
# %% Read the Image and Add Gaussian Noise
import torch
import pyiqa
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
from skimage import img_as_float, img_as_ubyte
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def to_rgb(gray_image):
    return np.stack((gray_image,) * 3, axis=-1)


image = cv2.imread('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/image2.png', cv2.IMREAD_GRAYSCALE)
image = img_as_float(image)  # Normalized image

noisy_image = random_noise(image, mode='gaussian', var=0.01)

cv2.imwrite('C:/Users/aryak/OneDrive/Desktop/MAM/HW03/image2_noisy.png', noisy_image * 255)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.tight_layout()
plt.axis("off")
plt.show()

plt.subplot(2, 1, 2)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image')
plt.axis("off")
plt.tight_layout()
plt.show()


# %% Implement Isotropic and Anisotropic Diffusion Filters

# % Isotropic Diffusion
def isodiff(image, lambda_param, constant, niter):
    im = image.copy()
    im = np.double(im)
    rows, cols = im.shape
    diff = im

    for _ in range(niter):
        diffl = np.zeros((rows + 2, cols + 2))
        diffl[1:rows+1, 1:cols+1] = diff

        deltaN = diffl[0:rows, 1:cols+1] - diff
        deltaS = diffl[2:rows+2, 1:cols+1] - diff
        deltaE = diffl[1:rows+1, 2:cols+2] - diff
        deltaW = diffl[1:rows+1, 0:cols] - diff

        cN = cS = cE = cW = constant

        diff = diff + lambda_param * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    return diff


lambda_param = 0.25
constant = 0.04
niter = 100

isotropic_denoised = isodiff(noisy_image, lambda_param, constant, niter)

plt.figure(figsize=(12, 8))
plt.imshow(isotropic_denoised, cmap='gray', vmin=0, vmax=1)
plt.title('Isotropic Denoised Image')
plt.tight_layout()
plt.axis("off")
plt.show()

# % Anisotropic Diffusion


def anisodiff(image, niter, kappa, lambda_param, option):
    im = image.copy()
    im = np.double(im)
    rows, cols = im.shape
    diff = im

    for _ in range(niter):
        diffl = np.zeros((rows + 2, cols + 2))
        diffl[1:rows+1, 1:cols+1] = diff

        deltaN = diffl[0:rows, 1:cols+1] - diff
        deltaS = diffl[2:rows+2, 1:cols+1] - diff
        deltaE = diffl[1:rows+1, 2:cols+2] - diff
        deltaW = diffl[1:rows+1, 0:cols] - diff

        if option == 1:
            cN = np.exp(-(deltaN / kappa)**2)
            cS = np.exp(-(deltaS / kappa)**2)
            cE = np.exp(-(deltaE / kappa)**2)
            cW = np.exp(-(deltaW / kappa)**2)
        elif option == 2:
            cN = 1 / (1 + (deltaN / kappa)**2)
            cS = 1 / (1 + (deltaS / kappa)**2)
            cE = 1 / (1 + (deltaE / kappa)**2)
            cW = 1 / (1 + (deltaW / kappa)**2)

        diff = diff + lambda_param * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    return diff


niter = 100
kappa = 0.15
lambda_param = 0.008

anisotropic_denoised_option1 = anisodiff(noisy_image, niter, kappa, lambda_param, option=1)
anisotropic_denoised_option2 = anisodiff(noisy_image, niter, kappa, lambda_param, option=2)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(anisotropic_denoised_option1, cmap='gray', vmin=0, vmax=1)
plt.title('Anisotropic Denoised Image - option 1')
plt.tight_layout()
plt.axis("off")
plt.show()

plt.subplot(2, 1, 2)
plt.imshow(anisotropic_denoised_option2, cmap='gray', vmin=0, vmax=1)
plt.title('Anisotropic Denoised Image - option 2')
plt.tight_layout()
plt.axis("off")
plt.show()

# %% Evaluate the Results
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.cuda.is_available = lambda: False

device = torch.device('cpu')

ssim_isotropic = ssim(image, isotropic_denoised, data_range=isotropic_denoised.max() - isotropic_denoised.min())
ssim_anisotropic_option1 = ssim(image, anisotropic_denoised_option1, data_range=anisotropic_denoised_option1.max() - anisotropic_denoised_option1.min())
ssim_anisotropic_option2 = ssim(image, anisotropic_denoised_option2, data_range=anisotropic_denoised_option2.max() - anisotropic_denoised_option2.min())


isotropic_denoised_rgb = to_rgb(img_as_ubyte(isotropic_denoised))
anisotropic_denoised_option1_rgb = to_rgb(img_as_ubyte(anisotropic_denoised_option1))
anisotropic_denoised_option2_rgb = to_rgb(img_as_ubyte(anisotropic_denoised_option2))

isotropic_denoised_tensor = torch.from_numpy(isotropic_denoised_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
anisotropic_denoised_option1_tensor = torch.from_numpy(anisotropic_denoised_option1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
anisotropic_denoised_option2_tensor = torch.from_numpy(anisotropic_denoised_option2_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

isotropic_denoised_tensor = isotropic_denoised_tensor.to(device)
anisotropic_denoised_option1_tensor = anisotropic_denoised_option1_tensor.to(device)
anisotropic_denoised_option2_tensor = anisotropic_denoised_option2_tensor.to(device)

niqe_model = pyiqa.create_metric('niqe').to(device)
niqe_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    niqe_isotropic = niqe_model(isotropic_denoised_tensor).item()
    niqe_anisotropic_option1 = niqe_model(anisotropic_denoised_option1_tensor).item()
    niqe_anisotropic_option2 = niqe_model(anisotropic_denoised_option2_tensor).item()

print(f"SSIM Isotropic: {ssim_isotropic}")
print(f"SSIM Anisotropic - option1: {ssim_anisotropic_option1}")
print(f"SSIM Anisotropic - option2: {ssim_anisotropic_option2}\n")
print(f"NIQE Isotropic: {niqe_isotropic}")
print(f"NIQE Anisotropic - option1: {niqe_anisotropic_option1}")
print(f"NIQE Anisotropic - option2: {niqe_anisotropic_option2}")


plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title('Original Image')
plt.tight_layout()
plt.axis("off")
plt.show()

plt.subplot(3, 2, 3)
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
plt.title('Noisy Image')
plt.axis("off")
plt.tight_layout()
plt.show()

plt.subplot(3, 2, 2)
plt.imshow(isotropic_denoised, cmap='gray', vmin=0, vmax=1)
plt.title('Isotropic Denoised Image')
plt.tight_layout()
plt.axis("off")
plt.show()


plt.subplot(3, 2, 4)
plt.imshow(anisotropic_denoised_option1, cmap='gray', vmin=0, vmax=1)
plt.title('Anisotropic Denoised Image - option 1')
plt.tight_layout()
plt.axis("off")
plt.show()

plt.subplot(3, 2, 6)
plt.imshow(anisotropic_denoised_option2, cmap='gray', vmin=0, vmax=1)
plt.title('Anisotropic Denoised Image - option 2')
plt.tight_layout()
plt.axis("off")
plt.show()

# %%
'''
### Comparison with Other Filters

Gaussian filters and other denoising techniques like Total Variation and Non-Local Means can be compared based on their denoising performance and preservation of image details.

- **Gaussian Filter**: Applies a Gaussian blur to the image, which reduces noise but also blurs edges.
- **Total Variation (TV)**: Minimizes the total variation of the image, preserving edges better than Gaussian.
- **Non-Local Means (NLM)**: Averages pixels with similar patches, providing good denoising with edge preservation.

Here is a brief comparison:
- **Isotropic Diffusion**: Uniform diffusion, does not consider image features.
- **Anisotropic Diffusion**: Directional diffusion based on image gradients, better at preserving edges.
- **Gaussian Filter**: Simple and fast, but tends to blur edges.
- **Total Variation**: Good edge preservation, suitable for piecewise-smooth images.
- **NLM**: Effective for detailed denoising, can be computationally intensive.

### Conclusion

By implementing and evaluating isotropic and anisotropic diffusion filters, we can observe their effectiveness in denoising while preserving image details. Using SSIM and NIQE metrics, we can quantify the quality of the denoised images and compare them with other denoising methods.
'''
