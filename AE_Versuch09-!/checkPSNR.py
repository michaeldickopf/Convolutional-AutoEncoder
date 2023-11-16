from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

# Load the original image
original_image = Image.open('image_Nr-300_0_0.tif')
original_image = np.array(original_image)/255.0

# Load the reconstructed image
reconstructed_image = np.load('reconstructed-image.npy')

# Convert the reconstructed image from (3, 128, 144) to (128, 144, 3) if needed
if reconstructed_image.shape != original_image.shape:
    reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))

# Convert both images to the same data type, for example, float32
original_image = original_image.astype(np.float32)
reconstructed_image = reconstructed_image.astype(np.float32)

print(f'orig: {original_image.shape}')
print(f'rec: {reconstructed_image.shape}')
print(f'orig-max: {original_image.max()}')
print(f'orig-max: {reconstructed_image.max()}')

# Calculate PSNR
psnr_value = psnr(original_image, reconstructed_image)

print("PSNR Value:", psnr_value)

# Calculate SSIM
ssim_value = ssim(original_image, reconstructed_image, multichannel=True, win_size=5, data_range=1)

print("SSIM Value:", ssim_value)
