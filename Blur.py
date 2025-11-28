import cv2
import numpy as np

def gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    return np.outer(k, k)

def apply_gaussian_blur(image, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    kernel = kernel / np.sum(kernel)

   
    img_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)

    blurred_fft = img_fft * kernel_fft
    blurred = np.fft.ifft2(blurred_fft).real

    return np.clip(blurred, 0, 255).astype(np.uint8)
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)

blurred = apply_gaussian_blur(img, 25, 7)

cv2.imwrite("blurred.png", blurred)

    