# implemented for kernel size used to blur the image  (25,25) and the sigma value used is 7.
# shoutout cf for the inspo question ->2095H
import cv2
import numpy as np

def gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    return np.outer(k, k)

def wiener_deconvolution(blurred, kernel, noise_power=5e-6):
    kernel = kernel / np.sum(kernel)
    
    H = np.fft.fft2(kernel, s=blurred.shape)
    G = np.fft.fft2(blurred)
    
    H_conj = np.conj(H)
    denom = (H * H_conj) + noise_power

    F_hat = (G * H_conj) / denom
    f = np.fft.ifft2(F_hat).real

    return np.clip(f, 0, 255).astype(np.uint8)

def deblur_image(path, kernel_size, sigma):
    blurred = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if blurred is None:
        print("Image not found.")
        return
    
    kernel = gaussian_kernel(kernel_size, sigma)
    restored = wiener_deconvolution(blurred, kernel)

    cv2.imwrite("deblurred.png", restored)
    print("Saved: deblurred.png")

deblur_image("blurred.png", 25, 7)
