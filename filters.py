import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """normalized Gaussian kernel."""
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)


def sharpen_kernel() -> np.ndarray:
    """simple sharpening kernel."""
    return np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)


def sobel_kernels():
    """Sobel kernels for edge detection."""
    kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    return kx, ky

def threshold_image(image: np.ndarray, threshold: float = 100.0) -> np.ndarray:
    return np.where(image >= threshold, 255, 0).astype(np.float32)


def spatial_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply spatial convolution."""
    return convolve2d(
        image,
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0
    ).astype(np.float32)


def fft_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply FFT-based convolution."""
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    padded_shape = (
        image_h + kernel_h - 1,
        image_w + kernel_w - 1
    )

    image_fft = fft2(image, padded_shape)
    kernel_fft = fft2(kernel, padded_shape)

    full_result = np.real(ifft2(image_fft * kernel_fft))

    start_h = kernel_h // 2
    start_w = kernel_w // 2

    same_result = full_result[
        start_h:start_h + image_h,
        start_w:start_w + image_w
    ]

    return same_result.astype(np.float32)