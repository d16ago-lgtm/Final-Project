import time
import numpy as np

from filters import (
    gaussian_kernel,
    sharpen_kernel,
    sobel_kernels,
    threshold_image,
    spatial_convolution,
    fft_convolution
)

from history import find_best_method_from_history, save_result


def heuristic_method(kernel_size):
    """
    Backup rule used only when no useful history exists.
    """
    if kernel_size >= 15:
        return "fft"
    else:
        return "spatial"


def choose_method(operation, image, kernel_size):


    image_height, image_width = image.shape

    history_method = find_best_method_from_history(
        operation=operation,
        image_height=image_height,
        image_width=image_width,
        kernel_size=kernel_size
    )

    if history_method is not None:
        print(f"Using history decision for {operation}: {history_method}")
        return history_method

    method = heuristic_method(kernel_size)
    print(f"Using heuristic decision for {operation}: {method}")
    return method


def run_filter_step(image, operation, kernel_size):
    """
    Run one filter step using either spatial or FFT.
    Used for Gaussian and sharpening.
    """
    if operation == "gaussian":
        sigma = kernel_size / 6
        kernel = gaussian_kernel(kernel_size, sigma)

    elif operation == "sharpen":
        kernel = sharpen_kernel()

    else:
        raise ValueError("This function only supports gaussian and sharpen.")

    method = choose_method(operation, image, kernel_size)

    start = time.perf_counter()

    if method == "spatial":
        output = spatial_convolution(image, kernel)
    elif method == "fft":
        output = fft_convolution(image, kernel)
    else:
        raise ValueError("Unknown method.")

    end = time.perf_counter()
    runtime = end - start

    image_height, image_width = image.shape

    save_result(
        operation=operation,
        image_height=image_height,
        image_width=image_width,
        kernel_size=kernel_size,
        method=method,
        runtime=runtime
    )

    print(
        f"Saved result: operation={operation}, "
        f"image={image_height}x{image_width}, "
        f"kernel={kernel_size}, method={method}, runtime={runtime:.6f}"
    )

    return output


def run_sobel_step(image):
    """
    Sobel uses two small 3x3 kernels.
    """
    kx, ky = sobel_kernels()

    start = time.perf_counter()

    gx = spatial_convolution(image, kx)
    gy = spatial_convolution(image, ky)

    output = np.sqrt(gx**2 + gy**2).astype(np.float32)

    end = time.perf_counter()
    runtime = end - start

    image_height, image_width = image.shape

    save_result(
        operation="sobel",
        image_height=image_height,
        image_width=image_width,
        kernel_size=3,
        method="spatial",
        runtime=runtime
    )

    print(
        f"Saved result: operation=sobel, "
        f"image={image_height}x{image_width}, "
        f"method=spatial, runtime={runtime:.6f}"
    )

    return output


def run_step(image, step):
    step_type = step["type"]

    if step_type == "gaussian":
        return run_filter_step(
            image=image,
            operation="gaussian",
            kernel_size=step["kernel"]
        )

    elif step_type == "sharpen":
        return run_filter_step(
            image=image,
            operation="sharpen",
            kernel_size=3
        )

    elif step_type == "sobel":
        return run_sobel_step(image)

    elif step_type == "threshold":
        threshold_value = step["value"]
        return threshold_image(image, threshold=threshold_value)

    else:
        raise ValueError(f"Unknown pipeline step: {step_type}")