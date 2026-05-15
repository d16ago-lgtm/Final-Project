import time
import numpy as np
import tracemalloc

def time_function(func, *args):
    """Run a function once and measure runtime."""
    start = time.perf_counter()
    output = func(*args)
    end = time.perf_counter()

    return output, end - start


def mean_absolute_error(output_a: np.ndarray, output_b: np.ndarray) -> float:
    """Measure average difference between two output images."""
    return float(np.mean(np.abs(output_a - output_b)))


def mean_squared_error(output_a: np.ndarray, output_b: np.ndarray) -> float:
    """Measure squared difference between two output images."""
    return float(np.mean((output_a - output_b) ** 2))


def speedup(baseline_time: float, optimized_time: float) -> float:
    """Compute speedup compared to baseline."""
    if optimized_time == 0:
        return 0.0

    return baseline_time / optimized_time

def psnr(image_a, image_b):
    mse = np.mean((image_a - image_b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def estimate_spatial_memory(image_shape, kernel_shape):
    h, w = image_shape
    kh, kw = kernel_shape

    float_bytes = np.dtype(np.float32).itemsize

    image_bytes = h * w * float_bytes
    kernel_bytes = kh * kw * float_bytes
    output_bytes = h * w * float_bytes

    total_bytes = image_bytes + kernel_bytes + output_bytes
    return total_bytes / (1024 ** 2)


def estimate_fft_memory(image_shape, kernel_shape):
    h, w = image_shape
    kh, kw = kernel_shape

    padded_h = h + kh - 1
    padded_w = w + kw - 1

    complex_bytes = np.dtype(np.complex128).itemsize
    float_bytes = np.dtype(np.float32).itemsize

    image_fft_bytes = padded_h * padded_w * complex_bytes
    kernel_fft_bytes = padded_h * padded_w * complex_bytes
    product_bytes = padded_h * padded_w * complex_bytes
    full_result_bytes = padded_h * padded_w * float_bytes
    output_bytes = h * w * float_bytes

    total_bytes = (
        image_fft_bytes
        + kernel_fft_bytes
        + product_bytes
        + full_result_bytes
        + output_bytes
    )

    return total_bytes / (1024 ** 2)


import tracemalloc

from pipeline import run_fixed_pipeline, run_adaptive_pipeline


def evaluate_execution_methods(image, pipeline):
    """
    Compare spatial, FFT, and adaptive execution.

    Measures:
    - runtime
    - peak memory usage
    - MAE
    - speedup
    """

    results = {}

    # Spatial
    tracemalloc.start()
    spatial_output, spatial_runtime = time_function(
        run_fixed_pipeline,
        image,
        pipeline,
        "spatial"
    )
    current, peak = tracemalloc.get_traced_memory()
    spatial_memory = peak / (1024 ** 2)
    tracemalloc.stop()

    # FFT
    tracemalloc.start()
    fft_output, fft_runtime = time_function(
        run_fixed_pipeline,
        image,
        pipeline,
        "fft"
    )
    current, peak = tracemalloc.get_traced_memory()
    fft_memory = peak / (1024 ** 2)
    tracemalloc.stop()

    # Adaptive
    tracemalloc.start()
    adaptive_output, adaptive_runtime = time_function(
        run_adaptive_pipeline,
        image,
        pipeline
    )
    current, peak = tracemalloc.get_traced_memory()
    adaptive_memory = peak / (1024 ** 2)
    tracemalloc.stop()

    results["spatial_runtime"] = spatial_runtime
    results["fft_runtime"] = fft_runtime
    results["adaptive_runtime"] = adaptive_runtime

    results["spatial_memory"] = spatial_memory
    results["fft_memory"] = fft_memory
    results["adaptive_memory"] = adaptive_memory

    results["fft_mae"] = mean_absolute_error(spatial_output, fft_output)
    results["adaptive_mae"] = mean_absolute_error(spatial_output, adaptive_output)

    return results