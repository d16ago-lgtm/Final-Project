import numpy as np

from filters import (
    gaussian_kernel,
    sharpen_kernel,
    sobel_kernels,
    threshold_image,
    spatial_convolution,
    fft_convolution
)

from runtime import run_step


class Pipeline:
    """
    DSL-like interface for defining image-processing pipelines.

    Example:
    pipeline = (
        Pipeline()
        .gaussian(5)
        .sharpen()
        .gaussian(15)
        .sobel()
    )
    """

    def __init__(self):
        self.steps = []

    def gaussian(self, kernel_size: int):
        self.steps.append({
            "type": "gaussian",
            "kernel": kernel_size
        })
        return self

    def sharpen(self):
        self.steps.append({
            "type": "sharpen"
        })
        return self

    def sobel(self):
        self.steps.append({
            "type": "sobel"
        })
        return self

    def get_steps(self):
        return self.steps

    def threshold(self, value: float = 100.0):
        self.steps.append({
            "type": "threshold",
            "value": value
        })
        return self



def run_fixed_pipeline(image: np.ndarray, pipeline: Pipeline, method: str) -> np.ndarray:
    """
    Run pipeline using only one method:
    - spatial
    - fft

    Sobel always uses spatial because it uses two small 3x3 kernels.
    """
    output = image.copy()

    for step in pipeline.get_steps():
        step_type = step["type"]

        if step_type == "gaussian":
            kernel_size = step["kernel"]
            sigma = kernel_size / 6
            kernel = gaussian_kernel(kernel_size, sigma)

            if method == "spatial":
                output = spatial_convolution(output, kernel)
            elif method == "fft":
                output = fft_convolution(output, kernel)
            else:
                raise ValueError("method must be 'spatial' or 'fft'")

        elif step_type == "sharpen":
            kernel = sharpen_kernel()

            if method == "spatial":
                output = spatial_convolution(output, kernel)
            elif method == "fft":
                output = fft_convolution(output, kernel)
            else:
                raise ValueError("method must be 'spatial' or 'fft'")

        elif step_type == "sobel":
            kx, ky = sobel_kernels()

            gx = spatial_convolution(output, kx)
            gy = spatial_convolution(output, ky)

            output = np.sqrt(gx**2 + gy**2).astype(np.float32)

        elif step_type == "threshold":
            threshold_value = step["value"]
            output = threshold_image(output, threshold=threshold_value)

        else:
            raise ValueError(f"Unknown pipeline step: {step_type}")

    return output


def run_adaptive_pipeline(image: np.ndarray, pipeline: Pipeline) -> np.ndarray:
    """
    Run pipeline using adaptive runtime decisions.
    """
    output = image.copy()

    for step in pipeline.get_steps():
        output = run_step(output, step)

    return output