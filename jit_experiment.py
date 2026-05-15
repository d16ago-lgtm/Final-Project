import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def load_grayscale_image(path: str, size = 1024) -> torch.Tensor:
    #Load image as grayscale PyTorch tensor.
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0

    # Shape: batch, channel, height, width
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    return tensor

def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    #Create Gaussian kernel as PyTorch tensor.
    ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / torch.sum(kernel)

    #Shape needed for conv2d: out_channels, in_channels, height, width
    return kernel.unsqueeze(0).unsqueeze(0)


def sharpen_kernel() -> torch.Tensor:

    kernel = torch.tensor([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=torch.float32)

    return kernel.unsqueeze(0).unsqueeze(0)


def sobel_kernels():

    kx = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    ky = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return kx, ky


def apply_kernel(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return F.conv2d(image, kernel, padding=padding)


def eager_pipeline(image: torch.Tensor) -> torch.Tensor:
    """
    Eager PyTorch image-processing pipeline:
    Gaussian blur -> sharpen -> Gaussian blur -> Sobel
    """
    k1 = gaussian_kernel(5, 5 / 6)
    k2 = gaussian_kernel(15, 15 / 6)
    sharp = sharpen_kernel()
    sx, sy = sobel_kernels()

    output = apply_kernel(image, k1)

    for _ in range(3):
        output = apply_kernel(output, sharp)
        output = apply_kernel(output, k2)

    gx = apply_kernel(output, sx)
    gy = apply_kernel(output, sy)

    output = torch.sqrt(gx**2 + gy**2)

    return output


def time_function(func, image, warmup=10, runs=30):
    with torch.no_grad():
        for _ in range(warmup):
            _ = func(image)

        times = []

        for _ in range(runs):
            start = time.perf_counter()
            output = func(image)
            end = time.perf_counter()
            times.append(end - start)

    median_time = np.median(times)
    return output, median_time

def main():
    image_path = "images/xray2.png"

    image = load_grayscale_image(image_path, size = 1024)

    print("Running eager pipeline...")
    eager_output, eager_time = time_function(eager_pipeline, image)

    print("Trying torch.compile...")

    try:
        # Safer backend for Windows when Inductor fails because C++ compiler is missing
        compiled_pipeline = torch.compile(eager_pipeline, backend="aot_eager")

        print("Running compiled pipeline...")
        jit_output, jit_time = time_function(compiled_pipeline, image)

        compile_status = "torch.compile ran successfully using backend='aot_eager'"

    except Exception as e:
        print("torch.compile failed. Falling back to eager output.")
        print("Reason:", e)

        eager_time = eager_time * 1000
        jit_output = eager_output
        jit_time = eager_time * 1000
        compile_status = "torch.compile failed; fallback used"

    mae = torch.mean(torch.abs(eager_output - jit_output)).item()

    if jit_time > 0:
        jit_speedup = eager_time/jit_time
    else:
        jit_speedup = 0

    print()
    print("JIT Experiment Results")
    print("-" * 40)
    print(compile_status)
    print(f"Eager time: {eager_time:.6f} ms")
    print(f"JIT time:   {jit_time:.6f} ms")
    print(f"JIT speedup: {jit_speedup:.2f}x")
    print(f"MAE eager vs JIT: {mae:.8f}")

    methods = ["Eager", "torch.compile"]
    times = [eager_time, jit_time]

    plt.figure(figsize=(7, 5))
    plt.bar(methods, times)
    plt.ylabel("Average Runtime (ms)")
    plt.title("Eager vs JIT Pipeline Runtime")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image.squeeze().numpy(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(eager_output.squeeze().numpy(), cmap="gray")
    axes[1].set_title("Eager Output")
    axes[1].axis("off")

    axes[2].imshow(jit_output.squeeze().numpy(), cmap="gray")
    axes[2].set_title("JIT Output")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()