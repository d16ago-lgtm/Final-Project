import numpy as np
from PIL import Image
import os
if os.path.exists("results/history.csv"):
    os.remove("results/history.csv")

from filters import (
    gaussian_kernel,
    spatial_convolution,
    fft_convolution,
    threshold_image
)

from pipeline import Pipeline, run_fixed_pipeline, run_adaptive_pipeline
from metrics import time_function, mean_absolute_error, speedup, estimate_spatial_memory, estimate_fft_memory, evaluate_execution_methods
from visualize import (
    plot_kernel_runtime,
    plot_runtime_bar,
    show_pipeline_outputs,
    show_pipeline_stages,
    plot_runtime_by_image_size,
    plot_runtime_memory_tradeoff
)

def load_grayscale_image(path: str) -> np.ndarray:
    #Load image as grayscale float32 array
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)

def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    #Resize image to size x size
    img = Image.fromarray(image.astype(np.uint8))
    img = img.resize((size, size))
    return np.array(img, dtype=np.float32)


def experiment_spatial_vs_fft():
    """
    Experiment 1:
    Compare spatial vs FFT convolution across different kernel sizes.

    This version tests multiple images and averages the runtime results.
    This makes the crossover graph more reliable than using only one image.
    """

    image_paths = [
        "images/xray1.png",
        "images/xray2.png",
        "images/xray3.png"
    ]

    kernel_sizes = [3, 5, 9, 15, 31, 63]
    repeats = 3

    avg_spatial_times = []
    avg_fft_times = []

    print("\nExperiment 1: Average Spatial vs FFT Runtime by Kernel Size")
    print("-" * 60)

    for kernel_size in kernel_sizes:
        sigma = kernel_size / 6
        kernel = gaussian_kernel(kernel_size, sigma)

        all_spatial_times = []
        all_fft_times = []
        all_mae_values = []

        for image_path in image_paths:
            image = load_grayscale_image(image_path)

            for run in range(repeats):
                spatial_result, spatial_time = time_function(
                    spatial_convolution,
                    image,
                    kernel
                )

                fft_result, fft_time = time_function(
                    fft_convolution,
                    image,
                    kernel
                )

                mae = mean_absolute_error(spatial_result, fft_result)

                all_spatial_times.append(spatial_time)
                all_fft_times.append(fft_time)
                all_mae_values.append(mae)

        avg_spatial = np.mean(all_spatial_times)
        avg_fft = np.mean(all_fft_times)
        avg_mae = np.mean(all_mae_values)

        avg_spatial_times.append(avg_spatial)
        avg_fft_times.append(avg_fft)

        print(f"Kernel size: {kernel_size}")
        print(f"Average spatial time: {avg_spatial:.6f}")
        print(f"Average FFT time:     {avg_fft:.6f}")
        print(f"Average MAE:          {avg_mae:.6f}")
        print()

    plot_kernel_runtime(
        kernel_sizes,
        avg_spatial_times,
        avg_fft_times
    )

def experiment_baseline_vs_adaptive(image):
    """
    Experiment 2:
    Compare always-spatial, always-FFT, and adaptive pipeline.
    """
    # Full pipeline for timing experiments
    pipeline = (
        Pipeline()
        .gaussian(5)
        .sharpen()
        .gaussian(15)
        .sobel()
        .threshold(100)
    )

    # Visualization pipeline (without threshold)
    visual_pipeline = (
        Pipeline()
        .gaussian(5)
        .sharpen()
        .gaussian(15)
        .sobel()
    )

    print("\nExperiment 2: Baseline vs Adaptive Pipeline")
    print("-" * 50)

    spatial_result, spatial_time = time_function(
        run_fixed_pipeline,
        image,
        pipeline,
        "spatial"
    )

    fft_result, fft_time = time_function(
        run_fixed_pipeline,
        image,
        pipeline,
        "fft"
    )

    adaptive_result, adaptive_time = time_function(
        run_adaptive_pipeline,
        image,
        pipeline
    )

    spatial_vs_adaptive_mae = mean_absolute_error(spatial_result, adaptive_result)
    fft_vs_adaptive_mae = mean_absolute_error(fft_result, adaptive_result)

    adaptive_speedup_vs_spatial = speedup(spatial_time, adaptive_time)
    adaptive_speedup_vs_fft = speedup(fft_time, adaptive_time)

    print("Pipeline:")
    for step in pipeline.get_steps():
        print(step)

    print()
    print(f"Always spatial time: {spatial_time:.6f}")
    print(f"Always FFT time:     {fft_time:.6f}")
    print(f"Adaptive time:       {adaptive_time:.6f}")

    print()
    print(f"Adaptive speedup vs spatial: {adaptive_speedup_vs_spatial:.2f}x")
    print(f"Adaptive speedup vs FFT:     {adaptive_speedup_vs_fft:.2f}x")

    print()
    print(f"MAE spatial vs adaptive: {spatial_vs_adaptive_mae:.6f}")
    print(f"MAE FFT vs adaptive:     {fft_vs_adaptive_mae:.6f}")

    methods = ["Always Spatial", "Always FFT", "Adaptive"]
    times = [spatial_time, fft_time, adaptive_time]

    plot_runtime_bar(
        methods,
        times,
        "Baseline vs Adaptive Pipeline Runtime"
    )
    # Generate visualization outputs without thresholding
    visual_spatial, _ = time_function(
        run_fixed_pipeline,
        image,
        visual_pipeline,
        "spatial"
    )

    visual_fft, _ = time_function(
        run_fixed_pipeline,
        image,
        visual_pipeline,
        "fft"
    )

    visual_adaptive, _ = time_function(
        run_adaptive_pipeline,
        image,
        visual_pipeline
    )
    show_pipeline_outputs(
        image,
        visual_spatial,
        visual_fft,
        visual_adaptive
    )

def experiment_pipeline_stages(image):
    """
    Experiment 3:
    Show each image-processing stage.
    """
    # Stage 1: Gaussian smoothing
    kernel_5 = gaussian_kernel(5, 5 / 6)
    gaussian_1 = spatial_convolution(image, kernel_5)

    # Stage 2: sharpening
    from filters import sharpen_kernel, sobel_kernels

    sharp_kernel = sharpen_kernel()
    sharpened = spatial_convolution(gaussian_1, sharp_kernel)

    # Stage 3: larger Gaussian smoothing
    kernel_15 = gaussian_kernel(15, 15 / 6)
    gaussian_2 = fft_convolution(sharpened, kernel_15)

    # Stage 4: Sobel edge detection
    kx, ky = sobel_kernels()
    gx = spatial_convolution(gaussian_2, kx)
    gy = spatial_convolution(gaussian_2, ky)
    sobel_edges = np.sqrt(gx**2 + gy**2).astype(np.float32)

    # BEFORE threshold
    sobel_edges = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    # Stage 5: thresholding
    thresholded = threshold_image(sobel_edges, threshold=100)


    # AFTER threshold
    thresholded = threshold_image(sobel_edges, threshold=100)

    stages = [
        ("Original", image),
        ("Gaussian Smooth", gaussian_1),
        ("Sharpen", sharpened),
        ("FFT Gaussian", gaussian_2),
        ("Sobel Edges", sobel_edges),
        ("Threshold", thresholded)
    ]

    show_pipeline_stages(stages)

def experiment_multiple_images_and_sizes():
    """
    Experiment 4:
    Compare spatial, FFT, and adaptive pipelines across multiple images
    and image sizes.

    This averages runtime across:
    - multiple chest X-ray images
    - repeated runs

    """

    image_paths = [
        "images/xray1.png",
        "images/xray2.png",
        "images/xray3.png"
    ]

    image_sizes = [256, 512, 1024]
    repeats = 3

    pipeline = (
        Pipeline()
        .gaussian(5)
        .sharpen()
        .gaussian(15)
        .sobel()
        .threshold(100)
    )

    results = []

    print("\nExperiment 4: Average Pipeline Runtime by Image Size")
    print("-" * 60)

    for image_path in image_paths:
        original_image = load_grayscale_image(image_path)

        for size in image_sizes:
            image = resize_image(original_image, size)

            spatial_times = []
            fft_times = []
            adaptive_times = []

            for run in range(repeats):
                spatial_result, spatial_time = time_function(
                    run_fixed_pipeline,
                    image,
                    pipeline,
                    "spatial"
                )

                fft_result, fft_time = time_function(
                    run_fixed_pipeline,
                    image,
                    pipeline,
                    "fft"
                )

                adaptive_result, adaptive_time = time_function(
                    run_adaptive_pipeline,
                    image,
                    pipeline
                )

                spatial_times.append(spatial_time)
                fft_times.append(fft_time)
                adaptive_times.append(adaptive_time)

            avg_spatial = np.mean(spatial_times)
            avg_fft = np.mean(fft_times)
            avg_adaptive = np.mean(adaptive_times)

            results.append({
                "image": image_path,
                "size": size,
                "spatial_time": avg_spatial,
                "fft_time": avg_fft,
                "adaptive_time": avg_adaptive
            })

            print(f"Image: {image_path}, Size: {size}x{size}")
            print(f"Average spatial time:  {avg_spatial:.6f}")
            print(f"Average FFT time:      {avg_fft:.6f}")
            print(f"Average adaptive time: {avg_adaptive:.6f}")
            print()

    return results

def experiment_memory_usage():
    image_sizes = [256, 512, 1024]

    spatial_runtime = [0.06, 0.23, 0.97]
    fft_runtime = [0.02, 0.13, 0.60]
    adaptive_runtime = [0.025, 0.11, 0.49]

    spatial_memory = [0.5, 2.0, 8.0]
    fft_memory = [4.5, 15.5, 59.0]
    adaptive_memory = [0.5, 15.5, 59.0]

    plot_runtime_memory_tradeoff(
        spatial_runtime,
        fft_runtime,
        adaptive_runtime,
        spatial_memory,
        fft_memory,
        adaptive_memory,
        image_sizes
    )
def experiment_multiple_images_and_sizes():
    """
    Run spatial, FFT, and adaptive pipelines across multiple images and sizes.
    Results are averaged across multiple images and repeated runs.
    """
    image_paths = [
        "images/xray1.png",
        "images/xray2.png",
        "images/xray3.png"
    ]

    image_sizes = [256, 512, 1024]
    repeats = 3

    pipeline = (
        Pipeline()
        .gaussian(5)
        .sharpen()
        .gaussian(15)
        .sobel()
        .threshold(100)
    )

    results = []

    print("\nExperiment: Average Runtime Across Multiple Images")
    print("-" * 60)

    for image_path in image_paths:
        original_image = load_grayscale_image(image_path)

        for size in image_sizes:
            image = resize_image(original_image, size)

            spatial_times = []
            fft_times = []
            adaptive_times = []

            for run in range(repeats):
                spatial_result, spatial_time = time_function(
                    run_fixed_pipeline,
                    image,
                    pipeline,
                    "spatial"
                )

                fft_result, fft_time = time_function(
                    run_fixed_pipeline,
                    image,
                    pipeline,
                    "fft"
                )

                adaptive_result, adaptive_time = time_function(
                    run_adaptive_pipeline,
                    image,
                    pipeline
                )

                spatial_times.append(spatial_time)
                fft_times.append(fft_time)
                adaptive_times.append(adaptive_time)

            avg_spatial = np.mean(spatial_times)
            avg_fft = np.mean(fft_times)
            avg_adaptive = np.mean(adaptive_times)

            results.append({
                "image": image_path,
                "size": size,
                "spatial_time": avg_spatial,
                "fft_time": avg_fft,
                "adaptive_time": avg_adaptive
            })

            print(f"Image: {image_path}, Size: {size}x{size}")
            print(f"Average Spatial:  {avg_spatial:.6f}")
            print(f"Average FFT:      {avg_fft:.6f}")
            print(f"Average Adaptive: {avg_adaptive:.6f}")
            print()

    return results

def experiment_final_metrics_summary():
    """
    Final summary experiment

    Measures average:
    - runtime
    - speedup
    - MAE
    - peak memory
    across multiple images and image sizes.
    """

    image_paths = [
        "images/xray1.png",
        "images/xray2.png",
        "images/xray3.png"
    ]

    image_sizes = [256, 512, 1024]

    pipeline = (
        Pipeline()
        .gaussian(5)
        .sharpen()
        .gaussian(15)
        .sobel()
        .threshold(100)
    )

    all_results = []

    print("\nFinal Metrics Summary")
    print("-" * 60)

    for image_path in image_paths:
        original_image = load_grayscale_image(image_path)

        for size in image_sizes:
            image = resize_image(original_image, size)

            result = evaluate_execution_methods(
                image,
                pipeline
            )

            all_results.append(result)

    avg_spatial_runtime = np.mean([r["spatial_runtime"] for r in all_results])
    avg_fft_runtime = np.mean([r["fft_runtime"] for r in all_results])
    avg_adaptive_runtime = np.mean([r["adaptive_runtime"] for r in all_results])

    avg_spatial_memory = np.mean([r["spatial_memory"] for r in all_results])
    avg_fft_memory = np.mean([r["fft_memory"] for r in all_results])
    avg_adaptive_memory = np.mean([r["adaptive_memory"] for r in all_results])

    avg_fft_mae = np.mean([r["fft_mae"] for r in all_results])
    avg_adaptive_mae = np.mean([r["adaptive_mae"] for r in all_results])

    adaptive_speedup_vs_spatial = avg_spatial_runtime / avg_adaptive_runtime
    adaptive_speedup_vs_fft = avg_fft_runtime / avg_adaptive_runtime

    print("Average Results Across All Images and Sizes")
    print("-" * 60)
    print(f"Spatial runtime:  {avg_spatial_runtime:.6f} s")
    print(f"FFT runtime:      {avg_fft_runtime:.6f} s")
    print(f"Adaptive runtime: {avg_adaptive_runtime:.6f} s")
    print()
    print(f"Spatial memory:   {avg_spatial_memory:.2f} MB")
    print(f"FFT memory:       {avg_fft_memory:.2f} MB")
    print(f"Adaptive memory:  {avg_adaptive_memory:.2f} MB")
    print()
    print(f"FFT MAE vs Spatial:      {avg_fft_mae:.6f}")
    print(f"Adaptive MAE vs Spatial: {avg_adaptive_mae:.6f}")
    print()
    print(f"Adaptive speedup vs Spatial: {adaptive_speedup_vs_spatial:.2f}x")
    print(f"Adaptive speedup vs FFT:     {adaptive_speedup_vs_fft:.2f}x")

def experiment_runtime_memory_tradeoff():
    """
    Estimate and plot runtime vs memory tradeoff for spatial, FFT, and adaptive.
    """

    image_sizes = [256, 512, 1024]

    # Use the average runtimes from your latest runtime-by-image-size graph
    spatial_runtime = [0.03, 0.17, 0.58]
    fft_runtime = [0.012, 0.087, 0.34]
    adaptive_runtime = [0.017, 0.073, 0.31]

    # Estimated memory values from your memory calculation
    spatial_memory = [0.5, 2.0, 8.0]
    fft_memory = [4.5, 15.5, 59.0]

    # Adaptive memory follows the method it selected
    adaptive_memory = [0.5, 15.5, 59.0]

    plot_runtime_memory_tradeoff(
        spatial_runtime,
        fft_runtime,
        adaptive_runtime,
        spatial_memory,
        fft_memory,
        adaptive_memory,
        image_sizes
    )

def main():
    image_path = "images/xray1.png"
    image = load_grayscale_image(image_path)

    #Average crossover analysis across multiple images
    experiment_spatial_vs_fft()

    #Single image example for pipeline output visuals
    experiment_baseline_vs_adaptive(image)

    #Image-processing stage visualization
    experiment_pipeline_stages(image)

    #Average runtime by image size across multiple images
    multi_results = experiment_multiple_images_and_sizes()
    plot_runtime_by_image_size(multi_results)

    #Memory comparison
    experiment_runtime_memory_tradeoff()

    experiment_final_metrics_summary()



if __name__ == "__main__":
    main()