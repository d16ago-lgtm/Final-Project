import numpy as np
import matplotlib.pyplot as plt


def show_pipeline_stages(stages):
    """
    Show each stage of the image-processing pipeline.

    stages example:
    [
        ("Original", image),
        ("Gaussian", gaussian_output),
        ("Sharpen", sharpen_output),
        ("Sobel", sobel_output),
        ("Threshold", threshold_output)
    ]
    """
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4))

    for ax, (title, image) in zip(axes, stages):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_frequency_spectrum(image, title="Frequency Spectrum"):

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

    plt.figure(figsize=(6, 5))
    plt.imshow(magnitude, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_frequency_comparison(original, processed):
    """
    Compare frequency spectrum before and after processing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    images = [original, processed]
    titles = ["Original Frequency Spectrum", "Processed Frequency Spectrum"]

    for ax, img, title in zip(axes, images, titles):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        ax.imshow(magnitude, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_kernel_runtime(kernel_sizes, spatial_times, fft_times):
    """
    Plot average spatial vs FFT runtime across multiple images
    for different kernel sizes.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(
        kernel_sizes,
        spatial_times,
        marker="o",
        label="Spatial Convolution"
    )

    plt.plot(
        kernel_sizes,
        fft_times,
        marker="o",
        label="FFT Convolution"
    )

    plt.xlabel("Kernel Size")
    plt.ylabel("Average Runtime (seconds)")

    plt.title(
        "Average Runtime by Kernel Size Across Multiple Images"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_runtime_bar(methods, times, title="Runtime Comparison"):
    """
    Bar chart for runtime comparison.
    """
    plt.figure(figsize=(8, 5))

    plt.bar(methods, times)

    plt.ylabel("Runtime (seconds)")
    plt.title(title)

    plt.tight_layout()
    plt.show()


def normalize_for_display(img):

    img = img.astype(np.float32)

    img = img - img.min()

    # brighten low-intensity edges
    img = np.sqrt(img)

    img = img / (img.max() + 1e-8)

    return img


def show_pipeline_outputs(original, spatial, fft, adaptive):

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    images = [
        normalize_for_display(original),
        normalize_for_display(spatial),
        normalize_for_display(fft),
        normalize_for_display(adaptive)
    ]

    titles = [
        "Original",
        "Always Spatial",
        "Always FFT",
        "Adaptive"
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_runtime_by_image_size(results):
    """
    Plot average runtime by image size for:
    - spatial
    - FFT
    - adaptive

    Results are averaged across:
    - multiple images
    - repeated runs
    """

    # Get unique image sizes
    sizes = sorted(list(set(row["size"] for row in results)))

    spatial_avg = []
    fft_avg = []
    adaptive_avg = []

    for size in sizes:

        # Get all rows for this image size
        size_rows = [
            row for row in results
            if row["size"] == size
        ]

        # Average runtimes across all images
        spatial_avg.append(
            np.mean([
                row["spatial_time"]
                for row in size_rows
            ])
        )

        fft_avg.append(
            np.mean([
                row["fft_time"]
                for row in size_rows
            ])
        )

        adaptive_avg.append(
            np.mean([
                row["adaptive_time"]
                for row in size_rows
            ])
        )

    x = np.arange(len(sizes))
    width = 0.25

    plt.figure(figsize=(9, 5))

    plt.bar(
        x - width,
        spatial_avg,
        width,
        label="Always Spatial"
    )

    plt.bar(
        x,
        fft_avg,
        width,
        label="Always FFT"
    )

    plt.bar(
        x + width,
        adaptive_avg,
        width,
        label="Adaptive"
    )

    plt.xticks(
        x,
        [f"{s}x{s}" for s in sizes]
    )

    plt.xlabel("Image Size")
    plt.ylabel("Average Runtime (seconds)")

    plt.title(
        "Average Pipeline Runtime by Image Size"
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_runtime_memory_tradeoff(
        spatial_runtime,
        fft_runtime,
        adaptive_runtime,
        spatial_memory,
        fft_memory,
        adaptive_memory,
        image_sizes):

    plt.figure(figsize=(8, 6))

    # Spatial points
    plt.scatter(
        spatial_runtime,
        spatial_memory,
        s=100,
        label="Spatial"
    )

    # FFT points
    plt.scatter(
        fft_runtime,
        fft_memory,
        s=100,
        label="FFT"
    )

    # Adaptive points
    plt.scatter(
        adaptive_runtime,
        adaptive_memory,
        s=100,
        label="Adaptive"
    )

    # Label adaptive points only
    for i, size in enumerate(image_sizes):

        plt.annotate(
            f"{size}x{size}",
            (adaptive_runtime[i], adaptive_memory[i]),
            textcoords="offset points",
            xytext=(8, 8)
        )

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Estimated Memory Usage (MB)")

    plt.title(
        "Execution Strategy Tradeoff"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()