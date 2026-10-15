"""
Example: Detect blobs in an image using the BlobBoards Python wrapper.

This example demonstrates:
- Loading an image with PIL/numpy
- Detecting blobs using detect_blobs()
- Visualizing results with matplotlib

Usage:
    pixi run python python/examples/detection/detect_blobs_example.py
"""
import sys
from pathlib import Path

# Add python package to path (for development)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from PIL import Image

from blobboards import detect_blobs


def main():
    # Load test image
    img_path = project_root / "test/data/blob_pattern_04a_a4abd.png"
    img = np.array(Image.open(img_path).convert('L')) / 255.0

    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")

    # Methods with tuned thresholds (DoG/Laplacian need ~15x higher threshold)
    methods = ["dog", "laplacian", "hessian_laplace", "hessian_hessian"]
    thresholds = [0.015, 0.015, 0.001, 0.001]

    # Detect with each method
    results = []
    for method, thresh in zip(methods, thresholds):
        print(f"\nRunning {method} (threshold={thresh})...")
        result = detect_blobs(img, method=method, peak_threshold=thresh, polarity="dark")
        print(f"Found {len(result)} blobs")
        results.append(result)

    # Plot results in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, method, result in zip(axes, methods, results):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{method} (N={len(result)})')

        # Draw circles for each detection
        for det in result:
            circle = MplCircle(
                (det.x, det.y),
                det.sigma * 3,  # 3-sigma radius
                fill=False,
                edgecolor='lime',
                linewidth=1.0
            )
            ax.add_patch(circle)

        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    plt.tight_layout()

    # Save and show
    output_path = project_root / "examples" / "python_detect_blobs.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
