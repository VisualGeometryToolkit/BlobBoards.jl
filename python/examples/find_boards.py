"""
find_boards demo — find the example boards in the example still, on the GPU
when one is usable (else the CPU), and save an overlay.

Mirrors the Julia entry point one-to-one.

Usage:
    pixi run -e py python python/examples/find_boards.py [IMAGE board1.json board2.json ...]

Without arguments it uses the bundled example data (downloaded on first use).
"""
import os
import sys

import numpy as np
from PIL import Image

import blobboards
from blobboards import blobs, descriptor_network_path, example_data, find_boards


def main():
    if len(sys.argv) > 1:
        image_path, *board_paths = sys.argv[1:]
    else:
        image_path = os.path.join(example_data("still"), "DSC02847.JPG")
        d = example_data("boards")
        board_paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    img = np.asarray(Image.open(image_path).convert("L"))

    print("device:", blobboards.device())
    found = find_boards(board_paths, descriptor_network_path(), img,
                        gallery_path="board_features.jld2")   # feature cache, created on first run
    for b in found:
        print(f"board {b['uid']}: quality={b['quality']:.1f} "
              f"inliers={b['n_inliers']}/{b['n_csponds']} ({b['execution_provider']})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    uid_to_json = {os.path.basename(p)[11:14]: p for p in board_paths}
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.imshow(img, cmap="gray")
    for b in found:
        X = blobs(uid_to_json[b["uid"]], reference_frame="world", units="mm")
        p = np.c_[X[:, :2], np.ones(len(X))] @ b["model"].T
        ax.plot(p[:, 0] / p[:, 2], p[:, 1] / p[:, 2], ".", ms=2)
        ax.annotate(b["uid"], p[0, :2] / p[0, 2], color="yellow")
    fig.savefig("found_boards.png", dpi=100)
    print("saved found_boards.png")


if __name__ == "__main__":
    main()
