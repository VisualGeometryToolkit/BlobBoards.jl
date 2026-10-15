#!/usr/bin/env python3
"""
Basic usage for the BlobBoards Python interface.

This example demonstrates generating patterns and boards using the clean
dataclass-based API.
"""

import sys
import os

# Add the parent directory to the path so we can import blobboards
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from blobboards import blob_pattern, blob_board


def run():
    os.makedirs("./patterns", exist_ok=True)
    os.makedirs("./boards", exist_ok=True)

    # Pattern: 800x600 px with blob_sizes in pixels
    print("Generating 800x600 pattern...")
    result = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234, dir="./patterns")
    print(f"  Pattern shape: {result.pattern.shape}")
    print(f"  Num blobs: {len(result.blobs)}")
    print(f"  Config: width={result.config.width}, height={result.config.height}")

    # Board: A4 at 300 dpi with blob_sizes in mm
    print("\nGenerating A4 board at 300 dpi...")
    result = blob_board("A4", 300, blob_sizes=(4.0, None), seed=0xBEEF, dir="./boards")
    print(f"  Num blobs: {len(result.blobs)}")
    print(f"  Board size: {result.preamble.board_config.canvas_width_mm:.1f} x {result.preamble.board_config.canvas_height_mm:.1f} mm")
    print(f"  DPI: {result.preamble.board_config.print_density_dpi}")

    # Custom size board: 150mm x 200mm at 200 dpi
    print("\nGenerating custom 150x200mm board...")
    result = blob_board((150, 200), 200, blob_sizes=(3.2, None), seed=0xCAFE, dir="./boards")
    print(f"  Num blobs: {len(result.blobs)}")
    print(f"  Board size: {result.preamble.board_config.canvas_width_mm:.1f} x {result.preamble.board_config.canvas_height_mm:.1f} mm")

    print("\nDone! Files saved to ./patterns and ./boards")


if __name__ == "__main__":
    run()
