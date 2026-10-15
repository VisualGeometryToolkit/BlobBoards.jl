#!/usr/bin/env python3
"""
Examples of blob extraction using the Python interface.

This demonstrates:
- Direct blob access from PatternResult and BoardResult
- Different coordinate conventions (opencv, colmap, julia)
- Loading blobs from JSON files
"""

import numpy as np
from pathlib import Path
from blobboards import blob_pattern, blob_board, blobs, short_side

def example_pattern_blobs():
    """
    Extract blobs directly from PatternResult.
    """
    print("=" * 60)
    print("Example 1: Direct blob access from pattern")
    print("=" * 60)

    # Generate a test pattern
    result = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234)

    print(f"Pattern shape: {result.pattern.shape}")
    print(f"Number of blobs: {len(result.blobs)}")
    print(f"Blob array shape: {result.blobs.shape}")
    print(f"\nFirst 5 blobs [x, y, σ] in pixels:")
    print(result.blobs[:5])
    print()

def example_board_blobs():
    """
    Extract blobs directly from BoardResult.
    """
    print("=" * 60)
    print("Example 2: Direct blob access from board")
    print("=" * 60)

    # Generate a test board
    result = blob_board("A4", blob_sizes=(4.0, 0.2 * short_side), seed=0xBEEF)

    print(f"Number of blobs: {len(result.blobs)}")
    print(f"Board size: {result.preamble.board_config.canvas_width_mm:.1f} x {result.preamble.board_config.canvas_height_mm:.1f} mm")
    print(f"\nFirst 5 blobs [x, y, σ] in pixels:")
    print(result.blobs[:5])
    print()

def example_blobs_from_json():
    """
    Extract blobs from saved JSON files with different coordinate conventions.
    """
    print("=" * 60)
    print("Example 3: Load blobs from JSON files")
    print("=" * 60)

    # Generate and save a pattern
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234, dir=tmpdir)

        # Find the saved JSON file
        json_files = list(Path(tmpdir).glob("*.json"))
        if not json_files:
            print("No JSON file found!")
            return
        json_path = str(json_files[0])
        print(f"Loading from: {json_path}")

        # Extract with different coordinate conventions
        blobs_opencv = blobs(json_path, image_origin="opencv")
        blobs_colmap = blobs(json_path, image_origin="colmap")
        blobs_julia = blobs(json_path, image_origin="julia")

        print(f"\nFirst blob in different conventions:")
        print(f"  OpenCV (0,0):    {blobs_opencv[0]}")
        print(f"  COLMAP (0.5,0.5): {blobs_colmap[0]}")
        print(f"  Julia (1,1):     {blobs_julia[0]}")

        print(f"\nDifference COLMAP - OpenCV: {blobs_colmap[0] - blobs_opencv[0]}")
        print(f"Difference Julia - OpenCV: {blobs_julia[0] - blobs_opencv[0]}")
    print()

def example_board_reference_frames():
    """
    Compare different reference frames for board blobs.
    """
    print("=" * 60)
    print("Example 4: Board reference frames")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate and save a board
        result = blob_board("A4", blob_sizes=(4.0, 0.2 * short_side), seed=0xBEEF, dir=tmpdir)

        # Find the saved JSON file
        json_files = list(Path(tmpdir).glob("*.json"))
        if not json_files:
            print("No JSON file found!")
            return
        json_path = str(json_files[0])

        # Extract with different reference frames
        blobs_pattern = blobs(json_path, units="mm", reference_frame="pattern")
        blobs_board = blobs(json_path, units="mm", reference_frame="board")

        print(f"First blob (pattern origin, mm): {blobs_pattern[0]}")
        print(f"First blob (board origin, mm):   {blobs_board[0]}")
        print(f"Difference (border offset, mm):  {blobs_board[0] - blobs_pattern[0]}")
    print()

def example_numpy_operations():
    """
    Demonstrate NumPy operations on extracted blobs.
    """
    print("=" * 60)
    print("Example 5: NumPy operations on blob data")
    print("=" * 60)

    result = blob_pattern(800, 600, blob_sizes=(12, None), seed=0x1234)
    blob_data = result.blobs

    # Extract columns
    x_coords = blob_data[:, 0]
    y_coords = blob_data[:, 1]
    scales = blob_data[:, 2]

    print(f"Number of blobs: {len(blob_data)}")
    print(f"X range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    print(f"Scale range: [{scales.min():.2f}, {scales.max():.2f}]")
    print(f"Mean scale: {scales.mean():.2f}")
    print()

if __name__ == "__main__":
    example_pattern_blobs()
    example_board_blobs()
    example_blobs_from_json()
    example_board_reference_frames()
    example_numpy_operations()

    print("✓ All blob extraction examples completed!")
