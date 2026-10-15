#!/usr/bin/env python3
"""
Generate Custom-Sized Calibration Board

Generate a calibration board with custom physical dimensions (150mm x 200mm).
Demonstrates how to use the width/height interface for non-standard paper sizes.

Files Created:
- blob_board_<UID>.pdf: High-resolution calibration board
- blob_board_<UID>.json: Complete metadata for pattern reproduction

Usage:
    pixi run python python/examples/boards/generate_custom_size_board.py
"""

from blobboards import blob_board

# Configuration
output_dir = "."
width = 150   # mm
height = 200  # mm
print_density = 300  # dpi
border_width = 15    # mm

print("Generating custom-sized calibration board...")
print(f"Output directory: {output_dir}")
print(f"Dimensions: {width} mm × {height} mm")
print(f"Print density: {print_density} dpi")
print(f"Border width: {border_width} mm")

# Generate custom calibration board
result = blob_board((width, height), print_density,
                    frame_width=border_width,
                    dir=output_dir)

# Report results
num_blobs = len(result.blobs)
print(f"\n✓ Generated {width}×{height}mm calibration board with {num_blobs} blobs")
print(f"  Files saved to: {output_dir}")
