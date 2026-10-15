#!/usr/bin/env python3
"""
Generate US Letter Calibration Board

Generate a US Letter sized calibration board for North American paper sizes.

Files Created:
- blob_board_<UID>.pdf: High-resolution calibration board
- blob_board_<UID>.json: Complete metadata for pattern reproduction

Usage:
    pixi run python python/examples/boards/generate_letter_board.py
"""

from blobboards import blob_board

# Configuration
output_dir = "."
print_density = 200  # dpi

print("Generating US Letter calibration board...")
print(f"Output directory: {output_dir}")
print(f"Print density: {print_density} dpi")

# Generate US Letter board
result = blob_board("Letter", print_density, dir=output_dir)

# Report results
num_blobs = len(result.blobs)
print(f"\n✓ Generated US Letter calibration board with {num_blobs} blobs")
print(f"  Files saved to: {output_dir}")
