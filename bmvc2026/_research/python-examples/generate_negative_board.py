#!/usr/bin/env python3
"""
Generate Negative Polarity (Inverted) Calibration Board

Generate and save an A4-sized calibration board with light blobs on dark background.
Negative polarity boards can be useful for certain detection algorithms or
when printing on dark materials.

Files Created:
- blob_board_<UID>.pdf: High-resolution calibration board with inverted colors
- blob_board_<UID>.json: Complete metadata for pattern reproduction

Usage:
    pixi run python python/examples/boards/generate_negative_board.py
"""

from blobboards import blob_board

# Configuration
output_dir = "."
print_density = 300  # dpi
border_width = 25    # mm

print("Generating negative polarity A4 calibration board...")
print(f"Output directory: {output_dir}")
print(f"Print density: {print_density} dpi")
print(f"Border width: {border_width} mm")
print("Polarity: light (light blobs on dark background)")

# Generate negative polarity A4 calibration board
result = blob_board("A4", print_density,
                    frame_width=border_width,
                    polarity="light",
                    dir=output_dir)

# Report results
num_blobs = len(result.blobs)
print(f"\n✓ Generated negative polarity A4 calibration board with {num_blobs} blobs")
print(f"  Files saved to: {output_dir}")
