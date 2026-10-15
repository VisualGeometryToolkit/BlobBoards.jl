#!/usr/bin/env python3
"""
Generate A4 Calibration Board

Generate and save an A4-sized calibration board as high-quality PDF.
Creates a calibration board with blob pattern, rulers, QR code,
and visual identifiers optimized for printing and computer vision applications.

Files Created:
- blob_board_<UID>.pdf: High-resolution calibration board
- blob_board_<UID>.json: Complete metadata for pattern reproduction

Usage:
    pixi run -e py python python/examples/boards/generate_a4_board.py
"""

from blobboards import blob_board, suggest_blob_sizes

# Configuration
output_dir = "."
printer_dpi = 300     # the printer the board is sized for (a board itself has no density)
ruler_width = 15      # mm: the ruler bands (0 turns the ruler off)

print("Generating A4 calibration board...")
print(f"Output directory: {output_dir}")
print(f"Sized for a {printer_dpi} dpi printer")
print(f"Ruler width: {ruler_width} mm")

# Generate A4 calibration board
result = blob_board("A4",
                    blob_sizes=suggest_blob_sizes(printer_dpi, "A4"),   # 12 mm … 20% of the 190 mm board side
                    ruler_width=ruler_width,
                    dir=output_dir)

# Report results
num_blobs = len(result.blobs)
print(f"\n✓ Generated A4 calibration board with {num_blobs} blobs")
print(f"  Files saved to: {output_dir}")
