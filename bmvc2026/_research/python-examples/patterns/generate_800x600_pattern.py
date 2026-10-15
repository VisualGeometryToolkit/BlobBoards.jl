#!/usr/bin/env python3
"""
Generate 800×600 Blob Pattern

Generate and save a blob pattern as PNG with automatic metadata saving.
Creates a blob pattern with specified dimensions and saves both the pattern
image and complete metadata for reproducibility and analysis.

Files Created:
- blob_pattern_<UID>.png: Raw pattern image
- blob_pattern_<UID>.json: Pattern metadata and blob coordinates

Usage:
    pixi run python python/examples/patterns/generate_800x600_pattern.py
"""

from blobboards import blob_pattern

# Configuration
output_dir = "."
width = 800
height = 600

print(f"Generating {width}×{height} blob pattern...")
print(f"Output directory: {output_dir}")

# Create a blob pattern
result = blob_pattern(width, height, dir=output_dir)

# Report results
num_blobs = len(result.blobs)
print(f"\n✓ Generated {width}×{height} blob pattern with {num_blobs} blobs")
print(f"  Files saved to: {output_dir}")
