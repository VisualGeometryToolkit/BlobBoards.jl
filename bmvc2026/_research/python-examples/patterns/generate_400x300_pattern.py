#!/usr/bin/env python3
"""
Generate 400×300 Blob Pattern (Small)

Generate a small blob pattern for testing or embedded applications.

Files Created:
- blob_pattern_<UID>.png: Raw pattern image
- blob_pattern_<UID>.json: Pattern metadata and blob coordinates

Usage:
    pixi run python python/examples/patterns/generate_400x300_pattern.py
"""

from blobboards import blob_pattern

# Configuration
output_dir = "."
width = 400
height = 300

print(f"Generating {width}×{height} small blob pattern...")
print(f"Output directory: {output_dir}")

# Generate small pattern
result = blob_pattern(width, height, dir=output_dir)

# Report results
num_blobs = len(result.blobs)
print(f"\n✓ Generated {width}×{height} small pattern with {num_blobs} blobs")
print(f"  Files saved to: {output_dir}")
