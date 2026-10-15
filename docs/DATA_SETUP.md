# Setting up BlobBoards data paths

BlobBoards examples load test data from a shared NFS directory. Each machine needs to specify its local mount point once.

## Setup (one-time)

1. **Pull latest**
   ```bash
   git pull
   ```

2. **Create `LocalPreferences.toml`** in the project root (`BlobBoards.jl/LocalPreferences.toml`):
   ```toml
   [BlobBoards]
   data_dir = "/path/to/your/nfs/mount/20251028_BlobBoards_Office"
   ```
   Replace the path with wherever the NFS share is mounted on your machine.

   This file is gitignored — it won't affect anyone else.

3. **Reinstantiate** (one-time, to pick up the new `Preferences` dependency):
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

## Verify

```julia
using BlobBoards
data_dir()  # Should print your NFS path
```

## Data directory layout

The examples expect this structure under `data_dir()`:

```
20251028_BlobBoards_Office/
├── blobboard_videos/          # Video sequences
├── cotracker_results/         # Video + homographies for training
├── dataset_sonyfx3_tamron20mm_f8/jpeg/  # Real camera images
├── evaluation/
│   ├── blobboards_original/   # Board JSON/PDF/PNG files
│   └── hallway_leica_01/png/8bit/  # Test images
├── models/                    # ONNX models
└── training/                  # Training output
```

## Alternative: environment variable

If you prefer not to create a file, set `BLOBBOARDS_DATA` instead:
```bash
export BLOBBOARDS_DATA="/path/to/20251028_BlobBoards_Office"
```
