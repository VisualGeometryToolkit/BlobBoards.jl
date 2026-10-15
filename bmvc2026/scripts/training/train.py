"""
Train a local descriptor on BlobBoards tracks using SupCon loss.

Usage:
    python examples/training/train.py --data training_tracks.tracks
    python examples/training/train.py --data training_tracks.tracks --epochs 200 --desc-dim 128

Requires:
    pip install torch h5py pytorch-metric-learning
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler

from blob_dataset import BlobTrackDataset


# ---- Model ----


class DescriptorNet(nn.Module):
    """Simple CNN descriptor for 64x64 grayscale patches.

    Architecture: 3 conv blocks (32→64→128) with BN+ReLU, max-pool between blocks,
    global average pool, linear projection to `desc_dim`, L2-normalized output.
    """

    def __init__(self, desc_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16x16 → global
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, desc_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)
        x = self.head(x)
        return F.normalize(x, dim=1)


# ---- Training ----


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = BlobTrackDataset(args.data, min_track_length=args.min_track_length)

    if dataset.n_classes < 2:
        raise ValueError(
            f"Need at least 2 tracks with length >= {args.min_track_length}, "
            f"got {dataset.n_classes}"
        )

    # MPerClassSampler: each batch gets m observations from batch_size/m tracks
    # m = samples per track per batch (need >= 2 for positive pairs)
    m = min(args.m_per_class, args.min_track_length)
    sampler = MPerClassSampler(
        dataset.labels, m=m, batch_size=args.batch_size, length_before_new_iter=len(dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model + optimizer
    model = DescriptorNet(desc_dim=args.desc_dim).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = SupConLoss(temperature=args.temperature)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, desc_dim={args.desc_dim}")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, m={m}")
    print(f"Loss: SupCon, temperature={args.temperature}")
    print()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)

            embeddings = model(patches)
            loss = loss_fn(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        lr = scheduler.get_last_lr()[0]

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.2e}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = Path(args.output_dir) / f"descriptor_epoch{epoch:04d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "desc_dim": args.desc_dim,
                },
                ckpt_path,
            )
            print(f"  Saved {ckpt_path}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Train blob descriptor with SupCon loss")
    parser.add_argument("--data", required=True, help="Path to training_tracks.tracks")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--desc-dim", type=int, default=128, help="Descriptor dimension")
    parser.add_argument("--batch-size", type=int, default=96, help="Batch size")
    parser.add_argument("--m-per-class", type=int, default=3, help="Samples per track per batch")
    parser.add_argument("--min-track-length", type=int, default=2, help="Min track length to include")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.1, help="SupCon temperature")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N epochs")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
