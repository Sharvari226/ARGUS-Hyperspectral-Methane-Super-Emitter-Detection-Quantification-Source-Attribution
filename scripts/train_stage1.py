"""
Quick-train script for Stage 1 plume segmenter.
Uses GEETROPOMIIngester (same live GEE path as the pipeline) for real data.
Falls back to synthetic if GEE is unavailable.

Run: python -m scripts.train_stage1 --epochs 10
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.data.gee import GEETROPOMIIngester
from src.models.stage1_sat import PlumeSegmenter, preprocess_tropomi, save_model


# ── Regions (same as pipeline scheduler) ─────────────────────────

REGIONS = [
    (31.0, 33.0, -104.0, -101.0),  # Permian Basin, USA
    (37.0, 40.0,   55.0,   60.0),  # Turkmenistan Gas Fields
    ( 4.0,  6.0,    5.0,    8.0),  # Niger Delta, Nigeria
    (60.0, 65.0,   70.0,   80.0),  # Siberia Gas Fields
    (18.0, 22.0,   68.0,   73.0),  # Mumbai Offshore, India
    (26.0, 28.0,   49.0,   51.0),  # Saudi Aramco East
]


# ── Shape normaliser ──────────────────────────────────────────────

def normalize_tensor(t: torch.Tensor, size: int = 224) -> torch.Tensor:
    """
    Guarantee output shape is (4, size, size) regardless of what
    preprocess_tropomi returns — handles (4,H,W) and (1,4,H,W).
    """
    # Drop spurious batch dim if present
    if t.dim() == 4:
        t = t.squeeze(0)                        # (1,4,H,W) → (4,H,W)

    # Resize spatial dims if needed
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(
            t.unsqueeze(0),                     # (4,H,W) → (1,4,H,W)
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                            # → (4,size,size)

    return t                                    # guaranteed (4,224,224)


# ── Synthetic fallback ────────────────────────────────────────────

def make_synthetic_batch(n: int = 200, size: int = 224):
    """Generate (input_tensor, binary_mask) pairs with Gaussian plume blobs."""
    rng   = np.random.default_rng(42)
    imgs  = rng.normal(0.5, 0.15, (n, 4, size, size)).astype(np.float32).clip(0, 1)
    masks = np.zeros((n, size, size), dtype=np.float32)

    for i in range(n):
        for _ in range(rng.integers(0, 3)):
            cy, cx = rng.integers(40, size - 40, size=2)
            sigma  = rng.integers(10, 30)
            yy, xx = np.ogrid[:size, :size]
            blob   = np.exp(-(((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2))
            masks[i] = np.clip(masks[i] + (blob > 0.5).astype(np.float32), 0, 1)

    return torch.from_numpy(imgs), torch.from_numpy(masks)


# ── Real + synthetic data loader ──────────────────────────────────

def make_training_data(n_synthetic: int = 200, size: int = 224):
    """
    Fetches real TROPOMI scenes from GEE for each pipeline region.
    GEETROPOMIIngester automatically falls back to its own mock if
    GEE is unavailable, so this never hard-fails.

    Weak supervision: XCH4 channel thresholded at mean + 1 std as plume mask.
    Replace with labeled masks when available for production quality.
    """
    real_imgs:  list[torch.Tensor] = []
    real_masks: list[torch.Tensor] = []

    ingester = GEETROPOMIIngester()

    for (lat_min, lat_max, lon_min, lon_max) in REGIONS:
        try:
            ds     = ingester.fetch(lat_min, lat_max, lon_min, lon_max, days_back=5)
            tensor = preprocess_tropomi(ds)

            if tensor is None:
                logger.warning(
                    f"preprocess_tropomi returned None for "
                    f"({lat_min},{lat_max},{lon_min},{lon_max}) — skipping"
                )
                continue

            # Normalise to (4, size, size) — handles both (4,H,W) and (1,4,H,W)
            tensor = normalize_tensor(tensor, size)

            # Weak label: pixels > mean + 1 std on the XCH4 channel (index 0)
            ch4  = tensor[0]
            mask = (ch4 > ch4.mean() + ch4.std()).float()   # (size, size)

            real_imgs.append(tensor)
            real_masks.append(mask)
            logger.info(
                f"Region ({lat_min},{lat_max},{lon_min},{lon_max}): "
                f"loaded | shape={tuple(tensor.shape)} | plume_px={mask.sum().int()}"
            )

        except Exception as exc:
            logger.warning(
                f"Region ({lat_min},{lat_max},{lon_min},{lon_max}) "
                f"failed: {exc} — skipping"
            )

    n_real   = len(real_imgs)
    n_needed = max(0, n_synthetic - n_real)

    if n_needed > 0:
        logger.info(f"Padding with {n_needed} synthetic samples (got {n_real} real scenes)")
        sx, sy     = make_synthetic_batch(n_needed, size)   # already (N,4,H,W)/(N,H,W)
        real_imgs  += list(sx)
        real_masks += list(sy)
    else:
        logger.info(f"Loaded {n_real} real scenes — no synthetic padding needed")

    # Final safety normalisation before stacking — catches any leftover dim issues
    real_imgs  = [normalize_tensor(t, size) for t in real_imgs]
    real_masks = [
        m.squeeze(0) if (m.dim() == 3 and m.shape[0] == 1) else m
        for m in real_masks
    ]

    X = torch.stack(real_imgs)   # (N, 4, 224, 224)
    Y = torch.stack(real_masks)  # (N, 224, 224)
    logger.info(f"Training set: {len(X)} total ({n_real} real GEE, {n_needed} synthetic) | X={tuple(X.shape)} Y={tuple(Y.shape)}")
    return X, Y


# ── Training loop ─────────────────────────────────────────────────

def train(epochs: int = 10, lr: float = 1e-4, batch: int = 8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device} for {epochs} epochs")

    X, Y   = make_training_data(n_synthetic=200)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)

    model     = PlumeSegmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # pos_weight=3 compensates for class imbalance (plume pixels << background)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit, _ = model(xb)
            loss = bce(logit.squeeze(1), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch:02d}/{epochs} | loss={avg_loss:.4f}")

    save_model(model)
    logger.info("Training complete — checkpoint saved.")


# ── Entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARGUS Stage 1 plume segmenter")
    parser.add_argument("--epochs", type=int,   default=10,   help="Number of training epochs")
    parser.add_argument("--lr",     type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch",  type=int,   default=8,    help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch=args.batch)