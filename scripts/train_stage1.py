"""
Quick-train script for Stage 1 plume segmenter on synthetic data.
Run: python scripts/train_stage1.py --epochs 10
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger

from src.models.stage1_sat import PlumeSegmenter, save_model
from src.data.tropomi import TROPOMIIngester
from src.models.stage1_sat import preprocess_tropomi


def make_synthetic_batch(n: int = 200, size: int = 224):
    """Generate (input_tensor, binary_mask) pairs for supervised training."""
    rng   = np.random.default_rng(0)
    imgs  = rng.normal(0.5, 0.15, (n, 4, size, size)).astype(np.float32).clip(0, 1)
    masks = np.zeros((n, size, size), dtype=np.float32)

    for i in range(n):
        # Inject 0–2 random Gaussian plumes per sample
        for _ in range(rng.integers(0, 3)):
            cy, cx = rng.integers(40, size - 40, size=2)
            σ      = rng.integers(10, 30)
            yy, xx = np.ogrid[:size, :size]
            blob   = np.exp(-(((yy - cy) / σ) ** 2 + ((xx - cx) / σ) ** 2))
            masks[i] = np.clip(masks[i] + (blob > 0.5).astype(np.float32), 0, 1)

    return torch.from_numpy(imgs), torch.from_numpy(masks)


def train(epochs: int = 10, lr: float = 1e-4, batch: int = 8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device} for {epochs} epochs")

    X, Y = make_synthetic_batch(200)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)

    model     = PlumeSegmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    bce       = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

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
        logger.info(f"Epoch {epoch:02d}/{epochs} | loss={total_loss/len(loader):.4f}")

    save_model(model)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr",     type=float, default=1e-4)
    parser.add_argument("--batch",  type=int, default=8)
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch)