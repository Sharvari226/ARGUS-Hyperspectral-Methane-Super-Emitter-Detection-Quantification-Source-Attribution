"""
Train the TGAN attribution model on synthetically generated plume-facility pairs.
Run: python scripts/train_stage3.py --epochs 20
"""
import argparse
import torch
import torch.nn.functional as F
from loguru import logger

from src.models.stage3_tgan import (
    TemporalGraphAttributor,
    build_attribution_graph,
    save_tgan,
)
from src.data.tropomi import TROPOMIIngester
from src.models.stage1_sat import preprocess_tropomi, mc_predict, load_model
from src.models.stage1_sat import extract_plume_detections
import numpy as np


def make_synthetic_detections(n: int = 6) -> list[dict]:
    rng = np.random.default_rng(7)
    return [
        {
            "label_id":           i,
            "centroid_lat":       float(rng.uniform(20, 60)),
            "centroid_lon":       float(rng.uniform(-10, 80)),
            "pixel_area":         int(rng.integers(10, 200)),
            "mean_probability":   float(rng.uniform(0.6, 0.99)),
            "epistemic_variance": float(rng.uniform(0.01, 0.12)),
            "flux_kg_hr":         float(rng.uniform(80, 800)),
            "transport_age_hr":   float(rng.uniform(0.5, 4.0)),
            "pixel_ys":           [],
            "pixel_xs":           [],
        }
        for i in range(n)
    ]


def train(epochs: int = 20, lr: float = 5e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TemporalGraphAttributor().to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # Generate a new random batch each epoch
        for _ in range(10):   # 10 graphs per epoch
            dets     = make_synthetic_detections(n=np.random.randint(2, 8))
            u_ms     = float(np.random.uniform(-8, 8))
            v_ms     = float(np.random.uniform(-5, 5))

            data, plume_ids, facility_ids = build_attribution_graph(
                dets, u_ms, v_ms
            )
            data = data.to(device)

            scores  = model(data)               # (P, F)
            P, Fac  = scores.shape

            # Synthetic ground-truth: nearest facility is always correct
            labels  = torch.zeros(P, dtype=torch.long, device=device)

            loss = F.cross_entropy(scores, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"avg_loss={total_loss/10:.4f}"
        )

    save_tgan(model)
    logger.info("TGAN training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr",     type=float, default=5e-4)
    args = parser.parse_args()
    train(args.epochs, args.lr)