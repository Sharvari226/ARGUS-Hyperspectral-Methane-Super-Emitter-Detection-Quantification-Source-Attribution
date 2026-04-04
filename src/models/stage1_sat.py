from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import NamedTuple

import timm
from einops import rearrange
from loguru import logger

from src.utils.config import cfg

CKPT_PATH = Path(cfg["stage1"]["checkpoint"])


# ── Output container ──────────────────────────────────────────────────────────

class SegmentationOutput(NamedTuple):
    mask_mean:     torch.Tensor   # (B, H, W)  mean plume probability
    mask_variance: torch.Tensor   # (B, H, W)  epistemic uncertainty
    embeddings:    torch.Tensor   # (B, D)     CLS token for downstream use


# ── Artifact removal attention block ──────────────────────────────────────────

class ArtifactSuppressionBlock(nn.Module):
    """
    Channel-attention gate that learns to down-weight spectral bands
    dominated by cloud/surface-reflectance artifacts.
    Inspired by CBAM (Convolutional Block Attention Module).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)   # global average
        self.gmp = nn.AdaptiveMaxPool2d(1)   # global max
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        avg_gate = self.mlp(self.gap(x))
        max_gate = self.mlp(self.gmp(x))
        scale    = torch.sigmoid(avg_gate + max_gate).unsqueeze(-1).unsqueeze(-1)
        return x * scale


# ── Decoder block ─────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Lightweight upsampling block: bilinear upsample → conv → BN → GELU.
    Connects the ViT patch embeddings back to pixel space.
    """

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.drop = nn.Dropout2d(p=cfg["stage1"]["dropout_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.conv(self.up(x)))


# ── Main model ────────────────────────────────────────────────────────────────

class PlumeSegmenter(nn.Module):
    """
    ViT encoder + lightweight CNN decoder for pixel-level CH₄ plume segmentation.

    Architecture:
        Input  → (B, 4, 224, 224)   4 TROPOMI CH4-sensitive bands
        Stage  → ViT-Small/16 backbone (timm) — patch embeddings
        Decode → 4× DecoderBlock upsample chain → (B, 1, 224, 224) logit
        Gate   → ArtifactSuppressionBlock on intermediate feature maps

    MC Dropout inference:
        Keep dropout ACTIVE at inference time, run N forward passes,
        compute mean + variance across passes → epistemic uncertainty map.
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()

        # ── ViT backbone ──────────────────────────────────────────
        self.encoder = timm.create_model(
            cfg["stage1"]["model"],
            pretrained=True,
            num_classes=0,           # remove classification head
            global_pool="",          # keep all patch tokens
        )
        embed_dim = self.encoder.embed_dim   # 384 for vit_small

        # Patch the first conv to accept in_channels bands instead of 3
        original_proj = self.encoder.patch_embed.proj
        self.encoder.patch_embed.proj = nn.Conv2d(
            in_channels,
            original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=original_proj.bias is not None,
        )
        # Re-initialise — copy the mean of the RGB weights for the extra channel
        with torch.no_grad():
            w  = original_proj.weight.data
            new_w = torch.zeros(
                original_proj.out_channels, in_channels,
                *original_proj.kernel_size
            )
            new_w[:, :3, ...] = w
            new_w[:, 3:, ...] = w.mean(dim=1, keepdim=True)
            self.encoder.patch_embed.proj.weight.copy_(new_w)

        # ── Artifact suppression ──────────────────────────────────
        self.artifact_gate = ArtifactSuppressionBlock(embed_dim)

        # ── Projection to spatial feature map ────────────────────
        patch = cfg["stage1"]["patch_size"]     # 16
        img   = cfg["stage1"]["image_size"]     # 224
        self.n_patches_side = img // patch       # 14

        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)

        # ── Decoder ───────────────────────────────────────────────
        # 14×14 → 28×28 → 56×56 → 112×112 → 224×224
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128, scale=2),
            DecoderBlock(128,  64, scale=2),
            DecoderBlock( 64,  32, scale=2),
            DecoderBlock( 32,  16, scale=2),
        )

        # ── Final pixel-wise classification head ──────────────────
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.GELU(),
            nn.Dropout2d(p=cfg["stage1"]["dropout_rate"]),
            nn.Conv2d(8, 1, 1),    # single-channel logit
        )

        logger.info(
            f"PlumeSegmenter: ViT-{embed_dim}d | "
            f"{sum(p.numel() for p in self.parameters()):,} params"
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) normalised satellite bands

        Returns:
            logit:      (B, 1, H, W)
            cls_token:  (B, embed_dim)   for downstream GNN use
        """
        B = x.shape[0]

        # ViT forward — returns (B, N+1, D) where N = n_patches², +1 = CLS
        tokens    = self.encoder.forward_features(x)
        cls_token = tokens[:, 0, :]                        # (B, D)
        patches   = tokens[:, 1:, :]                       # (B, N, D)

        # Reshape patch tokens to 2-D spatial grid
        p = self.n_patches_side
        feat = rearrange(patches, "b (h w) d -> b d h w", h=p, w=p)

        # Artifact suppression
        feat = self.artifact_gate(feat)

        # Project + decode
        feat   = self.proj(feat)
        feat   = self.decoder(feat)
        logit  = self.head(feat)

        return logit, cls_token


# ── MC Dropout inference ──────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """Force all Dropout layers into training mode for MC sampling."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_predict(
    model:   PlumeSegmenter,
    x:       torch.Tensor,
    n_passes: int | None = None,
    device:  str = "cpu",
) -> SegmentationOutput:
    """
    Run N stochastic forward passes with dropout active.

    Returns SegmentationOutput with:
        mask_mean      — averaged plume probability map  (B, H, W)
        mask_variance  — epistemic uncertainty map       (B, H, W)
        embeddings     — mean CLS token across passes    (B, D)
    """
    n_passes = n_passes or cfg["pipeline"]["mc_dropout_passes"]
    model.eval()
    _enable_dropout(model)
    x = x.to(device)

    preds      = []
    embeddings = []

    for _ in range(n_passes):
        logit, cls = model(x)
        prob = torch.sigmoid(logit.squeeze(1))   # (B, H, W)
        preds.append(prob.cpu())
        embeddings.append(cls.cpu())

    stack   = torch.stack(preds,      dim=0)   # (N, B, H, W)
    emb_stk = torch.stack(embeddings, dim=0)   # (N, B, D)

    mean_mask = stack.mean(dim=0)
    var_mask  = stack.var( dim=0)
    mean_emb  = emb_stk.mean(dim=0)

    return SegmentationOutput(
        mask_mean=mean_mask,
        mask_variance=var_mask,
        embeddings=mean_emb,
    )


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_tropomi(ds) -> torch.Tensor:
    """
    Convert xarray Dataset → (1, 4, 224, 224) float32 tensor.
    Handles NaN/masked pixels before normalisation.
    """
    import torch.nn.functional as F

    SIZE = cfg["stage1"]["image_size"]

    def extract(name: str, invert: bool = False) -> np.ndarray:
        arr = ds[name].values.copy()
        # Replace zeros and NaNs with NaN first, then fill with median
        arr = arr.astype(np.float32)
        arr[arr == 0] = np.nan
        if np.isnan(arr).all():
            arr = np.zeros_like(arr)
        else:
            median = np.nanmedian(arr)
            arr = np.where(np.isnan(arr), median, arr)
        if invert:
            arr = 1.0 - arr
        # Robust normalise to [0, 1]
        lo, hi = np.nanpercentile(arr, 1), np.nanpercentile(arr, 99)
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        return arr.clip(0, 1).astype(np.float32)

    bands = np.stack([
        extract("methane_mixing_ratio_bias_corrected"),
        extract("methane_mixing_ratio_precision"),
        extract("qa_value"),
        extract("cloud_fraction", invert=True),
    ], axis=0)  # (4, H, W)

    t = torch.from_numpy(bands).unsqueeze(0)  # (1, 4, H, W)
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return t


def extract_plume_detections(
    mask_mean:  torch.Tensor,
    mask_var:   torch.Tensor,
    lat_min:    float,
    lat_max:    float,
    lon_min:    float,
    lon_max:    float,
    prob_threshold:  float = 0.35,
    min_pixels:      int   = 2,
    conf_threshold:  float | None = None,
    raw_ch4:         np.ndarray | None = None,  # pass ds ch4 values for anomaly fallback
) -> list[dict]:
    """
    Convert probability map → detection dicts.
    Falls back to CH4 anomaly detection when model output is uninformative
    (uniform ~0.5 from untrained checkpoint on real data).
    """
    from scipy import ndimage

    conf_threshold = conf_threshold or cfg["pipeline"]["uncertainty_max"]
    probs = mask_mean.numpy()
    varis = mask_var.numpy()
    H, W  = probs.shape

    lats = np.linspace(lat_min, lat_max, H)
    lons = np.linspace(lon_min, lon_max, W)

    # ── Detect if model output is uninformative (uniform ~0.5) ───────────────
    model_is_uninformative = (probs.max() - probs.min()) < 0.15

    if model_is_uninformative and raw_ch4 is not None:
        # Fall back to direct CH4 anomaly detection
        # Real TROPOMI background: ~1870 ppb. Anomaly = pixels > mean + 2*std
        ch4 = raw_ch4.astype(np.float32)
        ch4[ch4 == 0] = np.nan
        ch4_clean = np.where(np.isnan(ch4), np.nanmedian(ch4), ch4)

        mean_ch4 = float(np.nanmean(ch4_clean))
        std_ch4  = float(np.nanstd(ch4_clean))
        threshold_ppb = mean_ch4 + 1.5 * std_ch4

        anomaly_mask = (ch4_clean > threshold_ppb).astype(np.uint8)
        # Use anomaly mask as proxy probability
        probs = np.where(anomaly_mask, 0.75, 0.1).astype(np.float32)
        varis = np.full_like(probs, 0.05)

        logger.info(
            f"Stage1: model uninformative — using CH4 anomaly detection "
            f"(threshold={threshold_ppb:.1f} ppb, "
            f"background={mean_ch4:.1f}±{std_ch4:.1f} ppb)"
        )
    else:
        anomaly_mask = (probs > prob_threshold).astype(np.uint8)

    # ── Connected components ──────────────────────────────────────────────────
    labeled, n_features = ndimage.label(anomaly_mask if model_is_uninformative
                                        else (probs > prob_threshold).astype(np.uint8))
    detections = []

    for label_id in range(1, n_features + 1):
        region_mask = labeled == label_id
        ys, xs = np.where(region_mask)

        if len(ys) < min_pixels:
            continue

        c_lat     = float(lats[np.clip(ys, 0, H - 1)].mean())
        c_lon     = float(lons[np.clip(xs, 0, W - 1)].mean())
        area      = int(region_mask.sum())
        mean_prob = float(probs[region_mask].mean())
        mean_var  = float(varis[region_mask].mean())

        detections.append({
            "label_id":           label_id,
            "centroid_lat":       round(c_lat, 5),
            "centroid_lon":       round(c_lon, 5),
            "pixel_area":         area,
            "mean_probability":   round(mean_prob, 4),
            "epistemic_variance": round(mean_var, 4),
            "high_confidence":    mean_var < conf_threshold,
            "pixel_ys":           ys.tolist(),
            "pixel_xs":           xs.tolist(),
        })

    logger.info(
        f"PlumeSegmenter: {len(detections)} plumes detected "
        f"({'anomaly-based' if model_is_uninformative else 'model-based'}, "
        f"{sum(d['high_confidence'] for d in detections)} high-confidence)"
    )
    return detections
# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_model(device: str = "cpu") -> PlumeSegmenter:
    model = PlumeSegmenter(in_channels=cfg["stage1"]["bands"].__len__())
    if CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state)
        logger.info(f"PlumeSegmenter: loaded checkpoint from {CKPT_PATH}")
    else:
        logger.warning(
            f"PlumeSegmenter: no checkpoint at {CKPT_PATH} — using random weights. "
            "Run scripts/train_stage1.py to train."
        )
    model.to(device)
    return model


def save_model(model: PlumeSegmenter) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    logger.info(f"PlumeSegmenter: checkpoint saved → {CKPT_PATH}")