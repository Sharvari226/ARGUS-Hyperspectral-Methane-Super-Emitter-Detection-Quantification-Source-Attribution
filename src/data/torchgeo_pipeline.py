"""
src/data/torchgeo_pipeline.py
──────────────────────────────
TorchGeo-based geospatial data pipeline for ARGUS.

TorchGeo provides:
  - CRS-aware tensor transforms
  - Geospatial dataset abstractions
  - Patch sampling with spatial indexing
  - Standard augmentations for satellite imagery
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any
import xarray as xr
from loguru import logger

try:
    from torchgeo.transforms import AugmentationSequential
    from torchgeo.transforms import indices as tg_indices
    import kornia.augmentation as K
    TORCHGEO_AVAILABLE = True
except ImportError:
    TORCHGEO_AVAILABLE = False
    logger.warning("TorchGeo: not installed — using basic transforms")


# ── Standard image size ───────────────────────────────────────────
TARGET_SIZE = 224


# ═════════════════════════════════════════════════════════════════
# Geospatial transforms
# ═════════════════════════════════════════════════════════════════

def get_train_transforms():
    """
    TorchGeo augmentation pipeline for training the Stage 1 ViT.
    Geospatially-aware: flips/rotations preserve lat/lon semantics.
    """
    if not TORCHGEO_AVAILABLE:
        return None

    return AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=15, p=0.3),
        K.RandomGaussianNoise(mean=0.0, std=0.02, p=0.4),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0), p=0.2),
        data_keys=["image", "mask"],
    )


def get_val_transforms():
    """Minimal transforms for validation — no augmentation."""
    if not TORCHGEO_AVAILABLE:
        return None
    return AugmentationSequential(
        data_keys=["image"],
    )


# ═════════════════════════════════════════════════════════════════
# xarray → TorchGeo-compatible tensor
# ═════════════════════════════════════════════════════════════════

class TROPOMITensorPipeline:
    """
    Converts a raw xarray Dataset (from GEETROPOMIIngester)
    into a model-ready tensor using TorchGeo conventions.

    Output tensor shape: (1, 4, 224, 224)
    Channel order:
        0 — CH4 mixing ratio (bias corrected)
        1 — CH4 precision (uncertainty)
        2 — QA value
        3 — Cloud fraction (inverted)

    TorchGeo convention: float32, values in [0, 1]
    """

    def __init__(self, size: int = TARGET_SIZE, augment: bool = False):
        self.size    = size
        self.augment = augment
        self.transforms = get_train_transforms() if augment else get_val_transforms()

    def __call__(self, ds: xr.Dataset) -> torch.Tensor:
        return self.transform(ds)

    def transform(self, ds: xr.Dataset) -> torch.Tensor:
        """Main entry point — returns (1, 4, H, W) tensor."""
        bands = self._extract_bands(ds)
        t     = torch.from_numpy(bands).unsqueeze(0)   # (1, 4, H, W)
        t     = F.interpolate(
            t, size=(self.size, self.size),
            mode="bilinear", align_corners=False
        )
        if self.augment and self.transforms is not None:
            t = self._apply_torchgeo_augmentation(t)
        return t

    def _extract_bands(self, ds: xr.Dataset) -> np.ndarray:
        """Extract and normalise 4 bands from the xarray dataset."""
        def norm(arr: np.ndarray, invert: bool = False) -> np.ndarray:
            arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)
            if invert:
                arr = 1.0 - arr
            lo  = np.percentile(arr, 1)
            hi  = np.percentile(arr, 99)
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            return arr.clip(0.0, 1.0)

        ch4   = norm(ds["methane_mixing_ratio_bias_corrected"].values)
        prec  = norm(ds["methane_mixing_ratio_precision"].values)
        qa    = norm(ds["qa_value"].values)
        cloud = norm(ds["cloud_fraction"].values, invert=True)

        return np.stack([ch4, prec, qa, cloud], axis=0)   # (4, H, W)

    def _apply_torchgeo_augmentation(self, t: torch.Tensor) -> torch.Tensor:
        """Apply TorchGeo augmentations. Falls back silently if unavailable."""
        try:
            if self.transforms is not None:
                sample = {"image": t}
                sample = self.transforms(sample)
                return sample["image"]
        except Exception as e:
            logger.debug(f"TorchGeo augmentation skipped: {e}")
        return t


# ═════════════════════════════════════════════════════════════════
# Patch sampler for training data generation
# ═════════════════════════════════════════════════════════════════

class SyntheticPlumeDataset(torch.utils.data.Dataset):
    """
    Synthetic geospatial dataset for Stage 1 training.

    Generates physically plausible CH4 fields with injected
    Gaussian plumes over randomised geographic regions.

    Compatible with TorchGeo DataLoader conventions:
        sample["image"] — (4, 224, 224) float32 tensor
        sample["mask"]  — (1, 224, 224) binary plume mask
        sample["bbox"]  — (lat_min, lat_max, lon_min, lon_max)
    """

    def __init__(
        self,
        n_samples:  int  = 500,
        size:       int  = TARGET_SIZE,
        augment:    bool = True,
        seed:       int  = 42,
    ):
        self.n_samples  = n_samples
        self.size       = size
        self.pipeline   = TROPOMITensorPipeline(size=size, augment=augment)
        self.rng        = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Random geographic bbox
        lat_min = float(self.rng.uniform(-60, 70))
        lat_max = lat_min + float(self.rng.uniform(1, 5))
        lon_min = float(self.rng.uniform(-160, 155))
        lon_max = lon_min + float(self.rng.uniform(1, 5))

        ds, mask = self._generate(lat_min, lat_max, lon_min, lon_max)

        image = self.pipeline.transform(ds)    # (1, 4, 224, 224)
        image = image.squeeze(0)               # (4, 224, 224)

        # Resize mask to match
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_t = F.interpolate(mask_t, size=(self.size, self.size), mode="nearest")
        mask_t = mask_t.squeeze(0)             # (1, 224, 224)

        return {
            "image": image,
            "mask":  mask_t,
            "bbox":  torch.tensor([lat_min, lat_max, lon_min, lon_max]),
        }

    def _generate(
        self,
        lat_min, lat_max, lon_min, lon_max,
        grid: int = 64,
    ) -> tuple[xr.Dataset, np.ndarray]:
        lats   = np.linspace(lat_min, lat_max, grid)
        lons   = np.linspace(lon_min, lon_max, grid)
        lon2d, lat2d = np.meshgrid(lons, lats)

        base  = 1870.0
        noise = self.rng.normal(0, 8, (grid, grid))
        ch4   = base + noise
        mask  = np.zeros((grid, grid), dtype=np.float32)

        # Inject 1–3 random plumes
        n_plumes = int(self.rng.integers(1, 4))
        for _ in range(n_plumes):
            clat   = float(self.rng.uniform(lat_min + 0.2, lat_max - 0.2))
            clon   = float(self.rng.uniform(lon_min + 0.2, lon_max - 0.2))
            strength = float(self.rng.uniform(80, 300))
            sigma    = float(self.rng.uniform(0.05, 0.15))

            plume  = strength * np.exp(
                -(((lat2d - clat) / sigma) ** 2
                  + ((lon2d - clon) / sigma) ** 2)
            )
            ch4  += plume
            mask  = np.clip(mask + (plume > strength * 0.3).astype(np.float32), 0, 1)

        cloud = self.rng.uniform(0, 0.25, (grid, grid)).astype(np.float32)

        ds = xr.Dataset(
            {
                "methane_mixing_ratio_bias_corrected": (["lat","lon"], ch4.astype(np.float32)),
                "methane_mixing_ratio_precision":      (["lat","lon"], np.full((grid,grid), 12.0, np.float32)),
                "qa_value":                            (["lat","lon"], np.ones((grid,grid), np.float32)),
                "cloud_fraction":                      (["lat","lon"], cloud),
            },
            coords={"lat": lats, "lon": lons},
        )
        return ds, mask


# ═════════════════════════════════════════════════════════════════
# Convenience wrapper — used by orchestrator
# ═════════════════════════════════════════════════════════════════

def preprocess_tropomi(ds: xr.Dataset) -> torch.Tensor:
    """
    Single-call wrapper for inference.
    Returns (1, 4, 224, 224) tensor — no augmentation.
    """
    pipeline = TROPOMITensorPipeline(size=TARGET_SIZE, augment=False)
    return pipeline.transform(ds)


def make_dataloader(
    n_samples:  int = 500,
    batch_size: int = 8,
    augment:    bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """
    Returns a DataLoader of synthetic plume samples for Stage 1 training.
    num_workers=0 for Windows compatibility.
    """
    dataset = SyntheticPlumeDataset(
        n_samples=n_samples,
        augment=augment,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )