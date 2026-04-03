from __future__ import annotations
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from sentinelsat import SentinelAPI
from loguru import logger

from src.utils.config import cfg

RAW_DIR = Path(cfg["data"]["tropomi_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)


class TROPOMIIngester:
    """
    Downloads and preprocesses Sentinel-5P TROPOMI Level-2 CH₄ products.
    Falls back to synthetic mock data if credentials are absent (hackathon mode).
    """

    def __init__(self):
        user = cfg["env"]["copernicus_user"]
        pwd  = cfg["env"]["copernicus_password"]
        self._live = bool(user and pwd)
        if self._live:
            self.api = SentinelAPI(
                user, pwd,
                "https://s5phub.copernicus.eu/dhus",
                show_progressbars=True,
            )
            logger.info("TROPOMI: connected to Copernicus hub")
        else:
            logger.warning("TROPOMI: no credentials — using synthetic mock data")

    # ------------------------------------------------------------------
    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        date: datetime | None = None,
        days_back: int = 3,
    ) -> xr.Dataset:
        if self._live:
            return self._fetch_live(lat_min, lat_max, lon_min, lon_max, date, days_back)
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    # ------------------------------------------------------------------
    def _fetch_live(self, lat_min, lat_max, lon_min, lon_max, date, days_back):
        date = date or datetime.utcnow()
        start = (date - timedelta(days=days_back)).strftime("%Y%m%d")
        end   = date.strftime("%Y%m%d")

        footprint = (
            f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
        )
        products = self.api.query(
            area=footprint,
            date=(start, end),
            producttype="L2__CH4___",
            platformname="Sentinel-5 Precursor",
        )
        if not products:
            logger.warning("No TROPOMI products found — falling back to mock")
            return self._mock(lat_min, lat_max, lon_min, lon_max)

        # Download most recent product
        pid   = list(products.keys())[-1]
        pinfo = products[pid]
        fpath = RAW_DIR / f"{pinfo['title']}.nc"
        if not fpath.exists():
            self.api.download(pid, directory_path=RAW_DIR)

        return self._parse_nc(fpath, lat_min, lat_max, lon_min, lon_max)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_nc(path: Path,
                  lat_min, lat_max, lon_min, lon_max) -> xr.Dataset:
        ds = xr.open_dataset(path, group="PRODUCT")
        # Spatial crop
        mask = (
            (ds.latitude  >= lat_min) & (ds.latitude  <= lat_max) &
            (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
        )
        ds = ds.where(mask, drop=True)

        # Keep only the channels useful for segmentation
        keep = ["methane_mixing_ratio_bias_corrected",
                "methane_mixing_ratio_precision",
                "qa_value",
                "cloud_fraction"]
        ds = ds[keep]
        logger.info(f"TROPOMI: loaded {ds.dims} pixels from {path.name}")
        return ds

    # ------------------------------------------------------------------
    @staticmethod
    def _mock(lat_min, lat_max, lon_min, lon_max,
              grid: int = 64) -> xr.Dataset:
        """
        Synthetic CH₄ field with two injected super-emitter plumes.
        Used for offline development and CI.
        """
        lats = np.linspace(lat_min, lat_max, grid)
        lons = np.linspace(lon_min, lon_max, grid)
        lon2d, lat2d = np.meshgrid(lons, lats)

        base_ppb = 1870.0
        noise    = np.random.normal(0, 8, (grid, grid))

        # Inject two Gaussian plumes
        def plume(clat, clon, strength=150, sigma=0.08):
            return strength * np.exp(
                -(((lat2d - clat) / sigma) ** 2 + ((lon2d - clon) / sigma) ** 2)
            )

        c_lat = (lat_min + lat_max) / 2
        c_lon = (lon_min + lon_max) / 2
        ch4 = base_ppb + noise + plume(c_lat, c_lon) + plume(c_lat + 0.3, c_lon - 0.2, 80)

        ds = xr.Dataset(
            {
                "methane_mixing_ratio_bias_corrected": (["lat", "lon"], ch4.astype(np.float32)),
                "methane_mixing_ratio_precision":      (["lat", "lon"], np.full((grid, grid), 12.0, dtype=np.float32)),
                "qa_value":                            (["lat", "lon"], np.ones((grid, grid),  dtype=np.float32)),
                "cloud_fraction":                      (["lat", "lon"], np.random.uniform(0, 0.2, (grid, grid)).astype(np.float32)),
            },
            coords={"lat": lats, "lon": lons},
        )
        logger.info("TROPOMI: generated synthetic mock dataset")
        return ds