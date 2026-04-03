from __future__ import annotations
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import cdsapi
from loguru import logger

from src.utils.config import cfg

RAW_DIR = Path("data/raw/ecmwf")
RAW_DIR.mkdir(parents=True, exist_ok=True)


class ECMWFIngester:
    """ERA5 wind vector retrieval via CDS API. Mocks if key absent."""

    def __init__(self):
        self._live = bool(cfg["env"]["ecmwf_api_key"])
        if self._live:
            self.client = cdsapi.Client()
            logger.info("ECMWF: CDS client initialised")
        else:
            logger.warning("ECMWF: no API key — using synthetic wind mock")

    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        date: datetime | None = None,
    ) -> xr.Dataset:
        if self._live:
            return self._fetch_live(lat_min, lat_max, lon_min, lon_max, date)
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    def _fetch_live(self, lat_min, lat_max, lon_min, lon_max, date):
        date = date or datetime.utcnow()
        out  = RAW_DIR / f"era5_uv_{date.strftime('%Y%m%d')}.nc"
        if not out.exists():
            self.client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable":     ["10m_u_component_of_wind",
                                     "10m_v_component_of_wind"],
                    "year":  str(date.year),
                    "month": f"{date.month:02d}",
                    "day":   f"{date.day:02d}",
                    "time":  "12:00",
                    "area":  [lat_max, lon_min, lat_min, lon_max],  # N,W,S,E
                    "format": "netcdf",
                },
                str(out),
            )
        ds = xr.open_dataset(out)
        logger.info(f"ECMWF: loaded ERA5 winds from {out.name}")
        return ds

    @staticmethod
    def _mock(lat_min, lat_max, lon_min, lon_max, grid: int = 32) -> xr.Dataset:
        lats = np.linspace(lat_min, lat_max, grid)
        lons = np.linspace(lon_min, lon_max, grid)
        # Steady 5 m/s westerly + slight northerly
        u10 = np.full((grid, grid),  5.0, dtype=np.float32)
        v10 = np.full((grid, grid), -1.5, dtype=np.float32)
        ds = xr.Dataset(
            {"u10": (["lat","lon"], u10),
             "v10": (["lat","lon"], v10)},
            coords={"lat": lats, "lon": lons},
        )
        logger.info("ECMWF: generated synthetic wind mock")
        return ds