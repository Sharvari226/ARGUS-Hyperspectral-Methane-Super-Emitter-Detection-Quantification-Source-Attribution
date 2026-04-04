from __future__ import annotations
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger

from src.utils.config import cfg

RAW_DIR = Path("data/raw/ecmwf")
RAW_DIR.mkdir(parents=True, exist_ok=True)


class ECMWFIngester:
    """
    ERA5 wind vector retrieval via CDS API.
    Falls back to synthetic mock if API key absent.
    """

    def __init__(self):
        self._live = bool(cfg["env"].get("ecmwf_api_key", ""))

        if self._live:
            try:
                import cdsapi
                self.client = cdsapi.Client()
                logger.info("ECMWF: CDS client initialised")
            except Exception as e:
                logger.warning(f"ECMWF: CDS init failed ({e}) — using mock")
                self._live = False
        else:
            logger.warning("ECMWF: no API key — using synthetic wind mock")

    # ------------------------------------------------------------------
    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        date: datetime | None = None,
    ) -> xr.Dataset:
        if self._live:
            try:
                return self._fetch_live(lat_min, lat_max, lon_min, lon_max, date)
            except Exception as e:
                logger.warning(f"ECMWF live fetch failed ({e}) — falling back to mock")
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    # ------------------------------------------------------------------
    def _fetch_live(self, lat_min, lat_max, lon_min, lon_max, date):
        date = date or datetime.utcnow()
        out  = RAW_DIR / f"era5_uv_{date.strftime('%Y%m%d')}.nc"

        if not out.exists():
            self.client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
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

    # ------------------------------------------------------------------
    @staticmethod
    def _mock(
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        grid: int = 32,
    ) -> xr.Dataset:
        """Steady 5 m/s westerly + slight northerly shear."""
        lats = np.linspace(lat_min, lat_max, grid)
        lons = np.linspace(lon_min, lon_max, grid)

        # Add slight spatial variation to make it more realistic
        lon2d, lat2d = np.meshgrid(lons, lats)
        u10 = (5.0 + 0.5 * np.sin(lat2d * 0.3)).astype(np.float32)
        v10 = (-1.5 + 0.3 * np.cos(lon2d * 0.3)).astype(np.float32)

        ds = xr.Dataset(
            {
                "u10": (["lat", "lon"], u10),
                "v10": (["lat", "lon"], v10),
            },
            coords={"lat": lats, "lon": lons},
        )
        logger.info("ECMWF: generated synthetic wind mock")
        return ds