from __future__ import annotations
from pathlib import Path

import numpy as np
import xarray as xr
import httpx
from loguru import logger

from src.utils.config import cfg

RAW_DIR = Path(cfg["data"]["emit_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)

EMIT_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


class EMITIngester:
    """
    NASA EMIT (Earth Surface Mineral Dust Source Investigation) CH₄ data.
    Higher spatial resolution (~60 m) vs TROPOMI (~7 km).
    Used for dual-sensor cross-validation.
    """

    def __init__(self):
        self._token = cfg["env"].get("earthdata_token", "")
        self._live  = bool(self._token)

        if not self._live:
            logger.warning("EMIT: no EarthData token — using synthetic mock")
        else:
            logger.info("EMIT: EarthData token found — live mode active")

    # ------------------------------------------------------------------
    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
    ) -> xr.Dataset:
        if self._live:
            try:
                return self._fetch_live(lat_min, lat_max, lon_min, lon_max)
            except Exception as e:
                logger.warning(f"EMIT live fetch failed ({e}) — falling back to mock")
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    # ------------------------------------------------------------------
    def _fetch_live(self, lat_min, lat_max, lon_min, lon_max):
        params = {
            "short_name":   "EMITL2BCH4ENH",
            "bounding_box": f"{lon_min},{lat_min},{lon_max},{lat_max}",
            "page_size":    1,
            "sort_key":     "-start_date",
        }
        headers = {"Authorization": f"Bearer {self._token}"}

        r = httpx.get(
            EMIT_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()

        entries = r.json().get("feed", {}).get("entry", [])
        if not entries:
            logger.warning("EMIT: no granules found — using mock")
            return self._mock(lat_min, lat_max, lon_min, lon_max)

        url  = entries[0]["links"][0]["href"]
        name = url.split("/")[-1]
        out  = RAW_DIR / name

        if not out.exists():
            logger.info(f"EMIT: downloading {name}")
            with httpx.stream("GET", url, headers=headers, timeout=120) as resp:
                resp.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)

        ds = xr.open_dataset(out)
        logger.info(f"EMIT: loaded {name}")
        return ds

    # ------------------------------------------------------------------
    @staticmethod
    def _mock(
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        grid: int = 128,
    ) -> xr.Dataset:
        """
        High-resolution synthetic CH₄ enhancement map.
        Grid is 128×128 (vs TROPOMI's 64×64) to simulate EMIT's finer resolution.
        """
        lats = np.linspace(lat_min, lat_max, grid)
        lons = np.linspace(lon_min, lon_max, grid)
        lon2d, lat2d = np.meshgrid(lons, lats)

        c_lat = (lat_min + lat_max) / 2
        c_lon = (lon_min + lon_max) / 2

        # Sharper, more localised plume (higher resolution than TROPOMI mock)
        enh = (
            2000 * np.exp(
                -(
                    ((lat2d - c_lat) / 0.02) ** 2
                    + ((lon2d - c_lon) / 0.02) ** 2
                )
            )
            + 500 * np.exp(
                -(
                    ((lat2d - c_lat - 0.08) / 0.015) ** 2
                    + ((lon2d - c_lon + 0.06) / 0.015) ** 2
                )
            )
        ).astype(np.float32)

        ds = xr.Dataset(
            {"ch4_enhancement": (["lat", "lon"], enh)},
            coords={"lat": lats, "lon": lons},
        )
        logger.info("EMIT: generated synthetic high-res mock (128×128)")
        return ds