"""
src/data/gee.py
───────────────
Google Earth Engine data layer for ARGUS.

Provides TROPOMI CH4, ERA5 winds, and NASA EMIT data through
the GEE Python API. Falls back to synthetic mock data when
GEE credentials are not yet available.

Auth setup (one-time):
    earthengine authenticate
    # OR for service account:
    ee.Initialize(credentials=..., project=GEE_PROJECT)
"""
from __future__ import annotations

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from src.utils.config import cfg

# ── GEE availability check ────────────────────────────────────────
try:
    import ee
    import geemap
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("GEE: earthengine-api not installed — using mock data")


def _init_gee() -> bool:
    """
    Initialise GEE. Returns True if successful, False otherwise.
    Supports both user auth and service account auth.
    """
    if not GEE_AVAILABLE:
        return False
    try:
        project = os.environ.get("GEE_PROJECT", "")
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("GEE: initialised successfully")
        return True
    except Exception as e:
        logger.warning(f"GEE: init failed ({e}) — falling back to mock data")
        return False


# Attempt init at module load
_GEE_READY = _init_gee()


# ═════════════════════════════════════════════════════════════════
# TROPOMI CH4 via GEE
# ═════════════════════════════════════════════════════════════════

class GEETROPOMIIngester:
    """
    Fetches Sentinel-5P TROPOMI Level-2 CH4 data from Google Earth Engine.

    GEE Collection: COPERNICUS/S5P/OFFL/L3_CH4
    Resolution:     ~7km x 7km
    Revisit:        Daily (global coverage)

    Falls back to synthetic plume data when GEE is unavailable.
    """

    COLLECTION = "COPERNICUS/S5P/OFFL/L3_CH4"
    BAND       = "CH4_column_volume_mixing_ratio_dry_air"
    QA_BAND    = "sensor_zenith_angle"

    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        date:    datetime | None = None,
        days_back: int = 5,
    ) -> xr.Dataset:
        if _GEE_READY:
            return self._fetch_gee(lat_min, lat_max, lon_min, lon_max, date, days_back)
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    def _fetch_gee(
        self,
        lat_min, lat_max, lon_min, lon_max,
        date, days_back,
    ) -> xr.Dataset:
        date     = date or datetime.utcnow()
        end_dt   = date.strftime("%Y-%m-%d")
        start_dt = (date - timedelta(days=days_back)).strftime("%Y-%m-%d")

        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        collection = (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(start_dt, end_dt)
            .filterBounds(region)
            .select([self.BAND])
        )

        # Median composite to reduce noise
        image  = collection.median().clip(region)
        scale  = 10000   # metres — ~0.1 degree at equator

        try:
            arr = geemap.ee_to_numpy(
                image,
                bands=[self.BAND],
                region=region,
                scale=scale,
            )
            if arr is None or arr.size == 0:
                raise ValueError("Empty GEE result")

            ch4     = arr[:, :, 0].astype(np.float32)
            grid    = ch4.shape[0]
            lats    = np.linspace(lat_min, lat_max, grid)
            lons    = np.linspace(lon_min, lon_max, ch4.shape[1])

            ds = xr.Dataset(
                {
                    "methane_mixing_ratio_bias_corrected": (["lat", "lon"], ch4),
                    "methane_mixing_ratio_precision":      (["lat", "lon"], np.full_like(ch4, 12.0)),
                    "qa_value":                            (["lat", "lon"], np.ones_like(ch4)),
                    "cloud_fraction":                      (["lat", "lon"], np.random.uniform(0, 0.15, ch4.shape).astype(np.float32)),
                },
                coords={"lat": lats, "lon": lons},
            )
            logger.info(f"GEE TROPOMI: fetched {ch4.shape} grid | {start_dt} → {end_dt}")
            return ds

        except Exception as e:
            logger.error(f"GEE TROPOMI: fetch failed ({e}) — falling back to mock")
            return self._mock(lat_min, lat_max, lon_min, lon_max)

    @staticmethod
    def _mock(
        lat_min, lat_max, lon_min, lon_max,
        grid: int = 64,
    ) -> xr.Dataset:
        """
        Synthetic CH4 field with injected super-emitter plumes.
        Physically realistic baseline + Gaussian plumes.
        """
        lats   = np.linspace(lat_min, lat_max, grid)
        lons   = np.linspace(lon_min, lon_max, grid)
        lon2d, lat2d = np.meshgrid(lons, lats)

        base  = 1870.0
        noise = np.random.normal(0, 8, (grid, grid))

        def plume(clat, clon, strength=150, sigma=0.08):
            return strength * np.exp(
                -(((lat2d - clat) / sigma) ** 2
                  + ((lon2d - clon) / sigma) ** 2)
            )

        c_lat = (lat_min + lat_max) / 2
        c_lon = (lon_min + lon_max) / 2
        ch4   = (base + noise
                 + plume(c_lat, c_lon, 180)
                 + plume(c_lat + 0.3, c_lon - 0.2, 95)
                 + plume(c_lat - 0.4, c_lon + 0.35, 120))

        ds = xr.Dataset(
            {
                "methane_mixing_ratio_bias_corrected": (["lat", "lon"], ch4.astype(np.float32)),
                "methane_mixing_ratio_precision":      (["lat", "lon"], np.full((grid, grid), 12.0, dtype=np.float32)),
                "qa_value":                            (["lat", "lon"], np.ones((grid, grid), dtype=np.float32)),
                "cloud_fraction":                      (["lat", "lon"], np.random.uniform(0, 0.2, (grid, grid)).astype(np.float32)),
            },
            coords={"lat": lats, "lon": lons},
        )
        logger.info("GEE TROPOMI: generated synthetic mock dataset")
        return ds


# ═════════════════════════════════════════════════════════════════
# ERA5 Wind via GEE
# ═════════════════════════════════════════════════════════════════

class GEEWindIngester:
    """
    Fetches ECMWF ERA5 10m wind components from Google Earth Engine.

    GEE Collection: ECMWF/ERA5_LAND/HOURLY
    Variables:      u_component_of_wind_10m, v_component_of_wind_10m
    Resolution:     ~9km

    Fix: ERA5_LAND/HOURLY has ~5-day latency. Always fetch a
    5-day window ending yesterday to guarantee non-empty results.
    """

    COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
    U_BAND     = "u_component_of_wind_10m"
    V_BAND     = "v_component_of_wind_10m"

    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        date:    datetime | None = None,
    ) -> xr.Dataset:
        if _GEE_READY:
            return self._fetch_gee(lat_min, lat_max, lon_min, lon_max, date)
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    def _fetch_gee(self, lat_min, lat_max, lon_min, lon_max, date) -> xr.Dataset:
        # ERA5-Land HOURLY has ~5 day latency on GEE.
        # Use a 7-day window ending 5 days ago to always get real data.
        ref_date  = (date or datetime.utcnow()) - timedelta(days=5)
        end_dt    = ref_date.strftime("%Y-%m-%d")
        start_dt  = (ref_date - timedelta(days=7)).strftime("%Y-%m-%d")
        region    = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        collection = (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(start_dt, end_dt)
            .filterBounds(region)
            .select([self.U_BAND, self.V_BAND])
        )

        try:
            # Guard: check collection is not empty before calling .mean()
            count = collection.size().getInfo()
            if count == 0:
                raise ValueError(f"No ERA5 images in {start_dt}→{end_dt} for this bbox")

            image = collection.mean().clip(region)

            arr = geemap.ee_to_numpy(
                image,
                bands=[self.U_BAND, self.V_BAND],
                region=region,
                scale=10000,
            )
            if arr is None or arr.size == 0:
                raise ValueError("Empty GEE result")

            u10  = arr[:, :, 0].astype(np.float32)
            v10  = arr[:, :, 1].astype(np.float32)
            grid = u10.shape[0]
            lats = np.linspace(lat_min, lat_max, grid)
            lons = np.linspace(lon_min, lon_max, u10.shape[1])

            ds = xr.Dataset(
                {
                    "u10": (["lat", "lon"], u10),
                    "v10": (["lat", "lon"], v10),
                },
                coords={"lat": lats, "lon": lons},
            )
            logger.info(
                f"GEE ERA5: fetched {count} images | "
                f"u={u10.mean():.2f} m/s v={v10.mean():.2f} m/s | "
                f"{start_dt} → {end_dt}"
            )
            return ds

        except Exception as e:
            logger.error(f"GEE ERA5: failed ({e}) — falling back to mock")
            return self._mock(lat_min, lat_max, lon_min, lon_max)

    @staticmethod
    def _mock(lat_min, lat_max, lon_min, lon_max, grid: int = 32) -> xr.Dataset:
        lats = np.linspace(lat_min, lat_max, grid)
        lons = np.linspace(lon_min, lon_max, grid)
        u10  = np.full((grid, grid),  5.2, dtype=np.float32)
        v10  = np.full((grid, grid), -1.8, dtype=np.float32)
        ds   = xr.Dataset(
            {"u10": (["lat", "lon"], u10), "v10": (["lat", "lon"], v10)},
            coords={"lat": lats, "lon": lons},
        )
        logger.info("GEE ERA5: generated synthetic wind mock")
        return ds


# ═════════════════════════════════════════════════════════════════
# NASA EMIT via GEE
# ═════════════════════════════════════════════════════════════════

class GEEEMITIngester:
    """
    Fetches NASA EMIT CH4 enhancement data from Google Earth Engine.

    GEE Collection: NASA/EMIT/L2B/CH4ENH
    Band:           methane_enhancement (ppm·m)
    Resolution:     ~60m
    Used for dual-sensor cross-validation.

    Note: EMIT has sparse spatial coverage — many regions will
    return empty collections. We handle this gracefully by checking
    collection size before computing and falling back to mock.
    """

    COLLECTION = "NASA/EMIT/L2B/CH4ENH"
    BAND       = "methane_enhancement"

    def fetch(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
        date:    datetime | None = None,
    ) -> xr.Dataset:
        if _GEE_READY:
            return self._fetch_gee(lat_min, lat_max, lon_min, lon_max, date)
        return self._mock(lat_min, lat_max, lon_min, lon_max)

    def _fetch_gee(self, lat_min, lat_max, lon_min, lon_max, date) -> xr.Dataset:
        date     = date or datetime.utcnow()
        end_dt   = date.strftime("%Y-%m-%d")
        # EMIT has sparse revisit — use a 90-day lookback to maximise hit rate
        start_dt = (date - timedelta(days=90)).strftime("%Y-%m-%d")
        region   = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        collection = (
            ee.ImageCollection(self.COLLECTION)
            .filterDate(start_dt, end_dt)
            .filterBounds(region)
        )

        try:
            # Guard: check collection size FIRST — calling .select() on an
            # empty collection returns an image with no bands, which causes
            # "Image has no band named X" downstream.
            count = collection.size().getInfo()
            if count == 0:
                raise ValueError(
                    f"No EMIT scenes in bbox for {start_dt}→{end_dt} "
                    f"(EMIT has sparse coverage — falling back to mock)"
                )

            image = (
                collection
                .select([self.BAND])
                .max()      # peak enhancement across all available scenes
                .clip(region)
            )

            arr = geemap.ee_to_numpy(
                image,
                bands=[self.BAND],
                region=region,
                scale=1000,   # 1km for reasonable array size
            )
            if arr is None or arr.size == 0:
                raise ValueError("Empty GEE EMIT result after fetch")

            enh  = arr[:, :, 0].astype(np.float32)
            grid = enh.shape[0]
            lats = np.linspace(lat_min, lat_max, grid)
            lons = np.linspace(lon_min, lon_max, enh.shape[1])

            ds = xr.Dataset(
                {"ch4_enhancement": (["lat", "lon"], enh)},
                coords={"lat": lats, "lon": lons},
            )
            logger.info(
                f"GEE EMIT: {count} scenes | "
                f"peak enhancement = {enh.max():.1f} ppm·m | "
                f"{start_dt} → {end_dt}"
            )
            return ds

        except Exception as e:
            logger.error(f"GEE EMIT: failed ({e}) — falling back to mock")
            return self._mock(lat_min, lat_max, lon_min, lon_max)

    @staticmethod
    def _mock(lat_min, lat_max, lon_min, lon_max, grid: int = 128) -> xr.Dataset:
        lats   = np.linspace(lat_min, lat_max, grid)
        lons   = np.linspace(lon_min, lon_max, grid)
        lon2d, lat2d = np.meshgrid(lons, lats)
        c_lat  = (lat_min + lat_max) / 2
        c_lon  = (lon_min + lon_max) / 2
        enh    = (2000 * np.exp(
            -(((lat2d - c_lat) / 0.02) ** 2
              + ((lon2d - c_lon) / 0.02) ** 2)
        )).astype(np.float32)
        ds = xr.Dataset(
            {"ch4_enhancement": (["lat", "lon"], enh)},
            coords={"lat": lats, "lon": lons},
        )
        logger.info("GEE EMIT: generated synthetic high-res mock")
        return ds


# ═════════════════════════════════════════════════════════════════
# GEE status helper — used by API health endpoint
# ═════════════════════════════════════════════════════════════════

def gee_status() -> dict:
    return {
        "available":   _GEE_READY,
        "package":     GEE_AVAILABLE,
        "project":     os.environ.get("GEE_PROJECT", "not set"),
        "collections": {
            "tropomi": "COPERNICUS/S5P/OFFL/L3_CH4",
            "era5":    "ECMWF/ERA5_LAND/HOURLY",
            "emit":    "NASA/EMIT/L2B/CH4ENH",
        }
    }