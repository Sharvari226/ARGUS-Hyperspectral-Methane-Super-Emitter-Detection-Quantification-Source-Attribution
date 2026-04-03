from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from loguru import logger

from src.utils.config import cfg
from src.utils.geo import haversine_km

FACILITY_PATH = Path(cfg["data"]["facility_geojson"])


@lru_cache(maxsize=1)
def load_facilities() -> gpd.GeoDataFrame:
    """
    Load facility polygons from GeoJSON.
    Falls back to a synthetic registry of 50 facilities if file not found.
    """
    if FACILITY_PATH.exists():
        gdf = gpd.read_file(FACILITY_PATH)
        logger.info(f"FacilityDB: loaded {len(gdf)} facilities from {FACILITY_PATH}")
        return gdf
    logger.warning("FacilityDB: GeoJSON not found — generating synthetic registry")
    return _synthetic_facilities()


def find_nearest_facilities(
    lat: float, lon: float,
    radius_km: float | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Return up to top_k facilities within radius_km of (lat, lon),
    sorted by distance ascending.
    """
    radius_km = radius_km or cfg["pipeline"]["attribution_radius_km"]
    gdf = load_facilities()

    gdf = gdf.copy()
    gdf["distance_km"] = gdf.apply(
        lambda r: haversine_km(lat, lon,
                               r.geometry.centroid.y,
                               r.geometry.centroid.x),
        axis=1,
    )
    nearby = gdf[gdf["distance_km"] <= radius_km].nsmallest(top_k, "distance_km")
    return nearby.reset_index(drop=True)


def get_facility_by_id(facility_id: str) -> dict | None:
    gdf = load_facilities()
    row = gdf[gdf["facility_id"] == facility_id]
    if row.empty:
        return None
    rec = row.iloc[0].to_dict()
    rec["geometry"] = json.loads(row.iloc[0].geometry.to_json())
    return rec


# ── Synthetic generator ────────────────────────────────────────────────────────

def _synthetic_facilities(n: int = 50) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(42)

    OPERATORS = [
        "OilCorp International", "GasField Ventures", "PetroDelta Ltd",
        "TurkGaz Holdings", "SovEnergy PJSC", "AlphaFuel Inc",
        "NordicPetro AS", "Gulf Stream Energy", "IndusGas Ltd", "ArcoFlare Co",
    ]
    TYPES = ["oil_wellpad", "gas_compressor", "lng_terminal",
             "pipeline_station", "refinery", "gas_storage"]

    records = []
    for i in range(n):
        lat  = rng.uniform(-60, 75)
        lon  = rng.uniform(-160, 160)
        records.append({
            "facility_id":   f"FAC-{i:04d}",
            "facility_name": f"Facility {i:04d}",
            "operator":      rng.choice(OPERATORS),
            "type":          rng.choice(TYPES),
            "country":       "Synthetic",
            "lat":           lat,
            "lon":           lon,
            "geometry":      Point(lon, lat).buffer(0.05),  # ~5 km polygon
            # Compliance history
            "violations_12mo":      int(rng.integers(0, 6)),
            "last_inspection_date": "2024-01-15",
            "compliance_score":     float(rng.uniform(40, 100)),
        })

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    # Save so subsequent runs don't regenerate
    FACILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(FACILITY_PATH, driver="GeoJSON")
    logger.info(f"FacilityDB: saved {n} synthetic facilities → {FACILITY_PATH}")
    return gdf