from __future__ import annotations
import numpy as np
from shapely.geometry import box, Point
from pyproj import Transformer

# WGS84 → metres (for distance calculations)
_to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def bbox_to_polygon(lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float):
    """Return a Shapely Polygon from lat/lon bounds."""
    return box(lon_min, lat_min, lon_max, lat_max)


def haversine_km(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def back_propagate_wind(
    lat: float, lon: float,
    u_ms: float, v_ms: float,
    duration_hours: float,
) -> tuple[float, float]:
    """
    Lagrangian back-trajectory: given a plume centroid and wind vector,
    estimate the source location duration_hours upwind.
    Returns (source_lat, source_lon).
    """
    # convert to metres, step backwards
    x, y = _to_m.transform(lon, lat)
    x -= u_ms * duration_hours * 3600
    y -= v_ms * duration_hours * 3600
    src_lon, src_lat = _to_ll.transform(x, y)
    return float(src_lat), float(src_lon)


def pixel_area_km2(lat: float, pixel_deg: float = 0.01) -> float:
    """Approximate area of a square pixel in km² at a given latitude."""
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians(lat))
    return (pixel_deg * lat_km) * (pixel_deg * lon_km)