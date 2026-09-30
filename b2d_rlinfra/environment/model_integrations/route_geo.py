"""Shared route geography helpers for model-specific route wrappers."""

from __future__ import annotations

import math
from typing import Any, Tuple
import xml.etree.ElementTree as ET

import numpy as np

EARTH_RADIUS_EQUA = 6378137.0
DEFAULT_LAT_REF = 42.0
DEFAULT_LON_REF = 2.0


def location_xy(value: Any, *, label: str = "route point") -> np.ndarray:
    """Extract CARLA world x/y from route points, waypoints, locations, or arrays."""
    if hasattr(value, "location"):
        loc = value.location
    elif hasattr(value, "transform") and hasattr(value.transform, "location"):
        loc = value.transform.location
    else:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size < 2:
            raise ValueError(f"{label} has no x/y coordinates: {value!r}")
        return array[:2].astype(np.float32, copy=True)
    return np.asarray([float(loc.x), float(loc.y)], dtype=np.float32)


def gps_latlon(value: Any, *, label: str = "GPS point") -> np.ndarray:
    """Extract latitude/longitude from dicts, CARLA GPS objects, or arrays."""
    if isinstance(value, dict):
        return np.asarray([float(value["lat"]), float(value["lon"])], dtype=np.float64)
    if hasattr(value, "lat") and hasattr(value, "lon"):
        return np.asarray([float(value.lat), float(value.lon)], dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2:
        raise ValueError(f"{label} has no latitude/longitude: {value!r}")
    return array[:2]


def gps_to_location_xy(gps: Any, lat_ref: float, lon_ref: float, *, label: str = "GPS value") -> np.ndarray:
    """Project GNSS latitude/longitude into the B2D world x/y frame."""
    lat, lon = gps_latlon(gps, label=label)[:2]
    scale = math.cos(float(lat_ref) * math.pi / 180.0)
    x = scale * (float(lon) - float(lon_ref)) * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = math.log(math.tan((float(lat) + 90.0) * math.pi / 360.0)) * EARTH_RADIUS_EQUA * scale
    ref_y = (
        scale
        * EARTH_RADIUS_EQUA
        * math.log(math.tan((90.0 + float(lat_ref)) * math.pi / 360.0))
    )
    y = ref_y - my
    return np.asarray([x, y], dtype=np.float32)


def actor_position_xy(
    transform: Any,
    *,
    offset_x: float = -1.4,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Return the GNSS-reference world x/y for an actor pose."""
    location = transform.location
    yaw_rad = math.radians(float(transform.rotation.yaw))
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world_x = float(location.x) + float(offset_x) * cos_yaw - float(offset_y) * sin_yaw
    world_y = float(location.y) + float(offset_x) * sin_yaw + float(offset_y) * cos_yaw
    return np.asarray([world_x, world_y], dtype=np.float32)


def latlon_ref_from_world(world: Any) -> Tuple[float, float]:
    """Read CARLA OpenDRIVE geoReference, matching leaderboard route projection."""
    lat_ref = DEFAULT_LAT_REF
    lon_ref = DEFAULT_LON_REF
    try:
        opendrive = world.get_map().to_opendrive()
        root = ET.fromstring(opendrive)
        geo_reference = root.find("./header/geoReference")
        text = geo_reference.text if geo_reference is not None else ""
        for item in (text or "").split():
            if item.startswith("+lat_0="):
                lat_ref = float(item.split("=", 1)[1])
            elif item.startswith("+lon_0="):
                lon_ref = float(item.split("=", 1)[1])
    except Exception:
        lat_ref = DEFAULT_LAT_REF
        lon_ref = DEFAULT_LON_REF
    return float(lat_ref), float(lon_ref)


def rollout_latlon_ref(
    world_entry: Tuple[Any, Any],
    gps_entry: Tuple[Any, Any],
    *,
    label: str = "route",
) -> Tuple[float, float]:
    """Match the official first-waypoint GPS/world georef solve."""
    world_xy = location_xy(world_entry[0], label=f"{label} world route point").astype(np.float64)
    gps = gps_latlon(gps_entry[0], label=f"{label} GPS route point")
    lat = float(gps[0])
    lon = float(gps[1])
    locx = float(world_xy[0])
    locy = float(world_xy[1])

    def y_equation(lat_ref: float) -> float:
        scale = math.cos(lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * EARTH_RADIUS_EQUA
        ref_y = EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
        return scale * my + locy - scale * ref_y

    lat_ref = 0.0
    for _ in range(50):
        value = y_equation(lat_ref)
        if abs(value) < 1e-7:
            break
        delta = 1e-5
        derivative = (y_equation(lat_ref + delta) - y_equation(lat_ref - delta)) / (2.0 * delta)
        if not math.isfinite(derivative) or abs(derivative) < 1e-12:
            raise RuntimeError(f"Unable to solve {label} latitude reference")
        lat_ref -= value / derivative
        if not math.isfinite(lat_ref) or lat_ref <= -89.0 or lat_ref >= 89.0:
            raise RuntimeError(f"Invalid {label} latitude reference")
    else:
        raise RuntimeError(f"{label} latitude reference solve did not converge")

    scale = math.cos(lat_ref * math.pi / 180.0)
    if abs(scale) < 1e-12:
        raise RuntimeError(f"Invalid {label} longitude reference scale")
    lon_ref = lon - ((locx * lat_ref * 180.0) / (math.pi * EARTH_RADIUS_EQUA)) / scale
    return float(lat_ref), float(lon_ref)
