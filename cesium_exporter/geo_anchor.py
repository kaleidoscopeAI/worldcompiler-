"""geo_anchor.py — WGS-84 / ECEF / ENU coordinate utilities.

All functions use the WGS-84 ellipsoid (GPS / Cesium standard).
No external dependencies — pure stdlib ``math``.

Coordinate systems used
-----------------------
* **LLA** — geodetic latitude (°), longitude (°), ellipsoidal height (m).
* **ECEF** — Earth-Centred Earth-Fixed Cartesian (m).  +X through (0°, 0°),
  +Z through north pole.
* **ENU** — local East-North-Up Cartesian (m) relative to a reference origin.
  Positive E = east, N = north, U = up.
"""
from __future__ import annotations

import math
from typing import List, Tuple

# ---------------------------------------------------------------------------
# WGS-84 ellipsoid constants
# ---------------------------------------------------------------------------
_WGS84_A: float = 6_378_137.0           # semi-major axis (m)
_WGS84_F: float = 1.0 / 298.257_223_563
_WGS84_B: float = _WGS84_A * (1.0 - _WGS84_F)   # semi-minor axis (m)
_WGS84_E2: float = 1.0 - (_WGS84_B / _WGS84_A) ** 2  # first eccentricity²


# ---------------------------------------------------------------------------
# LLA ↔ ECEF
# ---------------------------------------------------------------------------


def lla_to_ecef(
    lat_deg: float, lon_deg: float, height_m: float = 0.0
) -> Tuple[float, float, float]:
    """Convert geodetic (lat, lon, h) to ECEF Cartesian coordinates (m).

    Parameters
    ----------
    lat_deg:
        Geodetic latitude in degrees (−90 to +90).
    lon_deg:
        Longitude in degrees (−180 to +180).
    height_m:
        Ellipsoidal height in metres above the WGS-84 surface.

    Returns
    -------
    (X, Y, Z) in metres (ECEF).
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + height_m) * cos_lat * math.cos(lon)
    y = (N + height_m) * cos_lat * math.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + height_m) * sin_lat
    return x, y, z


def ecef_to_lla(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """Convert ECEF Cartesian to geodetic (lat, lon, h).

    Uses Bowring's iterative method (5 iterations give sub-mm precision
    everywhere on Earth).

    Returns
    -------
    (lat_deg, lon_deg, height_m)
    """
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    # Initial estimate
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(z + _WGS84_E2 * N * sin_lat, p)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    if abs(cos_lat) > 1e-10:
        h = p / cos_lat - N
    else:
        h = abs(z) / abs(sin_lat) - N * (1.0 - _WGS84_E2)
    return math.degrees(lat), math.degrees(lon), h


# ---------------------------------------------------------------------------
# ENU ↔ ECEF delta
# ---------------------------------------------------------------------------


def enu_delta_to_ecef(
    east_m: float,
    north_m: float,
    up_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
) -> Tuple[float, float, float]:
    """Convert a local ENU *offset* vector to an ECEF *offset* vector.

    Note: this converts a *displacement*, not an absolute position.
    Add the result to the origin's ECEF coordinates to obtain the target
    point in ECEF.

    Parameters
    ----------
    east_m, north_m, up_m:
        ENU displacement from the reference origin (metres).
    ref_lat_deg, ref_lon_deg:
        Geodetic coordinates of the reference origin.

    Returns
    -------
    (dX, dY, dZ) ECEF offset in metres.
    """
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    sn, cn = math.sin(lon), math.cos(lon)
    dx = -sn * east_m - sl * cn * north_m + cl * cn * up_m
    dy =  cn * east_m - sl * sn * north_m + cl * sn * up_m
    dz =  cl * north_m + sl * up_m
    return dx, dy, dz


# ---------------------------------------------------------------------------
# ENU → LLA (absolute)
# ---------------------------------------------------------------------------


def enu_to_lla(
    east_m: float,
    north_m: float,
    up_m: float,
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_height_m: float = 0.0,
) -> Tuple[float, float, float]:
    """Convert a local ENU offset from an origin to absolute geodetic LLA.

    Parameters
    ----------
    east_m, north_m, up_m:
        Offset from origin in metres (ENU frame).
    origin_lat_deg, origin_lon_deg:
        Geodetic coordinates of the ENU reference origin.
    origin_height_m:
        Ellipsoidal height of the reference origin (m).

    Returns
    -------
    (lat_deg, lon_deg, height_m) of the displaced point.
    """
    x0, y0, z0 = lla_to_ecef(origin_lat_deg, origin_lon_deg, origin_height_m)
    dx, dy, dz = enu_delta_to_ecef(east_m, north_m, up_m, origin_lat_deg, origin_lon_deg)
    return ecef_to_lla(x0 + dx, y0 + dy, z0 + dz)


# ---------------------------------------------------------------------------
# ENU-to-ECEF transform matrix (for 3D Tiles tile.transform)
# ---------------------------------------------------------------------------


def enu_to_ecef_transform(
    lat_deg: float, lon_deg: float, height_m: float = 0.0
) -> List[float]:
    """Return the 4×4 ENU-to-ECEF matrix stored in column-major order.

    This is the ``transform`` property expected by a 3D Tiles tile manifest.
    It maps the tile's local ENU coordinate frame to ECEF world space::

        ECEF_pos = M * [east, north, up, 1]^T

    The matrix columns are: east-axis, north-axis, up-axis, translation.

    Returns
    -------
    16 floats in column-major (Fortran) order.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    sn, cn = math.sin(lon), math.cos(lon)
    # Basis vectors expressed in ECEF:
    east  = (-sn,       cn,      0.0)
    north = (-sl * cn, -sl * sn,  cl)
    up    = ( cl * cn,  cl * sn,  sl)
    tx, ty, tz = lla_to_ecef(lat_deg, lon_deg, height_m)
    # Column-major: col0=east, col1=north, col2=up, col3=translation
    return [
        east[0],  east[1],  east[2],  0.0,
        north[0], north[1], north[2], 0.0,
        up[0],    up[1],    up[2],    0.0,
        tx,       ty,       tz,       1.0,
    ]


# ---------------------------------------------------------------------------
# Bounding-sphere helper for 3D Tiles
# ---------------------------------------------------------------------------


def ecef_bounding_sphere(
    lat_deg: float, lon_deg: float, height_m: float, radius_m: float
) -> List[float]:
    """Return ``[cx, cy, cz, radius]`` for a 3D Tiles boundingVolume.sphere.

    The sphere is centred at the given geodetic position.
    """
    cx, cy, cz = lla_to_ecef(lat_deg, lon_deg, height_m)
    return [cx, cy, cz, radius_m]
