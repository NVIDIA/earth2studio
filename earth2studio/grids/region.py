# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Geographic regions independent of model grids."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


def _finite(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Region parameters must be finite")
    return result


def _latitude(value: float, name: str) -> float:
    if not -90.0 <= value <= 90.0:
        raise ValueError(f"{name} must lie in [-90, 90]")
    return value


def _wrap(value: float, start: float) -> float:
    """Wrap into ``[start, start + 360)``, leaving in-range values untouched."""
    if start <= value < start + 360.0:
        return value
    wrapped = (value - start) % 360.0 + start
    # Rounding can land exactly on the open upper bound
    return start if wrapped >= start + 360.0 else wrapped


def _longitude(value: float) -> float:
    return _wrap(value, -180.0)


def _latitudes(south: float, north: float, kind: str) -> None:
    _latitude(south, "south")
    _latitude(north, "north")
    if south > north:
        raise ValueError(f"{kind} south must not exceed north")


def _box(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) != 4:
        raise ValueError("box requires (west, south, east, north)")
    west, south, east, north = params
    _latitudes(south, north, "box")
    if west != east and (east - west) % 360.0 == 0.0:
        raise ValueError(
            "box cannot span the full longitude range; west == east after "
            "normalization would describe a zero-width box. Use Region.band"
        )
    return (_longitude(west), south, _longitude(east), north)


def _band(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) != 2:
        raise ValueError("band requires (south, north)")
    _latitudes(*params, "band")
    return params


def _rectangle(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) != 5:
        raise ValueError(
            "rectangle requires (lat, lon, half_width_m, half_height_m, bearing_deg)"
        )
    lat, lon, half_width, half_height, bearing = params
    _latitude(lat, "lat")
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("rectangle half-widths must be positive")
    return (lat, _longitude(lon), half_width, half_height, _wrap(bearing, 0.0))


def _circle(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) != 3:
        raise ValueError("circle requires (lat, lon, radius_m)")
    lat, lon, radius = params
    _latitude(lat, "lat")
    if radius <= 0.0:
        raise ValueError("circle radius must be positive")
    # Every longitude names the same center at a pole
    return (lat, 0.0 if abs(lat) == 90.0 else _longitude(lon), radius)


_NORMALIZERS = {"box": _box, "band": _band, "rectangle": _rectangle, "circle": _circle}


@dataclass(frozen=True)
class Region:
    """Geographic request independent of any model grid.

    Build regions with :meth:`box`, :meth:`band`, :meth:`rectangle` or
    :meth:`circle`. Direct construction applies the same validation and
    normalization, so equal requests compare and hash equal. Longitudes are normalized to ``[-180, 180)`` and
    bearings to ``[0, 360)``; coordinates are never rounded.

    Parameters
    ----------
    kind : {"box", "band", "rectangle", "circle"}
        Region shape.
    params : tuple[float, ...]
        ``box``: ``(west, south, east, north)``. ``band``: ``(south, north)``.
        ``rectangle``: ``(lat, lon, half_width_m, half_height_m, bearing_deg)``.
        ``circle``: ``(lat, lon, radius_m)``.
    """

    kind: Literal["box", "band", "rectangle", "circle"]
    params: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.kind not in _NORMALIZERS:
            raise ValueError(f"Unknown region kind {self.kind!r}")
        normalize = _NORMALIZERS[self.kind]
        object.__setattr__(self, "params", normalize(_finite(self.params)))

    @classmethod
    def box(cls, west: float, south: float, east: float, north: float) -> Region:
        """Box bounded by meridians and parallels in CRS84 longitude/latitude.

        Parameters
        ----------
        west : float
            Western meridian in degrees; ``west > east`` crosses the antimeridian.
        south : float
            Southern parallel in degrees.
        east : float
            Eastern meridian in degrees; ``west == east`` is a zero-width box.
        north : float
            Northern parallel in degrees.

        Returns
        -------
        Region
            Box region.
        """
        return cls("box", (west, south, east, north))

    @classmethod
    def band(cls, south: float, north: float) -> Region:
        """Band between two parallels spanning every longitude.

        Covers latitude rings such as the tropics, polar caps (``band(60, 90)``)
        and the whole globe (``band(-90, 90)``).

        Parameters
        ----------
        south : float
            Southern parallel in degrees.
        north : float
            Northern parallel in degrees.

        Returns
        -------
        Region
            Latitude band region.
        """
        return cls("band", (south, north))

    @classmethod
    def rectangle(
        cls,
        lat: float,
        lon: float,
        half_width_m: float,
        half_height_m: float | None = None,
        bearing_deg: float = 0.0,
    ) -> Region:
        """Rectangle in the local azimuthal-equidistant plane centered on a point.

        Parameters
        ----------
        lat : float
            Center latitude in degrees.
        lon : float
            Center longitude in degrees; the reference meridian at a pole.
        half_width_m : float
            Half extent along the local x axis in meters.
        half_height_m : float | None, optional
            Half extent along the local y axis in meters; ``None`` makes a square,
            by default None.
        bearing_deg : float, optional
            Clockwise angle from true north to the local positive y axis, whose
            positive x axis is to its right, by default 0.0.

        Returns
        -------
        Region
            Oriented rectangle region.
        """
        height = half_width_m if half_height_m is None else half_height_m
        return cls("rectangle", (lat, lon, half_width_m, height, bearing_deg))

    @classmethod
    def circle(cls, lat: float, lon: float, radius_m: float) -> Region:
        """Points within a geodesic distance of a center point.

        Equivalent to a circle in the local azimuthal-equidistant plane centered on
        the point. At a pole the longitude is irrelevant and normalized to zero.

        Parameters
        ----------
        lat : float
            Center latitude in degrees.
        lon : float
            Center longitude in degrees.
        radius_m : float
            Geodesic radius in meters.

        Returns
        -------
        Region
            Circle region.
        """
        return cls("circle", (lat, lon, radius_m))
