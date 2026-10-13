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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

from pyproj import Geod

RegionKind = Literal["box", "band", "around", "polygon"]
PolygonEdges = Literal["geodesic"]

_GEOD = Geod(ellps="WGS84")


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


def _same_point(a: tuple[float, float], b: tuple[float, float]) -> bool:
    # All longitudes coincide at a pole
    return a[1] == b[1] and (abs(a[1]) == 90.0 or a[0] == b[0])


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


def _around(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) != 5:
        raise ValueError(
            "around requires (lat, lon, half_width_m, half_height_m, bearing_deg)"
        )
    lat, lon, half_width, half_height, bearing = params
    _latitude(lat, "lat")
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("around half-widths must be positive")
    return (lat, _longitude(lon), half_width, half_height, _wrap(bearing, 0.0))


def _polygon(params: tuple[float, ...]) -> tuple[float, ...]:
    if len(params) % 2:
        raise ValueError("polygon parameters must be (lon, lat) pairs")
    vertices = [
        (_longitude(lon), _latitude(lat, "polygon latitude"))
        for lon, lat in zip(params[::2], params[1::2])
    ]
    # Closure is implicit
    if len(vertices) > 1 and _same_point(vertices[0], vertices[-1]):
        vertices.pop()
    if len(vertices) < 3:
        raise ValueError("polygon requires at least three distinct vertices")
    for (lon_a, lat_a), (lon_b, lat_b) in zip(vertices, vertices[1:] + vertices[:1]):
        if _same_point((lon_a, lat_a), (lon_b, lat_b)):
            raise ValueError("polygon must not repeat consecutive vertices")
        if lat_a == -lat_b and (abs(lat_a) == 90.0 or abs(lon_a - lon_b) == 180.0):
            raise ValueError("polygon geodesic edges must not join antipodes")

    lons, lats = zip(*vertices)
    area, _ = _GEOD.polygon_area_perimeter(lons, lats)
    if area == 0.0:
        raise ValueError("polygon must enclose a nonzero area")
    # The interior is the smaller enclosed region; store it counterclockwise
    if area < 0.0:
        vertices = vertices[:1] + vertices[:0:-1]
    return tuple(value for vertex in vertices for value in vertex)


@dataclass(frozen=True)
class Region:
    """Geographic request independent of any model grid.

    Build regions with :meth:`box`, :meth:`band`, :meth:`around` or
    :meth:`polygon`. Direct construction applies the same validation and
    normalization, so equal requests compare and hash equal. Longitudes are
    normalized to ``[-180, 180)`` and bearings to ``[0, 360)``; coordinates are
    never rounded.

    Parameters
    ----------
    kind : {"box", "band", "around", "polygon"}
        Region shape.
    params : tuple[float, ...]
        ``box``: ``(west, south, east, north)``. ``band``: ``(south, north)``.
        ``around``: ``(lat, lon, half_width_m, half_height_m, bearing_deg)``.
        ``polygon``: flat ``(lon_0, lat_0, lon_1, lat_1, ...)``, stored
        counterclockwise around the smaller enclosed region, with implicit closure.
    edges : {"geodesic"} | None, optional
        Polygon edge interpretation; required for polygons and ``None`` otherwise,
        by default None.
    """

    kind: RegionKind
    params: tuple[float, ...]
    edges: PolygonEdges | None = None

    def __post_init__(self) -> None:
        if self.kind not in get_args(RegionKind):
            raise ValueError(f"Unknown region kind {self.kind!r}")
        if self.kind == "polygon":
            if self.edges not in get_args(PolygonEdges):
                raise ValueError(f"Unsupported polygon edges {self.edges!r}")
        elif self.edges is not None:
            raise ValueError(f"{self.kind} regions do not take edges")
        params = _finite(self.params)
        normalize = {
            "box": _box,
            "band": _band,
            "around": _around,
            "polygon": _polygon,
        }[self.kind]
        object.__setattr__(self, "params", normalize(params))

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
    def around(
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
        return cls("around", (lat, lon, half_width_m, height, bearing_deg))

    @classmethod
    def polygon(
        cls,
        vertices: Sequence[tuple[float, float]],
        edges: PolygonEdges = "geodesic",
    ) -> Region:
        """Polygon through ``(lon, lat)`` vertices enclosing the smaller region.

        Parameters
        ----------
        vertices : Sequence[tuple[float, float]]
            At least three ``(lon, lat)`` vertices in degrees; closure is implicit.
        edges : {"geodesic"}, optional
            How consecutive vertices are joined, by default "geodesic".

        Returns
        -------
        Region
            Polygon region.
        """
        if any(len(vertex) != 2 for vertex in vertices):
            raise ValueError("polygon vertices must be (lon, lat) pairs")
        return cls("polygon", tuple(v for vertex in vertices for v in vertex), edges)

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        """Polygon ``(lon, lat)`` vertices in stored counterclockwise order."""
        if self.kind != "polygon":
            raise AttributeError(f"{self.kind} regions have no vertices")
        return tuple(zip(self.params[::2], self.params[1::2]))
