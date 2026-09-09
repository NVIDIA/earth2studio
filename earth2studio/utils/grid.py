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

"""Spatial grid definitions and registry utilities.

Grid protocol
-------------
``GridDefinition`` describes geometry without owning field data or choosing a
regridding implementation.

Grid registry
-------------
The process-local registry assigns complete definitions stable names and aliases.
Use ``register_grid``, ``resolve_grid``, and ``list_grids`` to manage it.

Xarray coordinates
------------------
Use ``coordinates()`` for complete Xarray coordinates or
``coordinates(only_index=True)``
for dimension coordinates only. The result can be passed directly to
``DataArray(..., coords=...)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS, Transformer

E2S_CRS = "earth2studio_crs"
E2S_GRID_ID = "earth2studio_grid_id"

GridTopology = Literal["rectilinear", "projected", "curvilinear", "healpix", "points"]


def _array(values: Any, ndim: int, name: str) -> NDArray[Any]:
    array = np.asarray(values)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}D array")
    array = array.copy()
    array.setflags(write=False)
    return array


def _coordinate_hash(*arrays: NDArray[Any]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _is_crs(value: str) -> bool:
    try:
        CRS.from_user_input(value)
    except Exception:
        return False
    return True


def _geographic_subset_indexers(
    definition: GridDefinition, coordinates: xr.Coordinates, **selection: Any
) -> dict[str, Any]:
    unknown = set(selection) - {"bounds", "bounds_crs"}
    if unknown:
        raise ValueError(f"Unsupported grid subset options: {sorted(unknown)}")
    if "bounds_crs" in selection and "bounds" not in selection:
        raise ValueError("Grid subset bounds_crs requires bounds")
    if "bounds" not in selection:
        return {}

    bounds = selection["bounds"]
    if len(bounds) != 4:
        raise ValueError("Bounds must contain (min_x, min_y, max_x, max_y)")
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if min_y > max_y:
        raise ValueError("Bounds minimum y must not exceed maximum y")

    geographic = (
        xr.Coordinates({"lat": coordinates["lat"], "lon": coordinates["lon"]})
        if "lat" in coordinates and "lon" in coordinates
        else definition.coordinates(
            {
                dimension: np.asarray(coordinates[dimension])
                for dimension in definition.dims
            }
        )
    )
    latitude, longitude = xr.broadcast(geographic["lat"], geographic["lon"])
    latitude_values = np.asarray(latitude)
    longitude_values = np.asarray(longitude)
    target_crs = CRS.from_user_input(selection.get("bounds_crs", "OGC:CRS84"))
    if target_crs.is_geographic:
        if min_x >= 0 and max_x >= 0 and (min_x > 180 or max_x > 180):
            x_values = np.mod(longitude_values, 360)
        else:
            x_values = np.mod(longitude_values + 180, 360) - 180
        x_mask = (
            (x_values >= min_x) & (x_values <= max_x)
            if min_x <= max_x
            else (x_values >= min_x) | (x_values <= max_x)
        )
        y_values = latitude_values
    else:
        x_values, y_values = Transformer.from_crs(
            CRS.from_epsg(4326), target_crs, always_xy=True
        ).transform(longitude_values, latitude_values)
        x_mask = (x_values >= min_x) & (x_values <= max_x)
    mask = x_mask & (y_values >= min_y) & (y_values <= max_y)
    if not np.any(mask):
        raise ValueError("Grid subset bounds do not contain any cell centers")

    indexers: dict[str, Any] = {}
    for dimension in definition.dims:
        axis = latitude.dims.index(dimension)
        other_axes = tuple(index for index in range(mask.ndim) if index != axis)
        selected = np.flatnonzero(np.any(mask, axis=other_axes) if other_axes else mask)
        indexers[dimension] = (
            slice(int(selected[0]), int(selected[-1]) + 1)
            if np.all(np.diff(selected) == 1)
            else selected
        )
    return indexers


# sphinx - grid protocol start
@runtime_checkable
class GridDefinition(Protocol):
    """Describe spatial geometry through a structural interface."""

    @property
    def dims(self) -> tuple[str, ...]:
        """Return ordered spatial dimensions."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the spatial shape."""
        ...

    @property
    def topology(self) -> GridTopology:
        """Return the grid topology."""
        ...

    @property
    def crs(self) -> CRS | None:
        """Return the native coordinate reference system, if defined."""
        ...

    def coordinates(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return complete coordinates or only dimension indexes."""
        ...

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate a selection into positional dimension indexers."""
        ...

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> xr.Coordinates | None:
        """Return geographic cell boundaries when available."""
        ...

    def to_metadata(self) -> dict[str, Any]:
        """Return a serializable grid description."""
        ...

    def fingerprint(self) -> str:
        """Return a stable geometry fingerprint."""
        ...


# sphinx - grid protocol end


@dataclass(frozen=True)
class LatLonGrid:
    """Define a rectilinear latitude-longitude grid.

    Parameters
    ----------
    latitude : NDArray[Any]
        One-dimensional latitude coordinates.
    longitude : NDArray[Any]
        One-dimensional longitude coordinates.
    coordinate_reference_system : Any, optional
        Geographic CRS, by default ``EPSG:4326``
    """

    latitude: NDArray[Any]
    longitude: NDArray[Any]
    coordinate_reference_system: Any = "EPSG:4326"

    def __post_init__(self) -> None:
        object.__setattr__(self, "latitude", _array(self.latitude, 1, "latitude"))
        object.__setattr__(self, "longitude", _array(self.longitude, 1, "longitude"))
        object.__setattr__(
            self,
            "coordinate_reference_system",
            CRS.from_user_input(self.coordinate_reference_system),
        )

    @property
    def dims(self) -> tuple[str, ...]:
        """Return latitude then longitude."""
        return ("lat", "lon")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return latitude and longitude sizes."""
        return (self.latitude.size, self.longitude.size)

    @property
    def topology(self) -> GridTopology:
        """Return the rectilinear topology."""
        return "rectilinear"

    @property
    def crs(self) -> CRS:
        """Return the geographic CRS."""
        return self.coordinate_reference_system

    def coordinates(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return latitude and longitude coordinates."""
        indexes = indexes or {"lat": self.latitude, "lon": self.longitude}
        return xr.Coordinates({"lat": indexes["lat"], "lon": indexes["lon"]})

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate geographic bounds into dimension indexers."""
        return _geographic_subset_indexers(self, coordinates, **selection)

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell boundaries."""
        return None

    def to_metadata(self) -> dict[str, Any]:
        """Return rectilinear grid metadata."""
        return {"topology": self.topology}

    def fingerprint(self) -> str:
        """Return a coordinate fingerprint."""
        return _coordinate_hash(self.latitude, self.longitude) + self.crs.to_wkt()


@dataclass(frozen=True)
class ProjectedGrid:
    """Define a structured projected grid.

    Parameters
    ----------
    y : NDArray[Any]
        One-dimensional native y coordinates.
    x : NDArray[Any]
        One-dimensional native x coordinates.
    coordinate_reference_system : Any
        CRS accepted by :class:`pyproj.CRS`.
    """

    y: NDArray[Any]
    x: NDArray[Any]
    coordinate_reference_system: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "y", _array(self.y, 1, "y"))
        object.__setattr__(self, "x", _array(self.x, 1, "x"))
        object.__setattr__(
            self,
            "coordinate_reference_system",
            CRS.from_user_input(self.coordinate_reference_system),
        )

    @property
    def dims(self) -> tuple[str, ...]:
        """Return y then x."""
        return ("y", "x")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return y and x sizes."""
        return (self.y.size, self.x.size)

    @property
    def topology(self) -> GridTopology:
        """Return the projected topology."""
        return "projected"

    @property
    def crs(self) -> CRS:
        """Return the native projected CRS."""
        return self.coordinate_reference_system

    def coordinates(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return native and optional geographic coordinates."""
        indexes = indexes or {"y": self.y, "x": self.x}
        coordinates: dict[str, Any] = {"y": indexes["y"], "x": indexes["x"]}
        if only_index:
            return xr.Coordinates(coordinates)
        xx: NDArray[Any]
        yy: NDArray[Any]
        xx, yy = np.meshgrid(indexes["x"], indexes["y"])
        longitude, latitude = Transformer.from_crs(
            self.crs, CRS.from_epsg(4326), always_xy=True
        ).transform(xx, yy)
        coordinates.update(
            lat=(("y", "x"), latitude),
            lon=(("y", "x"), np.mod(longitude, 360)),
        )
        return xr.Coordinates(coordinates)

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate geographic bounds into dimension indexers."""
        return _geographic_subset_indexers(self, coordinates, **selection)

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell boundaries."""
        return None

    def to_metadata(self) -> dict[str, Any]:
        """Return projected grid metadata."""
        return {"topology": self.topology}

    def fingerprint(self) -> str:
        """Return a native-coordinate fingerprint."""
        return _coordinate_hash(self.y, self.x) + self.crs.to_wkt()


@dataclass(frozen=True)
class CurvilinearGrid:
    """Define a structured grid using two-dimensional geographic coordinates.

    Parameters
    ----------
    latitude : NDArray[Any]
        Two-dimensional latitude coordinates.
    longitude : NDArray[Any]
        Two-dimensional longitude coordinates.
    y : NDArray[Any] | None, optional
        One-dimensional y indexes, by default None
    x : NDArray[Any] | None, optional
        One-dimensional x indexes, by default None
    """

    latitude: NDArray[Any]
    longitude: NDArray[Any]
    y: NDArray[Any] | None = None
    x: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        latitude = _array(self.latitude, 2, "latitude")
        longitude = _array(self.longitude, 2, "longitude")
        if latitude.shape != longitude.shape:
            raise ValueError("latitude and longitude shapes must match")
        y = _array(np.arange(latitude.shape[0]) if self.y is None else self.y, 1, "y")
        x = _array(np.arange(latitude.shape[1]) if self.x is None else self.x, 1, "x")
        if (y.size, x.size) != latitude.shape:
            raise ValueError("y and x sizes must match geographic coordinates")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "x", x)

    @property
    def dims(self) -> tuple[str, ...]:
        """Return y then x."""
        return ("y", "x")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the two-dimensional grid shape."""
        return self.latitude.shape

    @property
    def topology(self) -> GridTopology:
        """Return the curvilinear topology."""
        return "curvilinear"

    @property
    def crs(self) -> None:
        """Return no native CRS."""
        return None

    def coordinates(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return index and optional geographic coordinates."""
        indexes = indexes or {"y": self.y, "x": self.x}
        coordinates = xr.Coordinates({"y": indexes["y"], "x": indexes["x"]})
        if only_index:
            return coordinates
        latitude = xr.DataArray(
            self.latitude, dims=self.dims, coords={"y": self.y, "x": self.x}
        ).sel(y=indexes["y"], x=indexes["x"])
        longitude = xr.DataArray(
            self.longitude, dims=self.dims, coords={"y": self.y, "x": self.x}
        ).sel(y=indexes["y"], x=indexes["x"])
        return coordinates.assign(lat=latitude, lon=longitude)

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate geographic bounds into dimension indexers."""
        return _geographic_subset_indexers(self, coordinates, **selection)

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell boundaries."""
        return None

    def to_metadata(self) -> dict[str, Any]:
        """Return curvilinear grid metadata."""
        return {"topology": self.topology}

    def fingerprint(self) -> str:
        """Return a geographic-coordinate fingerprint."""
        return _coordinate_hash(self.latitude, self.longitude)


@dataclass(frozen=True)
class PointGrid:
    """Define arbitrary geographic points along an x index.

    Parameters
    ----------
    latitude : NDArray[Any]
        One-dimensional latitude coordinates.
    longitude : NDArray[Any]
        One-dimensional longitude coordinates.
    x : NDArray[Any] | None, optional
        One-dimensional point indexes, by default None
    """

    latitude: NDArray[Any]
    longitude: NDArray[Any]
    x: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        latitude = _array(self.latitude, 1, "latitude")
        longitude = _array(self.longitude, 1, "longitude")
        if latitude.shape != longitude.shape:
            raise ValueError("latitude and longitude shapes must match")
        x = _array(np.arange(latitude.size) if self.x is None else self.x, 1, "x")
        if x.size != latitude.size:
            raise ValueError("x size must match geographic coordinates")
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "x", x)

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the x point dimension."""
        return ("x",)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the number of points."""
        return (self.latitude.size,)

    @property
    def topology(self) -> GridTopology:
        """Return the points topology."""
        return "points"

    @property
    def crs(self) -> None:
        """Return no native CRS."""
        return None

    def coordinates(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return point indexes and optional geographic coordinates."""
        indexes = indexes or {"x": self.x}
        coordinates = xr.Coordinates({"x": indexes["x"]})
        if only_index:
            return coordinates
        latitude = xr.DataArray(self.latitude, dims="x", coords={"x": self.x}).sel(
            x=indexes["x"]
        )
        longitude = xr.DataArray(self.longitude, dims="x", coords={"x": self.x}).sel(
            x=indexes["x"]
        )
        return coordinates.assign(lat=latitude, lon=longitude)

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate geographic bounds into point indexers."""
        return _geographic_subset_indexers(self, coordinates, **selection)

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell boundaries."""
        return None

    def to_metadata(self) -> dict[str, Any]:
        """Return point-grid metadata."""
        return {"topology": self.topology}

    def fingerprint(self) -> str:
        """Return a geographic-coordinate fingerprint."""
        return _coordinate_hash(self.latitude, self.longitude)


def _healpix_coordinates(
    nside: int, ordering: str, pixels: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    if ordering == "ring":
        npix = 12 * nside**2
        ncap = 2 * nside * (nside - 1)
        pixels = np.asarray(pixels, dtype=np.int64)
        z = np.empty(pixels.shape, dtype=float)
        longitude = np.empty(pixels.shape, dtype=float)

        north = pixels < ncap
        equatorial = (pixels >= ncap) & (pixels < npix - ncap)
        south = pixels >= npix - ncap
        if np.any(north):
            ipix = pixels[north]
            ring = np.floor(0.5 * (1 + np.sqrt(1 + 2 * ipix))).astype(int)
            azimuth = ipix + 1 - 2 * ring * (ring - 1)
            z[north] = 1 - ring**2 / (3 * nside**2)
            longitude[north] = (azimuth - 0.5) * np.pi / (2 * ring)
        if np.any(equatorial):
            ipix = pixels[equatorial] - ncap
            ring = ipix // (4 * nside) + nside
            azimuth = ipix % (4 * nside) + 1
            shift = 0.5 * (1 + (ring + nside) % 2)
            z[equatorial] = (2 * nside - ring) * 2 / (3 * nside)
            longitude[equatorial] = (azimuth - shift) * np.pi / (2 * nside)
        if np.any(south):
            ipix = npix - pixels[south]
            ring = np.floor(0.5 * (1 + np.sqrt(2 * ipix - 1))).astype(int)
            azimuth = 4 * ring + 1 - (ipix - 2 * ring * (ring - 1))
            z[south] = -1 + ring**2 / (3 * nside**2)
            longitude[south] = (azimuth - 0.5) * np.pi / (2 * ring)
        latitude = 90 - np.degrees(np.arccos(np.clip(z, -1, 1)))
        return latitude, np.mod(np.degrees(longitude), 360)

    level = nside.bit_length() - 1
    pixels = np.asarray(pixels, dtype=np.int64)
    local = pixels % nside**2
    x = np.zeros_like(local)
    y = np.zeros_like(local)
    for bit in range(level):
        x |= ((local >> (2 * bit)) & 1) << bit
        y |= ((local >> (2 * bit + 1)) & 1) << bit

    face = pixels // nside**2
    x = (x + 0.5) / nside
    y = (y + 0.5) / nside
    x_origin = np.array([1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3])
    y_origin = np.array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5])
    x_rot = x_origin[face] + x
    y_rot = -y_origin[face] + y
    xs = (x_rot - y_rot - 1) * np.pi / 4
    ys = (x_rot + y_rot) * np.pi / 4

    polar = np.abs(ys) > np.pi / 4
    longitude = xs.copy()
    longitude[polar] -= (
        (np.abs(ys[polar]) - np.pi / 4)
        / (np.abs(ys[polar]) - np.pi / 2)
        * (np.mod(xs[polar], np.pi / 2) - np.pi / 4)
    )
    z = 8 * ys / (3 * np.pi)
    term = 2 - 4 * np.abs(ys[polar]) / np.pi
    z[polar] = (1 - term**2 / 3) * np.sign(ys[polar])
    latitude = 90 - np.degrees(np.arccos(np.clip(z, -1, 1)))
    return latitude, np.mod(np.degrees(longitude), 360)


@dataclass(frozen=True)
class HEALPixGrid:
    """Define a HEALPix grid.

    Parameters
    ----------
    level : int
        HEALPix level where ``nside = 2**level``.
    ordering : Literal["nested", "ring"], optional
        Pixel ordering, by default ``nested``
    """

    level: int
    ordering: Literal["nested", "ring"] = "nested"

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("HEALPix level must be nonnegative")
        if self.ordering not in {"nested", "ring"}:
            raise ValueError("HEALPix ordering must be 'nested' or 'ring'")

    @property
    def nside(self) -> int:
        """Return the number of subdivisions per face edge."""
        return 2**self.level

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the HEALPix index dimension."""
        return ("hpx",)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the number of HEALPix pixels."""
        return (12 * self.nside**2,)

    @property
    def topology(self) -> GridTopology:
        """Return the HEALPix topology."""
        return "healpix"

    @property
    def crs(self) -> None:
        """Return no native CRS."""
        return None

    def coordinates(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return pixel indexes and optional geographic coordinates."""
        indexes = indexes or {"hpx": np.arange(self.shape[0])}
        coordinates = xr.Coordinates({"hpx": indexes["hpx"]})
        if only_index:
            return coordinates
        latitude, longitude = _healpix_coordinates(
            self.nside, self.ordering, indexes["hpx"]
        )
        return coordinates.assign(lat=("hpx", latitude), lon=("hpx", longitude))

    def subset_indexers(
        self, coordinates: xr.Coordinates, **selection: Any
    ) -> dict[str, Any]:
        """Translate bounds or face selection into pixel indexers.

        Parameters
        ----------
        coordinates : xr.Coordinates
            Current HEALPix coordinates.
        **selection : Any
            ``bounds``, ``bounds_crs``, or ``faces`` values.

        Returns
        -------
        dict[str, Any]
            Positional indexers for the hpx dimension.
        """
        unknown = set(selection) - {"bounds", "bounds_crs", "faces"}
        if unknown:
            raise ValueError(f"Unsupported grid subset options: {sorted(unknown)}")
        pixels = np.asarray(coordinates["hpx"])
        mask = np.ones(pixels.size, dtype=bool)
        if "faces" in selection:
            if self.ordering != "nested":
                raise NotImplementedError(
                    "HEALPix face selection requires NESTED ordering"
                )
            faces = np.atleast_1d(selection.pop("faces")).astype(int)
            if faces.size == 0 or np.any((faces < 0) | (faces > 11)):
                raise ValueError("HEALPix faces must be integers from 0 through 11")
            mask &= np.isin(pixels // self.nside**2, np.unique(faces))
        if selection:
            selected = xr.Coordinates({"hpx": ("hpx", pixels[mask])})
            bounds = _geographic_subset_indexers(self, selected, **selection)
            bounded = np.zeros(mask.sum(), dtype=bool)
            bounded[bounds["hpx"]] = True
            mask[np.flatnonzero(mask)] &= bounded
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            raise ValueError("Grid subset must contain at least one spatial cell")
        return {"hpx": positions}

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell boundaries."""
        return None

    def to_metadata(self) -> dict[str, Any]:
        """Return HEALPix grid metadata."""
        return {
            "topology": self.topology,
            "level": self.level,
            "nside": self.nside,
            "ordering": self.ordering,
        }

    def fingerprint(self) -> str:
        """Return a HEALPix definition fingerprint."""
        return f"healpix:{self.level}:{self.ordering}"


_GRID_REGISTRY: dict[str, GridDefinition] = {}
_GRID_ALIASES: dict[str, str] = {}


def _validate_definition(definition: GridDefinition) -> None:
    if not isinstance(definition, GridDefinition):
        raise TypeError("definition must implement GridDefinition")
    if not definition.dims or len(set(definition.dims)) != len(definition.dims):
        raise ValueError("Grid dimensions must be nonempty and unique")
    if len(definition.shape) != len(definition.dims) or any(
        size <= 0 for size in definition.shape
    ):
        raise ValueError("Grid shape must positively size every dimension")
    coordinates = definition.coordinates(only_index=True)
    for dimension, size in zip(definition.dims, definition.shape, strict=True):
        if dimension not in coordinates or coordinates[dimension].dims != (dimension,):
            raise ValueError(f"Grid must define a 1D '{dimension}' index coordinate")
        if coordinates[dimension].size != size:
            raise ValueError(f"Grid index coordinate '{dimension}' has the wrong size")
    try:
        json.dumps(definition.to_metadata())
    except TypeError as error:
        raise TypeError("Grid metadata must be JSON serializable") from error
    if not definition.fingerprint():
        raise ValueError("Grid fingerprint must not be empty")


def register_grid(
    name: str, definition: GridDefinition, *, aliases: Sequence[str] = ()
) -> None:
    """Register a complete grid definition under a name and optional aliases."""
    canonical = str(name)
    grid_aliases = tuple(str(alias) for alias in aliases)
    if not canonical:
        raise ValueError("Grid name must not be empty")
    if len(set(grid_aliases)) != len(grid_aliases) or canonical in grid_aliases:
        raise ValueError("Grid aliases must be unique and differ from the grid name")
    _validate_definition(definition)
    incoming = {canonical, *grid_aliases}
    for key in incoming:
        if _is_crs(key):
            raise ValueError(f"Grid name or alias '{key}' is already a valid CRS input")
        existing = key if key in _GRID_REGISTRY else _GRID_ALIASES.get(key)
        if existing is not None:
            if (
                existing == canonical
                and _GRID_REGISTRY[existing].fingerprint() == definition.fingerprint()
                and set(grid_aliases)
                == {
                    alias
                    for alias, target in _GRID_ALIASES.items()
                    if target == existing
                }
            ):
                return
            raise ValueError(
                f"Grid name or alias conflicts with registered grid '{existing}'"
            )
    _GRID_REGISTRY[canonical] = definition
    _GRID_ALIASES.update({alias: canonical for alias in grid_aliases})


def list_grids() -> tuple[str, ...]:
    """List canonical grid names in registration order."""
    return tuple(_GRID_REGISTRY)


def resolve_grid(grid: str) -> GridDefinition:
    """Resolve a canonical grid name or alias."""
    canonical = grid if grid in _GRID_REGISTRY else _GRID_ALIASES.get(grid)
    if canonical is not None:
        return _GRID_REGISTRY[canonical]
    if _is_crs(grid):
        raise ValueError(
            "A CRS does not define grid dimensions or geometry; create a grid "
            "definition and register it before use"
        )
    raise ValueError(f"Unknown Earth2Studio grid '{grid}'")


def infer_grid(array: xr.DataArray | xr.Dataset) -> GridDefinition:
    """Infer a grid definition from standard Xarray coordinates.

    Parameters
    ----------
    array : xr.DataArray | xr.Dataset
        Object containing spatial coordinates.

    Returns
    -------
    GridDefinition
        Inferred rectilinear, projected, curvilinear, or point grid.
    """
    grid_id = array.attrs.get(E2S_GRID_ID)
    if grid_id is not None:
        definition = resolve_grid(grid_id)
        spatial_dims = tuple(
            dimension for dimension in array.dims if dimension in definition.dims
        )
        if spatial_dims == definition.dims and all(
            array.sizes[dimension] == size
            for dimension, size in zip(definition.dims, definition.shape, strict=True)
        ):
            return definition
    coordinates = array.coords
    if "y" in coordinates and "x" in coordinates and E2S_CRS in array.attrs:
        return ProjectedGrid(
            np.asarray(coordinates["y"]),
            np.asarray(coordinates["x"]),
            array.attrs[E2S_CRS],
        )
    if "lat" in coordinates and "lon" in coordinates:
        latitude = coordinates["lat"]
        longitude = coordinates["lon"]
        if latitude.dims == ("lat",) and longitude.dims == ("lon",):
            return LatLonGrid(np.asarray(latitude), np.asarray(longitude))
        if latitude.dims == longitude.dims == ("x",):
            x = np.asarray(coordinates["x"]) if "x" in coordinates else None
            return PointGrid(np.asarray(latitude), np.asarray(longitude), x)
        if latitude.dims == longitude.dims == ("y", "x"):
            y = np.asarray(coordinates["y"]) if "y" in coordinates else None
            x = np.asarray(coordinates["x"]) if "x" in coordinates else None
            return CurvilinearGrid(np.asarray(latitude), np.asarray(longitude), y, x)
        raise ValueError(
            "Latitude and longitude coordinates have an unsupported layout"
        )
    raise ValueError(
        "Cannot infer grid geometry; provide lat/lon coordinates or y/x coordinates "
        f"with the '{E2S_CRS}' attribute"
    )


register_grid(
    "latlon-0.25deg",
    LatLonGrid(
        latitude=np.arange(90.0, -90.25, -0.25),
        longitude=np.arange(0.0, 360.0, 0.25),
    ),
    aliases=("latlon025",),
)
register_grid(
    "latlon-0.25deg-south-pole-excluded",
    LatLonGrid(
        latitude=np.arange(90.0, -90.0, -0.25),
        longitude=np.arange(0.0, 360.0, 0.25),
    ),
    aliases=("fcn1", "fcn1-global-0.25deg"),
)
register_grid(
    "hrrr-conus-3km",
    ProjectedGrid(
        y=-1587306.1525566636 + 3000.0 * np.arange(1059),
        x=-2697520.1425219304 + 3000.0 * np.arange(1799),
        coordinate_reference_system=(
            "+proj=lcc +lon_0=262.5 +lat_0=38.5 +lat_1=38.5 "
            "+lat_2=38.5 +R=6371229 +units=m +type=crs"
        ),
    ),
    aliases=("hrrr",),
)
register_grid(
    "healpix-l6-nested",
    HEALPixGrid(level=6, ordering="nested"),
    aliases=("hpx6",),
)
