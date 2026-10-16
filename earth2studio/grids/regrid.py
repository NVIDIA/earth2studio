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

"""Nearest-neighbor and bilinear regridders between grid definitions."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from earth2studio.grids.base import E2S_CRS, E2S_GRID_ID, GridDefinition
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import KDTree
except ImportError:
    OptionalDependencyFailure("utils")
    LinearNDInterpolator = None
    KDTree = None

try:
    import cupy as cp
except ImportError:
    cp = None

_EARTH_RADIUS_KM = 6371.0

# Attributes describing a field's geometry, replaced by the target grid's (R4).
_GRID_ATTRS = frozenset(
    {
        E2S_GRID_ID,
        E2S_CRS,
        "crs",
        "type",
        "dims",
        "shape",
        "topology",
        "level",
        "nside",
        "ordering",
        "layout",
        "origin",
        "clockwise",
    }
)

# Topologies whose signatures carry geographic auxiliaries (see ``coord_array``).
_GEOGRAPHIC_TOPOLOGIES = frozenset({"curvilinear", "points", "projected"})

# Host tables: neighbor flat indices [K, N], weights [K, N] or None, valid [N].
_Tables = tuple[NDArray[np.int64], NDArray[np.float64] | None, NDArray[np.bool_]]


def _resolve(grid: str | GridDefinition) -> tuple[GridDefinition, str | None]:
    if isinstance(grid, str):
        from earth2studio.grids import resolve_grid

        return resolve_grid(grid), grid
    if not isinstance(grid, GridDefinition):
        raise TypeError("Expected a GridDefinition or a registered grid name")
    return grid, None


def _geographic(grid: GridDefinition) -> tuple[NDArray[Any], NDArray[Any]]:
    """Flattened latitude and longitude of every grid point, in ``grid.dims`` order."""
    coords = grid.coords()
    if "lat" not in coords or "lon" not in coords:
        raise ValueError(
            f"{type(grid).__name__} does not provide geographic coordinates"
        )
    latitude, longitude = xr.broadcast(coords["lat"], coords["lon"])
    return (
        np.asarray(latitude.transpose(*grid.dims), dtype=np.float64).ravel(),
        np.asarray(longitude.transpose(*grid.dims), dtype=np.float64).ravel(),
    )


def _index_axes(grid: GridDefinition) -> tuple[NDArray[Any], NDArray[Any]]:
    coords = grid.coords(only_index=True)
    first, second = grid.dims
    return (
        np.asarray(coords[first], dtype=np.float64),
        np.asarray(coords[second], dtype=np.float64),
    )


def _longitude_period(axis: NDArray[Any]) -> float | None:
    """Return 360 when a longitude axis wraps the globe without repeating a point."""
    if axis.size < 2:
        return None
    span = abs(axis[-1] - axis[0])
    gap = 360.0 - span
    step = np.abs(np.diff(axis)).max()
    return 360.0 if 0 < gap <= step * (1 + 1e-6) else None


def _wrap_longitude(values: NDArray[Any], axis: NDArray[Any]) -> NDArray[Any]:
    """Shift longitudes to the convention of a non-periodic ascending ``axis``.

    Values outside the axis map to whichever end is nearer around the globe.
    """
    values = axis[0] + np.mod(values - axis[0], 360.0)
    beyond = values > axis[-1] + (360.0 - (axis[-1] - axis[0])) / 2
    return np.where(beyond, values - 360.0, values)


def _axis_neighbors(
    axis: NDArray[Any], values: NDArray[Any], *, period: float | None = None
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Bracket ``values`` on a strictly monotonic axis.

    Returns lower and upper neighbor indices, the weight of the upper neighbor, and
    whether each value lies within the axis. Periodic axes contain every finite
    value; values outside a bounded axis clamp to its nearest end.
    """
    size = axis.size
    if size < 2:
        raise ValueError("Grid axes need at least two points")
    step = np.diff(axis)
    if not (np.all(step > 0) or np.all(step < 0)):
        raise ValueError("Grid axes must be strictly monotonic")
    descending = bool(step[0] < 0)
    ordered = axis[::-1] if descending else axis
    finite = np.isfinite(values)
    values = np.where(finite, values, ordered[0])
    positions = np.arange(size, dtype=np.float64)
    if period is not None:
        values = ordered[0] + np.mod(values - ordered[0], period)
        position = np.interp(
            values,
            np.append(ordered, ordered[0] + period),
            np.append(positions, size),
        )
        lower = np.clip(np.floor(position), 0, size - 1).astype(np.int64)
        upper = (lower + 1) % size
        valid = finite
    else:
        position = np.interp(values, ordered, positions)
        lower = np.clip(np.floor(position), 0, size - 2).astype(np.int64)
        upper = lower + 1
        valid = finite & (values >= ordered[0]) & (values <= ordered[-1])
    weight = position - lower
    if descending:
        lower, upper = size - 1 - lower, size - 1 - upper
    return lower, upper, weight, valid


def _latlon_axes(grid: GridDefinition) -> tuple[NDArray[Any], NDArray[Any]]:
    """Latitude and longitude axes of a rectilinear grid.

    A longitude axis crossing the seam, such as ``[350, 0, 10]``, is unwrapped to
    become monotonic while every value keeps its position. Monotonic axes are
    kept as they are, since unwrapping would fold steps wider than 180 degrees.
    """
    lat_axis, lon_axis = _index_axes(grid)
    if not _monotonic(lon_axis):
        unwrapped = np.unwrap(lon_axis, period=360.0)
        if _monotonic(unwrapped):
            lon_axis = unwrapped
    return lat_axis, lon_axis


def _monotonic(axis: NDArray[Any]) -> bool:
    step = np.diff(axis)
    return bool(np.all(step > 0) or np.all(step < 0))


def _latlon_neighbors(
    grid: GridDefinition, latitude: NDArray[Any], longitude: NDArray[Any]
) -> tuple[tuple[NDArray[Any], ...], tuple[NDArray[Any], ...]]:
    """Bracket geographic points on a rectilinear latitude-longitude grid."""
    lat_axis, lon_axis = _latlon_axes(grid)
    period = _longitude_period(lon_axis)
    if period is None:
        ascending = lon_axis if lon_axis[-1] > lon_axis[0] else lon_axis[::-1]
        longitude = _wrap_longitude(longitude, ascending)
    return (
        _axis_neighbors(lat_axis, latitude),
        _axis_neighbors(lon_axis, longitude, period=period),
    )


def _rectilinear_nearest(
    grid: GridDefinition, latitude: NDArray[Any], longitude: NDArray[Any]
) -> tuple[NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Nearest point on the sphere of a rectilinear grid, without a search tree.

    Along every latitude row the nearest point lies in the column nearest in
    longitude, so the nearest grid point lies on that column's meridian. When the
    meridian's closest approach to the target lies between the poles, distance
    along it has a single minimum there, so the nearest row brackets that
    latitude. Otherwise the closest approach is on the far side of the globe,
    distance peaks between the poles, and the nearest row is the first or last.
    Both cases are covered by choosing among those four rows.
    """
    lat_axis, lon_axis = _latlon_axes(grid)
    columns = np.zeros(latitude.shape, dtype=np.int64)
    if lon_axis.size > 1:
        period = _longitude_period(lon_axis)
        wrapped = longitude
        if period is None:
            ascending = lon_axis if lon_axis[-1] > lon_axis[0] else lon_axis[::-1]
            wrapped = _wrap_longitude(longitude, ascending)
        lower, upper, weight, _ = _axis_neighbors(lon_axis, wrapped, period=period)
        columns = np.where(weight >= 0.5, upper, lower)
    meridian = lon_axis[columns]

    rows = np.zeros(latitude.shape, dtype=np.int64)
    if lat_axis.size > 1:
        lat, dlon = np.radians(latitude), np.radians(longitude - meridian)
        closest = np.degrees(np.arctan2(np.sin(lat), np.cos(lat) * np.cos(dlon)))
        lower, upper, _, _ = _axis_neighbors(lat_axis, np.clip(closest, -90, 90))
        ends = np.zeros_like(lower), np.full_like(lower, lat_axis.size - 1)
        # Upper first, so exact ties resolve as rounding the bracket would.
        candidates = np.stack((upper, lower, *ends))
        km = _great_circle_km(latitude, longitude, lat_axis[candidates], meridian)
        rows = np.take_along_axis(candidates, km.argmin(axis=0)[None], axis=0)[0]
    distance = _great_circle_km(latitude, longitude, lat_axis[rows], meridian)
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    return rows * lon_axis.size + columns, distance, valid


def _unit_vectors(latitude: NDArray[Any], longitude: NDArray[Any]) -> NDArray[Any]:
    lat, lon = np.radians(latitude), np.radians(longitude)
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1
    )


def _great_circle_km(
    lat0: NDArray[Any], lon0: NDArray[Any], lat1: NDArray[Any], lon1: NDArray[Any]
) -> NDArray[Any]:
    chord = np.linalg.norm(
        _unit_vectors(lat0, lon0) - _unit_vectors(lat1, lon1), axis=-1
    )
    return 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * _EARTH_RADIUS_KM


@check_optional_dependencies()
def _tree_neighbors(
    source: tuple[NDArray[Any], NDArray[Any]],
    target: tuple[NDArray[Any], NDArray[Any]],
) -> tuple[NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Nearest source point and its distance for each target point, on the sphere."""
    source_valid = np.isfinite(source[0]) & np.isfinite(source[1])
    if not source_valid.any():
        raise ValueError("Source grid has no finite geographic coordinates")
    target_valid = np.isfinite(target[0]) & np.isfinite(target[1])
    tree = KDTree(_unit_vectors(source[0][source_valid], source[1][source_valid]))
    chord = np.full(target[0].size, np.inf)
    nearest = np.zeros(target[0].size, dtype=np.int64)
    chord[target_valid], found = tree.query(
        _unit_vectors(target[0][target_valid], target[1][target_valid]), k=1
    )
    nearest[target_valid] = np.flatnonzero(source_valid)[found]
    distance = 2 * np.arcsin(np.clip(chord / 2, 0, 1)) * _EARTH_RADIUS_KM
    return nearest, distance, target_valid


@check_optional_dependencies()
def _triangulated_positions(
    source: GridDefinition, latitude: NDArray[Any], longitude: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Fractional source indices of geographic points by Delaunay triangulation.

    Interpolation is planar in degrees, so targets must not cross the longitude seam
    of the source. Points outside the source's convex hull are NaN.
    """
    source_lat, source_lon = _geographic(source)
    rows, columns = np.unravel_index(np.arange(source_lat.size), source.shape)
    valid = np.isfinite(source_lat) & np.isfinite(source_lon)
    points = np.stack((source_lat[valid], source_lon[valid]), axis=-1)
    longitude = source_lon[valid].min() + np.mod(
        longitude - source_lon[valid].min(), 360.0
    )
    queries = np.stack((latitude, longitude), axis=-1)
    return (
        LinearNDInterpolator(points, rows[valid].astype(np.float64))(queries),
        LinearNDInterpolator(points, columns[valid].astype(np.float64))(queries),
    )


def _cell_neighbors(
    position: NDArray[Any], size: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[Any], NDArray[np.bool_]]:
    """Bracket fractional indices, such as triangulated positions, on an axis."""
    valid = np.isfinite(position)
    position = np.where(valid, position, 0.0)
    lower = np.clip(np.floor(position), 0, size - 2).astype(np.int64)
    return lower, lower + 1, position - lower, valid


class _IndexRegridder:
    """Apply precomputed neighbor indices and weights on the field's own device.

    Subclasses build host tables once at construction (R1). Calls gather on the
    input's backing, so NumPy fields stay on the host and CuPy fields stay on their
    device (R5). Device copies of the tables are cached on first use, which is the
    only state a call may change (R6).
    """

    def __init__(
        self,
        source_grid: str | GridDefinition,
        target_grid: str | GridDefinition,
        fill_value: float,
    ) -> None:
        self._source_grid, _ = _resolve(source_grid)
        self._target_grid, self._target_id = _resolve(target_grid)
        self.fill_value = float(fill_value)
        index, weight, valid = self._build()
        for table in (index, weight, valid):
            if table is not None:
                table.setflags(write=False)
        self._host: _Tables = (index, weight, valid)
        self._devices: dict[int, tuple[Any, Any, Any]] = {}
        self._target_coords = self._target_grid.coords(
            only_index=self._target_grid.topology not in _GEOGRAPHIC_TOPOLOGIES
        )

    def _build(self) -> _Tables:
        raise NotImplementedError

    @property
    def source_grid(self) -> GridDefinition:
        """Grid the regridder accepts."""
        return self._source_grid

    @property
    def target_grid(self) -> GridDefinition:
        """Grid the regridder produces."""
        return self._target_grid

    @property
    def coverage(self) -> NDArray[np.bool_]:
        """Target points with source support, shaped like ``target_grid``.

        Points outside coverage receive ``fill_value``.
        """
        return self._host[2].reshape(self._target_grid.shape)

    def to(self, device: torch.device | str | int) -> _IndexRegridder:
        """Place index and weight tables on a device ahead of the first call.

        Calls follow the input's device regardless, so this is optional.

        Parameters
        ----------
        device : torch.device | str | int
            CPU or CUDA device; integers are CUDA indices.

        Returns
        -------
        _IndexRegridder
            This regridder.
        """
        device = torch.device("cuda", device) if isinstance(device, int) else device
        device = torch.device(device)
        if device.type == "cuda":
            if cp is None:
                raise ImportError("cupy is required to place regridders on CUDA")
            index = device.index if device.index is not None else cp.cuda.Device().id
            self._device_tables(index)
        elif device.type != "cpu":
            raise ValueError("Regridders support only CPU and CUDA devices")
        return self

    def _device_tables(self, device: int) -> tuple[Any, Any, Any]:
        if device not in self._devices:
            index, weight, valid = self._host
            with cp.cuda.Device(device):
                self._devices[device] = (
                    cp.asarray(index),
                    None if weight is None else cp.asarray(weight),
                    cp.asarray(valid),
                )
        return self._devices[device]

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Regrid the spatial dimensions of a field.

        Parameters
        ----------
        x : xr.DataArray
            NumPy- or CuPy-backed field containing every ``source_grid`` dimension
            at its grid size. Other dimensions pass through.

        Returns
        -------
        xr.DataArray
            Field on ``target_grid`` with the backing and device of ``x``: its
            non-spatial dimensions in order, followed by ``target_grid.dims``.
            Floating dtypes are preserved; other dtypes are promoted to float32.

        Raises
        ------
        ValueError
            If the spatial dimensions of ``x`` do not match ``source_grid``.
        """
        if not isinstance(x, xr.DataArray):
            raise TypeError("Regridders apply to xr.DataArray fields")
        source_dims = self._source_grid.dims
        sizes = tuple(x.sizes.get(dim) for dim in source_dims)
        if sizes != self._source_grid.shape:
            raise ValueError(
                f"Field dimensions {dict(x.sizes)} do not match the source grid "
                f"{dict(zip(source_dims, self._source_grid.shape))}"
            )
        leading = tuple(dim for dim in x.dims if dim not in source_dims)
        x = x.transpose(*leading, *source_dims)
        data = x.data
        context: AbstractContextManager[Any]
        if cp is not None and isinstance(data, cp.ndarray):
            xp = cp
            index, weight, valid = self._device_tables(data.device.id)
            # Allocate on the field's device, not the current one.
            context = cp.cuda.Device(data.device.id)
        else:
            xp, data = np, np.asarray(data)
            index, weight, valid = self._host
            context = nullcontext()
        dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float32
        leading_shape = data.shape[: len(leading)]
        with context:
            flat = data.reshape(*leading_shape, -1).astype(dtype, copy=False)
            if weight is None:
                values = xp.take(flat, index[0], axis=-1)
            else:
                values = xp.zeros((*leading_shape, index.shape[1]), dtype=dtype)
                for neighbor, factor in zip(index, weight.astype(dtype, copy=False)):
                    # Skip zero-weight neighbors so their NaNs do not leak into
                    # exact hits.
                    values += xp.where(
                        factor != 0, xp.take(flat, neighbor, axis=-1) * factor, 0
                    )
            fill = xp.asarray(self.fill_value, dtype=dtype)
            values = xp.where(valid, values, fill)
            values = values.reshape(*leading_shape, *self._target_grid.shape)

        coords = {
            name: coordinate.variable
            for name, coordinate in x.coords.items()
            if not set(coordinate.dims).intersection(source_dims)
        }
        coords.update(
            {name: value.variable for name, value in self._target_coords.items()}
        )
        return xr.DataArray(
            values,
            dims=(*leading, *self._target_grid.dims),
            coords=coords,
            name=x.name,
            attrs=self._attrs(x.attrs),
        )

    def _attrs(self, attrs: dict[Any, Any]) -> dict[Any, Any]:
        owned = _GRID_ATTRS.union(self._source_grid.attrs, self._target_grid.attrs)
        result = {key: value for key, value in attrs.items() if key not in owned}
        result.update(self._target_grid.attrs)
        result["dims"] = list(self._target_grid.dims)
        if self._target_grid.crs is not None:
            result[E2S_CRS] = self._target_grid.crs.to_string()
        if self._target_id is not None:
            result[E2S_GRID_ID] = self._target_id
        return result


class NearestRegridder(_IndexRegridder):
    """Map each target point to its nearest source point.

    Distances are great-circle, so the regridder is indifferent to the longitude
    convention of either grid. Source points without finite coordinates, such as
    off-disk satellite pixels, are never selected. On a latitude-longitude source
    the nearest point is found exactly from the grid axes, without a search tree.

    Parameters
    ----------
    source_grid : str | GridDefinition
        Grid of the fields to regrid, or a registered grid name.
    target_grid : str | GridDefinition
        Grid to produce, or a registered grid name. A name is recorded in the
        output's ``earth2studio_grid_id``.
    max_dist_km : float | None, optional
        Targets farther than this from their nearest source point are outside
        coverage. By default None, which accepts every target.
    fill_value : float, optional
        Value for targets outside coverage, by default NaN

    Note
    ----
    Sources other than latitude-longitude grids need SciPy (the ``utils`` extra)
    to build the search tree.
    """

    def __init__(
        self,
        source_grid: str | GridDefinition,
        target_grid: str | GridDefinition,
        *,
        max_dist_km: float | None = None,
        fill_value: float = np.nan,
    ) -> None:
        self.max_dist_km = max_dist_km
        super().__init__(source_grid, target_grid, fill_value)

    def _build(self) -> _Tables:
        source, target = self._source_grid, self._target_grid
        latitude, longitude = _geographic(target)
        if source.topology == "rectilinear" and all(
            map(_monotonic, _latlon_axes(source))
        ):
            index, distance, valid = _rectilinear_nearest(source, latitude, longitude)
        else:
            index, distance, valid = _tree_neighbors(
                _geographic(source), (latitude, longitude)
            )
        if self.max_dist_km is not None:
            valid = valid & (distance <= self.max_dist_km)
        return np.where(valid, index, 0)[None], None, valid


class BilinearRegridder(_IndexRegridder):
    """Interpolate between the four source points surrounding each target point.

    Source points are located by topology:

    - Rectilinear sources bracket each target along their latitude and longitude
      axes. Longitude wraps when the grid covers the globe, and target longitudes
      are matched to the source convention.
    - Projected sources transform targets into the source CRS and bracket them
      along the projected axes, which is exact for any target.
    - Curvilinear sources triangulate their geographic coordinates. This is planar
      in degrees, needs SciPy (the ``utils`` extra), and does not wrap longitude.

    Targets outside the source grid are outside coverage. Interpolation is in index
    space, so a source NaN propagates to targets that weight it.

    Parameters
    ----------
    source_grid : str | GridDefinition
        Rectilinear, projected or curvilinear grid of the fields to regrid, or a
        registered grid name.
    target_grid : str | GridDefinition
        Grid to produce, or a registered grid name. A name is recorded in the
        output's ``earth2studio_grid_id``.
    fill_value : float, optional
        Value for targets outside coverage, by default NaN
    """

    def __init__(
        self,
        source_grid: str | GridDefinition,
        target_grid: str | GridDefinition,
        *,
        fill_value: float = np.nan,
    ) -> None:
        super().__init__(source_grid, target_grid, fill_value)

    def _build(self) -> _Tables:
        source, target = self._source_grid, self._target_grid
        if source.topology == "rectilinear":
            latitude, longitude = _geographic(target)
            rows, columns = _latlon_neighbors(source, latitude, longitude)
        elif source.topology == "projected":
            if source.crs is None:
                raise ValueError("Projected source grids need a CRS")
            y_axis, x_axis = _index_axes(source)
            if target.topology == "projected" and target.crs == source.crs:
                target_y, target_x = _index_axes(target)
                y, x = (
                    values.ravel()
                    for values in np.meshgrid(target_y, target_x, indexing="ij")
                )
            else:
                latitude, longitude = _geographic(target)
                x, y = Transformer.from_crs(
                    CRS.from_epsg(4326), source.crs, always_xy=True
                ).transform(longitude, latitude)
            rows, columns = _axis_neighbors(y_axis, y), _axis_neighbors(x_axis, x)
        elif source.topology == "curvilinear":
            latitude, longitude = _geographic(target)
            row, column = _triangulated_positions(source, latitude, longitude)
            rows = _cell_neighbors(row, source.shape[0])
            columns = _cell_neighbors(column, source.shape[1])
        else:
            raise ValueError(
                f"Bilinear regridding needs a structured source grid, not "
                f"{source.topology}; use NearestRegridder"
            )
        width = source.shape[1]
        row0, row1, row_weight, row_valid = rows
        column0, column1, column_weight, column_valid = columns
        valid = row_valid & column_valid
        index = np.stack(
            (
                row0 * width + column0,
                row0 * width + column1,
                row1 * width + column0,
                row1 * width + column1,
            )
        )
        weight = np.stack(
            (
                (1 - row_weight) * (1 - column_weight),
                (1 - row_weight) * column_weight,
                row_weight * (1 - column_weight),
                row_weight * column_weight,
            )
        )
        return np.where(valid, index, 0), np.where(valid, weight, 0.0), valid
