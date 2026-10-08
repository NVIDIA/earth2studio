# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""DataArray-native field exchange and spatial transformation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import xarray as xr
from loguru import logger

from earth2studio.grids import PointGrid

from .clock import DeltaLike, as_datetime, as_timedelta, fmt_timedelta
from .component import Component
from .errors import CouplingError, IncompatibleFieldError, VerticalMismatchError
from .field import Field
from .mediator import _RunningReduction
from .vertical import HybridLevels, PressureLevels, interp_to_pressure

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CPU installations
    cp = None

Regridder = Callable[[xr.DataArray], xr.DataArray]


def _array_namespace(data: Any) -> Any:
    if cp is not None and isinstance(data, cp.ndarray):
        return cp
    return np


def _is_regular(values: np.ndarray) -> bool:
    return (
        values.ndim == 1
        and values.size > 1
        and np.allclose(np.diff(values), values[1] - values[0])
    )


def _bilinear_indices(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(source, target, side="right")
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    span = source[upper] - source[lower]
    weight = np.divide(
        target - source[lower],
        span,
        out=np.zeros_like(target, dtype=np.float64),
        where=span != 0,
    )
    return lower, upper, weight


def _regular_latlon_interpolate(
    array: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    target_dims: tuple[str, ...],
) -> xr.DataArray:
    if array.dims[-2:] != ("lat", "lon"):
        raise IncompatibleFieldError(
            f"Automatic interpolation requires trailing ('lat', 'lon') "
            f"dimensions, got {array.dims}"
        )
    source_lat = np.asarray(array.coords["lat"])
    source_lon = np.asarray(array.coords["lon"])
    if not (_is_regular(source_lat) and _is_regular(source_lon)):
        raise IncompatibleFieldError(
            "Automatic interpolation requires regular 1D source lat/lon"
        )
    data = array.data
    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        data = data[..., ::-1, :]
    flat_lat = np.asarray(target_lat).reshape(-1)
    flat_lon = np.asarray(target_lon).reshape(-1)
    il, iu, wy = _bilinear_indices(source_lat, flat_lat)
    jl, ju, wx = _bilinear_indices(source_lon, flat_lon)
    xp = _array_namespace(data)
    ilx, iux = xp.asarray(il), xp.asarray(iu)
    jlx, jux = xp.asarray(jl), xp.asarray(ju)
    wyx, wxx = xp.asarray(wy), xp.asarray(wx)
    v00 = data[..., ilx, jlx]
    v01 = data[..., ilx, jux]
    v10 = data[..., iux, jlx]
    v11 = data[..., iux, jux]
    output = (
        v00 * (1 - wyx) * (1 - wxx)
        + v01 * (1 - wyx) * wxx
        + v10 * wyx * (1 - wxx)
        + v11 * wyx * wxx
    )
    shape = (*output.shape[:-1], *np.asarray(target_lat).shape)
    output = output.reshape(shape)
    leading = array.dims[:-2]
    coords = {name: array.coords[name] for name in leading}
    for axis, name in enumerate(target_dims):
        coords[name] = np.arange(shape[len(leading) + axis])
    result = xr.DataArray(
        output,
        dims=(*leading, *target_dims),
        coords=coords,
        attrs=dict(array.attrs),
        name=array.name,
    )
    if target_dims == ("lat", "lon"):
        latitude = (
            np.asarray(target_lat)[:, 0]
            if np.asarray(target_lat).ndim == 2
            else np.asarray(target_lat)
        )
        longitude = (
            np.asarray(target_lon)[0, :]
            if np.asarray(target_lon).ndim == 2
            else np.asarray(target_lon)
        )
        result = result.assign_coords(lat=latitude, lon=longitude)
    elif target_dims == ("x",):
        result = result.assign_coords(
            lat=("x", np.asarray(target_lat)),
            lon=("x", np.asarray(target_lon)),
        )
    return result


def _nearest_indices(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    lat2d: np.ndarray
    lon2d: np.ndarray
    lat2d, lon2d = np.meshgrid(source_lat, source_lon, indexing="ij")
    return _nearest_point_indices(lat2d.ravel(), lon2d.ravel(), target_lat, target_lon)


def _nearest_point_indices(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    phi = np.deg2rad(np.asarray(source_lat)).ravel()
    lam = np.deg2rad(np.asarray(source_lon)).ravel()
    source_xyz = np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        axis=1,
    )
    phi, lam = np.deg2rad(target_lat).ravel(), np.deg2rad(target_lon).ravel()
    target_xyz = np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)],
        axis=1,
    )
    return cKDTree(source_xyz).query(target_xyz, k=1)[1]


def _sample_nearest(array: xr.DataArray, grid: PointGrid) -> xr.DataArray:
    source_lat = np.asarray(array.coords["lat"])
    source_lon = np.asarray(array.coords["lon"])
    index = _nearest_indices(source_lat, source_lon, grid.latitude, grid.longitude)
    data = array.data.reshape(*array.shape[:-2], -1)
    xp = _array_namespace(data)
    sampled = xp.take(data, xp.asarray(index), axis=-1)
    return xr.DataArray(
        sampled,
        dims=(*array.dims[:-2], "x"),
        coords={
            **{name: array.coords[name] for name in array.dims[:-2]},
            **dict(grid.coords()),
        },
        attrs=dict(array.attrs),
        name=array.name,
    )


class Connector:
    """Move matched fields from one component to another."""

    def __init__(
        self,
        src: Component,
        dst: Component,
        fields: list[str] | None = None,
        time_policy: Literal["constant", "linear"] = "constant",
        fill: Literal["none", "zero", "nearest"] = "none",
        regridder: Regridder | None = None,
        sample: Literal["nearest", "bilinear"] | None = None,
        window: DeltaLike | None = None,
        reduce: Literal["mean", "sum", "max", "min"] | None = None,
    ):
        self.src, self.dst = src, dst
        self.time_policy, self.fill = time_policy, fill
        if sample is not None and regridder is not None:
            raise CouplingError(
                f"Connector {self.name}: sample= and regridder= are exclusive"
            )
        if sample not in (None, "nearest", "bilinear"):
            raise CouplingError(f"Connector {self.name}: unsupported sample={sample!r}")
        if (window is None) != (reduce is None):
            raise CouplingError(
                f"Connector {self.name}: window= and reduce= must be set together"
            )
        if reduce not in (None, "mean", "sum", "max", "min"):
            raise CouplingError(f"Connector {self.name}: unsupported reduce={reduce!r}")
        self.sample = sample
        self.window = as_timedelta(window) if window is not None else None
        self.reduce = reduce
        self._user_regridder = regridder
        self._fields = list(fields) if fields is not None else None
        self._matched: list[str] | None = None
        self._derived: dict[str, str] = {}
        self._history: dict[str, tuple[Field | None, Field]] = {}
        self._linear_warned: set[str] = set()
        self._reduction = _RunningReduction()
        self._origin: np.datetime64 | None = None
        self.last_transfer: dict[str, Field] = {}

    @property
    def name(self) -> str:
        return f"{self.src.name}->{self.dst.name}"

    def match(self) -> list[str]:
        if self._matched is not None:
            return self._matched
        _, exports = self.src.advertise()
        imports, _ = self.dst.advertise()
        if self.window is not None:
            return self._match_windowed(exports, imports)
        if self._fields is None:
            matched = [name for name in imports if name in exports]
        else:
            matched = [
                name for name in self._fields if name in imports and name in exports
            ]
            missing = set(self._fields) - set(matched)
            if missing:
                raise IncompatibleFieldError(
                    f"Connector {self.name}: fields {sorted(missing)} are not "
                    "advertised by both endpoints"
                )
        if not matched:
            raise IncompatibleFieldError(
                f"Connector {self.name}: no fields match; source exports "
                f"{exports}, destination imports {imports}"
            )
        for name in matched:
            self.dst.dictionary.check_units(
                name,
                self.src.dictionary.resolve(name).canonical_units,
                src=self.src.name,
                dst=self.dst.name,
            )
        self._matched = matched
        return matched

    def _match_windowed(self, exports: list[str], imports: list[str]) -> list[str]:
        wanted = self._fields if self._fields is not None else exports
        for name in imports:
            method = self.dst.dictionary.resolve(name).cell_method
            if (
                method is not None
                and method.base in wanted
                and method.base in exports
                and method.method == self.reduce
                and as_timedelta(method.window) == self.window
            ):
                self._derived[method.base] = name
        missing = [name for name in wanted if name not in self._derived]
        if not self._derived or (self._fields is not None and missing):
            raise CouplingError(
                f"Connector {self.name}: destination has no matching derived "
                f"imports for {missing or exports}"
            )
        self._matched = [*self._derived, *self._derived.values()]
        return self._matched

    def _apply_time_policy(self, field: Field, time: np.datetime64) -> Field:
        previous, latest = self._history.get(field.standard_name, (None, None))
        if (
            latest is None
            or field.valid_time is None
            or latest.valid_time is None
            or as_datetime(field.valid_time) != as_datetime(latest.valid_time)
        ):
            previous, latest = latest, field
            self._history[field.standard_name] = (previous, latest)
        if self.time_policy == "constant" or previous is None:
            return field
        if "lead_time" in field.array.dims or "window" in field.array.dims:
            if field.standard_name not in self._linear_warned:
                logger.warning(
                    "Connector {}: linear time policy is undefined for "
                    "lead/window-resolved field {!r}; using constant",
                    self.name,
                    field.standard_name,
                )
                self._linear_warned.add(field.standard_name)
            return field
        if previous.valid_time is None or field.valid_time is None:
            return field
        history = (
            (as_datetime(field.valid_time) - as_datetime(previous.valid_time))
            .astype("timedelta64[ns]")
            .astype(np.int64)
        )
        ahead = (
            (as_datetime(time) - as_datetime(field.valid_time))
            .astype("timedelta64[ns]")
            .astype(np.int64)
        )
        if history <= 0 or ahead == 0:
            return field
        array = field.array + (field.array - previous.array) * (ahead / history)
        return replace(field, array=array, valid_time=as_datetime(time))

    def _apply_vertical(self, field: Field) -> Field:
        wanted = self.dst.import_vertical.get(field.standard_name)
        if wanted is None or wanted == field.vertical:
            return field
        if field.vertical is None or not isinstance(wanted, PressureLevels):
            raise VerticalMismatchError(
                f"Connector {self.name}: incompatible vertical coordinates"
            )
        surface_pressure = None
        if isinstance(field.vertical, HybridLevels):
            standard = self.src.dictionary.standard_name(field.vertical.ps_field)
            if standard not in self.src.export_state:
                raise VerticalMismatchError(
                    f"Connector {self.name}: hybrid interpolation needs {standard!r}"
                )
            surface_pressure = self.src.export_state[standard].array
        array = interp_to_pressure(
            field.array, field.vertical, wanted, surface_pressure
        )
        return replace(field, array=array, vertical=wanted)

    def _apply_fill(self, field: Field) -> Field:
        if field.mask is None or self.fill == "none":
            return field
        mask = field.mask.broadcast_like(field.array)
        if self.fill == "zero":
            return replace(field, array=field.array.where(mask, 0), mask=None)
        if field.array.dims[-2:] != ("lat", "lon"):
            raise IncompatibleFieldError(
                f"Connector {self.name}: nearest fill needs trailing lat/lon"
            )
        lat = np.asarray(field.array.coords["lat"])
        lon = np.asarray(field.array.coords["lon"])
        lat2d: np.ndarray
        lon2d: np.ndarray
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        spatial_size = lat2d.size
        data = field.array.data.reshape(-1, spatial_size)
        xp = _array_namespace(data)
        masks = np.asarray(mask).reshape(-1, spatial_size).astype(bool)
        rows = []
        for row, valid in zip(data, masks, strict=True):
            if not valid.any():
                raise IncompatibleFieldError("Mask fill impossible: no valid points")
            valid_flat = np.flatnonzero(valid)
            nearest = _nearest_point_indices(
                lat2d.ravel()[valid],
                lon2d.ravel()[valid],
                lat2d,
                lon2d,
            )
            rows.append(xp.take(row, xp.asarray(valid_flat[nearest]), axis=-1))
        filled = xp.stack(rows).reshape(field.array.shape)
        return replace(
            field,
            array=field.array.copy(data=filled),
            mask=None,
        )

    def _apply_regrid(self, field: Field) -> Field:
        destination = self.dst.grid_definition()
        if destination is None:
            return field
        try:
            source_fingerprint = field.grid_signature()
            same = source_fingerprint == (destination.fingerprint(),)
        except (AttributeError, ValueError):
            same = False
        if same and self._user_regridder is None:
            return field
        if self._user_regridder is not None:
            output = self._user_regridder(field.array)
            if not isinstance(output, xr.DataArray):
                raise TypeError("Custom regridder must return an xarray.DataArray")
            return replace(field, array=output)
        if isinstance(destination, PointGrid):
            return replace(field, array=self._sample(field.array, destination))
        coordinates = destination.coords()
        if "lat" not in coordinates or "lon" not in coordinates:
            raise IncompatibleFieldError(
                f"Connector {self.name}: automatic regrid supports lat/lon "
                "and point grids only; pass regridder="
            )
        lat = np.asarray(coordinates["lat"])
        lon = np.asarray(coordinates["lon"])
        if lat.ndim != 1 or lon.ndim != 1:
            raise IncompatibleFieldError(
                f"Connector {self.name}: curvilinear destination needs regridder="
            )
        return replace(
            field,
            array=_regular_latlon_interpolate(
                field.array, *np.meshgrid(lat, lon, indexing="ij"), ("lat", "lon")
            ),
        )

    def _sample(self, array: xr.DataArray, grid: PointGrid) -> xr.DataArray:
        if self.sample is None:
            raise CouplingError(
                f"Connector {self.name}: point destination requires sample="
            )
        if "lat" not in array.coords or "lon" not in array.coords:
            raise IncompatibleFieldError(
                f"Connector {self.name}: point sampling requires lat/lon source"
            )
        if self.sample == "nearest":
            return _sample_nearest(array, grid)
        return _regular_latlon_interpolate(
            array, grid.latitude, grid.longitude, ("x",)
        ).assign_coords(x=grid.x)

    def execute(self, time: np.datetime64) -> None:
        matched = self.match()
        if self.window is not None:
            self._execute_windowed(as_datetime(time))
            return
        for name in matched:
            if name in self._derived.values():
                continue
            self._deliver(self._apply_time_policy(self._source_field(name), time))

    def _source_field(self, name: str) -> Field:
        if name not in self.src.export_state:
            raise CouplingError(
                f"Connector {self.name}: source has not produced {name!r}"
            )
        return self.src.export_state[name]

    def _deliver(self, field: Field) -> None:
        field = self._apply_vertical(field)
        field = self._apply_fill(field)
        field = self._apply_regrid(field)
        self.dst.import_state.add(field)
        self.last_transfer[field.standard_name] = field

    def _execute_windowed(self, time: np.datetime64) -> None:
        if self.window is None or self.reduce is None:
            raise CouplingError(
                f"Connector {self.name}: windowed execution is not configured"
            )
        for base in self._derived:
            field = self._source_field(base)
            if self._origin is None:
                self._origin = (
                    as_datetime(field.valid_time)
                    if field.valid_time is not None
                    else time
                )
            self._reduction.add(base, field, self.reduce)
        elapsed: np.int64 = (
            (time - self._origin).astype("timedelta64[ns]").astype(np.int64)
        )
        if elapsed <= 0 or elapsed % self.window.astype(np.int64):
            return
        for base, derived in self._derived.items():
            array = self._reduction.emit(base, self.reduce)
            entry = self.dst.dictionary.resolve(derived)
            self._deliver(
                Field(
                    array,
                    derived,
                    entry.canonical_units,
                    valid_time=time,
                    source=self.src.name,
                )
            )
        self._reduction.reset()

    def reset(self) -> None:
        self._history.clear()
        self.last_transfer.clear()
        self._reduction.reset()
        self._origin = None

    def __repr__(self) -> str:
        fields = self._matched or self._fields or "auto"
        window = (
            f", window={fmt_timedelta(self.window)!r}, reduce={self.reduce!r}"
            if self.window is not None
            else ""
        )
        return f"Connector({self.name}, fields={fields}{window})"
