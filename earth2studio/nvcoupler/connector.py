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

"""Connector: moves Fields from one component's exports to another's imports.

The NUOPC_Connector analog. Matching is by standard name; each transfer runs
a pipeline of (1) time policy, (2) vertical interpolation when source and
destination vertical coordinates differ, (3) mask fill, (4) spatial regrid
onto the destination grid. Regridders and mask fillers are built lazily and
cached per grid signature (the dxwrapper.py pattern). All tensor math is
torch, so autograd graphs survive the exchange.
"""

from collections import OrderedDict
from dataclasses import replace
from typing import Callable, Literal

import numpy as np
import torch
from loguru import logger

from earth2studio.utils.interp import latlon_interpolation_regular
from earth2studio.utils.type import CoordSystem

from .clock import DeltaLike, as_datetime, as_timedelta, fmt_timedelta
from .component import Component
from .errors import (
    CouplingError,
    IncompatibleFieldError,
    VerticalMismatchError,
)
from .field import _SPATIAL_DIMS, Field
from .mediator import _RunningReduction
from .vertical import HybridLevels, PressureLevels, interp_to_pressure

Regridder = Callable[[torch.Tensor], torch.Tensor]


def _is_regular(v: np.ndarray) -> bool:
    return v.ndim == 1 and len(v) > 1 and np.allclose(np.diff(v), v[1] - v[0])


def _build_latlon_regridder(src: CoordSystem, dst: CoordSystem) -> Regridder:
    """Bilinear lat/lon regridder using earth2studio's regular-grid kernel.

    Requires 1D equally-spaced source lat/lon (the common case for global
    models); index clamping at the grid edge stands in for extrapolation.
    """
    src_lat, src_lon = np.asarray(src["lat"]), np.asarray(src["lon"])
    if not (_is_regular(src_lat) and _is_regular(src_lon)):
        raise IncompatibleFieldError(
            "Auto regrid requires a regular 1D source lat/lon grid; pass a "
            "custom regridder=... to the Connector for curvilinear or "
            "unstructured source grids"
        )
    flip_lat = src_lat[0] > src_lat[-1]
    lat0 = torch.as_tensor(src_lat[::-1].copy() if flip_lat else src_lat)
    lon0 = torch.as_tensor(src_lon)
    lat1g, lon1g = np.meshgrid(
        np.asarray(dst["lat"]), np.asarray(dst["lon"]), indexing="ij"
    )
    lat1 = torch.as_tensor(lat1g)
    lon1 = torch.as_tensor(lon1g)

    def regrid(data: torch.Tensor) -> torch.Tensor:
        if flip_lat:
            data = torch.flip(data, dims=(-2,))
        return latlon_interpolation_regular(
            data,
            lat0.to(device=data.device, dtype=data.dtype),
            lon0.to(device=data.device, dtype=data.dtype),
            lat1.to(device=data.device, dtype=data.dtype),
            lon1.to(device=data.device, dtype=data.dtype),
        )

    return regrid


def _build_mask_filler(coords: CoordSystem, mask: torch.Tensor) -> Regridder:
    """Nearest-valid fill on the source grid: every invalid point takes the
    value of its nearest valid neighbor (great-circle metric via unit-sphere
    KDTree). Differentiable (pure gather)."""
    from scipy.spatial import cKDTree

    lat = np.asarray(coords["lat"], dtype=np.float64)
    lon = np.asarray(coords["lon"], dtype=np.float64)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    phi, lam = np.deg2rad(lat2d).ravel(), np.deg2rad(lon2d).ravel()
    xyz = np.stack(
        [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=1
    )
    valid = mask.reshape(-1).cpu().numpy().astype(bool)
    if not valid.any():
        raise IncompatibleFieldError("Mask fill impossible: no valid source points")
    tree = cKDTree(xyz[valid])
    _, nearest = tree.query(xyz, k=1)
    valid_index = np.flatnonzero(valid)[nearest]
    index = torch.as_tensor(valid_index, dtype=torch.long)

    def fill(data: torch.Tensor) -> torch.Tensor:
        flat = data.reshape(*data.shape[:-2], -1)
        filled = torch.index_select(flat, -1, index.to(data.device))
        return filled.reshape(data.shape)

    return fill


class Connector:
    """Moves matched Fields src.exports -> dst.imports each time it executes.

    Parameters
    ----------
    src, dst : Component
    fields : list[str], optional
        Standard names to transfer; defaults to the intersection of src's
        advertised exports and dst's advertised imports.
    time_policy : "constant" | "linear"
        What the destination sees between source updates: hold the latest
        export (constant, the PhysicsNeMo ConstantCoupler behavior), or
        linearly extrapolate from the two most recent exports.
    fill : "none" | "zero" | "nearest"
        Treatment of masked (invalid) source points before regridding.
    regridder : callable, optional
        Override the spatial regrid for all fields of this connector
        (signature: tensor[..., H, W] -> tensor[..., H', W']). Required when
        the grids differ and the auto path cannot handle them (HEALPix
        'face' dims, curvilinear grids); identical grids — including
        identical face grids — pass through as identity without one.
    window, reduce : optional
        Set both to make this a *windowed* connector: each execute folds the
        source fields into a running reduction ("mean" | "sum" | "max" |
        "min"), and delivery happens only at execute times aligned to
        `window`. The delivered Field carries the DERIVED standard name — the
        destination must import a dictionary entry whose CellMethod is
        (base=source export, method=`reduce`, window=`window`). Between
        window boundaries the destination's previous import is untouched.
        The window origin is the valid_time of the first execute's source
        field (the clock start under lagged coupling, where the connector
        runs before the source in its slot), so no driver hook is needed.
        This replaces a single-source AccumulationMediator; `time_policy`
        does not apply on the windowed path.
    """

    def __init__(
        self,
        src: Component,
        dst: Component,
        fields: list[str] | None = None,
        time_policy: Literal["constant", "linear"] = "constant",
        fill: Literal["none", "zero", "nearest"] = "none",
        regridder: Regridder | None = None,
        window: DeltaLike | None = None,
        reduce: Literal["mean", "sum", "max", "min"] | None = None,
    ):
        self.src = src
        self.dst = dst
        self.time_policy = time_policy
        self.fill = fill
        if (window is None) != (reduce is None):
            raise CouplingError(
                f"Connector {src.name}->{dst.name}: window= and reduce= must "
                "be set together — a windowed reduction needs both the window "
                "length and the reduction method"
            )
        if reduce is not None and reduce not in ("mean", "sum", "max", "min"):
            raise CouplingError(
                f"Connector {src.name}->{dst.name}: unsupported reduce="
                f"{reduce!r}; choose 'mean', 'sum', 'max' or 'min'"
            )
        self.window = as_timedelta(window) if window is not None else None
        self.reduce = reduce
        self._reduction = _RunningReduction()
        self._derived: dict[str, str] = {}  # src export name -> derived dst name
        self._origin: np.datetime64 | None = None  # window alignment origin
        self._user_regridder = regridder
        self._fields = list(fields) if fields is not None else None
        self._matched: list[str] | None = None
        self._regridders: dict[tuple, Regridder] = {}
        self._fillers: dict[tuple, Regridder] = {}
        # 2-deep export history per field: (previous, latest), rotated only
        # when a genuinely new export (different valid_time) arrives
        self._history: dict[str, tuple[Field | None, Field]] = {}
        self._linear_warned: set[str] = set()
        self.last_transfer: dict[str, Field] = {}

    @property
    def name(self) -> str:
        return f"{self.src.name}->{self.dst.name}"

    # -- matching --------------------------------------------------------------
    def match(self) -> list[str]:
        """Resolve matched standard names (cached).

        Plain connectors match by name intersection. Windowed connectors
        match each source export against a destination import whose
        CellMethod derives from it; the returned list then contains both the
        consumed source names and the delivered derived names, so driver-side
        bookkeeping (fed imports, consumed exports) sees the full mapping.
        """
        if self._matched is not None:
            return self._matched
        _, src_exports = self.src.advertise()
        dst_imports, _ = self.dst.advertise()
        if self.window is not None:
            return self._match_windowed(src_exports, dst_imports)
        if self._fields is not None:
            missing = [
                f for f in self._fields if f not in src_exports or f not in dst_imports
            ]
            if missing:
                raise IncompatibleFieldError(
                    f"Connector {self.name}: fields {missing} are not in both "
                    f"{self.src.name!r} exports ({src_exports}) and "
                    f"{self.dst.name!r} imports ({dst_imports})"
                )
            self._matched = list(self._fields)
        else:
            self._matched = [n for n in dst_imports if n in src_exports]
        if not self._matched:
            raise IncompatibleFieldError(
                f"Connector {self.name}: no fields match — {self.src.name!r} "
                f"exports {src_exports}, {self.dst.name!r} imports {dst_imports}"
            )
        # units validation against the (shared) dictionary
        for n in self._matched:
            entry_src = self.src.dictionary.resolve(n)
            entry_dst = self.dst.dictionary.resolve(n)
            self.dst.dictionary.check_units(
                n, entry_src.canonical_units, src=self.src.name, dst=self.dst.name
            )
            del entry_dst
        return self._matched

    def _match_windowed(self, src_exports: list[str], dst_imports: list[str]) -> list[str]:
        """Pair source exports with the destination's derived imports.

        A source export `base` maps to a destination import whose dictionary
        entry carries CellMethod(base, self.reduce, self.window); the field
        is delivered under that derived name. No matching derived import is
        an error — the coupler never invents names.
        """
        wanted = self._fields if self._fields is not None else src_exports
        for name in dst_imports:
            cm = self.dst.dictionary.resolve(name).cell_method
            if (
                cm is not None
                and cm.method == self.reduce
                and as_timedelta(cm.window) == self.window
                and cm.base in wanted
                and cm.base in src_exports
            ):
                self._derived[cm.base] = name
        unmatched = [f for f in wanted if f not in self._derived]
        if not self._derived or (self._fields is not None and unmatched):
            w, r = fmt_timedelta(self.window), self.reduce
            raise CouplingError(
                f"Connector {self.name}: window={w!r}/reduce={r!r} is set but "
                f"{self.dst.name!r} imports no derived field for "
                f"{unmatched or src_exports} — register a "
                f"FieldEntry(cell_method=CellMethod(base, {r!r}, window={w!r})) "
                f"in the destination's dictionary and add its standard name to "
                f"{self.dst.name!r}'s imports (destination imports: "
                f"{dst_imports})"
            )
        for base, derived in self._derived.items():
            self.dst.dictionary.check_units(
                derived,
                self.src.dictionary.resolve(base).canonical_units,
                src=self.src.name,
                dst=self.dst.name,
            )
        self._matched = list(self._derived) + list(self._derived.values())
        return self._matched

    # -- pipeline stages ---------------------------------------------------------
    def _apply_time_policy(self, field: Field, time: np.datetime64) -> Field:
        prev, latest = self._history.get(field.standard_name, (None, None))
        # Rotate the (prev, latest) history only when the incoming export is
        # genuinely new (different valid_time); re-seeing the same export on
        # subsequent executes must not collapse the extrapolation baseline.
        is_new = (
            latest is None
            or field.valid_time is None
            or latest.valid_time is None
            or as_datetime(field.valid_time) != as_datetime(latest.valid_time)
        )
        if is_new:
            prev = latest
            self._history[field.standard_name] = (prev, field)
        if self.time_policy == "constant" or prev is None:
            return field
        if "lead_time" in field.coords or "window" in field.coords:
            # a lead-time-resolved field carries many valid times; a single
            # valid_time extrapolation is ill-defined for it
            if field.standard_name not in self._linear_warned:
                self._linear_warned.add(field.standard_name)
                logger.warning(
                    "Connector {}: time_policy='linear' is undefined for "
                    "field {!r} with a lead_time/window dimension — falling "
                    "back to 'constant' for it",
                    self.name,
                    field.standard_name,
                )
            return field
        if prev.valid_time is None or field.valid_time is None:
            return field
        dt_hist = (
            (as_datetime(field.valid_time) - as_datetime(prev.valid_time))
            .astype("timedelta64[ns]")
            .astype(np.int64)
        )
        if dt_hist <= 0:
            return field
        dt_ahead = (
            (as_datetime(time) - as_datetime(field.valid_time))
            .astype("timedelta64[ns]")
            .astype(np.int64)
        )
        if dt_ahead == 0:
            return field
        w = dt_ahead / dt_hist
        data = field.data + (field.data - prev.data) * w
        return replace(field, data=data, valid_time=as_datetime(time))

    def _apply_vertical(self, field: Field) -> Field:
        want = self.dst.import_vertical.get(field.standard_name)
        if want is None:
            return field
        have = field.vertical
        if have == want:
            return field
        if have is None:
            raise VerticalMismatchError(
                f"Connector {self.name}: {self.dst.name!r} expects "
                f"{field.standard_name!r} on {want}, but the source field has "
                "no vertical coordinate"
            )
        if not isinstance(want, PressureLevels):
            raise VerticalMismatchError(
                f"Connector {self.name}: only interpolation onto PressureLevels "
                f"is supported in v1 (destination wants {type(want).__name__})"
            )
        ps = None
        if isinstance(have, HybridLevels):
            ps_std = self.src.dictionary.standard_name(have.ps_field)
            if ps_std not in self.src.export_state:
                raise VerticalMismatchError(
                    f"Connector {self.name}: hybrid->pressure interpolation of "
                    f"{field.standard_name!r} needs {ps_std!r} in "
                    f"{self.src.name!r} exports — add it to the source's "
                    "export list"
                )
            ps = self.src.export_state[ps_std].data
        data, coords = interp_to_pressure(field.data, field.coords, have, want, ps)
        return replace(field, data=data, coords=coords, vertical=want)

    def _apply_fill(self, field: Field) -> Field:
        if field.mask is None or self.fill == "none":
            return field
        if self.fill == "zero":
            data = torch.where(field.mask.to(field.data.device), field.data, 0.0)
            return replace(field, data=data, mask=None)
        key = (field.grid_signature(), field.mask.cpu().numpy().tobytes())
        if key not in self._fillers:
            self._fillers[key] = _build_mask_filler(field.coords, field.mask)
        return replace(field, data=self._fillers[key](field.data), mask=None)

    def _apply_regrid(self, field: Field) -> Field:
        dst_grid = self.dst.grid_coords()
        if dst_grid is None:
            return field  # destination has no grid of its own (e.g. mediator)
        src_spatial = OrderedDict(
            (k, v) for k, v in field.coords.items() if k in _SPATIAL_DIMS
        )
        # Identity fast path: every spatial dim of the field (lat/lon, but
        # also HEALPix-style face/height/width) exists in the destination
        # grid with an equal coordinate array — nothing to regrid.
        same = bool(src_spatial) and all(
            k in dst_grid and np.array_equal(np.asarray(v), np.asarray(dst_grid[k]))
            for k, v in src_spatial.items()
        )
        if same and self._user_regridder is None:
            return field
        if ("face" in field.coords or "face" in dst_grid) and (
            self._user_regridder is None
        ):
            raise IncompatibleFieldError(
                f"Connector {self.name}: source and destination HEALPix "
                "'face' grids differ — pass a custom regridder= (e.g. built "
                "with earth2grid, see models/px/dlesym.py)"
            )
        if self._user_regridder is not None:
            # Explicit override: apply the user regridder to the trailing
            # spatial dims of ANY layout (lat/lon, HEALPix face/height/width,
            # curvilinear y/x) and rebuild coords from the destination grid.
            data = self._user_regridder(field.data)
            coords = OrderedDict(
                (k, v) for k, v in field.coords.items() if k not in _SPATIAL_DIMS
            )
            for k, v in dst_grid.items():
                coords[k] = np.asarray(v).copy()
            return replace(field, data=data, coords=coords)
        key = (field.grid_signature(),)
        if not (
            "lat" in src_spatial
            and "lon" in src_spatial
            and "lat" in dst_grid
            and "lon" in dst_grid
        ):
            raise IncompatibleFieldError(
                f"Connector {self.name}: auto regrid needs lat/lon on both "
                f"grids (source dims {list(src_spatial)}, destination dims "
                f"{list(dst_grid)}) — pass a custom regridder="
            )
        if key not in self._regridders:
            self._regridders[key] = _build_latlon_regridder(src_spatial, dst_grid)
        regrid = self._regridders[key]
        # regrid acts on the trailing two (lat, lon) dims
        spatial_last = list(field.coords)[-2:] == ["lat", "lon"]
        if not spatial_last:
            raise IncompatibleFieldError(
                f"Connector {self.name}: field {field.standard_name!r} must "
                f"have (lat, lon) as trailing dims, got {list(field.coords)}"
            )
        data = regrid(field.data)
        # preserve original dim order: everything except lat/lon, then dst grid
        coords = OrderedDict(
            (k, v) for k, v in field.coords.items() if k not in ("lat", "lon")
        )
        coords["lat"] = np.asarray(dst_grid["lat"]).copy()
        coords["lon"] = np.asarray(dst_grid["lon"]).copy()
        return replace(field, data=data, coords=coords)

    # -- execution ----------------------------------------------------------------
    def execute(self, time: np.datetime64) -> None:
        self.match()
        if self.window is not None:
            self._execute_windowed(as_datetime(time))
            return
        for name in self._matched:
            field = self._apply_time_policy(self._source_field(name), time)
            self._deliver(field)

    def _source_field(self, name: str) -> Field:
        if name not in self.src.export_state:
            raise CouplingError(
                f"Connector {self.name}: {self.src.name!r} has not produced "
                f"{name!r} yet — check the run sequence ordering"
            )
        return self.src.export_state[name]

    def _deliver(self, field: Field) -> None:
        """Run the spatial pipeline and hand the field to the destination."""
        field = self._apply_vertical(field)
        field = self._apply_fill(field)
        field = self._apply_regrid(field)
        self.dst.import_state.add(field)
        self.last_transfer[field.standard_name] = field
        logger.debug(
            "exchange {}: {} (valid {})",
            self.name,
            field.standard_name,
            field.valid_time,
        )

    def _execute_windowed(self, time: np.datetime64) -> None:
        """Fold sources into the running reduction; deliver on window boundaries.

        The origin for boundary alignment is the valid_time of the first
        execute's source field — under lagged coupling (connector before the
        source's RunAction in the slot) that is the clock start, so the first
        delivery lands exactly one window after t0 with no driver hook.
        """
        for base in self._derived:
            field = self._source_field(base)
            if self._origin is None:
                self._origin = (
                    as_datetime(field.valid_time)
                    if field.valid_time is not None
                    else time
                )
            self._reduction.add(base, field, self.reduce)
        elapsed_ns = (time - self._origin).astype("timedelta64[ns]").astype(np.int64)
        if elapsed_ns <= 0 or elapsed_ns % self.window.astype(np.int64) != 0:
            return  # mid-window: accumulate only, previous import stands
        for base, derived in self._derived.items():
            data, coords = self._reduction.emit(base, self.reduce)
            entry = self.dst.dictionary.resolve(derived)
            self._deliver(
                Field(
                    data=data,
                    coords=coords,
                    standard_name=derived,
                    units=entry.canonical_units,
                    valid_time=time,
                    source=self.src.name,
                )
            )
        self._reduction.reset()

    def reset(self) -> None:
        """Clear per-run exchange state (history, running reduction, probes)."""
        self._history.clear()
        self.last_transfer.clear()
        self._reduction.reset()
        self._origin = None

    def __repr__(self) -> str:
        fields = self._matched or self._fields or "auto"
        windowed = (
            f", window={fmt_timedelta(self.window)!r}, reduce={self.reduce!r}"
            if self.window is not None
            else ""
        )
        return (
            f"Connector({self.name}, fields={fields}, "
            f"time_policy={self.time_policy!r}, fill={self.fill!r}{windowed})"
        )
