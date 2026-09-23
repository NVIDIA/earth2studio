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

from collections.abc import Generator, Iterator
from copy import deepcopy
from typing import Protocol, cast

import numpy as np
import torch
import xarray as xr
from earth2studio.models._array_utils import _registered_grid

from earth2studio.grids import infer_grid
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordinateSystem


def _convert_to_2d(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if lat.ndim == lon.ndim == 1:
        return np.meshgrid(lat, lon, indexing="ij")
    return lat, lon


def _can_concat_directly(px: CoordinateSystem, dx: CoordinateSystem) -> bool:
    return (
        px.dims == dx.dims
        and all(
            d == "variable"
            or (
                px.sizes[d] == dx.sizes[d]
                and np.array_equal(px.coords[d], dx.coords[d])
            )
            for d in dx.dims
        )
        and all(
            k in px.coords and np.array_equal(px.coords[k], dx.coords[k])
            for k in ("lat", "lon")
            if k in dx.coords
        )
    )


def _can_concat_with_subregion(px: CoordinateSystem, dx: CoordinateSystem) -> bool:
    if px.dims != dx.dims or not {"lat", "lon"}.issubset(px.dims):
        return False
    for dim in px.dims:
        if dim == "variable":
            continue
        if dim not in ("lat", "lon"):
            if not np.array_equal(px.coords[dim], dx.coords[dim]):
                return False
        else:
            indices = px.get_index(dim).get_indexer(dx.coords[dim].values)
            if not len(indices) or np.any(indices < 0) or np.any(np.diff(indices) != 1):
                return False
    return True


class PrepareInputCoordsDefault:
    """Plan variable selection and geographic interpolation without field allocation."""

    def __call__(
        self, px_coords: CoordinateSystem, dx_coords: CoordinateSystem
    ) -> CoordinateSystem:
        if "variable" not in dx_coords.dims:
            return coord_array_like(px_coords).copy(deep=True)
        spatial = tuple(dx_coords.attrs.get("dims", dx_coords.dims[-2:]))
        source_spatial = tuple(px_coords.attrs.get("dims", px_coords.dims[-2:]))
        same_grid = spatial == source_spatial and all(
            k in px_coords.coords and np.array_equal(px_coords.coords[k], v)
            for k, v in dx_coords.coords.items()
            if set(v.dims).intersection(spatial)
        )
        if same_grid:
            result = coord_array_like(
                px_coords, {"variable": dx_coords.coords["variable"].variable}
            ).copy(deep=True)
            # A nested declaration may require the registered identity of this exact grid.
            for key in ("earth2studio_grid_id", "earth2studio_crs"):
                if key in dx_coords.attrs:
                    result.attrs[key] = deepcopy(dx_coords.attrs[key])
        else:
            grid = _registered_grid(infer_grid(dx_coords))
            removed = {*source_spatial, "variable"}
            coords = {
                k: v.variable.copy(deep=True)
                for k, v in px_coords.coords.items()
                if not removed.intersection(v.dims)
            }
            coords["variable"] = dx_coords.coords["variable"].variable.copy(deep=True)
            attrs = deepcopy(px_coords.attrs)
            for key in (
                "earth2studio_grid_id",
                "earth2studio_crs",
                "dims",
                "shape",
                "topology",
                "type",
                "crs",
            ):
                attrs.pop(key, None)
            leading = px_coords.dims[: -len(source_spatial)]
            result = coord_array(
                (*leading, *spatial),
                coords,
                grid=grid,
                sizes={d: px_coords.sizes[d] for d in leading if d != "variable"},
                dynamic=px_coords.attrs.get("earth2studio_dynamic_dims", ()),
                attrs=attrs,
                dtype=px_coords.dtype,
                name=px_coords.name,
            )
        result.encoding = deepcopy(px_coords.encoding)
        return result


class PrepareInputTensorDefault:
    """Select labelled variables and interpolate only when geographic grids differ."""

    def __init__(self) -> None:
        self.interp: torch.nn.Module | None = None
        self._grids: tuple[np.ndarray, ...] | None = None

    @torch.inference_mode()
    def __call__(self, x: xr.DataArray, dx_coords: CoordinateSystem) -> xr.DataArray:
        signature = PrepareInputCoordsDefault()(x, dx_coords)
        if "variable" not in dx_coords.dims:
            return x.copy(deep=True)
        selected = x.sel(variable=dx_coords.coords["variable"].values)
        grids = tuple(
            np.asarray(a)
            for a in (
                *_convert_to_2d(x.lat.values, x.lon.values),
                *_convert_to_2d(signature.lat.values, signature.lon.values),
            )
        )
        tensor = selected.e2s.to_torch()[0]
        if not (
            np.array_equal(grids[0], grids[2]) and np.array_equal(grids[1], grids[3])
        ):
            if (
                self.interp is None
                or self._grids is None
                or any(not np.array_equal(a, b) for a, b in zip(grids, self._grids))
            ):
                self.interp = LatLonInterpolation(*grids)
                self._grids = tuple(a.copy() for a in grids)
            tensor = self.interp.to(tensor.device)(tensor)
        result = from_torch(tensor.clone(), signature)
        result.encoding = deepcopy(x.encoding)
        return result


class PrepareOutputCoordsDefault:
    """Plan concatenation on the final diagnostic grid, with an optional base crop."""

    def __call__(
        self, px_coords: CoordinateSystem, dx_coords: list[CoordinateSystem]
    ) -> CoordinateSystem:
        target = dx_coords[-1]
        if not all(_can_concat_directly(c, target) for c in dx_coords):
            raise ValueError("Diagnostic output grids and dimensions must match")
        sources = dx_coords
        if _can_concat_directly(px_coords, target) or _can_concat_with_subregion(
            px_coords, target
        ):
            sources = [px_coords, *sources]
        result = coord_array_like(
            target,
            {
                "variable": np.concatenate(
                    [c.coords["variable"].values for c in sources]
                )
            },
        ).copy(deep=True)
        result.encoding = deepcopy(target.encoding)
        return result


class PrepareOutputTensorDefault(torch.nn.Module):
    """Concatenate diagnostics and, where compatible, the base field or its crop."""

    @torch.inference_mode()
    def forward(self, px_x: xr.DataArray, dx_x: list[xr.DataArray]) -> xr.DataArray:
        """Combine compatible fields on the final diagnostic's labelled grid."""
        signature = PrepareOutputCoordsDefault()(px_x, dx_x)
        target = dx_x[-1]
        sources = dx_x
        if _can_concat_directly(px_x, target):
            sources = [px_x, *sources]
        elif _can_concat_with_subregion(px_x, target):
            sources = [px_x.sel(lat=target.lat, lon=target.lon), *sources]
        tensors = [x.e2s.to_torch()[0] for x in sources]
        device = tensors[-1].device
        result = from_torch(
            torch.cat(
                [t.to(device) for t in tensors], dim=target.get_axis_num("variable")
            ),
            signature,
        )
        result.encoding = deepcopy(target.encoding)
        return result


class PrepareDxInputCoords(Protocol):
    """Signature-only preparation of diagnostic input."""

    def __call__(
        self, px_coords: CoordinateSystem, dx_coords: CoordinateSystem
    ) -> CoordinateSystem: ...


class PrepareDxInputTensor(Protocol):
    """Preparation of a diagnostic's labelled input field."""

    def __call__(
        self, x: xr.DataArray, dx_coords: CoordinateSystem
    ) -> xr.DataArray: ...


class PrepareOutputCoords(Protocol):
    """Signature-only preparation of composed output."""

    def __call__(
        self, px_coords: CoordinateSystem, dx_coords: list[CoordinateSystem]
    ) -> CoordinateSystem: ...


class PrepareOutputTensor(Protocol):
    """Preparation of the composed labelled output field."""

    def __call__(
        self, px_x: xr.DataArray, dx_x: list[xr.DataArray]
    ) -> xr.DataArray: ...


class DiagnosticWrapper(torch.nn.Module, DataArrayPrognosticMixin):
    """Compose a native DataArray prognostic with one or more diagnostics.

    Preparation callables customize signature planning, interpolation and output
    concatenation. Tensor-named preparation slots now consume labelled fields.
    The nested prognostic iterator retains ownership of its history and checkpoints.
    """

    def __init__(
        self,
        px_model: PrognosticModel,
        dx_model: DiagnosticModel | list[DiagnosticModel],
        prepare_dx_input_coords: (
            PrepareDxInputCoords | list[PrepareDxInputCoords] | None
        ) = None,
        prepare_dx_input_tensor: (
            PrepareDxInputTensor | list[PrepareDxInputTensor] | None
        ) = None,
        prepare_output_coords: PrepareOutputCoords | None = None,
        prepare_output_tensor: PrepareOutputTensor | None = None,
    ) -> None:
        super().__init__()
        self.px_model = px_model
        self.dx_model = torch.nn.ModuleList(
            dx_model if isinstance(dx_model, list) else [dx_model]
        )
        if not len(self.dx_model):
            raise ValueError("At least one diagnostic is required")
        count = len(self.dx_model)
        self.prepare_dx_input_coords = (
            [PrepareInputCoordsDefault() for _ in range(count)]
            if prepare_dx_input_coords is None
            else (
                prepare_dx_input_coords
                if isinstance(prepare_dx_input_coords, list)
                else [prepare_dx_input_coords] * count
            )
        )
        self.prepare_dx_input_tensor = (
            [PrepareInputTensorDefault() for _ in range(count)]
            if prepare_dx_input_tensor is None
            else (
                prepare_dx_input_tensor
                if isinstance(prepare_dx_input_tensor, list)
                else [prepare_dx_input_tensor] * count
            )
        )
        for name in ("prepare_dx_input_coords", "prepare_dx_input_tensor"):
            if len(getattr(self, name)) != count:
                raise ValueError(
                    f"Length of {name} must match number of diagnostic models"
                )
        self.prepare_output_coords = (
            prepare_output_coords or PrepareOutputCoordsDefault()
        )
        self.prepare_output_tensor = (
            prepare_output_tensor or PrepareOutputTensorDefault()
        )

    @property
    def front_hook_interval(self) -> int:  # type: ignore[override]
        return getattr(self.px_model, "front_hook_interval", 1)

    def input_coords(self) -> CoordinateSystem:
        """Return the nested prognostic's allocation-free input declaration."""
        return self.px_model.input_coords().copy(deep=True)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Compose nested signature transformations without evaluating fields."""
        px = self.px_model.output_coords(input_coords)
        dx = [
            m.output_coords(p(px.copy(deep=True), m.input_coords()))
            for m, p in zip(self.dx_model, self.prepare_dx_input_coords)
        ]
        return self.prepare_output_coords(px, dx)

    def _diagnose(self, x: xr.DataArray) -> xr.DataArray:
        outputs = [
            m(p(x.copy(deep=True), m.input_coords()))
            for m, p in zip(self.dx_model, self.prepare_dx_input_tensor)
        ]
        return self.prepare_output_tensor(x, outputs)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Advance the nested model once, then diagnose its labelled output."""
        return self._diagnose(self.px_model(x.copy(deep=True)))

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        iterator = self.px_model.create_iterator(x.copy(deep=True))
        try:
            first = True
            while True:
                front = getattr(self.px_model, "front_hook")
                had_front = "front_hook" in vars(self.px_model)
                advanced = False

                def apply_front(state: xr.DataArray) -> xr.DataArray:
                    nonlocal advanced
                    advanced = True
                    return self.front_hook(front(state).copy(deep=True))

                setattr(self.px_model, "front_hook", apply_front)
                try:
                    px = next(iterator)
                finally:
                    if had_front:
                        setattr(self.px_model, "front_hook", front)
                    else:
                        delattr(self.px_model, "front_hook")
                initial = (
                    first
                    and not advanced
                    and np.array_equal(px.lead_time.values, x.lead_time.values[-1:])
                )
                first = False
                if initial:
                    yield px.copy(deep=True)
                    continue
                yield self.rear_hook(self._diagnose(px)).copy(deep=True)
        finally:
            cast(Generator[xr.DataArray, None, None], iterator).close()

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the initial field, followed by diagnosed forecasts with hooks."""
        yield from self._default_generator(x)
