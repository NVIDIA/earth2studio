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

from collections.abc import Callable, Generator, Iterator
from copy import deepcopy
from datetime import datetime
from typing import cast

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils import coord_array
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from physicsnemo import Module as PhysicsNemoModule
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("interp-modafno")
    PhysicsNemoModule = None
    cos_zenith_angle = None


VARIABLES = [
    "u10m",
    "v10m",
    "u100m",
    "v100m",
    "t2m",
    "sp",
    "msl",
    "tcwv",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
]


@check_optional_dependencies()
class InterpModAFNO(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """ModAFNO interpolation for global prognostic models. Interpolates a forecast model
    to a shorter time-step size (by default from 6 to 1 hour). Operates on 0.25 degree
    lat-lon equirectangular grid with 73 variables.

    Note
    ----
    For more information on the model, please refer to:

    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/earth-2/models/afno_dx_fi-v1-era5
    - https://arxiv.org/abs/2410.18904

    Warning
    -------
    The model requires a base forecast model to be set before execution. This can be
    done by setting the `px_model` attribute or using the `load_model` method.

    Parameters
    ----------
    interp_model : torch.nn.Module
        The interpolation model that performs the time interpolation
    center : torch.Tensor
        Model center normalization tensor
    scale : torch.Tensor
        Model scale normalization tensors
    geop : torch.Tensor
        Geopotential height data used as a static feature
    lsm : torch.Tensor
        Land-sea mask data used as a static feature
    px_model : PrognosticModel, optional
        The base forecast model that produces the coarse time resolution forecasts. If
        not provide, should be set by the user before executing the model, by default
        None.
    num_interp_steps : int, optional
        Number of interpolation steps to perform between forecast steps, by default 6
    prepare_endpoint : Callable[[xr.DataArray], xr.DataArray], optional
        Prepare the left interpolation endpoint without advancing time, for example
        by computing diagnostics absent from the base input. Applied after front
        hooks, including restored history. The public initial yield is unchanged.

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos year:2024 gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        interp_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
        geop: torch.Tensor,
        lsm: torch.Tensor,
        px_model: PrognosticModel | None = None,
        num_interp_steps: int = 6,
        prepare_endpoint: Callable[[xr.DataArray], xr.DataArray] | None = None,
    ) -> None:
        super().__init__()
        self.px_model = px_model
        self.interp_model = interp_model
        self.num_interp_steps = num_interp_steps
        self.prepare_endpoint = prepare_endpoint
        if num_interp_steps < 1:
            raise ValueError("num_interp_steps must be positive")
        self.variables = np.array(VARIABLES)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("geop", geop)
        self.register_buffer("lsm", lsm)

    @property
    def front_hook_interval(self) -> int:  # type: ignore[override]
        return self.num_interp_steps * getattr(self.px_model, "front_hook_interval", 1)

    @staticmethod
    def _load_feature_from_file(fn: str, var: str) -> torch.Tensor:
        """Load a feature from a NetCDF file.

        Parameters
        ----------
        fn : str
            Path to the NetCDF file
        var : str
            Variable name to load from the file

        Returns
        -------
        torch.Tensor
            Loaded feature as a tensor with shape (1, 1, H, W)
        """
        with xr.open_dataset(fn) as ds:
            x = np.array(ds[var])
        return torch.Tensor(x).unsqueeze(0).unsqueeze(0)

    def _compute_latlon(self) -> None:
        # compute sin/cos of lat/lon
        coords = self.output_coords(self.input_coords())
        lat = np.deg2rad(coords["lat"])
        lon = np.deg2rad(coords["lon"])
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
        sincos_latlon = torch.Tensor(
            np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)], axis=0)
        ).unsqueeze(0)
        self.register_buffer("lat", torch.as_tensor(lat, device=self.center.device))
        self.register_buffer("lon", torch.as_tensor(lon, device=self.center.device))
        self.register_buffer(
            "sincos_latlon", torch.as_tensor(sincos_latlon, device=self.center.device)
        )

    def __str__(self) -> str:
        return "InterpModAFNO"

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        # Getter / Setters don't work with torch.nn.Module, need to check manually here
        if self.px_model is None:
            raise ValueError("Base forecast model, px_model, must be set")
        signature = self.px_model.input_coords().copy(deep=True)
        if "time" not in signature.dims:
            dims = list(signature.dims)
            dynamic = list(signature.attrs.get("earth2studio_dynamic_dims", ()))
            position = len(dynamic)
            dims.insert(position, "time")
            dynamic.append("time")
            signature = coord_array(
                dims,
                dict(signature.coords),
                sizes=dict(signature.sizes),
                dynamic=dynamic,
                attrs=deepcopy(signature.attrs),
                dtype=signature.dtype,
                name=signature.name,
            )
        return signature

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model
        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.
        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        self.input_coords()
        if self.px_model is None:
            raise ValueError("Base forecast model, px_model, must be set")
        coarse = self.px_model.output_coords(input_coords)
        final = input_coords.coords["lead_time"].values[-1:]
        delta = (coarse.coords["lead_time"].values[-1:] - final).astype(
            "timedelta64[ns]"
        )
        if (delta <= np.timedelta64(0, "ns")).any() or (
            delta.astype(np.int64) % self.num_interp_steps
        ).any():
            raise ValueError(
                "Coarse forecast interval must divide into positive interpolation steps"
            )
        return self._prediction_coords(coarse, final + delta // self.num_interp_steps)

    def _prediction_coords(self, x: xr.DataArray, lead: np.ndarray) -> CoordinateSystem:
        lead = lead.astype("timedelta64[ns]")
        # The interpolation network uses the north-pole-inclusive 720-row grid.
        target = coord_array(
            ("variable", "lat", "lon"),
            {"variable": self.variables},
            grid="latlon-0.25deg-south-pole-excluded",
        )
        for dim in ("variable", "lat", "lon"):
            if (
                dim not in x.dims
                or not np.isin(target.coords[dim], x.coords[dim]).all()
            ):
                raise ValueError(f"Base forecast is missing interpolation {dim} labels")
        changed = {"lead_time", "variable"}
        if not np.array_equal(x.lat, target.lat) or not np.array_equal(
            x.lon, target.lon
        ):
            changed.update(("lat", "lon"))
        attrs = deepcopy(x.attrs)
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
        coords = {
            k: v.variable.copy(deep=True)
            for k, v in x.coords.items()
            if not changed.intersection(v.dims)
        }
        coords.update(
            variable=self.variables,
            lead_time=x.coords["lead_time"].variable.copy(deep=True, data=lead),
        )
        if "time" in x.coords:
            time = x.coords["time"]
            valid = time + xr.DataArray(lead, dims="lead_time")
            coords["valid_time"] = valid.variable
        result = coord_array(
            x.dims,
            coords,
            grid="latlon-0.25deg-south-pole-excluded",
            dynamic=x.attrs.get("earth2studio_dynamic_dims", ()),
            sizes={
                d: x.sizes[d]
                for d in x.dims
                if d not in ("lead_time", "variable", "lat", "lon")
            },
            attrs=attrs,
            name=x.name,
            dtype=x.dtype,
        )
        result.encoding = deepcopy(x.encoding)
        return result

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "ngc://models/nvidia/earth-2/afno_dx_fi-v1-era5@v0.1.0",
            cache_options={
                "cache_storage": Package.default_cache("modafno_interpolation"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls, package: Package, px_model: PrognosticModel | None = None
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        px_model : PrognosticModel | None, optional
            The base forecast model that produces the coarse time resolution forecasts.
            If None, should be set manually, by default None

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        model = PhysicsNemoModule.from_checkpoint(
            package.resolve("fcinterp-modafno-2x2.mdlus")
        )
        model.eval()

        # Load center and std normalizations
        local_center = torch.Tensor(np.load(package.resolve("global_means.npy")))[
            :, :73
        ]
        local_std = torch.Tensor(np.load(package.resolve("global_stds.npy")))[:, :73]

        # load static variables
        geop = cls._load_feature_from_file(package.resolve("orography.nc"), "Z")[
            0, :, :, :720, :
        ]
        geop = (geop - geop.mean()) / geop.std()
        lsm = cls._load_feature_from_file(package.resolve("land_sea_mask.nc"), "LSM")[
            0, :, :, :720, :
        ]

        return cls(
            model,
            center=local_center,
            scale=local_std,
            geop=geop,
            lsm=lsm,
            px_model=px_model,
        )

    def _cos_zenith(self, times: list) -> torch.Tensor:
        """Calculate cosine of zenith angle for given times.

        Parameters
        ----------
        times : list
            List of times to calculate cosine of zenith angle for

        Returns
        -------
        torch.Tensor
            Cosine of zenith angle for each time
        """
        # Convert generator to list to fix type incompatibility
        times_list = list(datetime.fromisoformat(str(t)[:19]) for t in times)
        cos_zen = [
            cos_zenith_angle(t, self.lon.cpu().numpy(), self.lat.cpu().numpy())
            for t in times_list
        ]
        return torch.Tensor(np.stack(cos_zen, axis=0)).unsqueeze(0)

    @torch.inference_mode()
    def _interpolate(
        self,
        x0: xr.DataArray,
        x1: xr.DataArray,
    ) -> Generator[xr.DataArray, None, None]:
        """Interpolate between two forecast steps.

        Parameters
        ----------
        x0 : xr.DataArray
            First forecast step
        x1 : xr.DataArray
            Second forecast step

        Yields
        ------
        Generator[xr.DataArray, None, None]
            Labelled interpolated forecast steps.
        """
        if not hasattr(self, "sincos_latlon"):
            self._compute_latlon()
        left = (x0.e2s.to_torch()[0].to(self.center.device) - self.center) / self.scale
        right = (x1.e2s.to_torch()[0].to(self.center.device) - self.center) / self.scale
        shape = left.shape
        left, right = left.reshape(-1, *shape[-3:]), right.reshape(-1, *shape[-3:])
        leading = x0.dims[:-3]
        template = xr.DataArray(np.empty(shape[:-3]), dims=leading)
        if "time" not in x0.coords:
            raise ValueError("Interpolation requires a time coordinate")
        t0 = (
            (x0.time + x0.lead_time)
            .broadcast_like(template)
            .transpose(*leading)
            .values.reshape(-1)
        )
        t1 = (
            (x1.time + x1.lead_time)
            .broadcast_like(template)
            .transpose(*leading)
            .values.reshape(-1)
        )
        delta = (x1.lead_time.values - x0.lead_time.values) // self.num_interp_steps
        # Preserve ensemble batching even when time is an auxiliary coordinate
        # or members occupy several leading dimensions. Group both endpoints.
        groups: dict[tuple[int, int], list[int]] = {}
        for index, pair in enumerate(
            zip(
                t0.astype("datetime64[ns]").astype(np.int64),
                t1.astype("datetime64[ns]").astype(np.int64),
            )
        ):
            groups.setdefault(pair, []).append(index)
        for interp_step in range(1, self.num_interp_steps):
            signature = self._prediction_coords(
                x0, x0.lead_time.values + interp_step * delta
            )
            out = torch.empty_like(left)
            for members in groups.values():
                index = members[0]
                indices = torch.tensor(members, device=left.device)
                batch_size = len(members)
                t_ip = (
                    t0[index]
                    + interp_step * (t1[index] - t0[index]) // self.num_interp_steps
                )
                cos_zen = self._cos_zenith([t0[index], t1[index], t_ip]).to(
                    self.center.device
                )
                features = torch.cat(
                    [
                        left[indices],
                        right[indices],
                        cos_zen.expand(batch_size, -1, -1, -1),
                        self.sincos_latlon.expand(batch_size, -1, -1, -1),
                        self.geop.expand(batch_size, -1, -1, -1),
                        self.lsm.expand(batch_size, -1, -1, -1),
                    ],
                    dim=1,
                )
                t_norm = torch.tensor(
                    [interp_step / self.num_interp_steps], device=self.center.device
                )
                out[indices] = self.interp_model(features, t_norm)
            result = from_torch(
                out.reshape(shape) * self.scale + self.center, signature
            )
            result.encoding = deepcopy(x0.encoding)
            yield result

    def _select_prediction_grid(self, x: xr.DataArray) -> xr.DataArray:
        signature = self._prediction_coords(x, x.lead_time.values)
        return x.sel(
            {dim: signature.coords[dim].values for dim in ("variable", "lat", "lon")}
        )

    def _prepare_left_endpoint(self, x: xr.DataArray) -> xr.DataArray:
        if self.prepare_endpoint is not None:
            x = self.prepare_endpoint(x.copy(deep=True))
        return self._select_prediction_grid(x)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Return the first interpolated forecast from a labelled history, without hooks."""
        self.output_coords(x)
        if self.px_model is None:
            raise ValueError("Base forecast model, px_model, must be set")
        coarse = self._select_prediction_grid(self.px_model(x.copy(deep=True)))
        if self.num_interp_steps == 1:
            result = from_torch(
                coarse.e2s.to_torch()[0].to(self.center.device).clone(),
                self.output_coords(x),
            )
            result.encoding = deepcopy(coarse.encoding)
            return result
        initial = x.isel(lead_time=slice(-1, None))
        initial = self._prepare_left_endpoint(initial)
        gen = self._interpolate(initial, coarse)
        try:
            return next(gen)
        finally:
            gen.close()

    def _default_generator(
        self, x: xr.DataArray, hooks: bool = True
    ) -> Generator[xr.DataArray, None, None]:

        if self.px_model is None:
            raise ValueError(
                "Base forecast model, px_model, must be set before executing the model."
            )

        self.output_coords(x)
        iterator = self.px_model.create_iterator(x.copy(deep=True))
        try:
            first = True
            x0: xr.DataArray | None = None
            while True:
                front = getattr(self.px_model, "front_hook")
                had_front = "front_hook" in vars(self.px_model)
                advanced = False

                def apply_front(state: xr.DataArray) -> xr.DataArray:
                    nonlocal x0, advanced
                    advanced = True
                    state = front(state).copy(deep=True)
                    if hooks:
                        state = self.front_hook(state)
                    latest = state.isel(lead_time=slice(-1, None))
                    x0 = self._prepare_left_endpoint(latest).copy(deep=True)
                    return state

                setattr(self.px_model, "front_hook", apply_front)
                try:
                    x1 = next(iterator)
                finally:
                    if had_front:
                        setattr(self.px_model, "front_hook", front)
                    else:
                        delattr(self.px_model, "front_hook")
                initial = (
                    first
                    and not advanced
                    and np.array_equal(x1.lead_time.values, x.lead_time.values[-1:])
                )
                first = False
                if initial:
                    yield x1.copy(deep=True)
                    continue
                x1 = self._select_prediction_grid(x1)
                if self.num_interp_steps > 1:
                    if x0 is None:
                        raise ValueError(
                            "Resumed interpolation requires the restored left endpoint through the nested front hook"
                        )
                    for output in self._interpolate(x0, x1):
                        yield (self.rear_hook(output) if hooks else output).copy(
                            deep=True
                        )
                output = from_torch(
                    x1.e2s.to_torch()[0].to(self.center.device).clone(),
                    self._prediction_coords(x1, x1.lead_time.values),
                )
                output.encoding = deepcopy(x1.encoding)
                x0 = (self.rear_hook(output) if hooks else output).copy(deep=True)
                yield x0.copy(deep=True)
        finally:
            cast(Generator[xr.DataArray, None, None], iterator).close()

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the latest input before computing interpolated forecasts with hooks."""
        yield from self._default_generator(x)
