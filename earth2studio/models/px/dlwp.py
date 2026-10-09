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

import zipfile
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.grids import PointGrid
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.checkpoint import bind_checkpoint_state
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordinateSystem

try:
    import physicsnemo
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("dlwp")
    physicsnemo = None
    cos_zenith_angle = None

VARIABLES = ["t850", "z1000", "z700", "z500", "z300", "tcwv", "t2m"]


@dataclass
class _DLWPCheckpointState:
    x: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    public_metadata: dict[str, Any] = field(default_factory=dict)
    pending: bool = False


@check_optional_dependencies()
class DLWP(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Deep learning weather prediction (DLWP)  prognostic model. This is a parsimonious
    global forecast model with a time-step size of 6 hours. The core model is a
    convolutional encoder-decoder trained on [64,64] cubed sphere data that has an input
    of 18 fields (2x7 atmos variables + 4 prescriptive) and outputs 14 fields (2x7 atmos
    variables). This implementation provides a wrapper that accepts [721,1440] lat-lon
    equirectangular grid of just the atmospheric varaibles as an input for better
    compatability with common data sources. Prescriptive fields are added inside the
    model wrapper.

    Attributes
    ----------
    front_hook_interval : int
        One forecast chunk per front hook. Each twelve-hour core advance returns
        two six-hour frames together. The rear hook transforms each chunk before
        publication; the front hook runs before subsequent advances.

    Note
    ----
    For more information about this model see:

    - https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002502
    - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_dlwp_cubesphere

    Parameters
    ----------
    core_model : torch.nn.Module
        Core cubed-sphere DLWP model.
    landsea_mask : torch.Tensor
        Land sea mask in cubed sphere form [6,64,64]
    orography : torch.Tensor
        Surface geopotential (orography) in cubed sphere form [6,64,64]
    latgrid : torch.Tensor
        Cubed sphere latitude coordinates [6,64,64]
    longrid : torch.Tensor
        Cubed sphere longitude coordinates [6,64,64]
    cubed_sphere_transform : torch.Tensor
        Sparse pytorch tensor to transform equirectangular fields to cubed sphere of
        size [24576, 1038240]
    cubed_sphere_inverse : torch.Tensor
        Sparse pytorch tensor to transform cubed sphere fields to equirectangular of
        size [1038240, 24576]
    center : torch.Tensor
        Model atmospheric variable center normalization tensor of size [1,7,1,1]
    scale : torch.Tensor
        Model atmospheric variable scale normalization tensor of size [1,7,1,1]

    Badges
    ------
    region:global class:subseasonal-seasonal product:temp product:atmos year:2021 gpu:40gb
    provider:nvidia backend:pytorch
    """

    front_hook_interval: int = 1

    def __init__(
        self,
        core_model: torch.nn.Module,
        landsea_mask: torch.Tensor,
        orography: torch.Tensor,
        latgrid: torch.Tensor,
        longrid: torch.Tensor,
        cubed_sphere_transform: torch.Tensor,
        cubed_sphere_inverse: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ):

        super().__init__()
        self.model = core_model
        self.register_buffer("latgrid", latgrid)
        self.register_buffer("longrid", longrid)
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.register_buffer("landsea_mask", landsea_mask.unsqueeze(0))
        self.register_buffer(
            "topographic_height",
            (orography.unsqueeze(0).unsqueeze(0) - 3.724e03) / 8.349e03,
        )
        self.register_buffer("M", cubed_sphere_transform.T)
        self.register_buffer("N", cubed_sphere_inverse)
        # Six cubed-sphere faces are not HEALPix. Flatten their exact checkpoint
        # ordering into a point grid for labelled internal state and hooks.
        self._cube_grid = PointGrid(
            latgrid.cpu().numpy().reshape(-1), longrid.cpu().numpy().reshape(-1)
        )
        self.checkpoint = bind_checkpoint_state(_DLWPCheckpointState())

    def default_sources(self) -> DataSource:
        """Recommend ARCO ERA5 initial conditions.

        Returns
        -------
        DataSource
            Raw ARCO ERA5 source for the input slot.
        """
        from earth2studio.data import ARCO_ERA5

        return ARCO_ERA5()

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature for two history steps
            on the latitude/longitude grid.
        """
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(VARIABLES),
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature six and twelve hours
            after the final input lead time.
        """
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords,
            {"lead_time": lead[-1:] + np.array([6, 12], dtype="timedelta64[h]")},
        )

    @staticmethod
    def _metadata(x: xr.DataArray) -> dict[str, Any]:
        return deepcopy(
            {
                "dims": tuple(x.dims),
                "sizes": dict(x.sizes),
                "name": x.name,
                "coords": {
                    name: (tuple(value.dims), value.values.copy(), dict(value.attrs))
                    for name, value in x.coords.items()
                },
                "attrs": dict(x.attrs),
                "encoding": dict(x.encoding),
            }
        )

    @staticmethod
    def _signature(metadata: dict[str, Any]) -> CoordinateSystem:
        signature = coord_array(
            metadata["dims"],
            metadata["coords"],
            sizes=metadata["sizes"],
            attrs=metadata["attrs"],
            name=metadata["name"],
        )
        signature.encoding = metadata["encoding"].copy()
        return signature

    def _save_checkpoint_state(
        self, x: xr.DataArray, public: CoordinateSystem, pending: bool
    ) -> None:
        if self.checkpoint.checkpoint_enabled and self.checkpoint.checkpoint_level == 2:
            tensor, _ = x.e2s.to_torch()
            self.checkpoint.x = tensor.detach().clone().to(self.checkpoint.device)
            self.checkpoint.metadata = self._metadata(x)
            self.checkpoint.public_metadata = self._metadata(public)
            self.checkpoint.pending = pending
        else:
            self.checkpoint.x = None
            self.checkpoint.metadata = {}
            self.checkpoint.public_metadata = {}

    def _restore_checkpoint_state(
        self,
    ) -> tuple[xr.DataArray, CoordinateSystem, bool] | None:
        if (
            self.checkpoint.checkpoint_level == 2
            and self.checkpoint.checkpoint_state_loaded
            and self.checkpoint.x is not None
            and self.checkpoint.metadata
        ):
            signature = self._signature(self.checkpoint.metadata)
            x = from_torch(self.checkpoint.x.to(self.center.device), signature)
            x.encoding = signature.encoding.copy()
            return (
                x,
                self._signature(self.checkpoint.public_metadata),
                self.checkpoint.pending,
            )
        return None

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLWP model package on NGC"""
        return Package(
            "ngc://models/nvidia/modulus/modulus_dlwp_cubesphere@v0.2",
            cache_options={
                "cache_storage": Package.default_cache("dlwp"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package"""
        # Ghetto at the moment because NGC files are zipped. This will download zip and
        # unpack them then give the cached folder location from which we can then
        # access the needed files.
        dlwp_zip = Path(package.resolve("dlwp_cubesphere.zip"))
        # Have to manually unzip here. Should not zip checkpoints in the future
        with zipfile.ZipFile(dlwp_zip, "r") as zip_ref:
            zip_ref.extractall(dlwp_zip.parent)

        lsm = torch.Tensor(
            xarray.open_dataset(
                str(dlwp_zip.parent / Path("dlwp/land_sea_mask_rs_cs.nc"))
            )["lsm"].values
        )
        topographic_height = torch.Tensor(
            xarray.open_dataset(
                str(dlwp_zip.parent / Path("dlwp/geopotential_rs_cs.nc"))
            )["z"].values
        )
        latlon_grids = xarray.open_dataset(
            str(dlwp_zip.parent / Path("dlwp/latlon_grid_field_rs_cs.nc"))
        )
        latgrid = torch.Tensor(latlon_grids["latgrid"].values)
        longrid = torch.Tensor(latlon_grids["longrid"].values)
        # load maps
        input_map_wts = xarray.open_dataset(
            str(dlwp_zip.parent / Path("dlwp/map_LL721x1440_CS64.nc"))
        )
        output_map_wts = xarray.open_dataset(
            str(dlwp_zip.parent / Path("dlwp/map_CS64_LL721x1440.nc"))
        )

        i = input_map_wts.row.values - 1
        j = input_map_wts.col.values - 1
        data = input_map_wts.S.values
        cubed_sphere_transform = torch.sparse_coo_tensor(
            np.array((i, j)), data, dtype=torch.float
        )

        i = output_map_wts.row.values - 1
        j = output_map_wts.col.values - 1
        data = output_map_wts.S.values
        cubed_sphere_inverse = torch.sparse_coo_tensor(
            np.array((i, j)), data, dtype=torch.float
        )

        core_model = physicsnemo.Module.from_checkpoint(
            str(dlwp_zip.parent / Path("dlwp/dlwp.mdlus"))
        )

        center = torch.Tensor(
            np.load(str(dlwp_zip.parent / Path("dlwp/global_means.npy")))
        )
        scale = torch.Tensor(
            np.load(str(dlwp_zip.parent / Path("dlwp/global_stds.npy")))
        )

        return cls(
            core_model,
            landsea_mask=lsm,
            orography=topographic_height,
            latgrid=latgrid,
            longrid=longrid,
            cubed_sphere_transform=cubed_sphere_transform,
            cubed_sphere_inverse=cubed_sphere_inverse,
            center=center,
            scale=scale,
        )

    def to_cubedsphere(self, x: torch.Tensor) -> torch.Tensor:
        """[721,1440] eqr to [6,64,64] cs"""
        x = x.reshape(*x.shape[:-2], -1) @ self.M
        x = x.reshape(*x.shape[:-1], 6, 64, 64)
        return x

    def to_equirectangular(self, x: torch.Tensor) -> torch.Tensor:
        """[6,64,64] cs to [721,1440] eqr"""
        input_shape = x.shape[:-3]
        x = (self.N @ x.reshape(-1, 6 * 64 * 64).T).T
        x = x.reshape(*input_shape, 721, 1440)
        return x

    def get_cosine_zenith_fields(
        self, times: np.array, lead_time: timedelta, device: torch.device | str = "cuda"
    ) -> torch.Tensor:
        """Creates cosine zenith fields for input time array"""
        output = []
        for time in timearray_to_datetime(times):
            uvcossza = cos_zenith_angle(
                time + lead_time,
                self.longrid.cpu(),
                self.latgrid.cpu(),
            )
            # Normalize
            uvcossza = torch.Tensor(uvcossza).to(device)
            uvcossza = torch.clamp(uvcossza, min=0) - 1.0 / np.pi
            output.append(uvcossza)
        return torch.stack(output, axis=0)

    def _prepare_input(
        self, input: torch.Tensor, coords: CoordinateSystem
    ) -> torch.Tensor:
        """Prepares input cubed sphere tensor by adding land sea mask, uvcossza and
        orography fields to input atmospheric ([14,6,64,64] -> [18,6,64,64])
        """
        # Compress batch dim into time
        time_array = np.tile(
            coords["time"].values + coords["lead_time"].values[-1], input.shape[0]
        )
        input = input.reshape(-1, *input.shape[2:])

        uvcossza_6 = self.get_cosine_zenith_fields(
            time_array, timedelta(hours=-6), input.device
        ).unsqueeze(1)
        uvcossza_0 = self.get_cosine_zenith_fields(
            time_array, timedelta(hours=0), input.device
        ).unsqueeze(1)
        x = torch.cat([input[:, 0], uvcossza_6, input[:, 1], uvcossza_0], dim=1)

        input = torch.cat(
            (
                x,
                self.landsea_mask.repeat(x.shape[0], 1, 1, 1, 1),
                self.topographic_height.repeat(x.shape[0], 1, 1, 1, 1),
            ),
            dim=1,
        )
        return input

    def _prepare_output(
        self, output: torch.Tensor, coords: CoordinateSystem
    ) -> torch.Tensor:
        output = torch.split(output, output.shape[1] // 2, dim=1)
        # Add lead time dimension back in
        output = torch.stack(output, dim=1)
        # Add batch dimension back in
        output = output.view(-1, coords["time"].shape[0], *output.shape[1:])
        return output

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordinateSystem,
    ) -> torch.Tensor:

        center = self.center.unsqueeze(-1)
        scale = self.scale.unsqueeze(-1)

        x = (x - center) / scale
        x = self._prepare_input(x, coords)
        x = self.model(x)
        x = self._prepare_output(x, coords)
        x = scale * x + center
        return x

    def _to_cube(self, x: xr.DataArray) -> xr.DataArray:
        self.output_coords(x)
        handshake_time(x)
        tensor, _ = x.e2s.to_torch()
        tensor = self.to_cubedsphere(tensor.to(self.center.device))
        grid_keys = {
            "type",
            "dims",
            "shape",
            "topology",
            "crs",
            "earth2studio_grid_id",
            "earth2studio_crs",
        }
        attrs = {k: v for k, v in x.attrs.items() if k not in grid_keys}
        spatial_dim = "_dlwp_cell"
        while spatial_dim in x.dims or spatial_dim in x.coords:
            spatial_dim += "_"
        spatial = coord_array(("x",), grid=self._cube_grid).rename(x=spatial_dim)
        attrs.update(self._cube_grid.attrs)
        attrs["dims"] = [spatial_dim]
        signature = coord_array(
            (*x.dims[:-2], spatial_dim),
            {
                **dict(spatial.coords),
                **{
                    k: v
                    for k, v in x.coords.items()
                    if not set(v.dims).intersection(("lat", "lon"))
                },
            },
            attrs=attrs,
            name=x.name,
            sizes={dim: x.sizes[dim] for dim in x.dims[:-2]},
        )
        out = from_torch(tensor.flatten(-3), signature)
        out.encoding = x.encoding.copy()
        return out

    def _cube_step(self, x: xr.DataArray) -> xr.DataArray:
        tensor, _ = x.e2s.to_torch()
        shape = tensor.shape
        tensor = tensor.reshape(
            -1,
            x.sizes["time"],
            x.sizes["lead_time"],
            x.sizes["variable"],
            *self.latgrid.shape,
        )
        output = self._forward(tensor, x).reshape(shape)
        signature = coord_array_like(
            x, {"lead_time": x.lead_time.values + np.timedelta64(12, "h")}
        )
        out = from_torch(output, signature)
        out.encoding = x.encoding.copy()
        return out

    def _from_cube(self, x: xr.DataArray, public: CoordinateSystem) -> xr.DataArray:
        tensor, _ = x.e2s.to_torch()
        tensor = tensor.reshape(*tensor.shape[:-1], *self.latgrid.shape)
        spatial = coord_array_like(public, {"lead_time": x.lead_time.values})
        grid_keys = set(self._cube_grid.attrs) | {
            "crs",
            "earth2studio_grid_id",
            "earth2studio_crs",
        }
        # Only spatial metadata comes from the original public field. Rebuild
        # everything else from the hook result, including deletions and name=None.
        signature = coord_array(
            (*x.dims[:-1], "lat", "lon"),
            {
                **{
                    k: v.variable
                    for k, v in spatial.coords.items()
                    if set(v.dims).intersection(("lat", "lon"))
                },
                **{
                    k: v.variable
                    for k, v in x.coords.items()
                    if x.dims[-1] not in v.dims
                },
            },
            sizes={
                **{dim: x.sizes[dim] for dim in x.dims[:-1]},
                "lat": public.sizes["lat"],
                "lon": public.sizes["lon"],
            },
            attrs={
                **{k: v for k, v in public.attrs.items() if k in grid_keys},
                **{k: v for k, v in x.attrs.items() if k not in grid_keys},
            },
            name=x.name,
            dtype=x.dtype,
        )
        out = from_torch(self.to_equirectangular(tensor), signature, name=x.name)
        out.encoding = x.encoding.copy()
        return out

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict the complete twelve-hour forecast chunk without iterator hooks.

        Parameters
        ----------
        x : xr.DataArray
            Two-frame lat-lon history matching ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Six- and twelve-hour forecasts on the public lat-lon grid.
        """
        return self.initialize(x)[0]

    def initialize(
        self, x: xr.DataArray
    ) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]:
        """Compute both forecast frames and retain the native cubed-sphere state.

        Parameters
        ----------
        x : xr.DataArray
            Two-frame initial history matching ``input_coords()``.

        Returns
        -------
        tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]
            Six- and twelve-hour forecasts in one chunk, plus native cubed-sphere
            fields and a public-grid reference for reconciling edits. No hooks run.
        """
        handshake_nonempty(x)
        handshake_time(x)
        cube = self._cube_step(self._to_cube(x))
        y = self._from_cube(cube, x)
        return y, (cube.copy(deep=True), y.copy(deep=True))

    def step(
        self, y: xr.DataArray, state: tuple[xr.DataArray, xr.DataArray]
    ) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]:
        """Advance native recurrence, projecting public field edits onto the cube.

        Parameters
        ----------
        y : xr.DataArray
            Previous two-frame forecast chunk, including caller-applied edits.
        state : tuple[xr.DataArray, xr.DataArray]
            Native fields and published reference from ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]
            Next twelve-hour forecast chunk and updated state, without hooks.
        """
        cube, published = state
        current = self._to_cube(y)
        baseline = self._to_cube(published)
        tensor = cube.e2s.to_torch()[0] + (
            current.e2s.to_torch()[0] - baseline.e2s.to_torch()[0]
        )
        current = from_torch(tensor, coord_array_like(current))
        current.encoding = y.encoding.copy()
        cube = self._cube_step(current)
        output = self._from_cube(cube, y)
        return output, (cube.copy(deep=True), output.copy(deep=True))

    def create_iterator(self, x: xr.DataArray) -> Generator[xr.DataArray, None, None]:
        """Yield complete two-frame forecasts with hooks on the public grid.

        Parameters
        ----------
        x : xr.DataArray
            Initial history matching ``input_coords()``.

        Yields
        ------
        xr.DataArray
            Two-frame, twelve-hour chunks, starting with initialization's forecast.
            The rear hook runs before each yield and the front hook before later
            steps; edits are projected onto the native grid.
        """
        yield from self._default_create_iterator(x)
