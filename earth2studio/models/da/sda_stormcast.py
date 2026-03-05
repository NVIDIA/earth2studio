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

import warnings
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr

from earth2studio.data import GFS_FX, HRRR, DataSource, ForecastSource, fetch_data
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.utils import filter_time_range
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.coords import map_coords_xr
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import CoordSystem, FrameSchema, TimeTolerance

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.guidance import (
        DataConsistencyDPSGuidance,
        DPSDenoiser,
    )
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.preconditioners.legacy import EDMPrecond
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.diffusion.samplers.legacy_deterministic_sampler import (
        deterministic_sampler,
    )
    from physicsnemo.models.diffusion_unets import StormCastUNet
except ImportError:
    OptionalDependencyFailure("stormcast")
    StormCastUNet = None
    EDMPreconditioner = None
    OmegaConf = None
    deterministic_sampler = None


# Variables used in StormCastV1 paper
VARIABLES = (
    ["u10m", "v10m", "t2m", "msl"]
    + [
        var + str(level)
        for var, level in product(
            ["u", "v", "t", "q", "Z", "p"],
            map(
                lambda x: str(x) + "hl",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30],
            ),
        )
        if not ((var == "p") and (int(level.replace("hl", "")) > 20))
    ]
    + [
        "refc",
    ]
)

CONDITIONING_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "sp", "msl"] + [
    var + str(level)
    for var, level in product(["u", "v", "z", "t", "q"], [1000, 850, 500, 250])
]

INVARIANTS = ["lsm", "orography"]


@check_optional_dependencies()
class StormCast(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """StormCast generative convection-allowing model for regional forecasts consists of
    two core models: a regression and diffusion model. Model time step size is 1 hour,
    taking as input:

    - High-resolution (3km) HRRR state over the central United States (99 vars)
    - High-resolution land-sea mask and orography invariants
    - Coarse resolution (25km) global state (26 vars)

    The high-resolution grid is the HRRR Lambert conformal projection
    Coarse-resolution inputs are regridded to the HRRR grid internally.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2408.10958
    - https://huggingface.co/nvidia/stormcast-v1-era5-hrrr

    Parameters
    ----------
    regression_model : torch.nn.Module
        Deterministic model used to make an initial prediction
    diffusion_model : torch.nn.Module
        Generative model correcting the deterministic prediciton
    means : torch.Tensor
        Mean value of each input high-resolution variable
    stds : torch.Tensor
        Standard deviation of each input high-resolution variable
    invariants : torch.Tensor
        Static invariant  quantities
    hrrr_lat_lim : tuple[int, int], optional
        HRRR grid latitude limits, defaults to be the StormCastV1 region in central
        United States, by default (273, 785)
    hrrr_lon_lim : tuple[int, int], optional
        HRRR grid longitude limits, defaults to be the StormCastV1 region in central
        United States,, by default (579, 1219)
    variables : np.array, optional
        High-resolution variables, by default np.array(VARIABLES)
    conditioning_means : torch.Tensor | None, optional
        Means to normalize conditioning data, by default None
    conditioning_stds : torch.Tensor | None, optional
        Standard deviations to normalize conditioning data, by default None
    conditioning_variables : np.array, optional
        Global variables for conditioning, by default np.array(CONDITIONING_VARIABLES)
    conditioning_data_source : DataSource | ForecastSource | None, optional
        Data Source to use for global conditioning. Required for running in iterator mode, by default None
    sampler_args : dict[str, float  |  int], optional
        Arguments to pass to the diffusion sampler, by default {}
    tolerance : TimeTolerance, optional
        Time tolerance for filtering observations. Observations within the tolerance
        window around each requested time will be used for data assimilation,
        by default np.timedelta64(30, "m")
    """

    def __init__(
        self,
        regression_model: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        hrrr_lat_lim: tuple[int, int] = (273, 785),
        hrrr_lon_lim: tuple[int, int] = (579, 1219),
        variables: np.array = np.array(VARIABLES),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.array = np.array(CONDITIONING_VARIABLES),
        conditioning_data_source: DataSource | ForecastSource | None = None,
        sampler_args: dict[str, float | int] = {},
        tolerance: TimeTolerance = np.timedelta64(30, "m"),
    ):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)
        self.register_buffer("device_buffer", torch.empty(0))
        self.sampler_args = sampler_args
        self._tolerance = normalize_time_tolerance(tolerance)

        hrrr_lat, hrrr_lon = HRRR.grid()
        self.lat = hrrr_lat[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.lon = hrrr_lon[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]

        self.hrrr_x = HRRR.HRRR_X[hrrr_lon_lim[0] : hrrr_lon_lim[1]]
        self.hrrr_y = HRRR.HRRR_Y[hrrr_lat_lim[0] : hrrr_lat_lim[1]]

        # Build ordered boundary polygon from 2D grid perimeter for
        # point-in-grid testing (top row -> right col -> bottom row -> left col)
        self._grid_boundary = np.column_stack(
            [
                np.concatenate(
                    [
                        self.lat[0, :],
                        self.lat[1:, -1],
                        self.lat[-1, -2::-1],
                        self.lat[-2:0:-1, 0],
                    ]
                ),
                np.concatenate(
                    [
                        self.lon[0, :],
                        self.lon[1:, -1],
                        self.lon[-1, -2::-1],
                        self.lon[-2:0:-1, 0],
                    ]
                ),
            ]
        )  # [n_boundary, 2] ordered (lat, lon)

        self.variables = variables

        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None:
            warnings.warn(
                "No conditioning data source was provided to StormCast, "
                + "set the conditioning_data_source attribute of the model "
                + "before running inference."
            )

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)

        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    def init_coords(self) -> tuple[CoordSystem]:
        """Initialization coordinate system"""
        return (
            OrderedDict(
                {
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(self.variables),
                    "hrrr_y": self.hrrr_y,
                    "hrrr_x": self.hrrr_x,
                }
            ),
        )

    def input_coords(self) -> tuple[FrameSchema]:
        """Input coordinate system specifying required DataFrame fields."""
        return (
            FrameSchema(
                {
                    "time": np.empty(0, dtype="datetime64[ns]"),
                    "lat": np.empty(0, dtype=np.float32),
                    "lon": np.empty(0, dtype=np.float32),
                    "observation": np.empty(0, dtype=np.float32),
                    "variable": np.array(self.variables, dtype=str),
                }
            ),
        )

    def output_coords(self, input_coords: tuple[CoordSystem]) -> tuple[CoordSystem]:
        """Output coordinate system of diagnostic model

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

        output_coords = OrderedDict(
            {
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(1, "h")]),
                "variable": np.array(self.variables),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
            }
        )

        target_input_coords = self.init_coords()[0]

        handshake_dim(input_coords[0], "hrrr_x", 4)
        handshake_dim(input_coords[0], "hrrr_y", 3)
        handshake_dim(input_coords[0], "variable", 2)
        # Index coords are arbitrary as long its on the HRRR grid, so just check size
        handshake_size(input_coords[0], "hrrr_y", self.lat.shape[0])
        handshake_size(input_coords[0], "hrrr_x", self.lat.shape[1])
        handshake_coords(input_coords[0], target_input_coords, "variable")

        output_coords["time"] = input_coords[0]["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords[0]["lead_time"]
        )
        return (output_coords,)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://nvidia/stormcast-v1-era5-hrrr@6c89a0877a0d6b231033d3b0d8b9828a6f833ed8",
            cache_options={
                "cache_storage": Package.default_cache("stormcast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(verbose=False),
    ) -> DiagnosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        conditioning_data_source : DataSource | ForecastSource, optional
            Data source to use for global conditioning, by default GFS_FX

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            # Likely already registered so skip
            pass

        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        # TODO: remove strict=False once checkpoints/imports updated to new diffusion API
        regression = StormCastUNet.from_checkpoint(
            package.resolve("StormCastUNet.0.0.mdlus"),
            strict=False,
        )
        diffusion = EDMPrecond.from_checkpoint(
            package.resolve("EDMPrecond.0.0.mdlus"),
            strict=False,
        )

        # Load metadata: means, stds, grid
        store = zarr.storage.ZipStore(package.resolve("metadata.zarr.zip"), mode="r")
        metadata = xr.open_zarr(store, zarr_format=2)

        variables = metadata["variable"].values
        conditioning_variables = metadata["conditioning_variable"].values

        # Expand dims and tensorify normalization buffers
        means = torch.from_numpy(metadata["means"].values[None, :, None, None])
        stds = torch.from_numpy(metadata["stds"].values[None, :, None, None])
        conditioning_means = torch.from_numpy(
            metadata["conditioning_means"].values[None, :, None, None]
        )
        conditioning_stds = torch.from_numpy(
            metadata["conditioning_stds"].values[None, :, None, None]
        )

        # Load invariants
        invariants = metadata["invariants"].sel(invariant=config.data.invariants).values
        invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)

        # EDM sampler arguments
        if config.sampler_args is not None:
            sampler_args = config.sampler_args
        else:
            sampler_args = {}

        return cls(
            regression,
            diffusion,
            means,
            stds,
            invariants,
            variables=variables,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            sampler_args=sampler_args,
        )

    # @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        y_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        # Scale data
        if "conditioning_means" in self._buffers:
            conditioning = conditioning - self.conditioning_means
        if "conditioning_stds" in self._buffers:
            conditioning = conditioning / self.conditioning_stds

        x = (x - self.means) / self.stds

        # Run regression model
        invariant_tensor = self.invariants.repeat(x.shape[0], 1, 1, 1)
        concats = torch.cat((x, conditioning, invariant_tensor), dim=1)

        out = self.regression_model(concats)

        # Concat for diffusion conditioning
        condition = torch.cat((x, out, invariant_tensor), dim=1)
        latents = torch.randn_like(x, dtype=torch.float64)
        latents = self.sampler_args["sigma_max"] * latents

        class _CondtionalDiffusionWrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, img_lr: torch.Tensor):
                super().__init__()
                self.model = model
                self.img_lr = img_lr

            def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return self.model(x, t, condition=self.img_lr)

        scheduler = EDMNoiseScheduler(
            sigma_min=self.sampler_args["sigma_min"],
            sigma_max=self.sampler_args["sigma_max"],
            rho=self.sampler_args["rho"],
        )

        guidance = DataConsistencyDPSGuidance(
            mask=mask,
            y=y_obs,
            std_y=0.001,
            norm=1,  # L1 norm
            gamma=0.1,  # Enable SDA scaling
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
        )
        score_predictor = DPSDenoiser(
            x0_predictor=_CondtionalDiffusionWrapper(self.diffusion_model, condition),
            x0_to_score_fn=scheduler.x0_to_score,
            guidances=guidance,
        )
        denoiser = scheduler.get_denoiser(score_predictor=score_predictor)

        # denoiser = scheduler.get_denoiser(
        #     x0_predictor=_CondtionalDiffusionWrapper(self.diffusion_model, condition)
        # )

        edm_out = sample(
            denoiser,
            latents,
            noise_scheduler=scheduler,
            # num_steps=self.sampler_args["num_steps"],
            num_steps=2 * self.sampler_args["num_steps"],
            solver="edm_stochastic_heun",
            # solver_options={
            #     "S_churn": self.sampler_args["S_churn"],
            #     "S_min": self.sampler_args["S_min"],
            #     "S_max": self.sampler_args["S_max"],
            #     "S_noise": self.sampler_args["S_noise"],
            # },
        )

        out += edm_out
        out = out * self.stds + self.means

        return out.detach()

    @staticmethod
    def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Vectorized ray casting point-in-polygon test.

        Parameters
        ----------
        points : np.ndarray
            Points to test, shape [n, 2]
        polygon : np.ndarray
            Ordered polygon vertices, shape [m, 2]

        Returns
        -------
        np.ndarray
            Boolean array of shape [n], True if point is inside polygon
        """
        px, py = points[:, 0], points[:, 1]  # [n]
        vx, vy = polygon[:, 0], polygon[:, 1]  # [m]
        vx_next = np.roll(vx, -1)
        vy_next = np.roll(vy, -1)

        # For each edge (m) and each point (n), check if horizontal ray crosses
        # Broadcasting: [m, 1] vs [1, n] -> [m, n]
        crosses = (vy[:, None] > py[None, :]) != (vy_next[:, None] > py[None, :])
        x_intersect = (vx_next[:, None] - vx[:, None]) * (py[None, :] - vy[:, None]) / (
            vy_next[:, None] - vy[:, None]
        ) + vx[:, None]
        hits = crosses & (px[None, :] < x_intersect)

        # Odd number of crossings = inside
        return (np.sum(hits, axis=0) % 2) == 1

    def _build_obs_tensors(
        self,
        obs: pd.DataFrame | None,
        request_time: np.datetime64,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_var = len(self.variables)
        n_hrrr_y, n_hrrr_x = self.lat.shape

        y_obs = torch.zeros(
            1, n_var, n_hrrr_y, n_hrrr_x, device=device, dtype=torch.float32
        )
        mask = torch.zeros(
            1, n_var, n_hrrr_y, n_hrrr_x, device=device, dtype=torch.float32
        )

        if obs is None or len(obs) == 0:
            return y_obs, mask

        # Filter observations within tolerance window
        time_filtered = filter_time_range(
            obs, request_time, self._tolerance, time_column="time"
        )

        if len(time_filtered) == 0:
            return y_obs, mask

        # TODO, make native cudf support
        # Convert to pandas if cudf for reliable string/value access
        if hasattr(time_filtered, "to_pandas"):
            time_filtered = time_filtered.to_pandas()

        obs_lat = time_filtered["lat"].values.astype(np.float64)
        obs_lon = time_filtered["lon"].values.astype(np.float64)
        obs_var = time_filtered["variable"].values
        obs_val = time_filtered["observation"].values.astype(np.float32)

        # Normalize lon to 0-360 to match HRRR grid
        obs_lon = np.where(obs_lon < 0, obs_lon + 360.0, obs_lon)

        # Filter observations to those inside the curvilinear grid boundary
        # using ray casting point-in-polygon on the precomputed perimeter
        obs_points = np.column_stack([obs_lat, obs_lon])
        in_grid = self._points_in_polygon(obs_points, self._grid_boundary)

        if not in_grid.any():
            return y_obs, mask

        obs_lat = obs_lat[in_grid]
        obs_lon = obs_lon[in_grid]
        obs_var = obs_var[in_grid]
        obs_val = obs_val[in_grid]

        # Find nearest HRRR grid point for each observation (vectorized)
        grid_lat_flat = self.lat.ravel()  # [n_grid]
        grid_lon_flat = self.lon.ravel()  # [n_grid]
        lat_diff = obs_lat[:, None] - grid_lat_flat[None, :]  # [n_obs, n_grid]
        lon_diff = obs_lon[:, None] - grid_lon_flat[None, :]  # [n_obs, n_grid]
        dist_sq = lat_diff**2 + lon_diff**2
        nearest_flat = np.argmin(dist_sq, axis=1)  # [n_obs]
        nearest_y = nearest_flat // n_hrrr_x
        nearest_x = nearest_flat % n_hrrr_x

        # Map variable names to indices
        var_to_idx = {str(v): i for i, v in enumerate(self.variables)}
        var_indices = np.array([var_to_idx.get(str(v), -1) for v in obs_var])
        valid = var_indices >= 0

        if valid.any():
            vi = torch.tensor(var_indices[valid], device=device, dtype=torch.long)
            yi = torch.tensor(nearest_y[valid], device=device, dtype=torch.long)
            xi = torch.tensor(nearest_x[valid], device=device, dtype=torch.long)
            vals = torch.tensor(obs_val[valid], device=device, dtype=torch.float32)
            y_obs[0, vi, yi, xi] = vals
            mask[0, vi, yi, xi] = 1.0

        return y_obs, mask

    def __call__(
        self,
        x: xr.DataArray,
        obs: pd.DataFrame | None,
    ) -> xr.DataArray:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCast has been called without initializing the model's conditioning_data_source"
            )

        # Use registered buffer to track model's current device
        device = self.device_buffer.device

        c = fetch_data(
            self.conditioning_data_source,
            time=x.coords["time"].data,
            variable=self.conditioning_variables,
            lead_time=x.coords["lead_time"].data,
            device=self.device_buffer.device,
            legacy=False,
        )

        # Interpolate conditioning from regular lat/lon grid to HRRR curvilinear grid
        if cp is not None and isinstance(c.data, cp.ndarray):
            # GPU path: bilinear interpolation using cupy, data stays on GPU
            with cp.cuda.Device(device.index or 0):
                data = c.data  # cupy [time, lead_time, variable, lat, lon]
                src_lat = c.coords["lat"].values  # numpy 1D
                src_lon = c.coords["lon"].values  # numpy 1D

                # Compute fractional indices into the regular source grid
                target_lat_cp = cp.asarray(self.lat, dtype=cp.float64)
                target_lon_cp = cp.asarray(self.lon, dtype=cp.float64)
                lat_step = float(src_lat[1] - src_lat[0])
                lon_step = float(src_lon[1] - src_lon[0])
                lat_frac = (target_lat_cp - float(src_lat[0])) / lat_step
                lon_frac = (target_lon_cp - float(src_lon[0])) / lon_step

                # Floor indices and interpolation weights
                lat0 = cp.clip(
                    cp.floor(lat_frac).astype(cp.int64), 0, data.shape[-2] - 2
                )
                lon0 = cp.clip(
                    cp.floor(lon_frac).astype(cp.int64), 0, data.shape[-1] - 2
                )
                lat1 = lat0 + 1
                lon1 = lon0 + 1
                wlat = cp.clip(lat_frac - lat0.astype(cp.float64), 0.0, 1.0)
                wlon = cp.clip(lon_frac - lon0.astype(cp.float64), 0.0, 1.0)

                # Bilinear interpolation (fully vectorized over leading dims)
                interp_data = (
                    data[..., lat0, lon0] * (1 - wlat) * (1 - wlon)
                    + data[..., lat0, lon1] * (1 - wlat) * wlon
                    + data[..., lat1, lon0] * wlat * (1 - wlon)
                    + data[..., lat1, lon1] * wlat * wlon
                )

            c = xr.DataArray(
                data=interp_data,
                dims=["time", "lead_time", "variable", "hrrr_y", "hrrr_x"],
                coords={
                    "time": c.coords["time"],
                    "lead_time": c.coords["lead_time"],
                    "variable": c.coords["variable"],
                    "hrrr_y": self.hrrr_y,
                    "hrrr_x": self.hrrr_x,
                    "lat": (["hrrr_y", "hrrr_x"], self.lat),
                    "lon": (["hrrr_y", "hrrr_x"], self.lon),
                },
            )
        else:
            # CPU path: use xarray's built-in interpolation
            target_lat = xr.DataArray(self.lat, dims=["hrrr_y", "hrrr_x"])
            target_lon = xr.DataArray(self.lon, dims=["hrrr_y", "hrrr_x"])
            c = c.interp(lat=target_lat, lon=target_lon, method="linear")
            c = c.assign_coords(
                hrrr_y=("hrrr_y", self.hrrr_y),
                hrrr_x=("hrrr_x", self.hrrr_x),
                lat=(["hrrr_y", "hrrr_x"], self.lat),
                lon=(["hrrr_y", "hrrr_x"], self.lon),
            )

        # Build input CoordSystem from the xarray DataArray for handshake
        x_coords = OrderedDict({dim: x.coords[dim].values for dim in x.dims})
        output_coords = self.output_coords((x_coords,))

        # Zero copy from cupy / numpy
        x_tensor = torch.as_tensor(x.data)
        c_tensor = torch.as_tensor(c.data)

        # Build y_obs and mask from observations, then run forward
        # No batch dims at the moment
        for j, t in enumerate(x.coords["time"].data):
            # Build observation tensors for this time step
            y_obs, mask = self._build_obs_tensors(obs, t, device)
            for k, _ in enumerate(x.coords["lead_time"].data):
                x_tensor[j, k : k + 1] = self._forward(
                    x_tensor[j, k : k + 1], c_tensor[j, k : k + 1], y_obs, mask
                )

        # Convert output tensor to xarray DataArray
        (oc,) = output_coords
        if device.type == "cuda" and cp is not None:
            with cp.cuda.Device(device.index or 0):
                out_data = cp.asarray(x_tensor.detach())
        else:
            out_data = x_tensor.detach().cpu().numpy()

        return xr.DataArray(
            data=out_data,
            dims=list(oc.keys()),
            coords={
                k: ((["hrrr_y", "hrrr_x"], v) if k in ("lat", "lon") else v)
                for k, v in oc.items()
            }
            | {
                "lat": (["hrrr_y", "hrrr_x"], self.lat),
                "lon": (["hrrr_y", "hrrr_x"], self.lon),
            },
        )


if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    package = StormCast.load_default_package()
    model = StormCast.load_model(package)
    model = model.to("cuda")

    data = HRRR(verbose=False)
    x = fetch_data(
        data,
        np.array(["2024-01-01"], dtype=np.datetime64),
        model.input_coords()[0]["variable"],
        device="cuda",
        legacy=False,
    )

    x = map_coords_xr(x, model.init_coords()[0])

    # Create synthetic observation DataFrame with random points inside the HRRR grid
    # Sample a few lat/lon points from the grid interior
    rng = np.random.default_rng(42)
    n_obs = 10
    grid_lat, grid_lon = model.lat, model.lon
    yi = rng.integers(50, grid_lat.shape[0] - 50, size=n_obs)
    xi = rng.integers(50, grid_lat.shape[1] - 50, size=n_obs)
    obs_lats = grid_lat[yi, xi]
    obs_lons = grid_lon[yi, xi]

    obs_vars = rng.choice(["u10m", "v10m", "t2m"], size=n_obs)
    obs_vals = rng.normal(
        loc=[280.0 if v == "t2m" else 5.0 for v in obs_vars], scale=2.0
    )

    obs_df = pd.DataFrame(
        {
            "time": np.datetime64("2024-01-01", "ns"),
            "lat": obs_lats.astype(np.float32),
            "lon": obs_lons.astype(np.float32),
            "variable": obs_vars,
            "observation": obs_vals.astype(np.float32),
        }
    )
    obs_df.attrs = {"request_time": np.array(["2024-01-01"], dtype="datetime64[ns]")}

    out = model(x, obs_df)

    print(out)

    # Load stormcast_original.pt
    # torch.save(out, "stormcast.pt")
    original = torch.load("stormcast_original.pt", map_location=model.device)

    # Assume the dimensionality/order is the same as out
    diff = torch.as_tensor(out.data) - original

    print("Difference between out and stormcast_original.pt:")
    print("Max absolute difference:", diff.abs().max().item())
    print("Mean absolute difference:", diff.abs().mean().item())
    print("Shape of diff:", diff.shape)

    import matplotlib.pyplot as plt

    # Plot the first variable, first batch, first lead_time, first time
    # Infer axes: usually channels, y, x
    # out shape: (batch, time, lead_time, variable, y, x)
    var_axis = 3
    y_axis = 4
    x_axis = 5

    plt.figure(figsize=(8, 6))
    img = out.data[0, 0, 0].get()  # Shape: (y, x)
    plt.imshow(img, cmap="viridis", aspect="auto", vmin=-10, vmax=12.5)
    plt.title(f"Forecast: variable idx 0 (shape {img.shape})")
    plt.colorbar(label="Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("stormcast.jpg")
