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

from collections.abc import Generator
from itertools import product

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from loguru import logger

from earth2studio.data import GFS_FX, HRRR, DataSource, ForecastSource
from earth2studio.data.base import DataFrameSource
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import StatefulAssimilationModel
from earth2studio.models.da.utils import filter_time_range
from earth2studio.models.utils import fork_rng
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import CoordinateSystem, FrameSchema, TimeTolerance

try:
    from scipy.spatial import cKDTree
except ImportError:
    OptionalDependencyFailure("stormcast")
    cKDTree = None

try:
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.guidance import (
        DataConsistencyDPSGuidance,
        DPSScorePredictor,
    )
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.preconditioners.legacy import EDMPrecond
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.models.diffusion_unets import StormCastUNet
except ImportError:
    OptionalDependencyFailure("stormcast")
    StormCastUNet = None
    EDMPreconditioner = None
    OmegaConf = None


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
class StormCastSDA(torch.nn.Module, AutoModelMixin):
    """StormCast with score-based data assimilation (SDA) using diffusion posterior
    sampling for convection-allowing regional forecasts. Combines a regression and
    diffusion model with DPS guidance to assimilate observations during inference.
    Model time step size is 1 hour, taking as input:

    - High-resolution (3km) HRRR state over the central United States (99 vars)
    - High-resolution land-sea mask and orography invariants
    - Coarse resolution (25km) global state (26 vars)
    - Point observations for data assimilation

    The high-resolution grid is the HRRR Lambert conformal projection.
    The caller supplies conditioning regridded to the HRRR grid, followed by
    observations, in ``forcing_coords()`` order.

    ``initialize`` computes the first analysis and returns explicit Torch RNG
    snapshots for continuation with ``step``. ``create_iterator`` yields analyses
    only, starting at input plus one hour. Sampling uses scoped global Torch RNG
    state; run stochastic components serially within a process.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2408.10958
    - https://huggingface.co/nvidia/stormcast-v1-era5-hrrr
    - https://arxiv.org/abs/2306.10574

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
    variables : np.ndarray, optional
        High-resolution variables, by default np.array(VARIABLES)
    conditioning_means : torch.Tensor | None, optional
        Means to normalize conditioning data, by default None
    conditioning_stds : torch.Tensor | None, optional
        Standard deviations to normalize conditioning data, by default None
    conditioning_variables : np.ndarray, optional
        Global variables for conditioning, by default np.array(CONDITIONING_VARIABLES)
    time_tolerance : TimeTolerance, optional
        Time tolerance for filtering observations. Observations within the tolerance
        window around each requested time will be used for data assimilation,
        by default np.timedelta64(10, "m")
    sampler_steps : int, optional
        Number of diffusion sampler steps, by default 36
    sampler_args : dict[str, float  |  int] | None, optional
        Arguments to pass to the diffusion sampler, by default None
    sda_std_obs : float, optional
        Observation noise standard deviation for DPS guidance, by default 0.1
    sda_gamma : float, optional
        SDA scaling factor for DPS guidance, by default 0.001

    Badges
    ------
    region:na class:data-assimilation class:nowcasting product:wind product:temp product:atmos product:radar
    product:insitu year:2024 gpu:40gb
    provider:nvidia backend:pytorch
    """

    stochastic = True

    def __init__(
        self,
        regression_model: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        hrrr_lat_lim: tuple[int, int] = (273, 785),
        hrrr_lon_lim: tuple[int, int] = (579, 1219),
        variables: np.ndarray = np.array(VARIABLES),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.ndarray = np.array(CONDITIONING_VARIABLES),
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        sampler_steps: int = 36,
        sampler_args: dict[str, float | int] | None = None,
        sda_std_obs: float = 0.1,
        sda_gamma: float = 0.001,
    ):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)
        self.register_buffer("device_buffer", torch.empty(0))
        self.sampler_steps = sampler_steps
        self.sampler_args = {
            "sigma_min": 0.002,
            "sigma_max": 800,
            "rho": 7,
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": float("inf"),
            "S_noise": 1,
        }
        if sampler_args is not None:
            self.sampler_args.update(sampler_args)
        self._tolerance = normalize_time_tolerance(time_tolerance)
        self.sda_std_obs = sda_std_obs
        self.sda_dps_norm = 2
        self.sda_gamma = sda_gamma
        self._rng: torch.Generator | None = None

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

        # Build a KD-tree over (lat, lon) for efficient nearest-grid-point queries
        # TODO: Make cpu and gpu support
        self._grid_tree = cKDTree(np.column_stack([self.lat.ravel(), self.lon.ravel()]))

        self.variables = variables

        self.conditioning_variables = conditioning_variables

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)

        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the stream used to start new trajectories.

        Parameters
        ----------
        seed : int
            Seed for a local CPU generator. Global RNG state is unchanged.
        reset : bool, optional
            Restart the initialization stream when True. When False, initialize
            it only if absent, by default True

        Notes
        -----
        Existing trajectories use their explicit state and are unaffected by
        reseeding. Without seeding, initialization draws from the global CPU RNG.
        """
        if reset or self._rng is None:
            self._rng = torch.Generator().manual_seed(seed)

    def input_coords(self) -> CoordinateSystem:
        """Declare the initial HRRR state as an allocation-free signature."""
        return coord_array(
            ("time", "lead_time", "variable", "hrrr_y", "hrrr_x"),
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
                "lat": (("hrrr_y", "hrrr_x"), self.lat),
                "lon": (("hrrr_y", "hrrr_x"), self.lon),
            },
            dynamic=("time",),
        )

    def default_sources(
        self,
    ) -> tuple[DataSource, ForecastSource, DataFrameSource]:
        """Recommend HRRR state, global conditioning, and GHCNh observations.

        Returns
        -------
        tuple[DataSource, ForecastSource, DataFrameSource]
            ``HRRR`` for the initial state, ``GFS_FX`` for conditioning,
            then ``GHCNHourly`` for the
            surface station observation slot, restricted to the stations inside this
            model's domain and using the model's observation time tolerance.
        """
        from earth2studio.data import GHCNHourly

        # Model lon is in [0, 360); GHCN bbox for the western hemisphere CONUS
        # domain uses [-180, 180)
        bbox = (
            float(self.lat.min()),
            float(self.lon.min()) - 360.0,
            float(self.lat.max()),
            float(self.lon.max()) - 360.0,
        )
        stations = GHCNHourly.get_stations_bbox(bbox)
        return (
            HRRR(),
            GFS_FX(verbose=False),
            GHCNHourly(
                stations=stations, time_tolerance=self._tolerance, verbose=False
            ),
        )

    def forcing_coords(self) -> tuple[CoordinateSystem, FrameSchema]:
        """Declare global conditioning and sparse observation forcing.

        Returns
        -------
        tuple[CoordinateSystem, FrameSchema]
            Conditioning on the HRRR grid at the input valid time, followed by
            observations at the output valid time (input plus one hour), filtered
            with the configured tolerance. Pass None for the observation slot
            to run without assimilation; conditioning is always required.
        """
        return (
            coord_array_like(
                self.input_coords(), {"variable": self.conditioning_variables}
            ),
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

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the assimilation model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Coordinates of tensor used to initialize the forecast model.

        Returns
        -------
        CoordinateSystem
            Allocation-free analysis signature
            on the HRRR grid, one hour after the initialization lead time.
        """

        expected = coord_array_like(
            self.input_coords(), {"lead_time": input_coords.lead_time.values}
        )
        handshake_dataarray(input_coords, expected)
        if input_coords.sizes["lead_time"] != 1:
            raise ValueError("StormCastSDA requires one input lead time")
        return coord_array_like(
            input_coords,
            {"lead_time": input_coords.lead_time.values + np.timedelta64(1, "h")},
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load assimilation package"""
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
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        sampler_steps: int = 36,
        sda_std_obs: float = 0.1,
        sda_gamma: float = 0.001,
    ) -> StatefulAssimilationModel:
        """Load assimilation from package

        Parameters
        ----------
        package : Package
            Package to load model from
        time_tolerance : TimeTolerance, optional
            Time tolerance for filtering observations. Observations within the tolerance
            window around each requested time will be used for data assimilation,
            by default np.timedelta64(10, "m")
        sampler_steps : int, optional
            Number of diffusion sampler steps, by default 36
        sda_std_obs : float, optional
            Observation noise standard deviation for DPS guidance, by default 0.1
        sda_gamma : float, optional
            SDA scaling factor for DPS guidance, by default 0.001

        Returns
        -------
        StatefulAssimilationModel
            Assimilation model with explicit trajectory continuation
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
            conditioning_variables=conditioning_variables,
            time_tolerance=time_tolerance,
            sampler_steps=sampler_steps,
            sampler_args=sampler_args,
            sda_std_obs=sda_std_obs,
            sda_gamma=sda_gamma,
        )

    @torch.no_grad()
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
        y_obs = (y_obs - self.means) / self.stds

        # Run regression model
        invariant_tensor = self.invariants.repeat(x.shape[0], 1, 1, 1)
        concats = torch.cat((x, conditioning, invariant_tensor), dim=1)

        out = self.regression_model(concats)
        y_obs = y_obs - out  # Convert to residual obs

        # Concat for diffusion conditioning
        condition = torch.cat((x, out, invariant_tensor), dim=1)
        latents = torch.randn_like(x, dtype=torch.float64)
        latents = self.sampler_args["sigma_max"] * latents  # Initial guess

        def _conditional_diffusion(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.diffusion_model(x, t, condition=condition)

        scheduler = EDMNoiseScheduler(
            sigma_min=self.sampler_args["sigma_min"],
            sigma_max=self.sampler_args["sigma_max"],
            rho=self.sampler_args["rho"],
        )

        guidance = DataConsistencyDPSGuidance(
            mask=mask,
            y=y_obs,
            std_y=self.sda_std_obs,
            norm=self.sda_dps_norm,
            gamma=self.sda_gamma,
            sigma_fn=scheduler.sigma,
            alpha_fn=scheduler.alpha,
        )
        score_predictor = DPSScorePredictor(
            x0_predictor=_conditional_diffusion,
            x0_to_score_fn=scheduler.x0_to_score,
            guidances=guidance,
        )
        denoiser = scheduler.get_denoiser(score_predictor=score_predictor)

        edm_out = sample(
            denoiser,
            latents,
            noise_scheduler=scheduler,
            num_steps=self.sampler_steps,
            solver="edm_stochastic_heun",
            solver_options={
                "S_churn": self.sampler_args["S_churn"],
                "S_min": self.sampler_args["S_min"],
                "S_max": self.sampler_args["S_max"],
                "S_noise": self.sampler_args["S_noise"],
            },
        )

        out += edm_out
        out = out * self.stds + self.means
        return out

    @staticmethod
    def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Vectorized ray casting point-in-polygon test.
        TODO: Improved this (GPU and reduce memory requirement)
        make a general purpose util maybe...

        Note
        ----
        For more information see the following references:
        https://observablehq.com/@tmcw/understanding-point-in-polygon

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
        dvy = vy_next[:, None] - vy[:, None]
        safe_dvy = np.where(dvy == 0, 1.0, dvy)  # avoid division by zero; masked later
        x_intersect = (vx_next[:, None] - vx[:, None]) * (
            py[None, :] - vy[:, None]
        ) / safe_dvy + vx[:, None]
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

        # Find nearest HRRR grid point for each observation using a KD-tree
        # to avoid allocating the full [n_obs, n_grid] distance matrix.
        _, nearest_flat = self._grid_tree.query(np.column_stack([obs_lat, obs_lon]))
        nearest_y = nearest_flat // n_hrrr_x
        nearest_x = nearest_flat % n_hrrr_x

        # Map variable names to indices
        var_to_idx = {str(v): i for i, v in enumerate(self.variables)}
        var_indices = np.array([var_to_idx.get(str(v), -1) for v in obs_var])
        valid = var_indices >= 0

        # Average multiple observations that map to the same grid cell
        if valid.any():
            vi = torch.tensor(var_indices[valid], device=device, dtype=torch.long)
            yi = torch.tensor(nearest_y[valid], device=device, dtype=torch.long)
            xi = torch.tensor(nearest_x[valid], device=device, dtype=torch.long)
            vals = torch.tensor(obs_val[valid], device=device, dtype=torch.float32)

            # Flatten (vi, yi, xi) into a single linear index for scatter ops
            flat_idx = vi * (n_hrrr_y * n_hrrr_x) + yi * n_hrrr_x + xi
            flat_sum = torch.zeros(
                n_var * n_hrrr_y * n_hrrr_x, device=device, dtype=torch.float32
            )
            flat_cnt = torch.zeros_like(flat_sum)
            flat_sum.scatter_add_(0, flat_idx, vals)
            flat_cnt.scatter_add_(0, flat_idx, torch.ones_like(vals))

            occupied = flat_cnt > 0
            flat_avg = torch.where(occupied, flat_sum / flat_cnt, flat_sum)
            y_obs[0] = flat_avg.view(n_var, n_hrrr_y, n_hrrr_x)
            mask[0] = occupied.float().view(n_var, n_hrrr_y, n_hrrr_x)

        return y_obs, mask

    # NOTE: @torch.inference_mode() is intentionally omitted here.
    # DPS guidance requires gradient computation through the denoiser for
    # the score correction step; inference_mode would disable those gradients.
    def __call__(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Compute the first analysis, equivalent to ``initialize(...)[0]``.

        Parameters
        ----------
        x : xr.DataArray
            Input state on the HRRR curvilinear grid
        conditioning : xr.DataArray
            Conditioning on the same grid and valid time as x, matching the first
            ``forcing_coords()`` slot. Regridding is caller-owned.
        obs : pd.DataFrame | None, optional
            Sparse observations at the output valid time, by default None

        Returns
        -------
        xr.DataArray
            First analysis one hour into the future, without iterator hooks.

        Raises
        ------
        ValueError
            If state or conditioning coordinates do not match their declared slots.
        """
        return self.initialize(x, conditioning, obs)[0]

    def initialize(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> tuple[xr.DataArray, dict[str, torch.Tensor]]:
        """Compute the first analysis and start an independent trajectory.

        Parameters
        ----------
        x : xr.DataArray
            Initial HRRR state. Fields are moved to the model device.
        conditioning : xr.DataArray
            Required conditioning on the same grid and valid time as x.
        obs : pd.DataFrame | None, optional
            Observations at the first analysis time, by default None

        Returns
        -------
        tuple[xr.DataArray, dict[str, torch.Tensor]]
            First analysis at input plus one hour and serializable CPU/device
            RNG snapshots. Applies no hooks and does not modify inputs. Each
            initialization advances the model's initialization RNG stream.

        Raises
        ------
        ValueError
            If required conditioning is missing or coordinates are invalid.
        """
        seed = int(
            torch.randint(2**63 - 1, (), generator=self._rng, device="cpu").item()
        )
        return self._advance(x, conditioning, obs, {}, seed)

    def step(
        self,
        y: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None,
        state: dict[str, torch.Tensor],
    ) -> tuple[xr.DataArray, dict[str, torch.Tensor]]:
        """Advance an analysis using its explicit RNG continuation.

        Parameters
        ----------
        y : xr.DataArray
            Previous analysis. Fields are moved to the model device.
        conditioning : xr.DataArray
            Required conditioning at y's valid time on the same grid.
        obs : pd.DataFrame | None
            Observations at the next analysis time; None disables assimilation.
        state : dict[str, torch.Tensor]
            RNG snapshots returned by initialization or the previous step on the
            same device. Pass after the forcing slots, positionally or by keyword.

        Returns
        -------
        tuple[xr.DataArray, dict[str, torch.Tensor]]
            Next analysis one hour after y and updated RNG snapshots. Neither y,
            forcing nor state is modified; replay is independent of model reseeding
            and other trajectories. Applies no hooks.

        Raises
        ------
        ValueError
            If conditioning is missing, coordinates are invalid, or state lacks
            RNG snapshots for this device.
        """
        required = {"cpu"}
        if self.device.type == "cuda":
            required.add(str(self.device))
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError(
                "StormCastSDA state requires RNG snapshots for this device"
            )
        return self._advance(y, conditioning, obs, state)

    def _advance(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None,
        state: dict[str, torch.Tensor],
        seed: int = 0,
    ) -> tuple[xr.DataArray, dict[str, torch.Tensor]]:
        # The sampler also draws global Torch noise (e.g. stochastic churn).
        # Snapshot the whole numerical call, never a suspended iterator yield.
        next_state = {key: value.clone() for key, value in state.items()}
        with fork_rng(seed=seed, device=self.device, states=next_state):
            y = self._analysis(x, conditioning, obs)
        return y, next_state

    def _analysis(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None,
    ) -> xr.DataArray:
        device = self.device
        handshake_nonempty(x)
        output_coords = self.output_coords(x)
        if not isinstance(conditioning, xr.DataArray):
            raise ValueError("StormCastSDA requires conditioning as a DataArray")
        handshake_nonempty(conditioning)
        handshake_dataarray(
            conditioning, coord_array_like(x, {"variable": self.conditioning_variables})
        )
        x_tensor, _ = x.e2s.to_torch()
        x_tensor = x_tensor.to(device).clone()
        c_tensor, _ = conditioning.e2s.to_torch()
        c_tensor = c_tensor.to(device)

        for j, t in enumerate(x.coords["time"].data):
            obs_time = t + output_coords.lead_time.values[0]
            y_obs, mask = self._build_obs_tensors(obs, obs_time, device)
            x_tensor[j, :] = self._forward(x_tensor[j, :], c_tensor[j, :], y_obs, mask)

        return from_torch(x_tensor, output_coords)

    def create_iterator(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray,
        obs: pd.DataFrame | None = None,
    ) -> Generator[xr.DataArray, tuple[xr.DataArray, pd.DataFrame | None] | None, None]:
        """Yield hourly analyses, beginning with the initialization analysis.

        The generator yields states and receives conditioning and observations
        via ``send((conditioning, obs))`` in forcing-slot order. Observations
        are mapped to the HRRR grid, and the diffusion model produces the next
        forecast step.

        Parameters
        ----------
        x : xr.DataArray
            Initial state on the HRRR curvilinear grid
        conditioning : xr.DataArray
            Required initial conditioning on x's grid and valid time.
        obs : pd.DataFrame | None, optional
            Observations at the first analysis time, by default None

        Yields
        ------
        xr.DataArray
            Complete analyses, starting one hour after x; never the initial state.

        Receives
        --------
        tuple[xr.DataArray, pd.DataFrame | None] | None
            Conditioning at the current state valid time, then observations at the
            next analysis time. Pass ``(conditioning, None)`` without assimilation.
            Sending ``None`` raises ValueError because conditioning is required.

        Example
        -------
        >>> gen = model.create_iterator(x0, c0, obs_df)
        >>> analysis = next(gen)             # first analysis
        >>> analysis = gen.send((c1, None))   # second analysis, no observations
        """
        try:
            y, state = self.initialize(x, conditioning, obs)
            forcing = yield y
            while True:
                if not isinstance(forcing, tuple) or len(forcing) != 2:
                    raise ValueError(
                        "Send (conditioning, obs) in forcing_coords() order"
                    )
                y, state = self.step(y, forcing[0], forcing[1], state)
                forcing = yield y

        except GeneratorExit:
            logger.info("StormCast SDA clean up")
