# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.dx.corrdiff import CorrDiff
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, LeadTimeArray

try:
    from physicsnemo.models import Module as PhysicsNemoModule
    from physicsnemo.utils.corrdiff import diffusion_step, regression_step
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:  # pragma: no cover
    OptionalDependencyFailure("corrdiff")
    diffusion_step = None  # type: ignore[assignment]
    regression_step = None  # type: ignore[assignment]
    cos_zenith_angle = None  # type: ignore[assignment]


class CorrDiffCMIP6(CorrDiff):
    """CMIP6 to ERA5 downscaling model based on the CorrDiff architecture. This model
    can be used to downscale both in the spatial and temporal dimensions.
    This model works with the :py:class:`earth2studio.data.CMIP6MultiRealm` data source.

    Note
    ----
    For more information see the following references:

    - https://huggingface.co/nvidia/corrdiff-cmip6-era5

    Parameters
    ----------
    input_variables : Sequence[str]
        List of input variable names
    output_variables : Sequence[str]
        List of output variable names
    residual_model : torch.nn.Module
        Core pytorch model for diffusion step
    regression_model : torch.nn.Module
        Core pytorch model for regression step
    lat_input_grid : torch.Tensor
        Input latitude grid of size [in_lat]
    lon_input_grid : torch.Tensor
        Input longitude grid of size [in_lon]
    lat_output_grid : torch.Tensor
        Output latitude grid of size [out_lat]
    lon_output_grid : torch.Tensor
        Output longitude grid of size [out_lon]
    in_center : torch.Tensor
        Model input center normalization tensor of size [in_var]
    in_scale : torch.Tensor
        Model input scale normalization tensor of size [in_var]
    out_center : torch.Tensor
        Model output center normalization tensor of size [out_var]
    out_scale : torch.Tensor
        Model output scale normalization tensor of size [out_var]
    invariants : OrderedDict | None, optional
        Dictionary of invariant features, by default None
    invariant_center : torch.Tensor | None, optional
        Model invariant center normalization tensor, by default None
    invariant_scale : torch.Tensor | None, optional
        Model invariant scale normalization tensor, by default None
    number_of_samples : int, optional
        Number of high resolution samples to draw from diffusion model, by default 1
    number_of_steps : int, optional
        Number of langevin diffusion steps during sampling algorithm, by default 18
    solver : Literal["euler", "heun"], optional
        Discretization of diffusion process, by default "euler"
    sampler_type : Literal["deterministic", "stochastic"], optional
        Type of sampler to use, by default "stochastic"
    inference_mode : Literal["regression", "both"], optional
        Which inference mode to use ("both" or "regression"); diffusion-only
        is not supported in CorrDiffCMIP6. Default is "both".
    hr_mean_conditioning : bool, optional
        Whether to use high-res mean conditioning, by default True
    seed : int | None, optional
        Random seed for reproducibility, by default None
    grid_spacing_tolerance : float, optional
        Relative tolerance for checking regular grid spacing, by default 1e-5
    grid_bounds_margin : float, optional
        Fraction of input grid range to allow for extrapolation, by default 0.0
    sigma_min : float | None, optional
        Minimum noise level for diffusion process, by default None
    sigma_max : float | None, optional
        Maximum noise level for diffusion process, by default None
    time_feature_center : torch.Tensor | None, optional
        Normalization center for time features (sza, hod) of size [2], by default None
    time_feature_scale : torch.Tensor | None, optional
        Normalization scale for time features (sza, hod) of size [2], by default None
    output_lead_times: LeadTimeArray, optional
        Output lead times to sample at within the input time window. The default package
        is trained to support lead times between [-12, +11] hours at hourly intervals.
        This constraint ensures the input data remains aligned with the temporal features
        (SZA, HOD) calculated at the valid time. By default np.array([np.timedelta64(-12, "h")])

    Examples
    --------
    Run a single forward pass to predict CMIP6->ERA5 at two lead times within input window

    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model = CorrDiffCMIP6.load_model(
    ...     CorrDiffCMIP6.load_default_package(),
    ...     output_lead_times=np.array([np.timedelta64(-12, "h"), np.timedelta64(-6, "h")]),
    ... )
    >>> model.seed = 1 # Set seed for reprod
    >>> model.number_of_samples = 1 # Modify number of samples if needed
    >>> model = model.to(device)
    >>>
    >>> # Build CMIP6 multi-realm data source, about 60 Gbs of data will be fetched
    >>> cmip6_kwargs = dict(
    ...     experiment_id="ssp585",
    ...     source_id="CanESM5",
    ...     variant_label="r1i1p2f1",
    ...     exact_time_match=True,
    ... )
    >>> data = CMIP6MultiRealm([CMIP6(table_id=t, **cmip6_kwargs) for t in ("day", "Eday", "SIday")])
    >>>
    >>> x, coords = fetch_data(
    ...     source=data,
    ...     time=np.array([np.datetime64("2037-09-06T12:00")]), # Time must be 12:00 UTC
    ...     lead_time=model.input_coords()["lead_time"],
    ...     variable=model.input_coords()["variable"],
    ...     device=device,
    ... )
    >>>
    >>> # Run model forward pass
    >>> out, out_coords = model(x, coords)
    >>> da = xr.DataArray(data=out.cpu().numpy(), coords=out_coords, dims=list(out_coords.keys()))

    """

    # Variables that must be non-negative (clipped to min=0 during postprocessing)
    # These represent physical quantities that cannot be negative (temperature, pressure, etc.)
    _NONNEGATIVE_VARS = [
        "t2m",
        "sp",
        "msl",
        "tcwv",
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
        "sst",
        "d2m",
    ]

    # Padding applied during preprocessing (must be cropped in postprocessing)
    # Format: (top, bottom) for lat, (left, right) for lon
    _LAT_PAD = (23, 24)  # reflect padding in latitude
    _LON_PAD = (48, 48)  # circular padding in longitude

    # Valid range for output lead times in hours
    _MIN_LEAD_TIME_HOURS = -12
    _MAX_LEAD_TIME_HOURS = 11

    def __init__(
        self,
        input_variables: Sequence[str],
        output_variables: Sequence[str],
        residual_model: torch.nn.Module,
        regression_model: torch.nn.Module,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        invariants: OrderedDict | None = None,
        invariant_center: torch.Tensor | None = None,
        invariant_scale: torch.Tensor | None = None,
        number_of_samples: int = 1,
        number_of_steps: int = 18,
        solver: Literal["euler", "heun"] = "euler",
        sampler_type: Literal["deterministic", "stochastic"] = "stochastic",
        inference_mode: Literal["regression", "diffusion", "both"] = "both",
        hr_mean_conditioning: bool = True,
        seed: int | None = None,
        grid_spacing_tolerance: float = 1e-5,
        grid_bounds_margin: float = 0.0,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        time_feature_center: torch.Tensor | None = None,
        time_feature_scale: torch.Tensor | None = None,
        output_lead_times: LeadTimeArray = np.array([np.timedelta64(-12, "h")]),
    ) -> None:
        super().__init__(
            input_variables=input_variables,
            output_variables=output_variables,
            residual_model=residual_model,
            regression_model=regression_model,
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            in_center=in_center,
            in_scale=in_scale,
            out_center=out_center,
            out_scale=out_scale,
            invariants=invariants,
            invariant_center=invariant_center,
            invariant_scale=invariant_scale,
            number_of_samples=number_of_samples,
            number_of_steps=number_of_steps,
            solver=solver,
            sampler_type=sampler_type,
            inference_mode=inference_mode,
            hr_mean_conditioning=hr_mean_conditioning,
            seed=seed,
            grid_spacing_tolerance=grid_spacing_tolerance,
            grid_bounds_margin=grid_bounds_margin,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        # CMIP6 wrapper only supports "both" or "regression" modes (no diffusion-only)
        if self.inference_mode not in ("both", "regression"):
            raise ValueError(
                "CorrDiffCMIP6 supports inference_mode in {'both', 'regression'} only "
                f"but got {self.inference_mode!r}"
            )

        # Preprocess caches (built lazily on first use)
        self._cmip6_var_index: dict[str, int] | None = None
        self._cmip6_sai_kernel: torch.Tensor | None = None
        self._cmip6_lonlat_meshgrid: tuple[np.ndarray, np.ndarray] | None = None
        self._cmip6_reorder_indices: list[int] | None = None

        # When True, CorrDiffCMIP6 will accumulate multi-sample outputs on CPU to reduce GPU peak
        # memory for large `number_of_samples` / large output channel counts.
        # This does not change the generated samples, only where the final stacked tensor lives.
        self.stream_samples_to_cpu: bool = False

        # Controls the output lead times, should be hourly between [-12, +11]
        # Validate lead times are within the valid range
        self._validate_output_lead_times(output_lead_times)
        self.output_lead_times = output_lead_times

        # Extend in_center and in_scale to include time features (sza, hod) at the end
        # Note: During training, the last invariant position (coslat) mistakenly had hod VALUES,
        # but was normalized using coslat STATISTICS. This bug is replicated in preprocess_input
        # by putting hod values in the coslat position during channel reordering.
        if time_feature_center is not None and time_feature_scale is not None:
            # Reshape time features to match the 4D format [1, N, 1, 1]
            time_feature_center = time_feature_center.view(1, -1, 1, 1)
            time_feature_scale = time_feature_scale.view(1, -1, 1, 1)

            # Append time features after base variables and invariants
            self.in_center: torch.Tensor = torch.cat(  # type: ignore
                [self.in_center, time_feature_center], dim=1
            )
            self.in_scale: torch.Tensor = torch.cat(  # type: ignore
                [self.in_scale, time_feature_scale], dim=1
            )

        # Cache indices of output variables that must be non-negative so we don't
        # recompute them on every postprocess call.
        self._nonnegative_output_indices = [
            i
            for i, v in enumerate(self.output_variables)
            if v in self._NONNEGATIVE_VARS
        ]

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [
                        np.timedelta64(-24, "h"),
                        np.timedelta64(0, "h"),
                        np.timedelta64(24, "h"),
                    ]
                ),
                "variable": np.array(self.input_variables),
                "lat": self.lat_input_numpy,
                "lon": self.lon_input_numpy,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "time", 1)
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "lon", 5)
        handshake_coords(input_coords, target_input_coords, "lon")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords = OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "time": input_coords["time"],
                "lead_time": self.output_lead_times,
                "variable": np.array(self.output_variables),
                "lat": self.lat_output_numpy,
                "lon": self.lon_output_numpy,
            }
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load diagnostic package"""
        package = Package(
            "hf://nvidia/corrdiff-cmip6-era5@f756fad5b85efec64df4868aead14dda698b8aea",
            cache_options={
                "cache_storage": Package.default_cache("corrdiff_cmip6"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        output_lead_times: LeadTimeArray = np.array([np.timedelta64(-12, "h")]),
        device: str = "cpu",
    ) -> DiagnosticModel:
        """Load diagnostic from package

        Parameters
        ----------
        package : Package
            Package containing model weights and configuration
        output_lead_times : LeadTimeArray, optional
            Output lead times to sample at, by default np.array([np.timedelta64(-12, "h")])
        device : str, optional
            Device to load model on, by default "cpu"

        Returns
        -------
        DiagnosticModel
            Diagnostic model
        """
        # Load and validate metadata first (we need time_window for input expansion).
        metadata = cls._load_json_from_package(package, "metadata.json")
        stats = cls._load_json_from_package(package, "stats.json")

        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        # Load the base CorrDiff model from the package.
        residual = (
            PhysicsNemoModule.from_checkpoint(
                package.resolve("diffusion.mdlus"), strict=False
            )
            .eval()
            .to(device)
        )
        regression = (
            PhysicsNemoModule.from_checkpoint(
                package.resolve("regression.mdlus"), strict=False
            )
            .eval()
            .to(device)
        )

        # Apply inference optimizations
        residual.profile_mode = False
        regression.profile_mode = False

        residual = residual.to(memory_format=torch.channels_last)
        regression = regression.to(memory_format=torch.channels_last)

        torch._dynamo.config.cache_size_limit = 264
        torch._dynamo.reset()

        # Load meta data
        input_variables = metadata["input_variables"]
        output_variables = metadata["output_variables"]
        invariant_variables = metadata.get("invariant_variables", None)
        number_of_samples = metadata.get("number_of_samples", 1)
        number_of_steps = metadata.get("number_of_steps", 18)
        solver = metadata.get("solver", "euler")
        sampler_type = metadata.get("sampler_type", "stochastic")
        inference_mode = metadata.get("inference_mode", "both")
        hr_mean_conditioning = metadata.get("hr_mean_conditioning", True)
        sigma_max_metadata = metadata.get("sigma_max", 400)
        sigma_min_metadata = metadata.get("sigma_min", 1)
        grid_spacing_tolerance = metadata.get("grid_spacing_tolerance", 1e-5)
        grid_bounds_margin = metadata.get("grid_bounds_margin", 0.0)

        in_center_values = []
        in_scale_values = []
        for var in input_variables:
            if var not in stats["input"]:
                raise KeyError(
                    f"stats.json is missing normalization statistics for input variable '{var}'."
                )
            in_center_values.append(stats["input"][var]["mean"])
            in_scale_values.append(stats["input"][var]["std"])

        in_center = torch.tensor(in_center_values, device=device)
        in_scale = torch.tensor(in_scale_values, device=device)

        # Load output normalization parameters
        out_center = torch.tensor(
            [stats["output"][v]["mean"] for v in output_variables], device=device
        )
        out_scale = torch.tensor(
            [stats["output"][v]["std"] for v in output_variables], device=device
        )

        with xr.open_dataset(package.resolve("output_latlon_grid.nc")) as ds:
            lat_output_grid = torch.as_tensor(np.array(ds["lat"][:]), device=device)
            lon_output_grid = torch.as_tensor(np.array(ds["lon"][:]), device=device)

            # Validate output grid format and ordering
            cls._validate_grid_format(
                lat_output_grid, lon_output_grid, grid_name="output"
            )

        with xr.open_dataset(package.resolve("input_latlon_grid.nc")) as ds:
            lat_input_grid = torch.as_tensor(np.array(ds["lat"][:]), device=device)
            lon_input_grid = torch.as_tensor(np.array(ds["lon"][:]), device=device)

            # Validate input grid format and ordering
            cls._validate_grid_format(lat_input_grid, lon_input_grid, grid_name="input")

        with xr.open_dataset(package.resolve("invariants.nc")) as ds:
            # Determine which variables to load and in what order
            if invariant_variables is None:
                # Load all available variables
                var_names = list(ds.data_vars)
            else:
                # Load only specified variables in the specified order
                var_names = invariant_variables
                # Validate that all requested variables exist
                missing_vars = [v for v in var_names if v not in ds.data_vars]
                if missing_vars:
                    raise ValueError(
                        f"Invariant variables {missing_vars} not found in invariants.nc. "
                        f"Available variables: {list(ds.data_vars)}"
                    )

            invariants = OrderedDict(
                (var_name, torch.as_tensor(np.array(ds[var_name]), device=device))
                for var_name in var_names
            )
            # Load invariant normalization parameters
            invariant_center = torch.tensor(
                [stats["invariants"][v]["mean"] for v in invariants], device=device
            )
            invariant_scale = torch.tensor(
                [stats["invariants"][v]["std"] for v in invariants], device=device
            )

        # Create time feature normalization tensors on the same device as the base buffers.
        time_feature_center = torch.as_tensor(
            [stats["input"]["sza"]["mean"], stats["input"]["hod"]["mean"]],
            device=device,
        )
        time_feature_scale = torch.as_tensor(
            [stats["input"]["sza"]["std"], stats["input"]["hod"]["std"]], device=device
        )

        return cls(
            input_variables=input_variables,
            output_variables=output_variables,
            residual_model=residual,
            regression_model=regression,
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            in_center=in_center.squeeze(),
            in_scale=in_scale.squeeze(),
            invariants=invariants,
            invariant_center=invariant_center.squeeze(),
            invariant_scale=invariant_scale.squeeze(),
            out_center=out_center.squeeze(),
            out_scale=out_scale.squeeze(),
            number_of_samples=number_of_samples,
            number_of_steps=number_of_steps,
            solver=solver,
            sampler_type=sampler_type,
            inference_mode=inference_mode,
            hr_mean_conditioning=hr_mean_conditioning,
            seed=None,
            time_feature_center=time_feature_center,
            time_feature_scale=time_feature_scale,
            grid_spacing_tolerance=grid_spacing_tolerance,
            grid_bounds_margin=grid_bounds_margin,
            sigma_min=sigma_min_metadata,
            sigma_max=sigma_max_metadata,
            output_lead_times=output_lead_times,
        )

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)
        out_shape = tuple(len(v) for v in output_coords.values())
        out = torch.empty(out_shape, device=x.device, dtype=torch.float32)
        # Iterate of different time-stamps and lead time
        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                valid_time = output_coords["time"][i] + output_coords["lead_time"][j]
                # Input to forward should be [b, l, c, h, w]
                out[:, :, i, j] = self._forward(
                    x[:, i, :], pd.to_datetime(valid_time).to_pydatetime()
                )
        return out, output_coords

    def _get_lonlat_meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached lon/lat meshgrid on the output grid (numpy arrays)."""
        if self._cmip6_lonlat_meshgrid is None:
            self._cmip6_lonlat_meshgrid = np.meshgrid(
                self.lon_output_numpy, self.lat_output_numpy
            )
        return self._cmip6_lonlat_meshgrid

    def _get_sai_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Cached 3x3 averaging kernel (on correct device/dtype)."""
        k = self._cmip6_sai_kernel
        if k is None or k.device != x.device or k.dtype != x.dtype:
            k = x.new_ones((1, 1, 3, 3)) / 9.0
            self._cmip6_sai_kernel = k
        return k

    def _get_reorder_indices(self) -> list[int]:
        """Cached channel reorder indices after normalization."""
        if self._cmip6_reorder_indices is None:
            num_input = (
                len(self.input_variables) * self.input_coords()["lead_time"].shape[0]
            )
            num_inv = len(self.invariant_variables)
            self._cmip6_reorder_indices = (
                list(range(num_input))  # input variables
                + [num_input + num_inv]  # sza
                + list(
                    range(num_input, num_input + num_inv - 1)
                )  # invariants except coslat
                + [num_input + num_inv - 1, num_input + num_inv + 1]  # hod variants
            )
        return self._cmip6_reorder_indices

    def _apply_sai_cover(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sea-air-ice cover smoothing inplace on x ([B, C, L, H, W])."""
        kernel = self._get_sai_kernel(x)

        lead_len = self.input_coords()["lead_time"].shape[0]
        if "siconc" not in self.input_variables or "snc" not in self.input_variables:
            return x
        siconc_channel = self.input_variables.index("siconc")
        snc_channel = self.input_variables.index("snc")
        for i in range(lead_len):
            # [B, 1, H, W] so padding/conv2d operate on 4D tensors
            siconc = torch.nan_to_num(x[:, siconc_channel, i], nan=0.0).unsqueeze(1)
            snc = torch.nan_to_num(x[:, snc_channel, i], nan=0.0).unsqueeze(1)
            sai_cover = torch.clip(siconc + snc, 0.0, 100.0)

            # Pad: circular in lon (W), replicate in lat (H)
            sai_cover_pad = F.pad(sai_cover, (1, 1, 0, 0), mode="circular")
            sai_cover_pad = F.pad(sai_cover_pad, (0, 0, 1, 1), mode="replicate")
            sai_cover_smooth = F.conv2d(sai_cover_pad, kernel, padding="valid")

            x[:, siconc_channel, i] = sai_cover_smooth.squeeze(1)

        return x

    def _add_time_features(self, x: torch.Tensor, valid_time: datetime) -> torch.Tensor:
        """Append SZA and HOD features to x ([B,C,H,W])."""
        lon_grid, lat_grid = self._get_lonlat_meshgrid()
        cos_sza = cos_zenith_angle(valid_time, lon_grid, lat_grid).astype(np.float32)
        cos_sza_tensor = (
            torch.from_numpy(cos_sza).unsqueeze(0).unsqueeze(0).to(x.device)
        ).expand(x.shape[0], -1, -1, -1)
        x = torch.concat([x, cos_sza_tensor], dim=1)

        hour_tensor = torch.full_like(cos_sza_tensor, float(valid_time.hour))

        # Overwrite coslat slot with HOD values before normalization (training quirk replication).
        num_input = 3 * len(self.input_variables)
        num_inv = len(self.invariant_variables)
        if num_inv > 0:
            x[:, num_input + num_inv - 1] = hour_tensor.squeeze()

        # Also add HOD at the end
        x = torch.concat([x, hour_tensor], dim=1)
        return x

    def _normalize_pad_reorder(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, pad, and reorder channels to match training quirks."""
        x = self.normalize_input(x)
        x = torch.flip(x, [2])
        x = F.pad(x, (0, 0, self._LAT_PAD[0], self._LAT_PAD[1]), mode="reflect")
        x = F.pad(x, (self._LON_PAD[0], self._LON_PAD[1], 0, 0), mode="circular")

        indices = self._get_reorder_indices()
        x = x[:, indices]
        return x

    def preprocess_input(
        self, x: torch.Tensor, valid_time: datetime | None = None
    ) -> torch.Tensor:
        """Input preprocessing pipeline with optional time information."""
        if valid_time is None:
            raise ValueError(
                "CorrDiffCMIP6 requires valid_time for time-dependent features"
            )

        x = x.transpose(1, 2)  # [B, L, C, H, W] -> [B, C, L, H, W]
        x = self._apply_sai_cover(x)

        B, C, L, H, W = x.shape
        x = x.contiguous().view(
            B, -1, H, W
        )  # Flatten (variable, lead_time) -> [C*L, H, W]
        x = F.interpolate(x, self.img_shape, mode="bilinear")

        if self.invariants is not None:
            # Flip invars lat to match inverted cmip data
            invar = torch.flip(self.invariants.unsqueeze(0), [-2])
            x = torch.concat([x, invar.expand(x.shape[0], -1, -1, -1)], dim=1)

        x = self._add_time_features(x, valid_time)
        x = self._normalize_pad_reorder(x)

        return x

    def postprocess_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor using model's center and scale parameters.

        Parameters
        ----------
        x : torch.Tensor
            Normalized output tensor to denormalize

        Returns
        -------
        torch.Tensor
            Denormalized output tensor x * scale + center
        """
        # 1) Crop padding added during preprocessing [S, C, H, W] (see _LAT_PAD, _LON_PAD)
        x = x[
            :,
            :,
            self._LAT_PAD[0] : -self._LAT_PAD[1],
            self._LON_PAD[0] : -self._LON_PAD[1],
        ]
        # 2) Denormalize (reuse base implementation)
        x = super().postprocess_output(x)

        # 3) Enforce physical non-negativity constraints on selected channels
        if self._nonnegative_output_indices:
            # NOTE: advanced indexing returns a copy; assign back to update `x`.
            x[:, self._nonnegative_output_indices] = x[
                :, self._nonnegative_output_indices
            ].clamp(min=0)

        # 4) Flip latitude (model outputs S->N, we want N->S)
        return torch.flip(x, [2])

    @torch.inference_mode()
    def _forward(
        self, x: torch.Tensor, valid_time: datetime | None = None
    ) -> torch.Tensor:
        """Forward pass with optional CPU streaming for multi-sample inference.

        This override keeps the base `CorrDiff` class unchanged, but allows the CMIP6 wrapper
        to reduce GPU peak memory by avoiding a GPU-side concat of all samples.
        """
        if self.solver not in ["euler", "heun"]:
            raise ValueError(
                f"solver must be either 'euler' or 'heun' but got {self.solver}"
            )

        # Preprocess input (CMIP6 requires valid_time)
        image_lr = self.preprocess_input(x, valid_time)
        image_lr = image_lr.to(torch.float32).to(memory_format=torch.channels_last)

        # Regression model (mean)
        image_reg = torch.empty(
            (image_lr.shape[0], len(self.output_variables), *image_lr.shape[-2:]),
            device=x.device,
            dtype=image_lr.dtype,
        )
        latents_shape = (1, len(self.output_variables), *image_lr.shape[-2:])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # CorrDiff utils do not support batches, so we in-efficiently loop
            for i in range(image_lr.shape[0]):
                image_reg[i] = regression_step(
                    net=self.regression_model,
                    img_lr=image_lr[i : i + 1],
                    latents_shape=latents_shape,
                )

        # Regression-only: all samples are identical (deterministic mean)
        if self.inference_mode == "regression":
            return (
                self.postprocess_output(image_reg)
                .unsqueeze(1)
                .expand(-1, self.number_of_samples, -1, -1, -1)
            )

        # Compute base seed once (sample index added in loop)
        seed0 = (
            int(self.seed) if self.seed is not None else int(np.random.randint(2**32))
        )

        # Where to accumulate samples (CPU streaming reduces GPU peak memory)
        out_device = (
            torch.device("cpu") if self.stream_samples_to_cpu else image_lr.device
        )
        out = torch.empty(
            (
                image_reg.shape[0],
                self.number_of_samples,
                len(self.output_variables),
                self.lat_output_numpy.shape[0],
                self.lon_output_numpy.shape[0],
            ),
            device=out_device,
            dtype=image_reg.dtype,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # CorrDiff utils do not support batches, so we in-efficiently loop
            for i in range(out.shape[0]):
                mean_hr = image_reg[i : i + 1] if self.hr_mean_conditioning else None
                for j in range(self.number_of_samples):
                    image_res = diffusion_step(
                        net=self.residual_model,
                        sampler_fn=self.sampler,
                        img_shape=image_lr.shape[-2:],
                        img_out_channels=len(self.output_variables),
                        rank_batches=[[seed0 + j]],
                        img_lr=image_lr[i : i + 1],
                        rank=1,
                        device=image_lr.device,
                        mean_hr=mean_hr,
                    )
                    out[i, j] = self.postprocess_output(image_reg[i] + image_res).to(
                        out_device
                    )

        return out

    def _register_buffers(
        self,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
        in_center: torch.Tensor,
        in_scale: torch.Tensor,
        invariant_center: torch.Tensor,
        invariant_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        invariants: OrderedDict | None,
    ) -> None:
        """Register model buffers and handle invariants."""
        # Register grid coordinates
        self.register_buffer("lat_input_grid", lat_input_grid)
        self.register_buffer("lon_input_grid", lon_input_grid)
        self.register_buffer("lat_output_grid", lat_output_grid)
        self.register_buffer("lon_output_grid", lon_output_grid)

        self._interpolator = None  # Use efficient regular grid interpolation

        self.lat_input_numpy = lat_input_grid.cpu().numpy()
        self.lon_input_numpy = lon_input_grid.cpu().numpy()
        self.lat_output_numpy = lat_output_grid.cpu().numpy()
        self.lon_output_numpy = lon_output_grid.cpu().numpy()

        # Handle invariants
        if invariants:
            self.invariant_variables = list(invariants.keys())
            self.register_buffer(
                "invariants", torch.stack(list(invariants.values()), dim=0)
            )
        else:
            self.invariant_variables = []
            self.invariants = None

        # Combine input normalization with invariants (1D tensors)
        in_center = torch.concat([in_center, invariant_center], dim=0)
        in_scale = torch.concat([in_scale, invariant_scale], dim=0)

        # Repeat base variable stats for 3 lead times; keep invariants single
        num_invariants = len(self.invariant_variables)
        if num_invariants > 0:
            base_center, inv_center = (
                in_center[:-num_invariants],
                in_center[-num_invariants:],
            )
            base_scale, inv_scale = (
                in_scale[:-num_invariants],
                in_scale[-num_invariants:],
            )
        else:
            base_center, inv_center = in_center, in_center.new_empty((0,))
            base_scale, inv_scale = in_scale, in_scale.new_empty((0,))

        base_center = base_center.repeat_interleave(3, dim=0)
        base_scale = base_scale.repeat_interleave(3, dim=0)

        in_center = torch.concat([base_center, inv_center], dim=0)
        in_scale = torch.concat([base_scale, inv_scale], dim=0)

        # Register normalization parameters with final channel count
        num_inputs = int(in_center.shape[0])
        self.register_buffer("in_center", in_center.view(1, num_inputs, 1, 1))
        self.register_buffer("in_scale", in_scale.view(1, num_inputs, 1, 1))
        self.register_buffer(
            "out_center", out_center.view(1, len(self.output_variables), 1, 1)
        )
        self.register_buffer(
            "out_scale", out_scale.view(1, len(self.output_variables), 1, 1)
        )

    @staticmethod
    def _validate_output_lead_times(output_lead_times: LeadTimeArray) -> None:
        """Validate that output lead times are within the valid range [-12, +11] hours.

        The model requires input data at lead times [-24h, 0h, +24h] centered on the
        requested time. The temporal features (SZA, HOD) are calculated for the valid
        time (time + lead_time). To ensure alignment between input data and temporal
        features, output lead times must stay within [-12, +11] hours.

        For example, to get output for 2026-12-23 00:00:00:
        - Use time=2026-12-23 12:00:00 with lead_time=-12h (correct)
        - NOT time=2026-12-22 12:00:00 with lead_time=+12h (misaligned features)

        Parameters
        ----------
        output_lead_times : LeadTimeArray
            Array of lead times to validate

        Raises
        ------
        ValueError
            If any lead time is outside the valid range [-12, +11] hours
        """
        min_hours = CorrDiffCMIP6._MIN_LEAD_TIME_HOURS
        max_hours = CorrDiffCMIP6._MAX_LEAD_TIME_HOURS

        for lt in output_lead_times:
            # Convert to hours for validation
            hours = lt / np.timedelta64(1, "h")
            if hours < min_hours or hours > max_hours:
                raise ValueError(
                    f"output_lead_times must be within [{min_hours}, {max_hours}] hours, "
                    f"but got {hours}h. To get a specific valid time, adjust the input "
                    f"'time' parameter instead of using lead times outside this range. "
                    f"For example, to get 2026-12-23 00:00:00, use time=2026-12-23T12:00 "
                    f"with lead_time=-12h, not time=2026-12-22T12:00 with lead_time=+12h."
                )
