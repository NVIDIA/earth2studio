from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
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
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo.models import Module as PhysicsNemoModule
    from physicsnemo.utils.corrdiff import diffusion_step, regression_step
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:  # pragma: no cover
    OptionalDependencyFailure("corrdiff")
    diffusion_step = None  # type: ignore[assignment]
    regression_step = None  # type: ignore[assignment]
    cos_zenith_angle = None  # type: ignore[assignment]


class CorrDiffCMIP6New(CorrDiff):
    """CorrDiff model variant for CMIP6 data.

    This class extends the base CorrDiff model to work with CMIP6 climate model data.
    It provides access to time information in preprocessing for time-dependent operations.

    Key differences from base CorrDiff:
    - Adds time-dependent features (solar zenith angle, hour of day) to inputs
    - Uses relaxed grid validation tolerances suitable for Gaussian grids (1% spacing, 5% bounds)
    - Overrides default interpolation
    - Includes a time dimension in the coordinate system so timestamps are preserved through
      batching/compression and can be used for time-dependent features (e.g., solar zenith angle)

    Note
    ----
    Unlike CorrDiffTaiwan which has fixed input/output variables, CorrDiffCMIP6
    loads variables names from the model package. Input variables are
    expanded based on the ``time_window`` configuration in metadata.json
    (e.g., "t2m" → "t2m_t-6", "t2m_t-3", "t2m_t+0"). After loading, inspect
    ``model.input_variables`` and ``model.output_variables`` for the actual
    variable lists.
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
        time_window: dict | None = None,
    ) -> None:
        """Initialize CorrDiffCMIP6 model.

        Parameters
        ----------
        input_variables : Sequence[str]
            List of input variable names (time-windowed, e.g., "tas_t-1", "tas_t", "tas_t+1")
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
        time_window : dict | None, optional
            Time window configuration from metadata.json containing "offsets", "suffixes",
            "offsets_units", and "group_by". Used for create_time_window_wrapper(), by default None
        """
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

        # Store time_window config for create_time_window_wrapper() and input_coords()
        self.time_window = time_window
        print(time_window)
        self.time_suffixes = time_window.get("suffixes") if time_window else None

        # Preprocess caches (built lazily on first use)
        self._cmip6_var_index: dict[str, int] | None = None
        self._cmip6_sai_kernel: torch.Tensor | None = None
        self._cmip6_lonlat_meshgrid: tuple[np.ndarray, np.ndarray] | None = None
        self._cmip6_reorder_indices: list[int] | None = None

        # When True, CorrDiffCMIP6 will accumulate multi-sample outputs on CPU to reduce GPU peak
        # memory for large `number_of_samples` / large output channel counts.
        # This does not change the generated samples, only where the final stacked tensor lives.
        self.stream_samples_to_cpu: bool = False

        self.in_center = torch.cat(
            [
                self.in_center[:, : -len(self.invariant_variables)].repeat_interleave(  # type: ignore
                    3, dim=1
                ),
                self.in_center[:, -len(self.invariant_variables) :],  # type: ignore
            ],
            dim=1,
        )
        self.in_scale = torch.cat(
            [
                self.in_scale[:, : -len(self.invariant_variables)].repeat_interleave(  # type: ignore
                    3, dim=1
                ),
                self.in_scale[:, -len(self.invariant_variables) :],  # type: ignore
            ],
            dim=1,
        )

        # Extend in_center and in_scale to include time features (sza, hod) at the end
        # Note: During training, the last invariant position (coslat) mistakenly had hod VALUES,
        # but was normalized using coslat STATISTICS. This bug is replicated in preprocess_input
        # by putting hod values in the coslat position during channel reordering.
        if time_feature_center is not None and time_feature_scale is not None:
            # Reshape time features to match the 4D format [1, N, 1, 1]
            time_feature_center = time_feature_center.view(1, -1, 1, 1)
            time_feature_scale = time_feature_scale.view(1, -1, 1, 1)

            # print(self.in_center.shape)  # Should be torch.Size([1, 229, 1, 1])
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
            by default None, will use self.input_coords.

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

        if input_coords["time"].shape[0] != 1:
            raise ValueError('CorrDiffCMIP6 only supports a singleton "time" axis.')

        output_coords = OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "variable": np.array(self.output_variables),
                "lat": self.lat_output_numpy,
                "lon": self.lon_output_numpy,
            }
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Return the default pre-trained CorrDiffCMIP6 package.

        Notes
        -----
        The canonical NGC URI is not yet finalized.
        """
        package = Package(
            "ngc://models/<org>/<team>/<model>@<version>",
            cache_options={
                "cache_storage": Package.default_cache("corrdiff_cmip6"),
                "same_names": True,
            },
        )
        raise NotImplementedError(
            "CorrDiffCMIP6 default package URI is not configured yet. "
            "Please replace the placeholder URI in CorrDiffCMIP6.load_default_package()."
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package, device: str = "cpu") -> DiagnosticModel:
        """Load CorrDiffCMIP6 model from package with time feature normalization.

        This method extends the base CorrDiff loading to include time feature
        normalization parameters (sza and hod) from stats.json.

        Parameters
        ----------
        package : Package
            Package containing model weights and configuration

        Returns
        -------
        DiagnosticModel
            Initialized CorrDiffCMIP6 model
        """
        # Load and validate metadata first (we need time_window for input expansion).
        metadata = cls._load_json_from_package(package, "metadata.json")
        time_window_raw = metadata.get("time_window")
        if time_window_raw is None:
            raise ValueError(
                "metadata.json is missing required 'time_window' configuration."
            )
        time_window = cls._validate_time_window_metadata(time_window_raw)
        stats = cls._load_json_from_package(package, "stats.json")

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

        # Apply inference optimizations (following CorrDiffTaiwan patterns)
        # Disable profiling mode for both models
        residual.profile_mode = False
        regression.profile_mode = False

        # Convert to channels_last memory format for better GPU performance
        residual = residual.to(memory_format=torch.channels_last)
        regression = regression.to(memory_format=torch.channels_last)

        # Configure torch dynamo for potential compilation
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
        seed = metadata.get("seed", None)
        sigma_min_metadata = metadata.get(
            "sigma_min", None
        )  # TODO: Add override with load_model
        sigma_max_metadata = metadata.get(
            "sigma_max", None
        )  # TODO: Add override with load_model
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

        in_center = torch.tensor(in_center_values)
        in_scale = torch.tensor(in_scale_values)

        # Load output normalization parameters
        out_center = torch.tensor(
            [stats["output"][v]["mean"] for v in output_variables]
        )
        out_scale = torch.tensor([stats["output"][v]["std"] for v in output_variables])

        with xr.open_dataset(package.resolve("output_latlon_grid.nc")) as ds:
            lat_output_grid = torch.as_tensor(np.array(ds["lat"][:]))
            lon_output_grid = torch.as_tensor(np.array(ds["lon"][:]))

            # Validate output grid format and ordering
            cls._validate_grid_format(
                lat_output_grid, lon_output_grid, grid_name="output"
            )

        with xr.open_dataset(package.resolve("input_latlon_grid.nc")) as ds:
            lat_input_grid = torch.as_tensor(np.array(ds["lat"][:]))
            lon_input_grid = torch.as_tensor(np.array(ds["lon"][:]))

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
                (var_name, torch.as_tensor(np.array(ds[var_name])))
                for var_name in var_names
            )
            # Load invariant normalization parameters
            invariant_center = torch.tensor(
                [stats["invariants"][v]["mean"] for v in invariants]
            )
            invariant_scale = torch.tensor(
                [stats["invariants"][v]["std"] for v in invariants]
            )
            print(invariant_center.shape)

        # Create time feature normalization tensors on the same device as the base buffers.
        time_feature_center = torch.as_tensor(
            [stats["input"]["sza"]["mean"], stats["input"]["hod"]["mean"]],
        )
        time_feature_scale = torch.as_tensor(
            [stats["input"]["sza"]["std"], stats["input"]["hod"]["std"]],
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
            seed=seed,
            time_feature_center=time_feature_center,
            time_feature_scale=time_feature_scale,
            time_window=time_window,
            grid_spacing_tolerance=grid_spacing_tolerance,
            grid_bounds_margin=grid_bounds_margin,
            sigma_min=sigma_min_metadata,
            sigma_max=sigma_max_metadata,
        )

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)
        out_shape = tuple(len(v) for v in output_coords.values())
        out = torch.empty(out_shape, device=x.device, dtype=torch.float32)
        for i in range(out.shape[1]):  # Loop through time stamps
            valid_time = timearray_to_datetime(coords["time"])[i] - timedelta(hours=12)
            out[i] = self._forward(x[:, i], valid_time)
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
        print(x.shape, len(self.input_variables))
        for i in range(lead_len):
            # [B, 1, H, W] so padding/conv2d operate on 4D tensors
            siconc = torch.nan_to_num(x[:, siconc_channel, i], nan=0.0)
            snc = torch.nan_to_num(x[:, snc_channel, i], nan=0.0)
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
        )
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
        """Complete input preprocessing pipeline with optional time information.

        Performs interpolation to output grid, adds batch dimension,
        concatenates invariants if available, and normalizes the input.

        The ``valid_time`` parameter is used for time-dependent preprocessing
        operations (e.g., solar zenith angle, hour-of-day features).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [C, H_in, W_in]
        valid_time : datetime | None, optional
            Validity time associated with this input sample, required for
            time-dependent features in CorrDiffCMIP6. Default is None.

        Returns
        -------
        torch.Tensor
            Preprocessed and normalized input tensor [1, C+C_inv, H_out, W_out]

        Raises
        ------
        ValueError
            If valid_time is None (required for CorrDiffCMIP6)
        """
        if valid_time is None:
            raise ValueError(
                "CorrDiffCMIP6 requires valid_time for time-dependent features"
            )

        # [B, L, C, H, W] -> [B, C, L, H, W]
        print(x.shape, "====")
        x = x.transpose(1, 2)

        # 1) Sea-ice/snow derived feature smoothing (in-place update) per lead time
        x = self._apply_sai_cover(x)
        torch.save(x, "sai_input_new.pt")

        # 3) Flatten (variable, lead_time) -> channel to match model expectations
        B, C, L, H, W = x.shape
        x = x.contiguous().view(B, -1, H, W)  # [C*L, H, W]

        # 2) Interpolate each lead time slice to output grid
        x = F.interpolate(x, self.img_shape, mode="bilinear")
        print(x.shape)

        # 4) Concatenate invariants if available (single set, not per lead)
        if self.invariants is not None:
            x = torch.concat([x, torch.flip(self.invariants.unsqueeze(0), [2])], dim=1)
        print(x.shape, "~~`")
        torch.save(x, "invars_input_new.pt")

        # 5) Time-dependent features (SZA + HOD) appended after invariants
        x = self._add_time_features(x, valid_time)
        print(x.shape, "===")
        print(valid_time)  # Should be 2037-09-06 00:00:00
        torch.save(x, "times_input_new.pt")

        # 6) Normalize + pad + reorder channels
        x = self._normalize_pad_reorder(x)
        print(x.shape)
        # Debug: expose final channel ordering by rebuilding names from
        # (input_variables + invariant_variables + ["sza", "hod"]) and applying reorder indices.
        # pre_names = (
        #     list(self.input_variables) + list(self.invariant_variables) + ["sza", "hod"]
        # )
        # indices = self._get_reorder_indices()
        # if indices and max(indices) >= len(pre_names):
        #     raise RuntimeError(
        #         "Internal error: channel reorder indices are inconsistent with channel naming."
        #     )
        # self._last_preprocess_channel_names = [pre_names[i] for i in indices]

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
        # 1) Crop padding added during preprocessing (see _LAT_PAD, _LON_PAD)
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

        torch.save(x.cpu(), "input1_new.pt")
        # Preprocess input (CMIP6 requires valid_time)
        image_lr = self.preprocess_input(x, valid_time)
        print(image_lr.shape)  # ([1, 83, 768, 1536])
        image_lr = image_lr.to(torch.float32).to(memory_format=torch.channels_last)

        torch.save(image_lr.cpu(), "input2_new.pt")
        # Regression model (mean)
        image_reg = None
        if self.regression_model:
            latents_shape = (1, len(self.output_variables), *image_lr.shape[-2:])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_reg = regression_step(
                    net=self.regression_model,
                    img_lr=image_lr,
                    latents_shape=latents_shape,
                )

        # Validate required models
        if image_reg is None:
            raise RuntimeError(
                "Missing regression output: regression_model must be set."
            )

        # Regression-only: all samples are identical (deterministic mean)
        if self.inference_mode == "regression":
            out = self.postprocess_output(image_reg)
            out_device = (
                torch.device("cpu") if self.stream_samples_to_cpu else out.device
            )
            out = out.to(out_device)
            return out.expand(self.number_of_samples, -1, -1, -1).clone()

        # inference_mode == "both": need diffusion model
        if self.residual_model is None:
            raise RuntimeError(
                "Missing diffusion model: residual_model must be set for inference_mode='both'."
            )

        # Compute base seed once (sample index added in loop)
        seed0 = (
            int(self.seed) if self.seed is not None else int(np.random.randint(2**32))
        )
        mean_hr = image_reg[:1] if self.hr_mean_conditioning else None

        # Where to accumulate samples (CPU streaming reduces GPU peak memory)
        out_device = (
            torch.device("cpu") if self.stream_samples_to_cpu else image_lr.device
        )

        out = None

        for i in range(self.number_of_samples):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_res = diffusion_step(
                    net=self.residual_model,
                    sampler_fn=self.sampler,
                    img_shape=image_lr.shape[-2:],
                    img_out_channels=len(self.output_variables),
                    rank_batches=[[seed0 + i]],
                    img_lr=image_lr,
                    rank=1,
                    device=image_lr.device,
                    mean_hr=mean_hr,
                )
            torch.save(image_reg, "image_reg_new.pt")
            torch.save(image_res, "image_res_new.pt")
            yi = self.postprocess_output(image_reg + image_res)
            if out is None:
                out = torch.empty(
                    (self.number_of_samples, yi.shape[1], yi.shape[2], yi.shape[3]),
                    device=out_device,
                    dtype=yi.dtype,
                )
            out[i] = yi.to(out_device)[0]

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
        # Register grid coordinates and validate
        self.register_buffer("lat_input_grid", lat_input_grid)
        self.register_buffer("lon_input_grid", lon_input_grid)
        self.register_buffer("lat_output_grid", lat_output_grid)
        self.register_buffer("lon_output_grid", lon_output_grid)

        self._interpolator = None  # Use efficient regular grid interpolation

        self.lat_input_numpy = lat_input_grid.cpu().numpy()
        self.lon_input_numpy = lon_input_grid.cpu().numpy()
        self.lat_output_numpy = lat_output_grid.cpu().numpy()
        self.lon_output_numpy = lon_output_grid.cpu().numpy()

        if invariants:
            self.invariant_variables = list(invariants.keys())
            self.register_buffer(
                "invariants", torch.stack(list(invariants.values()), dim=0)
            )
        # Combine input normalization with invariants
        in_center = torch.concat([in_center, invariant_center], dim=0)
        in_scale = torch.concat([in_scale, invariant_scale], dim=0)

        # Register normalization parameters
        num_inputs = len(self.input_variables) + len(self.invariant_variables)
        self.register_buffer("in_center", in_center.view(1, num_inputs, 1, 1))
        self.register_buffer("in_scale", in_scale.view(1, num_inputs, 1, 1))
        self.register_buffer(
            "out_center", out_center.view(1, len(self.output_variables), 1, 1)
        )
        self.register_buffer(
            "out_scale", out_scale.view(1, len(self.output_variables), 1, 1)
        )

    @staticmethod
    def _validate_time_window_metadata(time_window: dict) -> dict:
        """Validate and normalize time window metadata loaded from package."""
        offsets = list(time_window["offsets"])
        suffixes = [str(suffix) for suffix in time_window["suffixes"]]
        offsets_units = time_window.get("offsets_units", "seconds")
        group_by = time_window.get("group_by", "variable")
        validated = {
            "offsets": offsets,
            "suffixes": suffixes,
            "offsets_units": offsets_units,
            "group_by": group_by,
        }
        if "description" in time_window:
            validated["description"] = time_window["description"]
        return validated
