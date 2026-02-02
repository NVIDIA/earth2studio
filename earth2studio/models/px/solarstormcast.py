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

"""SolarStormCast model for solar irradiance nowcasting from GOES satellite data."""

from collections import OrderedDict
from datetime import datetime
from typing import Any, Union

import numpy as np
import torch
import xarray as xr
import zarr
from loguru import logger

from earth2studio.models.auto import Package
from earth2studio.models.px.stormcast import StormCast
from earth2studio.utils import handshake_coords, handshake_dim, handshake_size
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.interp import NearestNeighborInterpolator
from earth2studio.utils.type import CoordSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo.models import Module as PhysicsNemoModule
    from physicsnemo.utils.generative import deterministic_sampler, stochastic_sampler
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("solarstormcast")
    PhysicsNemoModule = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
    deterministic_sampler = None  # type: ignore[assignment]
    stochastic_sampler = None  # type: ignore[assignment]
    cos_zenith_angle = None  # type: ignore[assignment]


@check_optional_dependencies()
class SolarStormCast(StormCast):
    """SolarStormCast model for solar irradiance (GHI) nowcasting.

    This model extends StormCast to predict Global Horizontal Irradiance (GHI) from
    GOES satellite imagery using a regression + diffusion architecture. It accepts
    conditioning data (GOES channels) and automatically computes:
    - Solar zenith angle (cos_zenith_angle) from timestamp and lat/lon
    - Positional encodings (cos_lat, sin_lat, cos_lon, sin_lon)

    The model outputs GHI predictions on a cropped HRRR grid domain, where the
    valid region corresponds to land points with NSRDB training data.

    Parameters
    ----------
    regression_model : torch.nn.Module | None
        Optional regression model for mean prediction. If None, only diffusion is used.
    diffusion_model : torch.nn.Module
        Diffusion model (EDMPrecond) for residual prediction.
    means : torch.Tensor
        Per-variable mean for state normalization, shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std for state normalization, shape [1, C, 1, 1].
    latitudes : torch.Tensor
        Latitudes of the grid, shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, shape [H, W].
    conditioning_means : torch.Tensor | None, optional
        Means to normalize conditioning (GOES) data, shape [1, C_cond, 1, 1].
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize conditioning (GOES) data, shape [1, C_cond, 1, 1].
    invariants : torch.Tensor | None, optional
        Static invariant features (e.g., altitude, land-sea mask), shape [1, N_inv, H, W].
    valid_mask : torch.Tensor | None, optional
        Boolean mask indicating valid land points, shape [H, W].
    variables : list[str], optional
        Names of output variables. Default is ["ghi"].
    conditioning_variables : list[str], optional
        Names of conditioning variables.
    sampler_args : dict[str, Any] | None, optional
        Arguments for the diffusion sampler.
    sampler_type : str, optional
        Type of sampler: "deterministic" or "stochastic". Default is "deterministic".
    input_interp_max_dist_km : float, optional
        Maximum distance in km for nearest neighbor interpolation. Default is 12.0.

    Note
    ----
    The model is trained on NSRDB data which only exists for land points in CONUS.
    Ocean and invalid points are masked with zeros in the output.
    """

    def __init__(
        self,
        regression_model: torch.nn.Module | None,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        invariants: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        variables: list[str] | None = None,
        conditioning_variables: list[str] | None = None,
        sampler_args: dict[str, Any] | None = None,
        sampler_type: str = "deterministic",
        input_interp_max_dist_km: float = 12.0,
    ):
        # Skip StormCast.__init__ and call nn.Module directly
        # because StormCast has different required parameters (hrrr grid, etc.)
        torch.nn.Module.__init__(self)

        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.sampler_args = sampler_args or {}
        self.sampler_type = sampler_type
        self._input_interp_max_dist_km = input_interp_max_dist_km

        self.variables = np.array(variables or ["ghi"])
        self.conditioning_variables = np.array(conditioning_variables or [])
        
        # Not used by SolarStormCast but needed for compatibility
        self.conditioning_data_source = None

        # Register normalization buffers
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)

        # Lat/lon grids (replaces HRRR grid from parent)
        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self._lat_cpu_copy = latitudes.cpu().numpy()
        self._lon_cpu_copy = longitudes.cpu().numpy()
        
        # Store as lat/lon for compatibility with some parent methods
        self.lat = self._lat_cpu_copy
        self.lon = self._lon_cpu_copy

        # Y/X coordinates (pixel indices, replaces hrrr_y/hrrr_x)
        self.y = np.arange(latitudes.shape[0])
        self.x = np.arange(latitudes.shape[1])

        # Invariants (optional)
        if invariants is not None:
            self.register_buffer("invariants", invariants)
        else:
            self.invariants = None

        # Valid mask (land points)
        if valid_mask is not None:
            self.register_buffer("valid_mask", valid_mask)
        else:
            self.register_buffer(
                "valid_mask", torch.ones_like(latitudes, dtype=torch.bool)
            )

        # Conditioning normalization
        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)
        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

        # Store full domain for subsetting (must be before _compute_sincos_latlon)
        self._full_latitudes = latitudes.clone()
        self._full_longitudes = longitudes.clone()
        self._full_valid_mask = valid_mask.clone() if valid_mask is not None else None
        self._full_invariants = invariants.clone() if invariants is not None else None
        self._full_sincos_latlon = None  # Will be set by _compute_sincos_latlon
        self._y_slice = slice(None)
        self._x_slice = slice(None)

        # Precompute sincos lat/lon positional encodings (after _full_sincos_latlon is initialized)
        self._compute_sincos_latlon()

        # Interpolators
        self.input_interp: NearestNeighborInterpolator | None = None
        self._interpolators: dict[str, NearestNeighborInterpolator] = {}

        # Input/output times (single step model, 10 min timestep)
        self.input_times = np.array([np.timedelta64(0, "m")])
        self.output_times = np.array([np.timedelta64(10, "m")])

    def _compute_sincos_latlon(self) -> None:
        """Compute sin/cos lat/lon tensors for positional encodings."""
        lat_rad = torch.deg2rad(self.latitudes)
        lon_rad = torch.deg2rad(self.longitudes)
        sincos_latlon = torch.stack(
            [
                torch.cos(lat_rad),
                torch.cos(lon_rad),
                torch.sin(lat_rad),
                torch.sin(lon_rad),
            ],
            dim=0,
        )  # [4, H, W]
        self.register_buffer("sincos_latlon", sincos_latlon)
        # Store full version for subsetting
        if self._full_sincos_latlon is None:
            self._full_sincos_latlon = sincos_latlon.clone()

    @property
    def domain_shape(self) -> tuple[int, int]:
        """Return current domain shape (H, W)."""
        return tuple(self.latitudes.shape)

    @property
    def full_domain_shape(self) -> tuple[int, int]:
        """Return full (original) domain shape (H, W)."""
        return tuple(self._full_latitudes.shape)

    def subset_domain(
        self,
        y_slice: slice | None = None,
        x_slice: slice | None = None,
    ) -> None:
        """Subset the model domain to a smaller region.

        This updates all spatial buffers (lat/lon, valid_mask, invariants, sincos_latlon)
        to the subsetted domain. Interpolators will need to be rebuilt after calling this.

        Parameters
        ----------
        y_slice : slice | None
            Slice for y (latitude) dimension. If None, uses full range.
        x_slice : slice | None
            Slice for x (longitude) dimension. If None, uses full range.

        Example
        -------
        >>> model.subset_domain(y_slice=slice(100, 900), x_slice=slice(200, 1600))
        >>> print(model.domain_shape)  # (800, 1400)
        """
        device = self.latitudes.device

        if y_slice is None:
            y_slice = slice(None)
        if x_slice is None:
            x_slice = slice(None)

        self._y_slice = y_slice
        self._x_slice = x_slice

        # Subset lat/lon
        self.latitudes = self._full_latitudes[y_slice, x_slice].to(device)
        self.longitudes = self._full_longitudes[y_slice, x_slice].to(device)
        self._lat_cpu_copy = self.latitudes.cpu().numpy()
        self._lon_cpu_copy = self.longitudes.cpu().numpy()
        self.lat = self._lat_cpu_copy
        self.lon = self._lon_cpu_copy

        # Update Y/X coordinates
        self.y = np.arange(self.latitudes.shape[0])
        self.x = np.arange(self.latitudes.shape[1])

        # Subset valid mask
        if self._full_valid_mask is not None:
            self.valid_mask = self._full_valid_mask[y_slice, x_slice].to(device)
        else:
            self.valid_mask = torch.ones_like(self.latitudes, dtype=torch.bool)

        # Subset invariants
        if self._full_invariants is not None:
            self.invariants = self._full_invariants[:, :, y_slice, x_slice].to(device)

        # Subset sincos_latlon
        if self._full_sincos_latlon is not None:
            self.sincos_latlon = self._full_sincos_latlon[:, y_slice, x_slice].to(device)

        # Clear interpolators (they need to be rebuilt)
        self._interpolators.clear()
        self.input_interp = None

        logger.info(
            f"Domain subsetted: {self.full_domain_shape} -> {self.domain_shape} "
            f"(y={y_slice}, x={x_slice})"
        )

    def reset_domain(self) -> None:
        """Reset domain to the full original domain."""
        self.subset_domain(y_slice=slice(None), x_slice=slice(None))
        logger.info(f"Domain reset to full: {self.domain_shape}")

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default SolarStormCast package."""
        package = Package(
            "/output/solar_package_3km_fd",
            cache_options={
                "cache_storage": Package.default_cache("solarstormcast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    def load_model(
        cls,
        package: Package,
        sampler_type: str = "deterministic",
    ) -> "SolarStormCast":
        """Load SolarStormCast model from a package.

        Parameters
        ----------
        package : Package
            Package containing model weights and metadata.
        sampler_type : str, optional
            Type of sampler: "deterministic" or "stochastic". Default is "deterministic".

        Returns
        -------
        SolarStormCast
            Instantiated model.
        """
        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            pass

        # Load config
        config = OmegaConf.load(package.resolve("model.yaml"))

        # Load regression model (optional)
        regression = None
        try:
            regression = PhysicsNemoModule.from_checkpoint(
                package.resolve("StormCastUNet.0.0.mdlus")
            )
            logger.info("Loaded regression model")
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"No regression model found: {e}. Running in diffusion-only mode.")

        # Load diffusion model (required)
        diffusion = PhysicsNemoModule.from_checkpoint(
            package.resolve("EDMPrecond.0.0.mdlus")
        )
        diffusion.amp_mode = True
        logger.info("Loaded diffusion model")

        # Load metadata
        store = zarr.storage.ZipStore(package.resolve("metadata.zarr.zip"), mode="r")
        metadata = xr.open_zarr(store, zarr_format=2)
        variables = list(metadata["variable"].values)
        conditioning_variables = list(metadata["conditioning_variable"].values)

        # Normalization stats
        means = torch.from_numpy(metadata["means"].values[None, :, None, None]).float()
        stds = torch.from_numpy(metadata["stds"].values[None, :, None, None]).float()
        conditioning_means = torch.from_numpy(
            metadata["conditioning_means"].values[None, :, None, None]
        ).float()
        conditioning_stds = torch.from_numpy(
            metadata["conditioning_stds"].values[None, :, None, None]
        ).float()

        # Load lat/lon grids
        lat_2d = torch.from_numpy(metadata["latitude"].values).float()
        lon_2d = torch.from_numpy(metadata["longitude"].values).float()
        logger.info(f"Loaded lat/lon grids: shape={lat_2d.shape}")

        # Load invariants (optional)
        invariant_names = getattr(config.data, "invariants", [])
        invariants = None
        if invariant_names and "invariants" in metadata:
            try:
                invariants = metadata["invariants"].sel(invariant=invariant_names).values
                invariants = torch.from_numpy(invariants).float().unsqueeze(0)
            except Exception as e:
                logger.warning(f"Could not load invariants: {e}")
                invariants = None

        # Load valid mask
        valid_mask = None
        if "valid_mask" in metadata:
            valid_mask = torch.from_numpy(metadata["valid_mask"].values).bool()
            valid_fraction = valid_mask.float().mean().item()
            logger.info(f"Loaded valid mask: shape={valid_mask.shape}, valid={valid_fraction:.1%}")

        # Sampler args from config
        sampler_args = dict(config.sampler_args) if config.sampler_args else {}

        logger.info(f"Model loaded:")
        logger.info(f"  Variables: {variables}")
        logger.info(f"  Conditioning variables: {len(conditioning_variables)}")
        logger.info(f"  Invariants: {invariant_names}")
        logger.info(f"  Sampler: {sampler_type}")

        return cls(
            regression_model=regression,
            diffusion_model=diffusion,
            means=means,
            stds=stds,
            latitudes=lat_2d,
            longitudes=lon_2d,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            invariants=invariants,
            valid_mask=valid_mask,
            variables=variables,
            conditioning_variables=conditioning_variables,
            sampler_args=sampler_args,
            sampler_type=sampler_type,
        )

    def build_input_interpolator(
        self,
        source_lats: np.ndarray,
        source_lons: np.ndarray,
        max_dist_km: float | None = None,
        name: str = "default",
    ) -> None:
        """Build interpolator for regridding input data to model grid.

        Parameters
        ----------
        source_lats : np.ndarray
            Source latitude grid.
        source_lons : np.ndarray
            Source longitude grid.
        max_dist_km : float | None, optional
            Maximum distance for nearest neighbor interpolation.
        name : str, optional
            Name for this interpolator. Default is "default".
        """
        if max_dist_km is None:
            max_dist_km = self._input_interp_max_dist_km

        interp = NearestNeighborInterpolator(
            source_lats=source_lats,
            source_lons=source_lons,
            target_lats=self._lat_cpu_copy,
            target_lons=self._lon_cpu_copy,
            max_dist_km=max_dist_km,
        )
        self._interpolators[name] = interp
        if name == "default":
            self.input_interp = interp

        logger.info(
            f"Built interpolator '{name}': {source_lats.shape} -> {self.latitudes.shape}"
        )

    def interpolate(
        self,
        x: torch.Tensor,
        fill_value: float = 0.0,
        interpolator: str = "default",
    ) -> torch.Tensor:
        """Interpolate input data to model grid.

        Parameters
        ----------
        x : torch.Tensor
            Input data with shape [..., H_src, W_src].
        fill_value : float, optional
            Value to fill NaN with. Default is 0.0.
        interpolator : str, optional
            Name of interpolator to use. Default is "default".

        Returns
        -------
        torch.Tensor
            Interpolated data with shape [..., H_tgt, W_tgt].
        """
        if interpolator not in self._interpolators:
            if interpolator == "default" and self.input_interp is not None:
                interp = self.input_interp
            else:
                raise ValueError(
                    f"Interpolator '{interpolator}' not found. "
                    f"Available: {list(self._interpolators.keys())}. "
                    "Call build_input_interpolator first."
                )
        else:
            interp = self._interpolators[interpolator]

        # Fill NaN values before interpolation (GOES has NaN outside Earth's disk)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=fill_value)

        result = interp(x)
        return torch.nan_to_num(result, nan=fill_value)

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state tensor."""
        return (x - self.means) / (self.stds + 1e-8)

    def denormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize state tensor."""
        return x * self.stds + self.means

    def normalize_conditioning(self, c: torch.Tensor) -> torch.Tensor:
        """Normalize conditioning tensor (GOES channels only)."""
        if hasattr(self, "conditioning_means") and hasattr(self, "conditioning_stds"):
            n_norm = self.conditioning_means.shape[1]
            c_goes = c[..., :n_norm, :, :]
            c_rest = c[..., n_norm:, :, :]
            c_goes_norm = (c_goes - self.conditioning_means) / (
                self.conditioning_stds + 1e-8
            )
            c = torch.cat([c_goes_norm, c_rest], dim=-3)
        return c

    def compute_sza(self, timestamp: Union[datetime, np.datetime64]) -> torch.Tensor:
        """Compute cosine of solar zenith angle for the given timestamp.

        Parameters
        ----------
        timestamp : datetime | np.datetime64
            Timestamp for SZA computation.

        Returns
        -------
        torch.Tensor
            Cosine of solar zenith angle, shape [H, W].
        """
        if isinstance(timestamp, np.datetime64):
            timestamp = timestamp.astype("datetime64[us]").astype(datetime)

        H, W = self.latitudes.shape
        sza_flat = cos_zenith_angle(timestamp, self._lon_cpu_copy, self._lat_cpu_copy)
        sza_2d = sza_flat.reshape(H, W).astype(np.float32)

        return torch.from_numpy(sza_2d).to(self.latitudes.device)

    def build_conditioning(
        self,
        goes_data: torch.Tensor,
        timestamp: Union[datetime, np.datetime64],
    ) -> torch.Tensor:
        """Build the full conditioning tensor from GOES data and timestamp.

        Combines:
        - GOES data
        - Invariants (if available)
        - Solar zenith angle
        - Positional encodings (cos_lat, cos_lon, sin_lat, sin_lon)

        Parameters
        ----------
        goes_data : torch.Tensor
            GOES satellite data, shape [B, L, C, H, W] or [B, C, H, W].
        timestamp : datetime | np.datetime64
            Timestamp for the data.

        Returns
        -------
        torch.Tensor
            Full conditioning tensor, shape [B, 1, C_total, H, W].
        """
        B = goes_data.shape[0]
        device = goes_data.device

        # Compute SZA
        cos_sza = self.compute_sza(timestamp).to(device)
        cos_sza = cos_sza.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Sincos latlon
        sincos = self.sincos_latlon.to(device)
        sincos = sincos.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Invariants
        if self.invariants is not None:
            inv = self.invariants.to(device).unsqueeze(0).expand(B, -1, -1, -1, -1)
        else:
            inv = None

        # Ensure goes_data is 5D [B, L, C, H, W]
        if goes_data.dim() == 4:
            goes_data = goes_data.unsqueeze(1)

        # Concatenate: [GOES, invariants, cos_sza, sincos]
        if inv is not None:
            conditioning = torch.cat([goes_data, inv, cos_sza, sincos], dim=2)
        else:
            conditioning = torch.cat([goes_data, cos_sza, sincos], dim=2)

        return conditioning

    def apply_valid_mask(
        self,
        x: torch.Tensor,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Apply valid mask to predictions.

        Parameters
        ----------
        x : torch.Tensor
            Predictions with shape [..., H, W].
        fill_value : float, optional
            Value for invalid pixels. Default is 0.0.

        Returns
        -------
        torch.Tensor
            Masked predictions.
        """
        if self.valid_mask is None:
            return x

        mask = self.valid_mask.to(x.device)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(0)

        return torch.where(mask.expand_as(x), x, torch.full_like(x, fill_value))

    def get_valid_mask_numpy(self) -> np.ndarray | None:
        """Return valid mask as numpy array for plotting."""
        if self.valid_mask is None:
            return None
        return self.valid_mask.cpu().numpy()

    def input_coords(self) -> CoordSystem:
        """Input coordinate system."""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.input_times,
                "variable": self.variables,
                "y": self.y,
                "x": self.x,
            }
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        CoordSystem
            Output coordinate system.
        """
        output_coords = OrderedDict(
            {
                "batch": input_coords.get("batch", np.empty(0)),
                "time": input_coords.get("time", np.empty(0)),
                "lead_time": self.output_times + input_coords["lead_time"][-1],
                "variable": self.variables,
                "y": self.y,
                "x": self.x,
            }
        )
        return output_coords

    @torch.inference_mode()
    def _forward(
        self,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        first_step: bool = False,
    ) -> torch.Tensor:
        """Run one prediction step.

        This overrides the parent StormCast._forward with SolarStormCast-specific logic.

        Parameters
        ----------
        state : torch.Tensor
            Previous state (GHI), shape [B, C, H, W] or [B, 1, C, H, W].
        conditioning : torch.Tensor
            Conditioning tensor, shape [B, 1, C_cond, H, W].
        first_step : bool, optional
            If True, skip state normalization. Default is False.

        Returns
        -------
        torch.Tensor
            Predicted GHI, shape [B, 1, C, H, W].
        """
        # Handle 5D input
        if state.dim() == 5:
            state = state[:, 0]

        B, C, H, W = state.shape

        # Normalize state
        if first_step:
            state_norm = state.unsqueeze(1)
        else:
            state_norm = self.normalize_state(state).unsqueeze(1)

        # Normalize conditioning
        cond_norm = self.normalize_conditioning(conditioning)

        # Build diffusion condition: [state, background]
        diffusion_cond = torch.cat([state_norm, cond_norm], dim=2)

        # Regression model (optional)
        if self.regression_model is not None:
            reg_input = cond_norm[:, 0]
            reg_out = self.regression_model(reg_input).unsqueeze(1)
        else:
            reg_out = torch.zeros_like(state_norm)

        # Diffusion model
        latents = torch.randn(B, C, H, W, device=state.device, dtype=state.dtype)
        diffusion_input = diffusion_cond[:, 0]

        if self.sampler_type == "stochastic":
            edm_out = stochastic_sampler(
                self.diffusion_model,
                latents=latents,
                img_lr=diffusion_input,
                **self.sampler_args,
            )
        else:
            edm_out = deterministic_sampler(
                self.diffusion_model,
                latents=latents,
                img_lr=diffusion_input,
                **self.sampler_args,
            )

        if edm_out.dim() == 4:
            edm_out = edm_out.unsqueeze(1)

        # Combine regression and diffusion
        out_norm = reg_out + edm_out

        # Denormalize
        out = self.denormalize_state(out_norm)
        out = torch.clamp(out, min=0)  # GHI cannot be negative

        # Apply valid mask
        out = self.apply_valid_mask(out, fill_value=0.0)

        # Mask where GOES unavailable (first channel is zero)
        goes_ch0 = conditioning[:, :, 0:1, :, :]
        goes_unavailable = goes_ch0 == 0
        out = torch.where(goes_unavailable.expand_as(out), torch.zeros_like(out), out)

        return out

    @torch.inference_mode()
    def __call__(
        self,
        state: torch.Tensor,
        conditioning: torch.Tensor,
        first_step: bool = False,
    ) -> torch.Tensor:
        """Run one prediction step.

        This is the main inference method for SolarStormCast.
        Note: This signature differs from the parent StormCast.__call__ because
        SolarStormCast expects pre-built conditioning rather than fetching it.

        Parameters
        ----------
        state : torch.Tensor
            Previous state (GHI).
        conditioning : torch.Tensor
            Full conditioning tensor (from build_conditioning).
        first_step : bool, optional
            If True, this is the first step (state is zeros).

        Returns
        -------
        torch.Tensor
            Predicted GHI.
        """
        return self._forward(state, conditioning, first_step=first_step)
