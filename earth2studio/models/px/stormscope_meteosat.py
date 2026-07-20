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

from collections import OrderedDict
from collections.abc import Callable, Generator
from datetime import datetime
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.utils.zenith_angle import zenith_azimuth_angles
except ImportError:
    OptionalDependencyFailure("stormscope")
    OmegaConf = None


# Variables used in StormCastMTG
VARIABLES = (
    "fci04vis",
    "fci05vis",
    "fci06vis",
    "fci08vis",
    "fci09vis",
    "fci13nir",
    "fci16nir",
    "fci22nir",
    "fci38ir",
    "fci63wv",
    "fci73wv",
    "fci87ir",
    "fci97ir",
    "fci105ir",
    "fci123ir",
    "fci133ir",
)


@check_optional_dependencies()
class StormScopeMeteosatEU(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Generative diffusion nowcasting model for MTG-I1 FCI satellite imagery.

    Predicts the next 10-minute MTG Full Combined Imager (FCI) frame from a sliding
    window of ``L`` consecutive input frames. The model uses a two-stage denoising
    strategy: a high-sigma denoiser captures coarse structure and a low-sigma denoiser
    refines fine details. Solar geometry angles are computed on the fly and provided as
    conditioning alongside static invariant fields.

    The model operates on a rectangular sub-region of the MTG full-disk image specified
    by ``mtg_ylim`` and ``mtg_xlim`` in native MTG pixel coordinates.

    Parameters
    ----------
    model_low : torch.nn.Module
        Denoising sub-model applied at low noise levels (sigma < ``sigma_threshold``).
    model_high : torch.nn.Module
        Denoising sub-model applied at high noise levels (sigma ≥ ``sigma_threshold``).
    means : torch.Tensor
        Per-channel mean of raw pixel counts used for normalisation, shape ``(C, 1, 1)``.
    stds : torch.Tensor
        Per-channel standard deviation of raw pixel counts used for normalisation,
        shape ``(C, 1, 1)``.
    scale_factor : torch.Tensor
        Per-channel multiplicative factor for the raw-to-physical radiance conversion,
        shape ``(C, 1, 1)``.
    add_offset : torch.Tensor
        Per-channel additive offset for the raw-to-physical radiance conversion,
        shape ``(C, 1, 1)``.
    invariants : torch.Tensor
        Static invariant fields (e.g. orography, land-sea mask) covering the full
        package extent, shape ``(1, C_inv, H_pkg, W_pkg)``.
    lat : torch.Tensor
        Latitude grid for the full package data extent, shape ``(H_pkg, W_pkg)``.
    lon : torch.Tensor
        Longitude grid for the full package data extent, shape ``(H_pkg, W_pkg)``.
    earth_mask : torch.Tensor
        Boolean mask where ``True`` indicates pixels over Earth (land or ocean),
        shape ``(H_pkg, W_pkg)``.
    mtg_y : np.ndarray
        MTG full-disk pixel row indices for the package domain, shape ``(H_pkg,)``.
    mtg_x : np.ndarray
        MTG full-disk pixel column indices for the package domain, shape ``(W_pkg,)``.
    mtg_ylim : tuple[int, int], optional
        Row pixel range ``(y_start, y_end)`` of the inference sub-region within the
        full MTG disk, by default ``(4320, 5440)``
    mtg_xlim : tuple[int, int], optional
        Column pixel range ``(x_start, x_end)`` of the inference sub-region within
        the full MTG disk, by default ``(1856, 4288)``
    inference_mtg_box : tuple[tuple[int, int], tuple[int, int]], optional
        Row/column bounding box of the loaded package data within the full MTG disk,
        used to compute array indices into package buffers,
        by default ``StormScopeMeteosatEU.Model_FCI_BBox``
    variables : np.ndarray, optional
        Channel names corresponding to the 16 FCI bands,
        by default ``np.array(VARIABLES)``
    sampler_args : dict[str, float | int] | None, optional
        Overrides for the EDM sampler/scheduler. Recognised keys:
        ``sigma_min``, ``sigma_max``, ``rho`` (scheduler) and
        ``S_churn``, ``S_min``, ``S_max``, ``S_noise`` (solver).
        Unspecified keys use sensible defaults, by default None
    input_times : np.ndarray, optional
        Context-window time offsets relative to the analysis time; each element is a
        ``np.timedelta64``. The number of elements ``L`` determines the sliding-window
        length, by default a 6-frame window ``[-50 min, …, 0 min]``
    output_times : np.ndarray, optional
        Output time offsets relative to the analysis time; single-step prediction
        only, by default ``[10 min]``
    ir_38_warm_scale_factor : float, optional
        Multiplier applied to the above-threshold branch of the ir_38 channel after
        raw-to-physical conversion, by default 0.024222141
    ir_38_warm_threshold : float, optional
        Raw digital-count boundary (the 12-bit maximum) above which the warm/HDR
        ir_38 scaling is applied, by default 4095.0
    num_diffusion_steps : int, optional
        Number of EDM diffusion sampling steps, by default 18
    sigma_threshold : float, optional
        Noise level that splits the forward pass between ``model_high``
        (sigma ≥ sigma_threshold) and ``model_low`` (sigma < sigma_threshold),
        by default 1.0
    batch_size : int, optional
        Maximum number of samples processed per forward pass, by default 1
    use_amp : bool, optional
        Whether to use automatic mixed precision (bfloat16) during the diffusion
        forward pass, by default True

    Badges
    ------
    region:eu class:nwc product:sat
    """

    Model_FCI_BBox: tuple[tuple[int, int], tuple[int, int]] = (
        (4320, 5440),
        (1856, 4288),
    )

    def __init__(
        self,
        model_low: torch.nn.Module,
        model_high: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        scale_factor: torch.Tensor,
        add_offset: torch.Tensor,
        invariants: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        earth_mask: torch.Tensor,
        mtg_y: np.ndarray,
        mtg_x: np.ndarray,
        mtg_ylim: tuple[int, int] = Model_FCI_BBox[0],
        mtg_xlim: tuple[int, int] = Model_FCI_BBox[1],
        inference_mtg_box: tuple[tuple[int, int], tuple[int, int]] = Model_FCI_BBox,
        variables: np.ndarray = np.array(VARIABLES),
        sampler_args: dict[str, float | int] | None = None,
        input_times: np.ndarray = np.arange(-5, 1) * np.timedelta64(10, "m"),
        output_times: np.ndarray = np.array([np.timedelta64(10, "m")]),
        ir_38_warm_scale_factor: float = 0.024222141,
        ir_38_warm_threshold: float = 4095.0,
        num_diffusion_steps: int = 48,
        sigma_threshold: float = 1.0,
        batch_size: int = 1,
        use_amp: bool = True,
    ):
        super().__init__()
        self.model_low = model_low
        self.model_high = model_high
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("scale_factor", scale_factor)
        self.register_buffer("add_offset", add_offset)
        self.register_buffer("invariants", invariants)
        self.sampler_args = {
            "sigma_min": 0.001,
            "sigma_max": 300,
            "sigma_data": 1.0,
            "rho": 7,
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": float("inf"),
            "S_noise": 1,
        }
        if sampler_args is not None:
            self.sampler_args.update(sampler_args)

        # Slice lat/lon/invariants to the requested sub-region of the package data.
        # Package data covers inference_mtg_box; compute indices relative to its origin.
        self.mtg_ylim = mtg_ylim
        self.mtg_xlim = mtg_xlim
        self._yi0 = mtg_ylim[0] - inference_mtg_box[0][0]
        self._yi1 = mtg_ylim[1] - inference_mtg_box[0][0]
        self._xi0 = mtg_xlim[0] - inference_mtg_box[1][0]
        self._xi1 = mtg_xlim[1] - inference_mtg_box[1][0]

        self.lat = lat[self._yi0 : self._yi1, self._xi0 : self._xi1]
        self.lon = lon[self._yi0 : self._yi1, self._xi0 : self._xi1]
        self.earth_mask = earth_mask[self._yi0 : self._yi1, self._xi0 : self._xi1]
        self.mtg_y = mtg_y[self._yi0 : self._yi1]
        self.mtg_x = mtg_x[self._xi0 : self._xi1]

        self.variables = variables

        # store some data needed for GPU ops in buffers
        self.register_buffer("lat_tensor", torch.as_tensor(self.lat))
        self.register_buffer("lon_tensor", torch.as_tensor(self.lon))
        self.register_buffer("off_earth_mask_tensor", torch.as_tensor(~self.earth_mask))

        # precalculate some normalization factors
        self.register_buffer("mean_inv_std", -self.means / self.stds)
        self.register_buffer("inv_std", 1.0 / self.stds)
        self.register_buffer("offset_inv_scale", -self.add_offset / self.scale_factor)
        self.register_buffer("inv_scale", 1.0 / self.scale_factor)

        # min and max values for clamping
        num_variables = len(variables)
        min_values = self.raw_to_normalized(
            torch.as_tensor([0] * num_variables).reshape(1, num_variables, 1, 1)
        )
        max_values = torch.as_tensor([4095] * num_variables)
        self.ir_38_channel: int | None = None
        if "fci38ir" in variables:
            # unlike other MTG-FCI channels, ir_38 can have uint16 values up to 2**13-1 == 8191
            # values > 2**12-1 == 4095 use a different scaling
            self.ir_38_channel = list(variables).index("fci38ir")
            self.register_buffer(
                "ir_38_warm_multiplier",
                torch.as_tensor(
                    ir_38_warm_scale_factor / scale_factor[self.ir_38_channel]
                ).to(dtype=torch.float32),
            )
            self.register_buffer(
                "ir_38_warm_threshold",
                torch.as_tensor(ir_38_warm_threshold).to(dtype=torch.float32),
            )
            max_values[self.ir_38_channel] = 8191

        max_values = self.raw_to_normalized(max_values.reshape(1, num_variables, 1, 1))
        self.register_buffer("min_values", min_values)
        self.register_buffer("max_values", max_values)

        _scheduler_keys = {"sigma_min", "sigma_max", "sigma_data", "rho"}
        self.scheduler = EDMNoiseScheduler(
            **{k: v for k, v in self.sampler_args.items() if k in _scheduler_keys}
        )

        self.num_diffusion_steps = num_diffusion_steps
        self.sigma_threshold = sigma_threshold
        self.batch_size = batch_size
        self.use_amp = use_amp

        self.input_times = input_times
        self.output_times = output_times

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.input_times,
                "variable": np.array(self.variables),
                "y": self.mtg_y,
                "x": self.mtg_x,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        CoordSystem
            Output coordinate system with ``lead_time`` advanced by one step.
        """

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.output_times,
                "variable": np.array(self.variables),
                "y": self.mtg_y,
                "x": self.mtg_x,
            }
        )

        target_input_coords = self.input_coords()

        handshake_dim(input_coords, "x", 5)
        handshake_dim(input_coords, "y", 4)
        handshake_dim(input_coords, "variable", 3)
        # Index coords are arbitrary as long as they are on the MTG grid, so just check size
        handshake_size(input_coords, "y", self.lat.shape[0])
        handshake_size(input_coords, "x", self.lat.shape[1])
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = self.output_times + input_coords["lead_time"][-1]
        return output_coords

    def compile_model(self) -> None:
        """Compile the diffusion model forward pass with ``torch.compile``."""
        self.model_low.forward = torch.compile(self.model_low.forward, dynamic=False)
        self.model_high.forward = torch.compile(self.model_high.forward, dynamic=False)

    def raw_to_physical(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Convert raw MTG data file values to physical values."""
        if self.ir_38_channel is not None:
            if not in_place:
                x = x.clone()
            x_ir_38 = x[..., self.ir_38_channel, :, :]
            ir_38_mask = x_ir_38 > self.ir_38_warm_threshold
            x_ir_38[ir_38_mask] = (
                x_ir_38[ir_38_mask] - self.ir_38_warm_threshold
            ) * self.ir_38_warm_multiplier + self.ir_38_warm_threshold

        return torch.addcmul(
            self.add_offset, x, self.scale_factor, out=x if in_place else None
        )

    def physical_to_raw(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Convert physical values to raw MTG data file values."""
        x = torch.addcmul(
            self.offset_inv_scale, x, self.inv_scale, out=x if in_place else None
        )

        if self.ir_38_channel is not None:
            x_ir_38 = x[..., self.ir_38_channel, :, :]
            ir_38_mask = x_ir_38 > self.ir_38_warm_threshold
            x_ir_38[ir_38_mask] = (x_ir_38[ir_38_mask] - self.ir_38_warm_threshold) * (
                1.0 / self.ir_38_warm_multiplier
            ) + self.ir_38_warm_threshold

        return x

    def raw_to_normalized(
        self, x: torch.Tensor, in_place: bool = False
    ) -> torch.Tensor:
        """Convert ``x`` from raw values to zero mean, unit variance."""
        return torch.addcmul(
            self.mean_inv_std, x, self.inv_std, out=x if in_place else None
        )

    def normalized_to_raw(
        self, x: torch.Tensor, in_place: bool = False
    ) -> torch.Tensor:
        """Convert ``x`` from zero mean, unit variance to raw values."""
        return torch.addcmul(self.means, x, self.stds, out=x if in_place else None)

    def normalize(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Normalise ``x`` to zero mean, unit variance."""
        x = self.physical_to_raw(x, in_place=in_place)
        x = self.raw_to_normalized(x, in_place=True)
        x.masked_fill_(self.off_earth_mask_tensor, 0)
        return x

    def denormalize(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Denormalise ``x`` back to physical units."""
        x = self.normalized_to_raw(x, in_place=in_place)
        x = self.raw_to_physical(x, in_place=True)
        x.masked_fill_(self.off_earth_mask_tensor, torch.nan)
        return x

    def _azimuth_zenith(
        self, times: list[np.datetime64], zen_azi: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute solar angles for a list of times.

        Parameters
        ----------
        times : list of np.datetime64
            Sequence of ``T`` times at which to evaluate the angles.
        zen_azi : torch.Tensor, optional
            Pre-allocated output buffer of shape ``(3, T, H, W)``.  If
            ``None`` (default) a new tensor is allocated on the same device
            and dtype as ``self``.

        Returns
        -------
        torch.Tensor
            Shape ``(3, T, H, W)``.  Dimension 0 contains, in order:
            cos-zenith, sin-azimuth, cos-azimuth.  The time-outer memory
            layout means ``view(3*T, H, W)`` gives the angle-type-outer
            ordering expected by the diffusion model.
        """

        if zen_azi is None:
            (H, W) = self.lon_tensor.shape
            T = len(times)
            zen_azi = torch.empty(
                3, T, H, W, dtype=self.lon_tensor.dtype, device=self.lon_tensor.device
            )

        # Compute solar angles for every time step in the list
        for i, t in enumerate(times):
            t_dt = datetime.fromisoformat(str(t))
            (_, zen_azi[0, i], zen_azi[1, i], zen_azi[2, i]) = zenith_azimuth_angles(
                t_dt, self.lon_tensor, self.lat_tensor
            )

        zen_azi.masked_fill_(self.off_earth_mask_tensor, 0)

        return zen_azi

    @torch.no_grad()
    def _forward(
        self,
        x: torch.Tensor,
        zen_azi: torch.Tensor,
    ) -> torch.Tensor:
        """Run one diffusion forward step on a normalised context window.

        Builds the conditioning tensor from solar angles, the flattened
        context history, and invariant fields; samples the diffusion model;
        then advances the context window in-place by dropping the oldest
        frame and appending the new prediction.

        Parameters
        ----------
        x : torch.Tensor
            Normalised context window, shape ``(batch, L, C, H, W)``, where
            ``L = len(self.input_times)`` and frames are ordered oldest first.
            Modified in-place: on return the window is shifted by one step
            (oldest frame dropped, new prediction appended at index ``-1``).
        zen_azi : torch.Tensor
            Solar-angle tensor of shape ``(3, L+1, H, W)`` covering the
            ``L`` context times plus the one target time, as returned by
            :meth:`_azimuth_zenith`.

        Returns
        -------
        torch.Tensor
            Updated context window, shape ``(batch, L, C, H, W)``, normalised.
            The last slice ``[:, -1]`` holds the predicted next frame.
        """
        # reshape x to the order needed by the model
        (B, L, C, H, W) = x.shape
        x = x.reshape(B, C * L, H, W)

        zen_azi = zen_azi.view(1, 3 * (L + 1), H, W).expand(B, -1, -1, -1)

        invariant_tensor = self.invariants[
            :, :, self._yi0 : self._yi1, self._xi0 : self._xi1
        ].repeat(B, 1, 1, 1)

        # Channel order matches training background_channels in mtg.py:
        #   [cos_zen × (N+1), sin_azi × (N+1), cos_azi × (N+1),
        #    MTG_history × N (oldest first), invariants]
        condition = torch.cat(
            (zen_azi, x, invariant_tensor),
            dim=1,
        )
        denoiser = self.scheduler.get_denoiser(
            x0_predictor=self._conditioned_x0_predictor(condition)
        )
        return self._sample(
            denoiser, shape=(B, C, H, W), dtype=x.dtype, device=x.device
        )

    def _conditioned_x0_predictor(
        self, condition: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return an x0-predictor closure conditioned on ``condition``."""

        def x0_predictor(x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            diffusion_model = (
                self.model_high if t[0] >= self.sigma_threshold else self.model_low
            )
            with torch.autocast(
                x_noisy.device.type, dtype=torch.bfloat16, enabled=self.use_amp
            ):
                x0 = diffusion_model(x_noisy, t, condition=condition)
                x0 = torch.clamp(x0, min=self.min_values, max=self.max_values, out=x0)
                return x0

        return x0_predictor

    def _sample(
        self,
        denoiser: Any,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Draw a diffusion sample.

        Parameters
        ----------
        state : torch.Tensor
            Reference tensor; its shape, device, and dtype are used for the initial latents.
        denoiser :
            Configured denoiser (EDM or SDA) returned by ``scheduler.get_denoiser``.
        num_steps : int, optional
            Number of diffusion steps; defaults to ``self.num_diffusion_steps``.

        Returns
        -------
        torch.Tensor
            Normalised diffusion output, same shape as ``state``.
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps
        latents = self.sampler_args["sigma_max"] * torch.randn(
            shape, dtype=dtype, device=device
        )

        return sample(
            denoiser,
            latents,
            noise_scheduler=self.scheduler,
            num_steps=num_steps,
            solver="edm_stochastic_heun",
            solver_options={
                "S_churn": self.sampler_args["S_churn"],
                "S_min": self.sampler_args["S_min"],
                "S_max": self.sampler_args["S_max"],
                "S_noise": self.sampler_args["S_noise"],
            },
        )

    @torch.no_grad()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run the prognostic model one step forward.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, time, lead_time, variable, y, x)``
            containing ``L = len(self.input_times)`` consecutive raw MTG frames per
            ``(batch, time)`` entry, ordered oldest first along the ``lead_time``
            dimension.
        coords : CoordSystem
            Input coordinate system.  ``coords["lead_time"]`` must equal
            ``self.input_times``.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Predicted next frame as a denormalised tensor of shape
            ``(batch, time, 1, variable, y, x)`` and the corresponding output
            coordinate system.
        """

        output_coords = self.output_coords(coords)

        # x: (batch, time, n_input_times, C, H, W)
        (B, T, L, C, H, W) = x.shape
        x = self.normalize(x)

        for j, time in enumerate(coords["time"]):
            # all_times: N input times + 1 output time (N+1 entries for solar angles)
            all_times = list(time + self.input_times) + [time + self.output_times[0]]
            zen_azi = self._azimuth_zenith(all_times)

            for i0 in range(0, B, self.batch_size):
                i1 = i0 + self.batch_size
                x[i0:i1, j, -1] = self._forward(x[i0:i1, j], zen_azi)

        return self.denormalize(x[:, :, -1:]), output_coords

    @torch.no_grad()
    def create_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Create a generator for autoregressive rollout.

        The first ``next()`` call yields the initial condition unchanged.
        Subsequent calls each advance the sliding context window by one step
        and yield the newly predicted frame alongside its output coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, time, lead_time, variable, y, x)``
            containing raw MTG frames; must conform to ``self.input_coords()``.
        coords : CoordSystem
            Input coordinate system; must conform to ``self.input_coords()``.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted frame tensor of shape ``(batch, time, 1, variable, y, x)``
            (denormalised) and its output coordinate system.
        """
        yield x, coords

        (B, T, L, C, H, W) = x.shape
        x = self.normalize(x)
        times = coords["time"].copy()
        time_step = self.output_times[0]
        time_offsets = np.concatenate([self.input_times, self.output_times])
        zen_azi = torch.stack(
            [self._azimuth_zenith(time + time_offsets) for time in times], dim=0
        )

        try:
            while True:
                for j, time in enumerate(coords["time"]):
                    for i0 in range(0, B, self.batch_size):
                        i1 = i0 + self.batch_size
                        x_next = self._forward(x[i0:i1, j], zen_azi[j])
                        x_next.masked_fill_(self.off_earth_mask_tensor, 0)
                        for k in range(L - 1):  # copyless roll of tensor
                            x[i0:i1, j, k] = x[i0:i1, j, k + 1]
                        x[i0:i1, j, -1] = x_next

                yield (self.denormalize(x[:, :, -1:]), self.output_coords(coords))

                # roll time step
                coords["lead_time"] = coords["lead_time"] + time_step
                for k in range(L - 1):
                    zen_azi[:, :, k] = zen_azi[:, :, k + 1]
                for j, time in enumerate(coords["time"]):
                    self._azimuth_zenith(
                        [time + coords["lead_time"][-1] + time_step], zen_azi[j, :, -1:]
                    )

        except GeneratorExit:
            pass

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Create an iterator for autoregressive rollout.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor; must conform to ``self.input_coords()``.
        coords : CoordSystem
            Input coordinate system; must conform to ``self.input_coords()``.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted frame tensor and output coordinate system after each step.
        """
        yield from self.create_generator(x, coords)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        # package = Package(
        #     "hf://nvidia/stormcast-conus@...",
        #     cache_options={
        #         "cache_storage": Package.default_cache("stormcast-conus"),
        #         "same_names": True,
        #     },
        # )
        # return package
        raise NotImplementedError("No default package.")

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        mtg_ylim: tuple[int, int] | None = None,
        mtg_xlim: tuple[int, int] | None = None,
    ) -> PrognosticModel:
        """Load the prognostic model from a package.

        Parameters
        ----------
        package : Package
            Model package containing checkpoints and metadata.
        mtg_ylim : tuple[int, int] | None, optional
            Row pixel range ``(y_start, y_end)`` of the inference sub-region;
            defaults to the full package extent when ``None``, by default None
        mtg_xlim : tuple[int, int] | None, optional
            Column pixel range ``(x_start, x_end)`` of the inference sub-region;
            defaults to the full package extent when ``None``, by default None

        Returns
        -------
        PrognosticModel
            Loaded and initialised :class:`StormScopeMeteosatEU` model.
        """
        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        model_low = EDMPreconditioner.from_checkpoint(
            package.resolve("StormscopeMTG-low.mdlus")
        ).eval()
        model_high = EDMPreconditioner.from_checkpoint(
            package.resolve("StormscopeMTG-high.mdlus")
        ).eval()

        # Load metadata: means, stds, grid
        with xr.open_dataset(package.resolve("metadata.nc")) as metadata:
            variables = np.array(metadata["variable_state"], copy=True)

            # Expand dims and tensorify normalization buffers
            means = torch.from_numpy(metadata["means"].values[:, None, None])
            stds = torch.from_numpy(metadata["stds"].values[:, None, None])
            scale_factor = torch.from_numpy(
                metadata["scale_factor"].values[:, None, None]
            )
            add_offset = torch.from_numpy(metadata["add_offset"].values[:, None, None])
            ir_38_warm_scale_factor = metadata.attrs["ir_38_warm_scale_factor"]
            ir_38_warm_threshold = metadata.attrs["ir_38_warm_threshold"]

            # Load invariants
            invariants = metadata["invariants"].values
            invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)
            lat = torch.from_numpy(metadata["lat"].values)
            lon = torch.from_numpy(metadata["lon"].values)
            earth_mask = torch.from_numpy(metadata["earth_mask"].values)
            mtg_y = metadata["mtg_y"].values
            mtg_x = metadata["mtg_x"].values

        # EDM sampler arguments
        sampler_args = config.sampler_args
        inference_mtg_box = (
            tuple(config.get("mtg_ylim", cls.Model_FCI_BBox[0])),
            tuple(config.get("mtg_xlim", cls.Model_FCI_BBox[1])),
        )
        # Sub-region defaults to the full package box
        if mtg_ylim is None:
            mtg_ylim = inference_mtg_box[0]
        if mtg_xlim is None:
            mtg_xlim = inference_mtg_box[1]

        num_input_time_steps = config.get(
            "num_input_time_steps", config.get("num_input_steps", 6)
        )
        step_interval = config.get("step_interval_minutes", 10)
        input_times = np.arange(-num_input_time_steps + 1, 1) * np.timedelta64(
            step_interval, "m"
        )
        output_times = np.array([np.timedelta64(step_interval, "m")])

        return cls(
            model_low,
            model_high,
            means=means,
            stds=stds,
            invariants=invariants,
            scale_factor=scale_factor,
            add_offset=add_offset,
            ir_38_warm_scale_factor=ir_38_warm_scale_factor,
            ir_38_warm_threshold=ir_38_warm_threshold,
            lat=lat,
            lon=lon,
            earth_mask=earth_mask,
            variables=variables,
            sampler_args=sampler_args,
            inference_mtg_box=inference_mtg_box,
            sigma_threshold=config["sigma_threshold"],
            mtg_y=mtg_y,
            mtg_x=mtg_x,
            mtg_ylim=mtg_ylim,
            mtg_xlim=mtg_xlim,
            input_times=input_times,
            output_times=output_times,
        )
