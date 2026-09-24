# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import torch
from fsspec.implementations.cache_mapper import BasenameCacheMapper

from earth2studio.data import HRRR
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import handshake_coords, handshake_dim, interp
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import timearray_to_datetime
from earth2studio.utils.type import CoordSystem

try:
    import natten  # noqa: F401  # the DiT needs NATTEN (neighborhood attention)
    from physicsnemo.diffusion.noise_schedulers import (
        EDMNoiseScheduler,
        RectifiedFlowNoiseScheduler,
    )
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.diffusion.utils import ConcatConditionWrapper
    from physicsnemo.models.dit import DiT
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
    from tensordict import TensorDict
except ImportError:
    OptionalDependencyFailure("corrdiff-era5-hrrr")
    EDMNoiseScheduler = None
    RectifiedFlowNoiseScheduler = None
    EDMPreconditioner = None
    sample = None
    ConcatConditionWrapper = None
    DiT = None
    cos_zenith_angle = None
    TensorDict = None

# Default channel order from the pretrained x_pred package.
ERA5_VARIABLES = (
    "u10m",
    "v10m",
    "t2m",
    "tcwv",
    "sp",
    "msl",
    *(
        f"{variable}{level}"
        for variable in ("u", "v", "z", "t", "q")
        for level in (1000, 850, 500, 250)
    ),
)
OUTPUT_VARIABLES = (
    "u10m",
    "v10m",
    "t2m",
    "mslp",
    *(
        f"{variable}{level}hl"
        for variable in ("u", "v", "t", "q", "Z")
        for level in (*range(1, 12), 13, 15, 20, 25, 30)
    ),
    *(f"p{level}hl" for level in (*range(1, 12), 13, 15, 20)),
    "refc",
)

# hours in a mean tropical-ish year used by the training pipeline (365.25 days)
_HOURS_PER_YEAR = 8766.0

# Sub-folders of the hosted package, one per generative formulation. Every
# artifact of a variant (checkpoint, metadata, statistics, grids, invariants)
# lives under ``<variant>/`` so one package URI serves every formulation.
SUPPORTED_VARIANTS = ("x_pred",)


@check_optional_dependencies()
class CorrDiffEra5Hrrr(torch.nn.Module, AutoModelMixin):
    """Generative downscaling from 0.25-degree ERA5 to 3 km HRRR over CONUS.

    Note
    ----
    For more information see the following references:

    - https://huggingface.co/nvidia/corrdiff-era5-hrrr

    Parameters
    ----------
    network : torch.nn.Module
        Conditioned network. For ``network_kind="rectified_flow"`` a
        ``ConcatConditionWrapper(DiT)`` whose output is the velocity or the clean
        data (see ``prediction_type``); for ``network_kind="edm"`` an
        ``EDMPreconditioner(ConcatConditionWrapper(DiT))`` (an x0-predictor).
    lat_input_grid, lon_input_grid : torch.Tensor
        1-D regular ERA5 input grid (the native training footprint). Latitude may
        be ascending or descending; input longitudes may use either the ``[0, 360)``
        or the ``[-180, 180)`` convention (compared modulo 360).
    lat_output_grid, lon_output_grid : torch.Tensor
        2-D HRRR latitude / longitude ``[H, W]`` of the output crop (degrees).
    hrrr_y, hrrr_x : torch.Tensor
        1-D native HRRR projection coordinates (m) of the output crop.
    era5_center, era5_scale : torch.Tensor
        ERA5 input normalization (mean / std), size ``[n_era5]``.
    out_center, out_scale : torch.Tensor
        Output normalization (mean / std), size ``[n_out]``.
    invariants : torch.Tensor
        Normalized static invariant channels ``[n_inv, H, W]``, appended to the
        conditioning after the cosine-zenith channel.
    network_kind : Literal["rectified_flow", "edm"], optional
        Generative formulation, by default "rectified_flow"
    era5_variables : Sequence[str], optional
        Input channel order, by default :data:`ERA5_VARIABLES`
    output_variables : Sequence[str], optional
        Output channel order, by default :data:`OUTPUT_VARIABLES`
    presence_flags : Sequence[str]
        ERA5 variables that training randomly dropped from the input; one scalar
        "present" flag (always 1 at inference) per name.
    day_of_year : bool
        Whether the scalar conditioning carries ``[sin, cos]`` of the day-of-year
        phase at the validity time.
    prediction_type : Literal["x0", "flow"], optional
        Rectified-flow output parameterization. Ignored for EDM, by default "x0"
    time_scale : float
        Multiplier applied to the rectified-flow time ``t in [0, 1]`` before the
        network's timestep embedder (the examples train with ``999.0``).
    number_of_samples : int
        Ensemble members per input; settable between calls.
    number_of_steps : int
        ODE solver steps (each Heun step costs two network evaluations).
    solver : {"heun", "euler"}
        ODE solver.
    shift : float
        SD3 resolution shift ``a`` applied to the rectified-flow time grid,
        ``t -> a t / (1 + (a - 1) t)``; ``1.0`` disables it. The examples found
        ``a`` of 12-32 optimal for full-domain sampling (the networks train on
        256 x 256 patches). Ignored for EDM.
    t_max : float
        Rectified-flow start time (the noise end); kept below 1.
    x0v_clip : float
        Lower clamp of the denominator in the x-prediction to velocity conversion
        ``v = (x_t - x0_hat) / max(t, x0v_clip)`` near the data end (``t -> 0``).
    sigma_min, sigma_max, rho : float
        EDM Karras schedule parameters. Ignored for rectified flow.
    seed : int | None
        Base RNG seed for the sampling latents; member ``i`` uses ``seed + i``.
        ``None`` leaves sampling unseeded.
    amp : bool
        Run network forwards under bf16 autocast while the ODE integration stays in
        fp32 (the examples' evaluation setting; roughly halves inference time).

    Badges
    ------
    region:na class:downscaling product:wind product:temp product:atmos product:radar
    year:2026 gpu:80gb provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        network: torch.nn.Module,
        lat_input_grid: torch.Tensor,
        lon_input_grid: torch.Tensor,
        lat_output_grid: torch.Tensor,
        lon_output_grid: torch.Tensor,
        hrrr_y: torch.Tensor,
        hrrr_x: torch.Tensor,
        era5_center: torch.Tensor,
        era5_scale: torch.Tensor,
        out_center: torch.Tensor,
        out_scale: torch.Tensor,
        invariants: torch.Tensor,
        network_kind: Literal["rectified_flow", "edm"] = "rectified_flow",
        era5_variables: Sequence[str] = ERA5_VARIABLES,
        output_variables: Sequence[str] = OUTPUT_VARIABLES,
        presence_flags: Sequence[str] = (),
        day_of_year: bool = True,
        prediction_type: Literal["x0", "flow"] = "x0",
        time_scale: float = 999.0,
        number_of_samples: int = 1,
        number_of_steps: int = 50,
        solver: Literal["heun", "euler"] = "heun",
        shift: float = 32.0,
        t_max: float = 0.99,
        x0v_clip: float = 0.05,
        sigma_min: float = 0.01,
        sigma_max: float = 200.0,
        rho: float = 7.0,
        seed: int | None = None,
        amp: bool = True,
    ):
        super().__init__()
        if network_kind not in ("rectified_flow", "edm"):
            raise ValueError(
                f"network_kind must be 'rectified_flow' or 'edm', got {network_kind!r}"
            )
        if prediction_type not in ("x0", "flow"):
            raise ValueError(
                f"prediction_type must be 'x0' or 'flow', got {prediction_type!r}"
            )
        if solver not in ("heun", "euler"):
            raise ValueError(f"solver must be 'heun' or 'euler', got {solver!r}")
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("number_of_samples must be a positive integer")
        if number_of_steps < 1:
            raise ValueError("number_of_steps must be at least 1")
        if not 0.0 < t_max < 1.0:
            raise ValueError("t_max must lie in (0, 1)")
        if shift <= 0.0:
            raise ValueError("shift must be positive")
        if lat_output_grid.shape != lon_output_grid.shape or lat_output_grid.ndim != 2:
            raise ValueError("lat_output_grid / lon_output_grid must be 2-D [H, W]")
        n_era5, n_out = len(era5_variables), len(output_variables)
        if era5_center.numel() != n_era5 or era5_scale.numel() != n_era5:
            raise ValueError(
                "era5_center / era5_scale must have one entry per ERA5 variable"
            )
        if out_center.numel() != n_out or out_scale.numel() != n_out:
            raise ValueError(
                "out_center / out_scale must have one entry per output variable"
            )
        if invariants.ndim != 3 or invariants.shape[1:] != lat_output_grid.shape:
            raise ValueError("invariants must be [n_inv, H, W] on the output grid")
        if (
            hrrr_y.numel() != lat_output_grid.shape[0]
            or hrrr_x.numel() != lat_output_grid.shape[1]
        ):
            raise ValueError("hrrr_y / hrrr_x must match the output grid shape")

        self.network = network
        self.network_kind = network_kind
        self.prediction_type = prediction_type
        self.time_scale = float(time_scale)
        self.era5_variables = np.array(list(era5_variables))
        self.output_variables = np.array(list(output_variables))
        self.presence_flags = tuple(presence_flags)
        self.day_of_year = bool(day_of_year)
        self.number_of_samples = number_of_samples
        self.number_of_steps = int(number_of_steps)
        self.solver = solver
        self.shift = float(shift)
        self.t_max = float(t_max)
        self.x0v_clip = float(x0v_clip)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.seed = seed
        self.amp = bool(amp)

        lat_in = torch.as_tensor(lat_input_grid, dtype=torch.float32).reshape(-1)
        lon_in = torch.as_tensor(lon_input_grid, dtype=torch.float32).reshape(-1)
        # the bilinear helper needs an ascending regular latitude axis; remember
        # whether the native axis (and so the incoming data) is descending
        self._lat_descending = bool(lat_in[0] > lat_in[-1])
        self.register_buffer("lat_input_grid", lat_in)
        self.register_buffer("lon_input_grid", lon_in)
        self.register_buffer("lat_output_grid", lat_output_grid.to(torch.float32))
        self.register_buffer("lon_output_grid", lon_output_grid.to(torch.float32))
        self.register_buffer(
            "era5_center", era5_center.reshape(1, -1, 1, 1).to(torch.float32)
        )
        self.register_buffer(
            "era5_scale", era5_scale.reshape(1, -1, 1, 1).to(torch.float32)
        )
        self.register_buffer(
            "out_center", out_center.reshape(1, -1, 1, 1).to(torch.float32)
        )
        self.register_buffer(
            "out_scale", out_scale.reshape(1, -1, 1, 1).to(torch.float32)
        )
        self.register_buffer("invariants", invariants.to(torch.float32))
        # keep the native HRRR projection coordinates and the ERA5 grid on the CPU
        # as numpy for coordinate systems
        self.hrrr_y = np.asarray(
            torch.as_tensor(hrrr_y).cpu().numpy(), dtype=np.float64
        )
        self.hrrr_x = np.asarray(
            torch.as_tensor(hrrr_x).cpu().numpy(), dtype=np.float64
        )
        self.lat_input_numpy = lat_in.cpu().numpy().copy()
        self.lon_input_numpy = lon_in.cpu().numpy().copy()
        self._lat_out_cpu = self.lat_output_grid.cpu().numpy().astype(np.float64)
        self._lon_out_cpu = self.lon_output_grid.cpu().numpy().astype(np.float64)

    def __str__(self) -> str:
        return "CorrDiffEra5Hrrr"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system: the native ERA5 CONUS footprint.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary; regrid ERA5 onto ``lat`` / ``lon``.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "variable": self.era5_variables.copy(),
                "lat": self.lat_input_numpy.copy(),
                "lon": self.lon_input_numpy.copy(),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system: HRRR crop on the native projection grid.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords.

        Returns
        -------
        CoordSystem
            ``[batch, sample, time, variable, hrrr_y, hrrr_x]``; the 2-D latitude
            and longitude of the crop are available as ``lat_output_grid`` /
            ``lon_output_grid``.
        """
        target = self.input_coords()
        handshake_dim(input_coords, "time", 1)
        handshake_dim(input_coords, "variable", -3)
        handshake_dim(input_coords, "lat", -2)
        handshake_dim(input_coords, "lon", -1)
        handshake_coords(input_coords, target, "variable")
        lat = np.asarray(input_coords["lat"], dtype=np.float64)
        lon = np.asarray(input_coords["lon"], dtype=np.float64)
        # longitudes compare modulo 360 so both the [0, 360) and the [-180, 180)
        # conventions match the native grid; the data order is what matters
        if (
            lat.shape != self.lat_input_numpy.shape
            or lon.shape != self.lon_input_numpy.shape
            or not np.allclose(lat, self.lat_input_numpy, atol=1e-4)
            or not np.allclose(
                np.mod(lon, 360.0), np.mod(self.lon_input_numpy, 360.0), atol=1e-4
            )
        ):
            raise ValueError(
                "CorrDiffEra5Hrrr requires the native ERA5 input grid from "
                "input_coords() (regrid the ERA5 input onto it first)."
            )
        return OrderedDict(
            {
                "batch": input_coords["batch"],
                "sample": np.arange(self.number_of_samples),
                "time": input_coords["time"],
                "variable": self.output_variables.copy(),
                "hrrr_y": self.hrrr_y.copy(),
                "hrrr_x": self.hrrr_x.copy(),
            }
        )

    def _interpolate(self, x: torch.Tensor) -> torch.Tensor:
        """Bilinear ERA5 ``[C, H_in, W_in]`` -> HRRR crop ``[C, H, W]``."""
        lat0 = self.lat_input_grid
        if self._lat_descending:
            x = x.flip(-2)
            lat0 = lat0.flip(0)
        return interp.latlon_interpolation_regular(
            x, lat0, self.lon_input_grid, self.lat_output_grid, self.lon_output_grid
        )

    def _cos_zenith(self, valid_time: datetime) -> torch.Tensor:
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone.utc)
        cz = cos_zenith_angle(valid_time, self._lon_out_cpu, self._lat_out_cpu)
        return torch.as_tensor(
            np.asarray(cz, dtype=np.float32), device=self.invariants.device
        )

    def _scalar_conditions(self, valid_time: datetime) -> torch.Tensor:
        parts: list[float] = []
        if self.day_of_year:
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=timezone.utc)
            start = datetime(valid_time.year, 1, 1, tzinfo=valid_time.tzinfo)
            phase = (
                2.0
                * math.pi
                * ((valid_time - start).total_seconds() / (_HOURS_PER_YEAR * 3600.0))
            )
            parts += [math.sin(phase), math.cos(phase)]
        parts += [1.0] * len(self.presence_flags)  # every input present
        return torch.tensor(parts, dtype=torch.float32, device=self.invariants.device)

    def preprocess_input(
        self, era5: torch.Tensor, valid_time: datetime
    ) -> "TensorDict":
        """Build the network conditioning for one ERA5 state.

        Parameters
        ----------
        era5 : torch.Tensor
            ``[n_era5, H_in, W_in]`` physical-unit ERA5 fields on the native grid.
        valid_time : datetime
            Validity time (UTC) of the state.

        Returns
        -------
        TensorDict
            ``cond_concat`` ``[1, n_era5 + 1 + n_inv, H, W]`` (normalized ERA5 on
            the HRRR grid, cosine zenith angle, invariants) and ``cond_vec``
            ``[1, n_scalar]``.
        """
        era5_r = self._interpolate(era5.to(torch.float32)).unsqueeze(0)
        era5_r = (era5_r - self.era5_center) / self.era5_scale
        cz = self._cos_zenith(valid_time).view(1, 1, *self.lat_output_grid.shape)
        background = torch.cat([era5_r, cz, self.invariants.unsqueeze(0)], dim=1)
        if not torch.isfinite(background).all():
            raise ValueError(
                "non-finite values in the network conditioning; check the ERA5 "
                "input for missing data or fill values"
            )
        return TensorDict(
            {
                "cond_concat": background.to(torch.float32),
                "cond_vec": self._scalar_conditions(valid_time).view(1, -1),
            },
            batch_size=[1],
            device=background.device,
        )

    def _inference_context(self, device: torch.device) -> AbstractContextManager:
        if self.amp:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return nullcontext()

    def _rf_time_steps(self, device: torch.device, scheduler: Any) -> torch.Tensor:
        t = scheduler.timesteps(
            self.number_of_steps, device=device, dtype=torch.float32
        )
        if self.shift != 1.0:
            t = self.shift * t / (1.0 + (self.shift - 1.0) * t)
        return t

    def _sample_one(self, condition: "TensorDict", seed: int | None) -> torch.Tensor:
        """Draw one member: ``[1, n_out, H, W]`` in normalized units."""
        device = self.invariants.device
        H, W = self.lat_output_grid.shape
        n_out = len(self.output_variables)
        gen = (
            torch.Generator(device=device).manual_seed(seed)
            if seed is not None
            else None
        )
        latents = torch.randn((1, n_out, H, W), device=device, generator=gen)
        ctx = self._inference_context(device)

        def net(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            with ctx:
                out = self.network(x.float(), t, condition=condition)
            return out.float()

        if self.network_kind == "edm":
            scheduler = EDMNoiseScheduler(
                sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho
            )
            denoiser = scheduler.get_denoiser(
                x0_predictor=lambda x, s: net(x, s.to(torch.float32).reshape(-1))
            )
            return sample(
                denoiser,
                latents * self.sigma_max,
                scheduler,
                num_steps=self.number_of_steps,
                solver=self.solver,
            ).float()

        scheduler = RectifiedFlowNoiseScheduler(t_max=self.t_max)
        t_steps = self._rf_time_steps(device, scheduler)
        xN = scheduler.sigma(t_steps[0]) * latents

        def flow(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            tt = t.to(torch.float32).reshape(-1)
            out = net(x, tt * self.time_scale)
            if self.prediction_type == "flow":
                return out
            # x-prediction: v = (x_t - x0_hat) / t with the denominator clamped
            # near the data end (t -> 0), where x0_hat -> x_t
            tc = torch.clamp(tt, min=self.x0v_clip).view(-1, 1, 1, 1)
            return (x.float() - out) / tc

        denoiser = scheduler.get_denoiser(flow_predictor=flow)
        return sample(
            denoiser,
            xN,
            scheduler,
            num_steps=self.number_of_steps,
            solver=self.solver,
            time_steps=t_steps,
        ).float()

    @torch.inference_mode()
    def _forward(self, era5: torch.Tensor, valid_time: datetime) -> torch.Tensor:
        """Downscale one ERA5 state -> ``[sample, n_out, H, W]`` in physical units."""
        condition = self.preprocess_input(era5, valid_time)
        members = []
        for i in range(self.number_of_samples):
            seed = None if self.seed is None else self.seed + i
            members.append(self._sample_one(condition, seed))
        out = torch.cat(members, dim=0)
        return out * self.out_scale + self.out_center

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Downscale ERA5 states to the HRRR CONUS crop.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``[batch, time, variable, lat, lon]`` (the wrapper
            flattens leading batch dimensions automatically).
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            ``[batch, sample, time, variable, hrrr_y, hrrr_x]`` and its coordinates.
        """
        output_coords = self.output_coords(coords)
        valid_times = timearray_to_datetime(np.asarray(output_coords["time"]))
        H, W = self.lat_output_grid.shape
        out = torch.empty(
            (
                output_coords["batch"].shape[0],
                self.number_of_samples,
                len(valid_times),
                len(self.output_variables),
                H,
                W,
            ),
            device=x.device,
            dtype=torch.float32,
        )
        for b in range(out.shape[0]):
            for t in range(out.shape[2]):
                out[b, :, t] = self._forward(x[b, t], valid_times[t])
        return out, output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default pre-trained model package.

        Returns
        -------
        Package
            Model package with default checkpoint location
        """
        return Package(
            "hf://nvidia/corrdiff-era5-hrrr@c95089642d19985714eebebdbd5b0b72c86ed1a3",
            cache_options={
                "cache_storage": Package.default_cache("corrdiff_era5_hrrr"),
                # Keep the variant directory so files with matching basenames in
                # different sub-folders (for example ``x_pred/metadata.json``)
                # get distinct cache entries.
                "cache_mapper": BasenameCacheMapper(directory_levels=1),
            },
        )

    @staticmethod
    def _load_json(package: Package, filename: str) -> dict:
        with open(package.resolve(filename), encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"{filename} is empty")
        return json.loads(content)

    @staticmethod
    def _load_npy(package: Package, filename: str) -> np.ndarray:
        return np.load(package.resolve(filename))

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        device: str | None = None,
        number_of_samples: int | None = None,
        number_of_steps: int | None = None,
        shift: float | None = None,
        seed: int | None = None,
        amp: bool | None = None,
        variant: Literal["x_pred"] = "x_pred",
    ) -> DiagnosticModel:
        """Load the model from a package.

        Parameters
        ----------
        package : Package
            Model package to load from.
        device : str | None, optional
            Device to place the model on, by default None (CPU).
        number_of_samples : int | None, optional
            Ensemble members per input; defaults to the package metadata.
        number_of_steps : int | None, optional
            ODE steps; defaults to the package metadata.
        shift : float | None, optional
            Rectified-flow resolution shift; defaults to the package metadata.
        seed : int | None, optional
            Base RNG seed, by default None (unseeded).
        amp : bool | None, optional
            bf16 autocast for the network; defaults to the package metadata.
        variant : {"x_pred"}, optional
            Sub-folder of the package to load, by default ``"x_pred"`` (the
            x-prediction rectified-flow model).

        Returns
        -------
        DiagnosticModel
            Loaded model.
        """
        # Check the selector up front so a bad value fails with a clear message
        # rather than a missing-file error when resolving "<variant>/...".
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variant must be one of {list(SUPPORTED_VARIANTS)}")
        try:
            package.resolve("config.json")  # download bookkeeping, if hosted
        except (FileNotFoundError, ValueError):
            pass
        # ``prefix`` locates every artifact of the selected model.
        prefix = f"{variant}/"
        metadata = cls._load_json(package, prefix + "metadata.json")
        stats = cls._load_json(package, prefix + "stats.json")
        era5_variables = list(metadata["era5_variables"])
        output_variables = list(metadata["output_variables"])
        network_meta = metadata["network"]
        sampler_meta = metadata.get("sampler", {})
        scalar_meta = metadata.get("scalar_conditions", {})

        dit = DiT.from_checkpoint(package.resolve(prefix + metadata["checkpoint"]))
        dit = dit.eval().requires_grad_(False)
        network: torch.nn.Module = ConcatConditionWrapper(dit)
        kind = network_meta["kind"]
        if kind == "edm":
            network = EDMPreconditioner(
                network, sigma_data=float(network_meta.get("sigma_data", 0.5))
            )
        elif kind != "rectified_flow":
            raise ValueError(f"unsupported network kind {kind!r} in package metadata")
        if device is not None:
            network = network.to(device)

        def _vec(group: str, names: Sequence[str], key: str) -> torch.Tensor:
            missing = [v for v in names if v not in stats.get(group, {})]
            if missing:
                raise ValueError(f"stats.json['{group}'] lacks {missing}")
            return torch.tensor([float(stats[group][v][key]) for v in names])

        hrrr_lat = torch.from_numpy(
            cls._load_npy(package, prefix + "hrrr_lat.npy").astype(np.float32)
        )
        hrrr_lon = torch.from_numpy(
            cls._load_npy(package, prefix + "hrrr_lon.npy").astype(np.float32)
        )
        invariants = torch.from_numpy(
            cls._load_npy(package, prefix + "invariants.npy").astype(np.float32)
        )
        rows = metadata["hrrr_window"]["rows"]
        cols = metadata["hrrr_window"]["cols"]
        hrrr_y = torch.from_numpy(HRRR.HRRR_Y[rows[0] : rows[1]].copy())
        hrrr_x = torch.from_numpy(HRRR.HRRR_X[cols[0] : cols[1]].copy())

        model = cls(
            network=network,
            network_kind=kind,
            era5_variables=era5_variables,
            output_variables=output_variables,
            lat_input_grid=torch.from_numpy(
                cls._load_npy(package, prefix + "era5_lat.npy")
            ),
            lon_input_grid=torch.from_numpy(
                cls._load_npy(package, prefix + "era5_lon.npy")
            ),
            lat_output_grid=hrrr_lat,
            lon_output_grid=hrrr_lon,
            hrrr_y=hrrr_y,
            hrrr_x=hrrr_x,
            era5_center=_vec("era5", era5_variables, "mean"),
            era5_scale=_vec("era5", era5_variables, "std"),
            out_center=_vec("hrrr", output_variables, "mean"),
            out_scale=_vec("hrrr", output_variables, "std"),
            invariants=invariants,
            presence_flags=scalar_meta.get("presence_flags", []),
            day_of_year=scalar_meta.get("day_of_year", True),
            prediction_type=network_meta.get("prediction_type", "x0"),
            time_scale=float(network_meta.get("time_scale", 999.0)),
            number_of_samples=(
                number_of_samples
                if number_of_samples is not None
                else int(metadata.get("number_of_samples", 1))
            ),
            number_of_steps=(
                number_of_steps
                if number_of_steps is not None
                else int(sampler_meta.get("num_steps", 50))
            ),
            solver=sampler_meta.get("solver", "heun"),
            shift=shift if shift is not None else float(sampler_meta.get("shift", 1.0)),
            t_max=float(sampler_meta.get("t_max", 0.99)),
            x0v_clip=float(sampler_meta.get("x0v_clip", 0.05)),
            sigma_min=float(sampler_meta.get("sigma_min", 0.01)),
            sigma_max=float(sampler_meta.get("sigma_max", 200.0)),
            rho=float(sampler_meta.get("rho", 7.0)),
            seed=seed,
            amp=amp if amp is not None else bool(sampler_meta.get("amp", True)),
        )
        if device is not None:
            model = model.to(device)
        return model
