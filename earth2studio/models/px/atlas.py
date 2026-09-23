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

import json
import os
from collections.abc import Generator, Iterator
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.nn.atlas import StochasticInterpolant
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem, CoordSystem

try:
    from physicsnemo import Module
except ImportError:
    OptionalDependencyFailure("atlas")
    Module = None


VARIABLES: list[str] = [
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
    "sst",
    "tp06",
]

# Helper for datetime convention compatible with custom cos zenith calculation
_EPOCH = datetime(1970, 1, 1)


def npdt64_to_naive_utc(t: np.datetime64) -> datetime:
    delta_us = (
        t.astype("datetime64[us]") - np.datetime64("1970-01-01T00:00:00", "us")
    ).astype(int)
    return _EPOCH + timedelta(microseconds=int(delta_us))


@check_optional_dependencies()
class Atlas(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """Atlas prognostic model for ERA5 variables on a 0.25° global lat-lon grid.

    Atlas consumes two input lead times (t-6h and t) and predicts a single step at
    t+6h on a 721x1440 latitude-longitude grid.

    Note
    ----
    For more information see the following references:

    - https://huggingface.co/nvidia/atlas-era5

    Parameters
    ----------
    autoencoders : nn.ModuleList
        List of autoencoders for the full-resolution physical state.
    autoencoder_processors : nn.ModuleList
        List of autoencoder processors for the full-resolution physical state.
    model : nn.Module
        Model for the full-resolution physical state.
    model_processor : nn.Module
        Model processor for the full-resolution physical state.
    sinterpolant : nn.Module
        Stochastic interpolant for the low-resolution latent state.
    sinterpolant_sample_steps : int
        Number of steps to sample for the stochastic interpolant.

    Warning
    ----------
    This model is expected to use the iterator interface for autoregressive
    rollouts longer than one step. Iteratively using the ``__call__`` and
    ``prep_next_input`` methods will not produce correct results, since the model
    performs autoregressive timestepping using a full-resolution physical state
    and an internal low-resolution latent state.

    Note
    ----
    For best inference performance, set the environment variable `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`.
    This is on by default in NGC containers, but other environments may need to set it manually.

    Badges
    ------
    region:global class:medium-range product:wind product:precip product:temp product:atmos year:2026
    gpu:80gb
    provider:nvidia backend:pytorch
    """

    DT = np.timedelta64(6, "h")

    def __init__(
        self,
        autoencoders: nn.ModuleList,
        autoencoder_processors: nn.ModuleList,
        model: nn.Module,
        model_processor: nn.Module,
        sinterpolant: nn.Module,
        sinterpolant_sample_steps: int = 60,
    ) -> None:
        super().__init__()
        self.autoencoders = autoencoders
        self.autoencoder_processors = autoencoder_processors
        self.model = model
        self.model_processor = model_processor
        self.sinterpolant = sinterpolant
        self.sinterpolant_sample_steps = sinterpolant_sample_steps
        self.register_buffer("device_buffer", torch.empty(0))

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system expected by Atlas.

        Notes
        -----
        - Lead times are fixed to [-6h, 0h].
        - Variables are defined by the module-level `VARIABLES`.
        - Spatial grid is 0.25° lat-lon: 721 latitudes, 1440 longitudes.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature with coordinates:
            - 'lead_time' : np.ndarray[np.timedelta64] of shape (2,)
            - 'variable' : np.ndarray[str] of shape (n_variables,)
            - 'lat' : np.ndarray[float] of shape (721,)
            - 'lon' : np.ndarray[float] of shape (1440,)
        """
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([-self.DT, np.timedelta64(0, "h")]),
                "variable": ["tp:sum:6h" if v == "tp06" else v for v in VARIABLES],
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system produced by a single Atlas step (t+6h).

        Parameters
        ----------
        input_coords : CoordinateSystem
            Coordinate system associated with the input to the forward pass.

        Returns
        -------
        CoordinateSystem
            Allocation-free output signature with coordinates:
            - 'time' : np.ndarray[np.datetime64] (copied from input if present)
            - 'lead_time' : np.timedelta64 set to +6h
            - 'variable' : np.ndarray[str] matching `VARIABLES`
            - 'lat' : np.ndarray[float] (copied from input if present, else 721 values)
            - 'lon' : np.ndarray[float] (copied from input if present, else 1440 values)
        """
        if "lead_time" not in input_coords.coords:
            raise ValueError("Input lead_time coordinate is required")
        lead = input_coords.lead_time.values
        if (
            input_coords.lead_time.dims != ("lead_time",)
            or lead.size != 2
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("lead_time must contain two finite timedeltas")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(input_coords, {"lead_time": lead[-1:] + self.DT})

    def prep_next_input(
        self,
        x_pred: xr.DataArray,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Prepare the next input for the Atlas model. Since the input requires two lead times
        but the model predicts one, we update a sliding window to make autoregressive predictions.

        Parameters
        ----------
        x_pred : xr.DataArray
            Forecast from the previous step.
        x : xr.DataArray
            Previous two-frame history.

        Returns
        -------
        xr.DataArray
            Updated two-frame history, carrying prediction metadata.
        """
        previous = x.isel(lead_time=slice(-1, None))
        # Use the prediction's metadata, including hook removals, for the new history.
        signature = coord_array_like(
            x_pred,
            {
                "lead_time": np.concatenate(
                    (previous.lead_time.values, x_pred.lead_time.values)
                )
            },
        )
        axis = x.get_axis_num("lead_time")
        a, _ = previous.e2s.to_torch()
        b, _ = x_pred.e2s.to_torch()
        a = a.to(b.device)
        result = from_torch(torch.cat((a, b), dim=axis), signature, name=x_pred.name)
        result.encoding = x_pred.encoding.copy()
        return result

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        prev_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the prognostic model, integrating a single 6h step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., lead_time, variable, lat, lon) corresponding
            to the coordinate system. Lead times expected: [-6h, 0h].
        coords : CoordSystem
            Coordinate dictionary describing `x`.
        prev_latent : torch.Tensor, optional
            Low-resolution latent from the previous forecast step. If provided, it will be
            reused instead of downsampling the input high-resolution state, by default None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple containing the decoded forecast at t+6h and the corresponding latent
            (low-resolution) prediction.
        """

        if x.ndim != 4:
            raise ValueError(
                f"Internal forward pass expects x of shape (lead_time, variable, lat, lon), got {x.shape}"
            )
        if len(coords["lead_time"]) != 2:
            raise ValueError(
                f"Internal forward pass expects coords['lead_time'] of length 2, got {len(coords['lead_time'])}"
            )

        # Prepare input tensor and date metadata
        x_cur, x_prev = x[-1:, :, :, :], x[:1, :, :, :]
        t = coords["time"][0] + coords["lead_time"][-1]
        current_date = np.array([[npdt64_to_naive_utc(t)]])

        # Preprocess to build high/low-res latent/state
        self.model_processor.add_noise = False
        high_res, low_res = self.model_processor.preprocess_input(x_cur, current_date)
        if prev_latent is not None:
            low_res = prev_latent.clone()
        prev = self.model_processor.normalizer_in.normalize(x_prev)
        prev = self.model_processor.intep(
            prev, self.model_processor.downsample_grid_shape
        )

        # Condition dictionary
        cond = {"x_1": low_res.clone(), "x_2": prev.clone()}

        # Stochastic interpolant sampling in latent space
        prediction_latent = self.sinterpolant.sample(
            self.model,
            low_res.clone(),
            steps=self.sinterpolant_sample_steps,
            cond=cond,
            verbose=False,
            compute_normalization=True,
        )

        # Decode
        pred = self.autoencoders[0](high_res, prediction_latent)

        # Update latent difference prediction into latent state prediction
        prediction_latent = self.model_processor.normalizer_out.unnormalize(
            prediction_latent
        )
        prediction_latent = (
            prediction_latent + self.model_processor.normalizer_in.unnormalize(low_res)
        )
        prediction_latent = self.model_processor.normalizer_in.normalize(
            prediction_latent
        )

        # Postprocess to state space
        pred = self.autoencoder_processors[0].postprocess(pred, x_cur)
        return pred, prediction_latent

    @torch.inference_mode()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict a six-hour DataArray from two input frames, without hooks."""

        out, _ = self._call_with_latent(x)
        return out

    @torch.inference_mode()
    def _call_with_latent(
        self,
        x: xr.DataArray,
        prev_latents: list[list[torch.Tensor | None]] | None = None,
    ) -> tuple[xr.DataArray, list[list[torch.Tensor]]]:
        """Internal helper that handles cached latents during autoregressive rollout."""

        self.output_coords(x)
        if (
            "time" not in x.coords
            or x.time.dims != ("time",)
            or not np.issubdtype(x.time.dtype, np.datetime64)
            or np.isnat(x.time.values).any()
        ):
            raise ValueError("time must contain finite datetimes")
        packed, restore = batch_func()._compress_array(self, x)
        signature = self.output_coords(packed)
        tensor, coords = packed.e2s.to_torch()
        tensor = tensor.to(self.device_buffer.device).clone()
        if torch.isnan(tensor).any():
            logger.info("Atlas input contains NaNs, replacing with 0.0")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        out = torch.empty_like(tensor[:, :, :1])
        latents_out: list[list[torch.Tensor | None]] = [
            [None for _ in coords["time"]] for _ in coords["batch"]
        ]

        for i, _ in enumerate(coords["batch"]):
            for j, _ in enumerate(coords["time"]):
                slice_coords = coords.copy()
                slice_coords["time"] = slice_coords["time"][j : j + 1]
                prev_latent = None
                if prev_latents is not None:
                    prev_latent = prev_latents[i][j]
                pred, pred_latent = self._forward(
                    tensor[i, j, :], slice_coords, prev_latent
                )
                out[i, j, :] = pred
                latents_out[i][j] = pred_latent

        result = from_torch(out, signature, name=x.name)
        result.encoding = x.encoding.copy()
        return restore(result), latents_out

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the final input frame then six-hour forecasts with cached latents."""
        yield from self._default_generator(x)

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        self.output_coords(x)
        yield x.isel(lead_time=slice(-1, None)).copy(deep=True)
        latent_cache: list[list[torch.Tensor | None]] | None = None
        while True:
            if self.front_hook is not self._default_hook:
                x = self.front_hook(x.copy(deep=True))
            x_pred, latent_cache = self._call_with_latent(x, prev_latents=latent_cache)
            x_pred = self.rear_hook(x_pred)
            yield x_pred
            x = self.prep_next_input(x_pred, x)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default package for the Atlas model."""
        package = Package(
            "hf://nvidia/atlas-era5@893a38550aa313c97c41382a5003d209d60a840b",
            cache_options={
                "cache_storage": Package.default_cache("atlas"),
                "same_names": False,  # prevents overwrites from files with same name in different directories
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> PrognosticModel:
        """Instantiate and load Atlas from a package."""
        if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1":
            logger.warning(
                "Atlas inference expects TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 for "
                "best performance. Set this environment variable before starting "
                "Python, as documented in the Atlas API docs."
            )

        # Resolve the package-root config.json so HuggingFace records the download
        # (its content is not used here now that the merged si/crps package layout
        # moved the functional config under si/config.json); tolerate its absence.
        try:
            package.resolve("config.json")
        except FileNotFoundError:
            pass

        config_path = None
        for candidate in ("si/config.json", "config.json"):
            try:
                config_path = package.resolve(candidate)
                break
            except FileNotFoundError:
                continue
        if config_path is None:
            raise FileNotFoundError(
                "Could not locate Atlas config.json in package "
                "(checked si/config.json and config.json)"
            )

        with open(config_path) as f:
            config = json.load(f)

        modelpkg = config["package"]

        autoencoders = nn.ModuleList()
        autoencoder_processors = nn.ModuleList()
        for i, ae_cfg in enumerate(modelpkg["autoencoders"]):
            ae_path = package.resolve(ae_cfg["model_path"])
            aeprocessor_path = package.resolve(ae_cfg["processor_path"])
            ae = Module.from_checkpoint(ae_path)
            ae.eval()
            aeprocessor = Module.from_checkpoint(aeprocessor_path)
            aeprocessor.eval()

            autoencoders.append(ae)
            autoencoder_processors.append(aeprocessor)

        model = Module.from_checkpoint(
            package.resolve(modelpkg["genmodel"]["model_path"])
        )
        model.eval()
        model_processor = Module.from_checkpoint(
            package.resolve(modelpkg["genmodel"]["processor_path"])
        )
        model_processor.eval()

        sinterpolant = StochasticInterpolant(
            alpha=config["sinterpolant"]["alpha"],
            beta=config["sinterpolant"]["beta"],
            sigma=config["sinterpolant"]["sigma"],
            g=config["sinterpolant"]["g"],
            epsilon=config["sinterpolant"]["epsilon"],
            noise_sampler=config["sinterpolant"]["noise_sampler"],
            time_sampler=config["sinterpolant"]["time_sampler"],
            sample_method=config["sinterpolant"]["sample_method"],
            studentt_deg=config["sinterpolant"]["studentt_deg"],
        )

        return cls(
            autoencoders=autoencoders,
            autoencoder_processors=autoencoder_processors,
            model=model,
            model_processor=model_processor,
            sinterpolant=sinterpolant,
            sinterpolant_sample_steps=config["sinterpolant"]["sample_steps"],
        )
