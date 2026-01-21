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

import json
from collections.abc import Generator, Iterator
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.models.auto.package import Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.nn.atlas import StochasticInterpolant
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

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
    "tp",
]

# Helper for datetime convention compatible with custom cos zenith calculation
_EPOCH = datetime(1970, 1, 1)


def npdt64_to_naive_utc(t: np.datetime64) -> datetime:
    delta_us = (
        t.astype("datetime64[us]") - np.datetime64("1970-01-01T00:00:00", "us")
    ).astype(int)
    return _EPOCH + timedelta(microseconds=int(delta_us))


@check_optional_dependencies()
class Atlas(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Atlas prognostic model for ERA5 variables on a 0.25° global lat-lon grid.

    Atlas consumes two input lead times (t-6h and t) and predicts a single step at
    t+6h on a 721x1440 latitude-longitude grid.

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

    def input_coords(self) -> CoordSystem:
        """Input coordinate system expected by Atlas.

        Notes
        -----
        - Lead times are fixed to [-6h, 0h].
        - Variables are defined by the module-level `VARIABLES`.
        - Spatial grid is 0.25° lat-lon: 721 latitudes, 1440 longitudes.

        Returns
        -------
        CoordSystem
            Ordered dictionary with keys:
            - 'lead_time' : np.ndarray[np.timedelta64] of shape (2,)
            - 'variable' : np.ndarray[str] of shape (n_variables,)
            - 'lat' : np.ndarray[float] of shape (721,)
            - 'lon' : np.ndarray[float] of shape (1440,)
        """
        coords = CoordSystem(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([-self.DT, np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90.0, -90.0, 721, dtype=np.float32),
                "lon": np.linspace(
                    0.0,
                    360.0,
                    1440,
                    dtype=np.float32,
                    endpoint=False,
                ),
            }
        )
        return coords

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system produced by a single Atlas step (t+6h).

        Parameters
        ----------
        input_coords : CoordSystem
            Coordinate system associated with the input to the forward pass.

        Returns
        -------
        CoordSystem
            Ordered dictionary with keys:
            - 'time' : np.ndarray[np.datetime64] (copied from input if present)
            - 'lead_time' : np.timedelta64 set to +6h
            - 'variable' : np.ndarray[str] matching `VARIABLES`
            - 'lat' : np.ndarray[float] (copied from input if present, else 721 values)
            - 'lon' : np.ndarray[float] (copied from input if present, else 1440 values)
        """
        output_coords = CoordSystem(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([self.DT]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90.0, -90.0, 721, dtype=np.float32),
                "lon": np.linspace(
                    0.0,
                    360.0,
                    1440,
                    dtype=np.float32,
                    endpoint=False,
                ),
            }
        )

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]

        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )

        return output_coords

    @batch_func()
    def prep_next_input(
        self,
        x_pred: torch.Tensor,
        coords_pred: CoordSystem,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare the next input for the Atlas model. Since the input requires two lead times
        but the model predicts one, we update a sliding window to make autoregressive predictions.

        Parameters
        ----------
        x_pred : torch.Tensor
            Predicted tensor from the previous step.
        coords_pred : CoordSystem
            Coordinates describing `x_pred`.
        x : torch.Tensor
            Input tensor from the previous step.
        coords : CoordSystem
            Coordinates describing `x`.
        """
        x_next = x.clone()
        # Fill latest step with most recent prediction
        x_next[:, :, 1:, :, :, :] = x_pred[:, :, :1, :, :, :]
        # Shift the previous latest step to the earlier position
        x_next[:, :, :1, :, :, :] = x[:, :, 1:, :, :, :]
        coords_next = coords.copy()
        coords_next["lead_time"] = coords_next["lead_time"] + self.DT
        return x_next, coords_next

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
    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of the prognostic model, integrating a single 6h step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., lead_time, variable, lat, lon) corresponding
            to the coordinate system. Lead times expected: [-6h, 0h].
        coords : CoordSystem
            Coordinate dictionary describing `x`.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor advanced to t+6h and its coordinate system.
        """

        # Sanitize NaNs in input sst
        if torch.isnan(x).any():
            logger.info("Atlas input contains NaNs, replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)

        output_coords = self.output_coords(coords)
        out = torch.empty_like(x[:, :, :1])

        # Loop over init times
        for i, _ in enumerate(coords["batch"]):
            for j, _ in enumerate(coords["time"]):
                slice_coords = coords.copy()
                slice_coords["time"] = slice_coords["time"][j : j + 1]
                pred, _ = self._forward(x[i, j, :], slice_coords)
                out[i, j, :] = pred

        return out, output_coords

    @torch.inference_mode()
    def _call_with_latent(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        prev_latents: list[list[torch.Tensor | None]] | None = None,
    ) -> tuple[torch.Tensor, CoordSystem, list[list[torch.Tensor]]]:
        """Internal helper that handles cached latents during autoregressive rollout."""

        # Sanitize NaNs in input sst
        if torch.isnan(x).any():
            logger.info("Atlas input contains NaNs, replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)

        output_coords = self.output_coords(coords)
        out = torch.empty_like(x[:, :, :1])
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
                pred, pred_latent = self._forward(x[i, j, :], slice_coords, prev_latent)
                out[i, j, :] = pred
                latents_out[i][j] = pred_latent

        return out, output_coords, latents_out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create an iterator that yields the initial state then successive 6h steps.

        Parameters
        ----------
        x : torch.Tensor
            Initial data tensor on device representing the initial condition.
        coords : CoordSystem
            Coordinate system for the initial data tensor.

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator yielding successive model outputs and their coordinates.
        """
        yield from self._default_generator(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()

        # Validate coords
        _ = self.output_coords(coords)

        # Sanitize NaNs in input sst
        if torch.isnan(x).any():
            logger.info("Atlas input contains NaNs, replacing with 0.0")
            x = torch.nan_to_num(x, nan=0.0)

        # Yield initial condition
        ic_coords = coords.copy()
        ic_coords["lead_time"] = ic_coords["lead_time"][-1:]
        yield x[:, :, -1:, :, :, :], ic_coords

        latent_cache: list[list[torch.Tensor | None]] | None = None
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            # Forward
            x_pred, coords_pred, latent_cache = self._call_with_latent(
                x, coords, prev_latents=latent_cache
            )
            # Rear hook
            x_pred, coords_pred = self.rear_hook(x_pred, coords_pred)
            yield x_pred, coords_pred.copy()

            # Prepare next input
            x, coords = self.prep_next_input(x_pred, coords_pred, x, coords)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default package for the Atlas model."""
        package = Package(
            "hf://nvidia/atlas-era5@fdce0480c5e6f03d409089bf285f4bcc1d84519e",
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

        with open(package.resolve("config.json")) as f:
            config = json.load(f)

        modelpkg = config["package"]

        autoencoders = nn.ModuleList()
        autoencoder_processors = nn.ModuleList()
        for i, ae_cfg in enumerate(modelpkg["autoencoders"]):
            ae_path = package.resolve(ae_cfg["model_path"])
            aeprocessor_path = package.resolve(ae_cfg["processor_path"])
            ae = Module.from_checkpoint(ae_path)
            aeprocessor = Module.from_checkpoint(aeprocessor_path)

            autoencoders.append(ae)
            autoencoder_processors.append(aeprocessor)

        model = Module.from_checkpoint(
            package.resolve(modelpkg["genmodel"]["model_path"])
        )
        model_processor = Module.from_checkpoint(
            package.resolve(modelpkg["genmodel"]["processor_path"])
        )

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
