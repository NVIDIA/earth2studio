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
from typing import Iterator, Generator
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timezone
from loguru import logger

from earth2studio.models.batch import batch_func
from earth2studio.models.auto.mixin import AutoModelMixin
from earth2studio.models.auto.package import Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.imports import check_optional_dependencies, OptionalDependencyFailure
from earth2studio.utils.type import CoordSystem
from earth2studio.utils.coords import handshake_coords, handshake_dim

try:
    from nvw.models.base_model import BaseModel as AtlasBaseModel
    from nvw import models as nvw_models
    from nvw import training as nvw_training
    from nvw.stochastic.interpolants import StochasticInterpolant
except ImportError:
    OptionalDependencyFailure("atlas")
    AtlasBaseModel = None
    nvw_models = None
    nvw_training = None
    StochasticInterpolant = None


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
    "tp"
]


@check_optional_dependencies()
class Atlas(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Atlas prognostic model for ERA5 variables on a 0.25° global lat-lon grid.

    Atlas consumes two input lead times (t-6h and t) and predicts a single step at
    t+6h on a 721x1440 latitude-longitude grid.
    """

    DT = np.timedelta64(6, "h")

    def __init__(self,
        autoencoders: nn.ModuleList,
        autoencoder_processors: nn.ModuleList,
        model: nn.Module,
        model_processor: nn.Module,
        sinterpolant: nn.Module,
        means: np.ndarray,
        stds: np.ndarray,
        sinterpolant_sample_steps: int = 60,
    ) -> None:
        super().__init__()
        self.autoencoders = autoencoders
        self.autoencoder_processors = autoencoder_processors
        self.model = model
        self.model_processor = model_processor
        self.sinterpolant = sinterpolant
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.sinterpolant_sample_steps = sinterpolant_sample_steps

    def input_coords(self) -> CoordSystem:
        """Input coordinate system expected by Atlas.

        Notes
        -----
        - Lead times are fixed to [-6h, 0h].
        - Variables are defined by the module-level `VARIABLES`.
        - Spatial grid is 0.25° lat-lon: 721 latitudes (-90..90), 1440 longitudes (0..360).

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
                "lon": np.linspace(0.0, 360.0 - (360.0 / 1440.0), 1440, dtype=np.float32),
            }
        )
        return coords

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
                "lon": np.linspace(0.0, 360.0 - (360.0 / 1440.0), 1440, dtype=np.float32),
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

    #@batch_func()
    def prep_next_input(self, x_pred: torch.Tensor, coords_pred: CoordSystem, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
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
        x_next[:, :, -1:, :, :, :] = x_pred[:, :, :1, :, :, :] # Full latest step with most recent prediction
        x_next[:, :, :-1, :, :, :] = x[:, :, 1:, :, :, :] # Shift the previous step to the previous position
        coords_next = coords.copy()
        coords_next["lead_time"] = coords_next["lead_time"] + self.DT
        return x_next, coords_next

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor, coords: CoordSystem) -> torch.Tensor:
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
        torch.Tensor
            Output tensor advanced to t+6h and its coordinate system.
        """

        if x.ndim != 4:
            raise ValueError(f"Internal forward pass expects x of shape (lead_time, variable, lat, lon), got {x.shape}")
        if len(coords["lead_time"]) != 2:
            raise ValueError(f"Internal forward pass expects coords['lead_time'] of length 2, got {len(coords['lead_time'])}")

        # Prepare input tensor and date metadata 
        x_cur, x_prev = x[:1, :, :, :], x[1:, :, :, :]
        t = coords['time'][0] + coords["lead_time"][-1]
        current_date = np.array([[datetime.fromtimestamp(t.astype('datetime64[s]').astype(int))]])

        # Preprocess to build high/low-res latent/state
        self.model_processor.add_noise = False # TODO needed or not?
        high_res, low_res = self.model_processor.preprocess_input(x_cur, current_date)
        prev = self.model_processor.normalizer_in.normalize(x_prev)
        prev = self.model_processor.intep(prev, self.model_processor.downsample_grid_shape)

        # Condition dictionary
        cond = {'x_1': low_res.clone(), 'x_2': prev.clone()}

        # Stochastic interpolant sampling in latent space
        prediction_latent = self.sinterpolant.sample(
            self.model,
            low_res.clone(),
            steps= self.sinterpolant_sample_steps,
            cond=cond,
            verbose=False,
            compute_normalization=True
        )

        # Decode according to configuration of model/autoencoder heads
        pred = self.autoencoders[0](high_res, prediction_latent)
        prediction_latent = self.model_processor.normalizer_out.unnormalize(prediction_latent)
        prediction_latent = prediction_latent + self.model_processor.normalizer_in.unnormalize(low_res)
        prediction_latent = self.model_processor.normalizer_in.normalize(prediction_latent)
        pred = self.autoencoders[0](high_res, prediction_latent)

        # Postprocess to state space
        pred = self.autoencoder_processors[0].postprocess(pred, x_cur)
        return pred

    @torch.inference_mode()
    @batch_func()
    def __call__(self, x: torch.Tensor, coords: CoordSystem) -> tuple[torch.Tensor, CoordSystem]:
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
                out[i, j, :] = self._forward(
                    x[i, j, :], slice_coords
                )

        return out, output_coords

    def create_iterator(self, x: torch.Tensor, coords: CoordSystem) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
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

    
    def _default_generator(self, x: torch.Tensor, coords: CoordSystem) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
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
        
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)
            # Forward
            x_pred, coords_pred = self.__call__(x, coords)
            # Rear hook
            x_pred, coords_pred = self.rear_hook(x_pred, coords_pred)
            yield x_pred, coords_pred.copy()

            # Prepare next input
            x, coords = self.prep_next_input(x_pred, coords_pred, x, coords)

    @classmethod
    def load_default_package(cls):
        """Load the default package for the Atlas model."""
        package = Package(
            "/lustre/fsw/portfolios/nvr/projects/nvr_earth2_e2/users/pharrington/model_pkg/atlas",
            cache_options={
                "cache_storage": Package.default_cache("atlas"),
                "same_names": True,
            },
        )
        return package


    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package):
        """Instantiate and load Atlas from a package."""
        
        with open(package.resolve("config.json"), "r") as f:
            config = json.load(f)

        modelpkg = config["package"]

        autoencoders = nn.ModuleList()
        autoencoder_processors = nn.ModuleList()
        for i, ae_cfg in enumerate(modelpkg["autoencoders"]):
            ae_path = package.resolve(ae_cfg["model_path"])
            ae_cls = getattr(nvw_models, config["model_meta"]["autoencoder_model"][i])
            aeprocessor_path = package.resolve(ae_cfg["processor_path"])
            aeprocessor_cls = getattr(nvw_training, config["model_meta"]["autoencoder_processor"][i])

            ae = ae_cls.from_checkpoint(ae_path)
            aeprocessor = aeprocessor_cls.from_checkpoint(aeprocessor_path)

            autoencoders.append(ae)
            autoencoder_processors.append(aeprocessor)
        
        model_cls = getattr(nvw_models, config["model_meta"]["genmodel_model"])
        model_processor_cls = getattr(nvw_training, config["model_meta"]["genmodel_processor"])
        model = model_cls.from_checkpoint(package.resolve(modelpkg["genmodel"]["model_path"]))
        model_processor = model_processor_cls.from_checkpoint(package.resolve(modelpkg["genmodel"]["processor_path"]))

        means_path = package.resolve(modelpkg["stats"]["means"][0])
        stds_path = package.resolve(modelpkg["stats"]["stds"][0])
        means = torch.from_numpy(np.load(means_path)).to(dtype=torch.float32)
        stds = torch.from_numpy(np.load(stds_path)).to(dtype=torch.float32)

        sinterpolant = StochasticInterpolant(
            alpha=config["sinterpolant"]["alpha"],
            beta=config["sinterpolant"]["beta"],
            sigma=config["sinterpolant"]["sigma"],
            g=config["sinterpolant"]["g"],
            epsilon=config["sinterpolant"]["epsilon"],
            noise_sampler=config["sinterpolant"]["noise_sampler"],
            time_sampler=config["sinterpolant"]["time_sampler"],
            sample_method=config["sinterpolant"]["sample_method"],
            studentt_deg=config["sinterpolant"]["studentt_deg"]
        )

        return cls(
            autoencoders=autoencoders,
            autoencoder_processors=autoencoder_processors,
            model=model,
            model_processor=model_processor,
            means=means,
            stds=stds,
            sinterpolant=sinterpolant,
            sinterpolant_sample_steps=config["sinterpolant"]["sample_steps"],
        )


