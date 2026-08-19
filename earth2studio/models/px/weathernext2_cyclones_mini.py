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

import copy
import dataclasses
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from typing import Any

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.lexicon.wb2 import WB2Lexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import map_coords
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    import chex
    import haiku as hk
    import jax
    import pandas as pd
    from weathernext.cyclones import constants as cyclone_constants
    from weathernext.cyclones import direct_tracker_6h_v1_config
    from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
    from weathernext.weathernext2 import fgn
except ImportError:
    OptionalDependencyFailure("weathernext")
    chex = None
    checkpoint = None
    data_utils = None
    cyclone_constants = None
    direct_tracker_6h_v1_config = None
    fgn = None
    fiddle_config_io = None
    hk = None
    jax = None
    pd = None
    rollout = None


SURFACE_INPUT_VARIABLES = [
    "t2m",
    "msl",
    "v10m",
    "u10m",
    "sst",
]
SURFACE_OUTPUT_VARIABLES = SURFACE_INPUT_VARIABLES + ["tp06"]
ATMOS_VARIABLES = ["t", "z", "u", "v", "w", "q"]
PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

INPUT_VARIABLES = SURFACE_INPUT_VARIABLES + [
    f"{var}{level}" for var in ATMOS_VARIABLES for level in PRESSURE_LEVELS
]
OUTPUT_VARIABLES = SURFACE_OUTPUT_VARIABLES + [
    f"{var}{level}" for var in ATMOS_VARIABLES for level in PRESSURE_LEVELS
]

WN2_TARGET_VARIABLES = tuple(
    dict.fromkeys(WB2Lexicon.VOCAB[var].split("::")[0] for var in OUTPUT_VARIABLES)
)
INV_VOCAB = {v: k for k, v in WB2Lexicon.VOCAB.items()}

MODEL_NAME = "WeatherNextCyclones_Mini"
MODEL_SPLIT = "2024"
PARAMS_PATH = f"params/{MODEL_NAME}_<{MODEL_SPLIT}.npz"
SAMPLE_PATH = (
    "dataset/source-hres_forecast_init-2024-10-07 00:00:00_"
    "res-1.0_levels-13_steps-01.nc"
)


@check_optional_dependencies()
class WeatherNext2CyclonesMini(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """WeatherNext 2 Cyclones Mini medium-range forecast model.

    WeatherNext 2 is Google DeepMind's global medium-range weather forecasting
    model family. This wrapper currently uses the public
    ``WeatherNextCyclones_Mini`` checkpoint and 1 degree sample grid, which is the
    mini model configuration that can be validated on the available single-GPU
    test hardware.

    The model requires two input states, valid at ``-6h`` and ``0h`` lead time,
    and predicts 6 hours forward per model call. By default this wrapper returns
    only the gridded weather fields expected by Earth2Studio prognostic models.
    Cyclone tracking can be enabled with ``track_cyclones=True`` to accumulate
    WeatherNext's tropical cyclone track diagnostics in the ``cyclone_tracks``
    property without changing the model output type.

    Examples
    --------
    Access tropical cyclone tracks after a model call:

    >>> model = WeatherNext2CyclonesMini.load_model(
    ...     WeatherNext2CyclonesMini.load_default_package(),
    ...     track_cyclones=True,
    ... )
    >>> x, coords = model(x, coords)
    >>> tracks = model.cyclone_tracks

    The tracker filters short-lived cyclogenesis tracks, so short rollouts can
    return an empty dataframe even when cyclone tracking is active. The active
    duration threshold is set by
    `model._cyclone_tracker.cyclogenesis_minimum_duration`.

    Note
    ----
    For more information see the following references:

    - https://doi.org/10.1038/s41586-026-10953-2
    - https://github.com/google-deepmind/weathernext

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ckpt : fgn.CheckPoint
        Model checkpoint containing weights.
    land_sea_mask : np.ndarray
        Land-sea mask on the WeatherNext grid.
    geopotential_at_surface : np.ndarray
        Surface geopotential on the WeatherNext grid.
    seed : int, optional
        Initial random seed for the stochastic FGN noise generator, by default 0.
    jit_compile : bool, optional
        JIT-compile the model forward pass, by default True.
    track_cyclones : bool, optional
        Accumulate tropical cyclone tracks in the ``cyclone_tracks`` property,
        by default False.

    Badges
    ------
    region:global class:medium-range product:wind product:precip product:temp product:atmos
    product:ocean year:2026 gpu:40gb provider:google backend:jax
    """

    def __init__(
        self,
        ckpt: "fgn.CheckPoint",
        land_sea_mask: np.ndarray,
        geopotential_at_surface: np.ndarray,
        seed: int = 0,
        jit_compile: bool = True,
        track_cyclones: bool = False,
    ):
        super().__init__()

        self.ckpt = ckpt
        self.land_sea_mask = land_sea_mask
        self.geopotential_at_surface = geopotential_at_surface
        self.seed = seed
        self.prng_key = jax.random.PRNGKey(seed)
        self.track_cyclones = track_cyclones
        self._cyclone_tracks = pd.DataFrame()
        self._cyclone_prediction_history: list[xr.Dataset] = []
        self._cyclone_tracker = None
        if self.track_cyclones:
            tracker_config = direct_tracker_6h_v1_config.get_config()
            self._cyclone_tracker = tracker_config.tracker_constructor(
                **tracker_config.tracker_kwargs
            )
        self.task_config = self._load_task_config()
        self.run_forward = self._load_run_forward_from_checkpoint(
            jit_compile=jit_compile
        )

        n_lat = land_sea_mask.shape[0]
        n_lon = land_sea_mask.shape[1]
        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, n_lat, endpoint=True),
                "lon": np.linspace(0, 360, n_lon, endpoint=False),
            }
        )
        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(OUTPUT_VARIABLES),
                "lat": np.linspace(90, -90, n_lat, endpoint=True),
                "lon": np.linspace(0, 360, n_lon, endpoint=False),
            }
        )

    @property
    def cyclone_tracks(self) -> "pd.DataFrame":
        """Tropical cyclone tracks accumulated during the latest model run."""
        if not self.track_cyclones:
            logger.warning("Cyclone tracking is currently not active on this model.")
            return pd.DataFrame()
        return self._cyclone_tracks.copy()

    def _reset_cyclone_tracks(self) -> None:
        """Reset accumulated cyclone track diagnostics."""
        self._cyclone_tracks = pd.DataFrame()
        self._cyclone_prediction_history = []

    @staticmethod
    def _empty_initial_storms() -> "pd.DataFrame":
        """Create an empty initial storm table for pure cyclogenesis tracking."""
        return pd.DataFrame(
            columns=[
                cyclone_constants.TRACK_ID,
                cyclone_constants.LEAD_TIME,
                cyclone_constants.VALID_TIME,
                cyclone_constants.LAT,
                cyclone_constants.LON,
            ]
        )

    def _update_cyclone_tracks(
        self,
        predictions: xr.Dataset,
        coords: CoordSystem,
        accumulate_predictions: bool,
    ) -> None:
        """Update cyclone tracks from native WeatherNext prediction fields."""
        if not self.track_cyclones:
            return
        if self._cyclone_tracker is None:
            logger.warning("Cyclone tracking is active, but no tracker is available.")
            return

        init_times = np.asarray(coords["time"]).reshape(-1)
        if len(init_times) != 1:
            logger.warning(
                "Cyclone tracking currently supports one init time per model run."
            )
            return

        cyclone_vars = [
            var for var in predictions.data_vars if var.startswith("cyclone")
        ]
        if not cyclone_vars:
            logger.warning(
                "Cyclone tracking is active, but this prediction did not include "
                "cyclone fields."
            )
            return

        cyclone_predictions = predictions[cyclone_vars].copy()
        if "batch" in cyclone_predictions.dims:
            if cyclone_predictions.sizes["batch"] != 1:
                logger.warning("Cyclone tracking currently supports batch size one.")
                return
            cyclone_predictions = cyclone_predictions.isel(batch=0, drop=True)
        cyclone_predictions = cyclone_predictions.assign_coords(
            time=np.asarray(coords["lead_time"]), init_time=init_times[0]
        )

        if accumulate_predictions:
            self._cyclone_prediction_history.append(cyclone_predictions)
            tracker_input = xr.concat(self._cyclone_prediction_history, dim="time")
            tracker_input = tracker_input.sortby("time")
            self._cyclone_tracks = self._cyclone_tracker(
                tracker_input, initial_storms_df=self._empty_initial_storms()
            )
            return

        tracks = self._cyclone_tracker(
            cyclone_predictions, initial_storms_df=self._empty_initial_storms()
        )
        if self._cyclone_tracks.empty:
            self._cyclone_tracks = tracks
        elif not tracks.empty:
            self._cyclone_tracks = pd.concat(
                [self._cyclone_tracks, tracks], ignore_index=True
            )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        output_coords = self._output_coords.copy()
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load default pre-trained WeatherNext 2 package from Google Cloud.

        Returns
        -------
        Package
            Model package.
        """
        return Package(
            "gs://dm_graphcast/weathernext2",
            cache_options={
                "cache_storage": Package.default_cache("weathernext2"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        seed: int = 0,
        jit_compile: bool = True,
        track_cyclones: bool = False,
    ) -> PrognosticModel:
        """Load prognostic model from package.

        Parameters
        ----------
        package : Package
            Package to load model from.
        seed : int, optional
            Initial random seed for the stochastic FGN noise generator, by default 0.
        jit_compile : bool, optional
            JIT-compile the model forward pass, by default True.
        track_cyclones : bool, optional
            Accumulate tropical cyclone tracks in the ``cyclone_tracks`` property,
            by default False.

        Returns
        -------
        PrognosticModel
            Prognostic model.
        """
        params_path = package.resolve(PARAMS_PATH)
        with open(params_path, "rb") as f:
            ckpt = checkpoint.load(f, fgn.CheckPoint)

        sample_input = xr.load_dataset(package.resolve(SAMPLE_PATH))
        land_sea_mask = sample_input["land_sea_mask"].values
        geopotential_at_surface = sample_input["geopotential_at_surface"].values

        return cls(
            ckpt,
            land_sea_mask,
            geopotential_at_surface,
            seed=seed,
            jit_compile=jit_compile,
            track_cyclones=track_cyclones,
        )

    def _load_task_config(self) -> Any:
        config = fiddle_config_io.get_fiddle_config_by_name(
            f"weathernext2/configs/{MODEL_NAME}"
        )
        target_variables = (
            config.task.target_variables
            if self.track_cyclones
            else WN2_TARGET_VARIABLES
        )
        return dataclasses.replace(config.task, target_variables=target_variables)

    def _load_run_forward_from_checkpoint(self, jit_compile: bool = True) -> Callable:
        """Build WeatherNext 2 inference function from checkpoint."""
        config = copy.deepcopy(
            fiddle_config_io.get_fiddle_config_by_name(
                f"weathernext2/configs/{MODEL_NAME}"
            )
        )
        task_config = self.task_config
        noisy_function_kwargs = config.predictor_kwargs["noisy_function_kwargs"]
        noisy_function_kwargs["per_var_activation_fns"] = {
            key: value
            for key, value in noisy_function_kwargs.get(
                "per_var_activation_fns", {}
            ).items()
            if key in task_config.target_variables
        }
        transformer_kwargs = noisy_function_kwargs["mesh_model_ctor"].keywords[
            "transformer_kwargs"
        ]
        if jax.default_backend() == "gpu":
            transformer_kwargs["attention_type"] = "triblockdiag_mha"

        config_inference = fgn.PredictorConfig(
            task=task_config,
            predictor_constructor=config.predictor_constructor,
            predictor_kwargs=config.predictor_kwargs,
            predictor_wrappers=config.predictor_wrappers[:-1],
        )

        @hk.transform
        def run_forward(
            inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset
        ) -> xr.Dataset:
            predictor = fgn.construct_predictor(config_inference)
            return predictor(
                inputs, targets_template=targets_template, forcings=forcings
            )

        def apply(
            rng: "chex.PRNGKey",
            inputs: xr.Dataset,
            targets_template: xr.Dataset,
            forcings: xr.Dataset,
        ) -> xr.Dataset:
            return run_forward.apply(
                self.ckpt.params, rng, inputs, targets_template, forcings
            )

        if jit_compile:
            return jax.jit(apply)
        return apply

    def iterator_result_to_tensor(self, dataset: xr.Dataset) -> torch.Tensor:
        """Convert an xarray Dataset prediction to an Earth2Studio tensor."""
        dataset = dataset[
            [var for var in dataset.data_vars if var in WN2_TARGET_VARIABLES]
        ]
        for var in list(dataset.data_vars):
            if "level" in dataset[var].dims:
                for level in dataset[var].level:
                    dataset[f"{var}::{level.values}"] = dataset[var].sel(level=level)
                dataset = dataset.drop_vars(var)
            else:
                dataset = dataset.rename({var: f"{var}::"})

        if "level" in dataset.dims:
            dataset = dataset.drop_dims("level")
        if len(dataset.time) > 1:
            dataset = dataset.rename({"time": "lead_time"})
            dataset = dataset.expand_dims(dim="time")
        else:
            dataset = dataset.expand_dims(dim="lead_time")
        if "sample" in dataset.dims:
            dataset = dataset.isel(sample=0, drop=True)

        dataset = dataset.rename({key: INV_VOCAB[key] for key in dataset.data_vars})
        if "batch" in dataset.dims:
            dataarray = (
                dataset[OUTPUT_VARIABLES]
                .to_dataarray()
                .T.transpose(
                    ..., "batch", "time", "lead_time", "variable", "lat", "lon"
                )
            )
        else:
            dataarray = (
                dataset[OUTPUT_VARIABLES]
                .to_dataarray()
                .T.transpose(..., "time", "lead_time", "variable", "lat", "lon")
            )
        out = torch.from_numpy(dataarray.to_numpy().copy())
        return out.flip(-2)

    @staticmethod
    def get_jax_device_from_tensor(x: torch.Tensor) -> "jax.Device":
        """From a tensor, get device and corresponding JAX device."""
        device_id = x.get_device()
        if device_id == -1:
            return jax.devices("cpu")[0]
        return jax.devices("gpu")[device_id]

    def from_dataarray_to_dataset(
        self, data: xr.DataArray, lead_time: int = 6, hour_steps: int = 6
    ) -> tuple[xr.Dataset, list[str]]:
        """Convert an Earth2Studio DataArray to a WeatherNext 2 Dataset."""
        if len(data.time.values) > 1:
            raise TypeError("WeatherNext 2 only supports one init_time per JAX call.")
        if "lead_time" in data.dims:
            data["lead_time"] = [
                data.time.values[0] + level for level in data.lead_time.values
            ]
            data = data.isel(time=0).reset_coords("time", drop=True)
            data = data.rename({"lead_time": "time"})

        lead_times = range(hour_steps, lead_time + hour_steps, hour_steps)
        target_lead_times = [f"{h}h" for h in lead_times]
        time_deltas = np.concatenate(
            (
                self._input_coords["lead_time"],
                [np.timedelta64(h, "h") for h in lead_times],
            )
        )
        start_date = data.time.values[-1]
        all_datetimes = [start_date + time_delta for time_delta in time_deltas]

        data = data.to_dataset(dim="variable")
        data = data.rename({key: WB2Lexicon.VOCAB[key] for key in data.data_vars})
        out_data = xr.Dataset(
            coords={
                "time": all_datetimes[0:2],
                "lat": data.lat,
                "lon": data.lon,
                "level": PRESSURE_LEVELS,
            }
        )

        pressure_level_vars: dict[str, list[xr.DataArray]] = {}
        for var in data.data_vars:
            wb2_variable, level = var.split("::")
            if level:
                pressure_level_vars.setdefault(wb2_variable, []).append(
                    data[var].expand_dims(dim=dict(level=[int(level)]))
                )
            else:
                out_data[wb2_variable] = data[var]
        for var in pressure_level_vars:
            out_data[var] = xr.concat(pressure_level_vars[var], dim="level")

        out_data = out_data.assign_coords(
            datetime=all_datetimes[: len(out_data.time.values)]
        )
        out_data = out_data.assign_coords(time=time_deltas[: len(out_data.time.values)])
        out_data["datetime"] = out_data.datetime.expand_dims(dict(batch=1))
        for var in out_data.data_vars:
            if "batch" not in out_data[var].dims:
                out_data[var] = out_data[var].expand_dims(dict(batch=1))

        out_data = out_data.pad(pad_width=dict(time=(0, len(lead_times))))
        out_data = out_data.assign_coords(
            coords=dict(time=time_deltas, datetime=(("batch", "time"), [all_datetimes]))
        )
        out_data = out_data.reindex(lat=sorted(out_data.lat.values))
        out_data = out_data.transpose("batch", "time", "level", "lat", "lon", ...)
        out_data["land_sea_mask"] = xr.DataArray(
            self.land_sea_mask, dims=("lat", "lon")
        )
        out_data["geopotential_at_surface"] = xr.DataArray(
            self.geopotential_at_surface, dims=("lat", "lon")
        )
        out_data["total_precipitation_6hr"] = xr.full_like(
            out_data["2m_temperature"], np.nan
        )
        for var in self.task_config.target_variables:
            if var.startswith("cyclone") and var not in out_data:
                out_data[var] = xr.full_like(out_data["2m_temperature"], np.nan)
        for var in out_data.data_vars:
            out_data[var] = out_data[var].astype(np.float32)
        return out_data, target_lead_times

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model one step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future.
        """
        self._reset_cyclone_tracks()
        device = x.device
        with jax.default_device(self.get_jax_device_from_tensor(x)):
            x, coords = map_coords(x, coords, self.input_coords())
            time_dim = list(coords.keys()).index("time")
            results = []
            for t in range(len(coords["time"])):
                x_t = x.narrow(time_dim, t, 1)
                coords_t = coords.copy()
                coords_t["time"] = coords["time"][t : t + 1]
                data, target_lead_times = self.from_dataarray_to_dataset(
                    xr.DataArray(x_t.cpu(), coords=coords_t), 6
                )
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    data,
                    target_lead_times=target_lead_times,
                    **dataclasses.asdict(self.task_config),
                )
                self.prng_key, rng = jax.random.split(self.prng_key)
                predictions = rollout.chunked_prediction(
                    self.run_forward,
                    rng=rng,
                    inputs=inputs,
                    targets_template=targets * np.nan,
                    forcings=forcings,
                )
                self._update_cyclone_tracks(
                    predictions,
                    self.output_coords(coords_t),
                    accumulate_predictions=False,
                )
                results.append(self.iterator_result_to_tensor(predictions))

            out = torch.cat(results, dim=1) if len(results) > 1 else results[0]
            return out.to(device), self.output_coords(coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem]]:
        coords = coords.copy()
        self.output_coords(coords)
        device = x.device
        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][1:]
        yield x[:, :, 1:, ...], coords_out

        while True:
            coords = self.output_coords(coords)
            predictions = [next(it) for it in self.iterators]
            if len(predictions) == 1:
                self._update_cyclone_tracks(
                    predictions[0], coords, accumulate_predictions=True
                )
            elif self.track_cyclones:
                logger.warning(
                    "Cyclone tracking currently supports one init time per iterator."
                )
            results = [self.iterator_result_to_tensor(pred) for pred in predictions]
            x = torch.cat(results, dim=1) if len(results) > 1 else results[0]
            x, coords = self.rear_hook(x, coords)
            yield x.to(device), coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create a time-integration iterator for the prognostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates model time steps.
        """
        self.output_coords(coords)
        self._reset_cyclone_tracks()
        with jax.default_device(self.get_jax_device_from_tensor(x)):
            time_dim = list(coords.keys()).index("time")
            self.iterators = []
            for t in range(len(coords["time"])):
                x_t = x.narrow(time_dim, t, 1)
                coords_t = coords.copy()
                coords_t["time"] = coords["time"][t : t + 1]
                data, target_lead_times = self.from_dataarray_to_dataset(
                    xr.DataArray(x_t.cpu(), coords=coords_t), 6
                )
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    data,
                    target_lead_times=target_lead_times,
                    **dataclasses.asdict(self.task_config),
                )
                self.prng_key, rng = jax.random.split(self.prng_key)
                self.iterators.append(
                    rollout.chunked_prediction_generator(
                        predictor_fn=self.run_forward,
                        rng=rng,
                        inputs=inputs,
                        targets_template=targets * np.nan,
                        forcings=forcings,
                        num_steps_per_chunk=1,
                    )
                )
            yield from self._default_generator(x, coords)
