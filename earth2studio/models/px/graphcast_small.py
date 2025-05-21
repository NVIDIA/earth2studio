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

import dataclasses
import functools
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator

import numpy as np
import torch
import xarray as xr

try:
    import chex
    import haiku as hk
    import jax
    from graphcast import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        graphcast,
        normalization,
        rollout,
    )
except ImportError:
    hk = None
    jax = None
    chex = None
    autoregressive = None
    casting = None
    checkpoint = None
    data_utils = None
    graphcast = None
    normalization = None
    rollout = None

from earth2studio.lexicon.wb2 import WB2Lexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import check_extra_imports
from earth2studio.utils.coords import map_coords
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "t2m",
    "msl",
    "u10m",
    "v10m",
    "tp06",
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
    "w50",
    "w100",
    "w150",
    "w200",
    "w250",
    "w300",
    "w400",
    "w500",
    "w600",
    "w700",
    "w850",
    "w925",
    "w1000",
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
    "z",
    "lsm",
]

STATIC_VARS = (
    "geopotential_at_surface",  # z
    "land_sea_mask",  # lsm
)
EXTERNAL_FORCING_VARS = ("toa_incident_solar_radiation",)  # tisr
GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)
FORCING_VARIABLES = EXTERNAL_FORCING_VARS + GENERATED_FORCING_VARS

ATMOS_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

INV_VOCAB = {v: k for k, v in WB2Lexicon.VOCAB.items()}


@check_extra_imports("graphcast", ["haiku", "jax", "graphcast"])
class GraphCastSmall(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """GraphCast Small 1.0 degree model

    A smaller, low-resolution version of GraphCast (1 degree resolution, 13 pressure
    levels and a smaller mesh), trained on ERA5 data from 1979 to 2015. This model is
    useful for running with lower memory and compute constraints while maintaining good
    forecast skill. The model operates on a 1-degree lat-lon grid (south-pole including)
    equirectangular grid with 85 variables including:

    - Surface variables (2m temperature, 10m winds, etc.)
    - Pressure level variables (temperature, winds, geopotential, etc.)
    - Static variables (land-sea mask, surface geopotential)

    Note
    ----
    This model and checkpoint are based on the GraphCast architecture from DeepMind.
    For more information see the following references:

    - https://arxiv.org/abs/2212.12794
    - https://github.com/google-deepmind/graphcast
    - https://www.science.org/doi/10.1126/science.adi2336

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ckpt : graphcast.CheckPoint
        Model checkpoint containing weights and configuration
    diffs_stddev_by_level : xr.Dataset
        Standard deviation of differences by level for normalization
    mean_by_level : xr.Dataset
        Mean values by level for normalization
    stddev_by_level : xr.Dataset
        Standard deviation by level for normalization
    """

    def __init__(
        self,
        ckpt: "graphcast.CheckPoint",
        diffs_stddev_by_level: xr.Dataset,
        mean_by_level: xr.Dataset,
        stddev_by_level: xr.Dataset,
    ):
        super().__init__()

        self.ckpt = ckpt
        self.diffs_stddev_by_level = diffs_stddev_by_level
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.prng_key = jax.random.PRNGKey(0)

        self.run_forward = self._load_run_forward_from_checkpoint()

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [
                        np.timedelta64(-6, "h"),
                        np.timedelta64(0, "h"),
                    ]
                ),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(-90, 90, 181, endpoint=True),
                "lon": np.linspace(0, 360, 360, endpoint=False),
            }
        )

        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(-90, 90, 181, endpoint=True),
                "lon": np.linspace(0, 360, 360, endpoint=False),
            }
        )

    def _load_run_forward_from_checkpoint(self) -> "autoregressive.Predictor":
        """This function is mostly copied from
        https://github.com/google-deepmind/graphcast/tree/main

        License info:

        # Copyright 2023 DeepMind Technologies Limited.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #      http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS-IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        """
        state: dict = {}
        params = self.ckpt.params
        model_config = self.ckpt.model_config
        task_config = self.ckpt.task_config

        def construct_wrapped_graphcast(
            model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
        ) -> autoregressive.Predictor:
            """Constructs and wraps the GraphCast Predictor."""
            # Deeper one-step predictor.
            predictor = graphcast.GraphCast(model_config, task_config)

            # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
            # from/to float32 to/from BFloat16.
            predictor = casting.Bfloat16Cast(predictor)

            # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
            # BFloat16 happens after applying normalization to the inputs/targets.
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level,
            )

            # Wraps everything so the one-step model can produce trajectories.
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor

        @hk.transform_with_state
        def run_forward(
            model_config: graphcast.ModelConfig,
            task_config: graphcast.TaskConfig,
            inputs: xr.Dataset,
            targets_template: xr.Dataset,
            forcings: xr.Dataset,
        ) -> autoregressive.Predictor:
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(
                inputs, targets_template=targets_template, forcings=forcings
            )

        # Jax doesn't seem to like passing configs as args through the jit. Passing it
        # in via partial (instead of capture by closure) forces jax to invalidate the
        # jit cache if you change configs.
        def with_configs(fn: Callable) -> Callable:
            return functools.partial(
                fn, model_config=model_config, task_config=task_config
            )

        # Always pass params and state, so the usage below are simpler
        def with_params(fn: Callable) -> Callable:
            return functools.partial(fn, params=params, state=state)

        # Our models aren't stateful, so the state is always empty, so just return the
        # predictions. This is requiredy by our rollout code, and generally simpler.
        def drop_state(fn: Callable) -> Callable:
            return lambda **kw: fn(**kw)[0]

        return drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    def _chunked_prediction_generator(
        self,
        predictor_fn: "autoregressive.PredictorFn",
        rng: "chex.PRNGKey",
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        batch: xr.Dataset,
        forcings: xr.Dataset,
    ) -> Generator[xr.Dataset, None, None]:
        """This is used to construct the iterator for the prognostic model.

        This function is mostly copied from
        https://github.com/google-deepmind/graphcast/tree/main

        License info:

        # Copyright 2023 DeepMind Technologies Limited.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        #      http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS-IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        """

        # Create copies to avoid mutating inputs.
        inputs = xr.Dataset(inputs)
        targets_template = xr.Dataset(targets_template)
        forcings = xr.Dataset(forcings)

        # Our template targets will always have a time axis corresponding for the
        # timedeltas for the first chunk.
        targets_chunk_time = targets_template.time.isel(time=slice(0, 1))

        current_inputs = inputs

        def split_rng_fn(rng: chex.PRNGKey) -> tuple[chex.PRNGKey, chex.PRNGKey]:
            # Note, this is *not* equivalent to `return jax.random.split(rng)`, because
            # by assigning to a tuple, the single numpy array returned by
            # `jax.random.split` actually gets split into two arrays, so when calling
            # the function with pmap the output is Tuple[Array, Array], where the
            # leading axis of each array is `num devices`.
            rng1, rng2 = jax.random.split(rng)
            return rng1, rng2

        index = 0
        while True:

            # Reset forcings time to targets_chunk_time
            forcings = forcings.assign_coords(time=targets_chunk_time)
            forcings = forcings.compute()

            # Make predictions for the chunk.
            rng, this_rng = split_rng_fn(rng)
            predictions = predictor_fn(
                rng=this_rng,
                inputs=current_inputs,
                targets_template=targets_template,
                forcings=forcings,
            )
            next_frame = xr.merge([predictions, forcings])

            next_inputs = rollout._get_next_inputs(current_inputs, next_frame)

            # Shift timedelta coordinates, so we don't recompile at every iteration.
            next_inputs = next_inputs.assign_coords(time=current_inputs.coords["time"])
            current_inputs = next_inputs

            # At this point we can assign the actual targets time coordinates.
            predictions = predictions.assign_coords(
                time=targets_template.coords["time"] + index * np.timedelta64(6, "h")
            )
            yield predictions
            del predictions

            # Update batch time 6 hours and rename time to datetime
            batch = batch.assign_coords(
                datetime=batch.coords["datetime"] + np.timedelta64(6, "h")
            )

            # Pop forcings from batch if needed
            try:
                batch = batch.drop_vars(
                    list(FORCING_VARIABLES) + ["year_progress", "day_progress"]
                )
            except ValueError:
                pass

            # Compute forcings
            data_utils.add_derived_vars(batch)
            data_utils.add_tisr_var(batch)

            # Compute batch
            batch = batch.compute()

            # Get new forcings
            forcings = batch.isel(time=slice(-1, None))[list(FORCING_VARIABLES)]
            forcings = forcings.reset_coords("datetime", drop=True)
            forcings = forcings.compute()

            # Increment index
            index += 1

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem]]:
        coords = coords.copy()

        self.output_coords(coords)

        # Get device
        device = x.device

        # first batch has 2 times
        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][1:]
        yield x[:, :, 1:, ...], coords_out

        while True:

            # Forward is identity operator
            coords = self.output_coords(coords)

            # Get next prediction
            x = self.iterator_result_to_tensor(next(self.iterator))

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            # Convert to device
            x = x.to(device)

            coords = coords.copy()
            yield x, coords

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system


        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """

        with jax.default_device(self.get_jax_device_from_tensor(x)):
            batch, target_lead_times = self.from_dataarray_to_dataset(
                xr.DataArray(x.cpu(), coords=coords), 6
            )

            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                batch,
                target_lead_times=target_lead_times,
                **dataclasses.asdict(self.ckpt.task_config),
            )

            self.iterator = self._chunked_prediction_generator(
                predictor_fn=self.run_forward,
                rng=self.prng_key,
                inputs=inputs,
                targets_template=targets * np.nan,
                batch=batch,
                forcings=forcings,
            )

            yield from self._default_generator(x, coords)

    def iterator_result_to_tensor(self, dataset: xr.Dataset) -> torch.Tensor:
        """Convert a iterator result to a tensor"""
        for var in dataset.data_vars:
            if "level" in dataset[var].dims:
                for level in dataset[var].level:
                    dataset[f"{var}::{level.values}"] = dataset[var].sel(level=level)
                dataset = dataset.drop_vars(var)
            else:
                dataset = dataset.rename({var: f"{var}::"})
        dataset = dataset.drop_dims("level")
        if len(dataset.time) > 1:
            # Coming from call
            dataset = dataset.rename({"time": "lead_time"})
            dataset = dataset.expand_dims(dim="time")
        else:
            dataset = dataset.expand_dims(dim="lead_time")
        dataset["land_sea_mask::"] = xr.zeros_like(dataset["2m_temperature::"])
        dataset["geopotential_at_surface::"] = xr.zeros_like(
            dataset["2m_temperature::"]
        )

        if "ensemble" in dataset.dims:
            dataset = dataset.squeeze("batch", drop=True)

        dataset = dataset.rename({key: INV_VOCAB[key] for key in dataset.data_vars})

        if "batch" in dataset.dims:
            dataarray = (
                dataset[VARIABLES]
                .to_dataarray()
                .T.transpose(
                    ..., "batch", "time", "lead_time", "variable", "lat", "lon"
                )
            )
        else:
            dataarray = (
                dataset[VARIABLES]
                .to_dataarray()
                .T.transpose(..., "time", "lead_time", "variable", "lat", "lon")
            )

        return torch.from_numpy(dataarray.to_numpy().copy())

    @staticmethod
    def get_jax_device_from_tensor(x: torch.Tensor) -> "jax.Device":
        """From a tensor, get device and corresponding jax device"""
        device_id = x.get_device()
        if device_id == -1:  # -1 is CPU
            device = jax.devices("cpu")[0]
        else:
            device = jax.devices("gpu")[device_id]
        return device

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future
        """
        # Get device
        device = x.device

        with jax.default_device(self.get_jax_device_from_tensor(x)):
            # Map lat and lon if needed
            x, coords = map_coords(x, coords, self.input_coords())

            data, target_lead_times = self.from_dataarray_to_dataset(
                xr.DataArray(x.cpu(), coords=coords), 6
            )

            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                data,
                target_lead_times=target_lead_times,
                **dataclasses.asdict(self.ckpt.task_config),
            )

            predictions = rollout.chunked_prediction(
                self.run_forward,
                rng=self.prng_key,
                inputs=inputs,
                targets_template=targets * np.nan,
                forcings=forcings,
            )
            out = self.iterator_result_to_tensor(predictions)
            output_coords = self.output_coords(coords)

            # Convert to device
            out = out.to(device)

            return out, output_coords

    def from_dataarray_to_dataset(
        self, data: xr.DataArray, lead_time: int = 6, hour_steps: int = 6
    ) -> xr.Dataset:
        """From a datarray get a dataset"""
        if len(data.time.values) > 1:
            raise TypeError("GraphCast model only supports 1 init_time.")
        # time
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

        # 2nd date is center
        if len(data.time.values) == 1:
            start_date = (data.time.values + data.lead_time.values)[1]
        else:
            start_date = data.time.values[1]
        all_datetimes = [start_date + time_delta for time_delta in time_deltas]

        data = data.to_dataset(dim="variable")
        data = data.rename({key: WB2Lexicon.VOCAB[key] for key in data.data_vars})
        out_data = xr.Dataset(
            coords={
                "time": all_datetimes[0:2],
                "lat": data.lat,
                "lon": data.lon,
                "level": ATMOS_LEVELS,
            }
        )

        # Pressure levels back together
        pressure_level_vars = {}
        for var in data.data_vars:
            arco_variable, level = var.split("::")
            if level:
                if arco_variable not in pressure_level_vars:
                    pressure_level_vars[arco_variable] = [
                        data[var].expand_dims(dim=dict(level=[int(level)]))
                    ]
                else:
                    pressure_level_vars[arco_variable] += [
                        data[var].expand_dims(dim=dict(level=[int(level)]))
                    ]
            elif arco_variable in STATIC_VARS:
                out_data[arco_variable] = data[var].isel(time=0).squeeze()
            else:
                out_data[arco_variable] = data[var]
        for var in pressure_level_vars:
            out_data[var] = xr.concat(pressure_level_vars[var], dim="level")

        # Shape up for  data_utils.extract_inputs_targets_forcings, need 3 timesteps
        out_data = out_data.assign_coords(
            datetime=all_datetimes[: len(out_data.time.values)]
        )
        out_data = out_data.assign_coords(time=time_deltas[: len(out_data.time.values)])
        out_data["datetime"] = out_data.datetime.expand_dims(dict(batch=1))

        # add batch dimension
        for var in out_data.data_vars:
            if var not in STATIC_VARS:
                if "batch" not in out_data[var].dims:
                    out_data[var] = out_data[var].expand_dims(dict(batch=1))

        # pad times for target
        out_data = out_data.pad(pad_width=dict(time=(0, len(lead_times))))
        out_data = out_data.assign_coords(
            coords=dict(time=time_deltas, datetime=(("batch", "time"), [all_datetimes]))
        )
        # make sure lat is -90 to 90
        out_data = out_data.reindex(lat=sorted(out_data.lat.values))
        out_data = out_data.transpose("batch", "time", "level", "lat", "lon", ...)

        # change dtype
        for var in out_data.data_vars:
            out_data[var] = out_data[var].astype(np.float32)

        return out_data, target_lead_times

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(
        self,
        input_coords: CoordSystem,
    ) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
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
        """Load prognostic package"""
        return Package(
            "gs://dm_graphcast/graphcast",
            cache_options={
                "cache_storage": Package.default_cache("graphcast"),
                "same_names": True,
            },
        )

    @classmethod
    @check_extra_imports("graphcast", ["haiku", "jax", "graphcast"])
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        # Import the stats
        diffs_stddev_by_level = xr.load_dataset(
            package.resolve("stats/diffs_stddev_by_level.nc")
        ).compute()
        mean_by_level = xr.load_dataset(
            package.resolve("stats/mean_by_level.nc")
        ).compute()
        stddev_by_level = xr.load_dataset(
            package.resolve("stats/stddev_by_level.nc")
        ).compute()

        # Load model
        params = package.resolve(
            "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
        )
        with open(params, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)

        return cls(ckpt, diffs_stddev_by_level, mean_by_level, stddev_by_level)
