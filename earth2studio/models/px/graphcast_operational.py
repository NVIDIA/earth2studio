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

import dataclasses
import functools
from collections.abc import Callable, Generator, Hashable, Iterator
from typing import Any

import numpy as np
import torch
import xarray as xr
from earth2studio.models._array_utils import _registered_grid

from earth2studio.grids import LatLonGrid
from earth2studio.lexicon.wb2 import WB2Lexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.aurora import _aurora_history
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_size,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    import chex
    import haiku as hk
    import jax
    from weathernext.utils import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        normalization,
        rollout,
    )
    from weathernext.weathernext1_graph import graphcast
except ImportError:
    OptionalDependencyFailure("graphcast")
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

VARIABLES = [
    "t2m",
    "msl",
    "u10m",
    "v10m",
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
]

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


def _jax_signature(
    variables: list[str], hours: int, shape: tuple[int, int]
) -> CoordinateSystem:
    grid = _registered_grid(
        LatLonGrid(
            np.linspace(90, -90, shape[0], endpoint=True),
            np.linspace(0, 360, shape[1], endpoint=False),
        )
    )
    return coord_array(
        ("batch", "time", "lead_time", "variable", "lat", "lon"),
        {
            "lead_time": np.array([-hours, 0], dtype="timedelta64[h]"),
            "variable": _jax_variables(variables),
        },
        dynamic=("batch", "time"),
        grid=grid,
    )


def _jax_variables(variables: list[str]) -> list[str]:
    return ["tp:sum:12h" if v == "tp12" else v for v in variables]


def _jax_output_coords(
    model: torch.nn.Module, x: CoordinateSystem, variables: list[str], hours: int
) -> CoordinateSystem:
    handshake_time(x, allow_dynamic=True)
    handshake_time(x, "lead_time")
    lead = x.lead_time.values
    handshake_dataarray(
        x.assign_coords(lead_time=lead - lead[-1]), model.input_coords()
    )
    replacements: dict[Hashable, np.ndarray | list[str]] = {
        "lead_time": x.lead_time.values[-1:] + np.timedelta64(hours, "h")
    }
    labels = _jax_variables(variables)
    if not np.array_equal(x.coords["variable"], labels):
        replacements["variable"] = labels
    return coord_array_like(x, replacements)


def _add_tisr_batched(data: xr.Dataset, backend_data_utils: Any) -> None:
    """Compute pinned WeatherNext solar forcing on singleton batches in order."""
    name = "toa_incident_solar_radiation"
    if name in data:
        return
    if data.sizes.get("batch", 1) == 1:
        backend_data_utils.add_tisr_var(data)
        return
    fields = []
    for index in range(data.sizes["batch"]):
        sample = data.isel(batch=slice(index, index + 1)).copy()
        backend_data_utils.add_tisr_var(sample)
        fields.append(sample[name])
    data[name] = xr.concat(fields, dim="batch")


def _jax_inputs(
    model: torch.nn.Module, x: xr.DataArray, hours: int, backend_data_utils: Any
) -> tuple:
    # Only native dimension coordinates belong in the backend dataset. User
    # auxiliaries remain on the public array and are restored at the boundary.
    tensor, coords = x.e2s.to_torch()
    coords["variable"] = np.array(
        ["tp12" if v == "tp:sum:12h" else v for v in coords["variable"]]
    )
    data, leads = model.from_dataarray_to_dataset(
        xr.DataArray(tensor.cpu().numpy().copy(), dims=x.dims, coords=coords), hours
    )
    task = (
        model.task_config if hasattr(model, "task_config") else model.ckpt.task_config
    )
    if "toa_incident_solar_radiation" in task.forcing_variables:
        _add_tisr_batched(data, backend_data_utils)
    inputs, targets, forcings = backend_data_utils.extract_inputs_targets_forcings(
        data, target_lead_times=leads, **dataclasses.asdict(task)
    )
    return data, inputs, targets, forcings


def _same_jax_input(a: xr.DataArray, b: xr.DataArray) -> bool:
    return a.variable.equals(b.variable) and all(
        a.coords[d].variable.equals(b.coords[d].variable)
        for d in ("time", "lead_time", "variable", "lat", "lon")
    )


def _jax_iterator(
    model: torch.nn.Module,
    x: xr.DataArray,
    hours: int,
    backend_jax: Any,
    backend_data_utils: Any,
    generated_forcings: bool = False,
) -> Iterator[xr.DataArray]:
    handshake_nonempty(x)
    handshake_time(x)
    model.output_coords(x)
    # Reserve per-time streams before the initial yield, as the original
    # iterators did. Hooks may replace fields without restarting these streams.
    with backend_jax.default_device(
        model.get_jax_device_from_tensor(model.device_buffer)
    ):
        rngs = [model._next_rng(t) for t in range(x.sizes["time"])]
    yield x.isel(lead_time=slice(-1, None)).copy(deep=True)
    handshake_time(x)
    iterators: list[Generator[xr.Dataset, Any, None]] = []
    refresh = False
    while True:
        history = (
            x
            if model.front_hook is model._default_hook
            else model.front_hook(x.copy(deep=True))
        )
        refresh = refresh or (history is not x and not _same_jax_input(history, x))
        model.output_coords(history)
        handshake_time(history)
        packed, restore = batch_func()._compress_array(model, history)
        signature = model.output_coords(packed)
        device = model.device_buffer.device
        with backend_jax.default_device(
            model.get_jax_device_from_tensor(model.device_buffer)
        ):
            replacements = []
            started = bool(iterators)
            if not iterators or refresh:
                for t in range(packed.sizes["time"]):
                    data, inputs, targets, forcings = _jax_inputs(
                        model,
                        packed.isel(time=slice(t, t + 1)),
                        hours,
                        backend_data_utils,
                    )
                    replacements.append((data, inputs, forcings))
                    if len(iterators) <= t:
                        kwargs = (
                            {"init_datetime": data.coords["datetime"].values[0, 1]}
                            if generated_forcings
                            else {"batch": data}
                        )
                        iterators.append(
                            model._chunked_prediction_generator(
                                predictor_fn=model.run_forward,
                                rng=rngs[t],
                                inputs=inputs,
                                targets_template=targets * np.nan,
                                forcings=forcings,
                                **kwargs,
                            )
                        )
            predictions = [
                it.send(replacements[t]) if started and refresh else next(it)
                for t, it in enumerate(iterators)
            ]
            if hasattr(model, "_update_cyclone_tracks"):
                if len(predictions) == 1:
                    model._update_cyclone_tracks(
                        predictions[0], signature, accumulate_predictions=True
                    )
                elif model.track_cyclones:
                    from loguru import logger

                    logger.warning(
                        "Cyclone tracking currently supports one init time per iterator."
                    )
            results = [model.iterator_result_to_tensor(pred) for pred in predictions]
        out = from_torch(
            torch.cat(results, dim=1).to(device), signature, name=history.name
        )
        out.encoding = history.encoding.copy()
        out = restore(out)
        prediction = (
            out
            if model.rear_hook is model._default_hook
            else model.rear_hook(out.copy(deep=True))
        )
        refresh = prediction is not out and not _same_jax_input(prediction, out)
        variables = model.input_coords().coords["variable"].values
        latest = prediction.reindex(variable=variables)
        # GenCast's SST is an input-only invariant.
        missing = ~np.isin(variables, prediction.coords["variable"].values)
        if missing.any():
            a, _ = history.isel(lead_time=slice(-1, None)).e2s.to_torch()
            b, _ = latest.e2s.to_torch()
            b[..., missing, :, :] = a.to(b.device)[..., missing, :, :]
        x = _aurora_history(history, latest)
        yield prediction


@check_optional_dependencies()
class GraphCastOperational(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """GraphCast operational model

    A high-resolution model (0.25 degree resolution, 13 pressure levels) pre-trained on ERA5 data
    from 1979 to 2017 and fine-tuned on HRES data from 2016 to 2021. This model can be initialized
    from HRES data (does not require precipitation inputs).

    The model operates on a 0.25-degree lat-lon grid (south-pole including) equirectangular grid
    with 85 variables including:

    - Surface variables (2m temperature, 10m winds, etc.)
    - Pressure level variables (temperature, winds, geopotential, etc.)
    - Static variables (land-sea mask, surface geopotential)

    Note
    ----
    As of July 2026, GraphCast was renamed to WeatherNext 1-Graph in the
    WeatherNext codebase. This Earth2Studio wrapper keeps the GraphCast name
    for backwards compatibility.

    For more information see the following references:

    - https://arxiv.org/abs/2212.12794
    - https://github.com/google-deepmind/weathernext
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
    land_sea_mask : np.array
        Quater degree resolution [721x1440] land sea mask on lat-lon grid
    geopotential_at_surface : np.array
        Quater degree resolution [721x1440] geopotential at surface on lat-lon grid

    Badges
    ------
    region:global class:medium-range product:wind product:precip product:temp product:atmos year:2022
    gpu:40gb provider:google backend:jax
    """

    def __init__(
        self,
        ckpt: "graphcast.CheckPoint",
        diffs_stddev_by_level: xr.Dataset,
        mean_by_level: xr.Dataset,
        stddev_by_level: xr.Dataset,
        land_sea_mask: np.array,
        geopotential_at_surface: np.array,
    ):
        super().__init__()

        self.ckpt = ckpt
        self.diffs_stddev_by_level = diffs_stddev_by_level
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.land_sea_mask = land_sea_mask
        self.geopotential_at_surface = geopotential_at_surface
        self.prng_key = jax.random.PRNGKey(0)

        self.run_forward = self._load_run_forward_from_checkpoint()

        self.register_buffer("device_buffer", torch.empty(0))

    def _next_rng(self, time_index: int) -> "chex.PRNGKey":
        return self.prng_key

    def _load_run_forward_from_checkpoint(self) -> "autoregressive.Predictor":
        """This function is mostly copied from
        https://github.com/google-deepmind/weathernext/tree/main

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
    ) -> Generator[xr.Dataset, tuple | None, None]:
        """This is used to construct the iterator for the prognostic model.

        This function is mostly copied from
        https://github.com/google-deepmind/weathernext/tree/main

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
            replacement = yield predictions
            if replacement is not None:
                batch, current_inputs, forcings = replacement
                index += 1
                continue
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
            _add_tisr_batched(batch, data_utils)

            # Compute batch
            batch = batch.compute()

            # Get new forcings
            forcings = batch.isel(time=slice(-1, None))[list(FORCING_VARIABLES)]
            forcings = forcings.reset_coords("datetime", drop=True)
            forcings = forcings.compute()

            # Increment index
            index += 1

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the final input then native six-hour rollout predictions."""
        yield from _jax_iterator(self, x, 6, jax, data_utils)

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

        if "ensemble" in dataset.dims:
            dataset = dataset.squeeze("batch", drop=True)

        dataset = dataset.rename({key: INV_VOCAB[key] for key in dataset.data_vars})

        if "batch" in dataset.dims:
            dataarray = (
                dataset[VARIABLES + ["tp06"]]
                .to_dataarray()
                .T.transpose(
                    ..., "batch", "time", "lead_time", "variable", "lat", "lon"
                )
            )
        else:
            dataarray = (
                dataset[VARIABLES + ["tp06"]]
                .to_dataarray()
                .T.transpose(..., "time", "lead_time", "variable", "lat", "lon")
            )

        out = torch.from_numpy(dataarray.to_numpy().copy())
        out = out.flip(-2)  # Flip lat from ascending (-90->90, JAX native) to (90->-90)
        return out

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
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict a six-hour DataArray without hooks."""
        signature = self.output_coords(x)
        handshake_time(x)
        device = self.device_buffer.device
        with jax.default_device(self.get_jax_device_from_tensor(self.device_buffer)):
            results = []
            for t in range(x.sizes["time"]):
                _, inputs, targets, forcings = _jax_inputs(
                    self, x.isel(time=slice(t, t + 1)), 6, data_utils
                )

                predictions = rollout.chunked_prediction(
                    self.run_forward,
                    rng=self.prng_key,
                    inputs=inputs,
                    targets_template=targets * np.nan,
                    forcings=forcings,
                )
                results.append(self.iterator_result_to_tensor(predictions))

            out = from_torch(
                torch.cat(results, dim=1).to(device), signature, name=x.name
            )
            out.encoding = x.encoding.copy()
            return out

    def from_dataarray_to_dataset(
        self, data: xr.DataArray, lead_time: int = 6, hour_steps: int = 6
    ) -> xr.Dataset:
        """From a datarray get a dataset"""
        handshake_time(data)
        handshake_size(data, "time", 1)
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
                self.input_coords().lead_time.values,
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
            else:
                out_data[arco_variable] = data[var]
        for var in pressure_level_vars:
            out_data[var] = xr.concat(pressure_level_vars[var], dim="level")

        # Shape up for  data_utils.extract_inputs_targets_forcings, need 3 timesteps
        out_data = out_data.assign_coords(
            datetime=all_datetimes[: len(out_data.time.values)]
        )
        out_data = out_data.assign_coords(time=time_deltas[: len(out_data.time.values)])
        batch_size = out_data.sizes.get("batch", 1)
        out_data["datetime"] = out_data.datetime.expand_dims(dict(batch=batch_size))

        # add batch dimension
        for var in out_data.data_vars:
            if "batch" not in out_data[var].dims:
                out_data[var] = out_data[var].expand_dims(dict(batch=1))

        # pad times for target
        out_data = out_data.pad(pad_width=dict(time=(0, len(lead_times))))
        out_data = out_data.assign_coords(
            coords=dict(
                time=time_deltas,
                datetime=(("batch", "time"), np.tile(all_datetimes, (batch_size, 1))),
            )
        )
        # make sure lat is -90 to 90
        out_data = out_data.reindex(lat=sorted(out_data.lat.values))
        out_data = out_data.transpose("batch", "time", "level", "lat", "lon", ...)

        # add in zeros tp06 (operational model does not need tp06)
        shape = out_data["2m_temperature"].shape
        dims = out_data["2m_temperature"].dims
        coords = {dim: out_data["2m_temperature"].coords[dim] for dim in dims}
        out_data["total_precipitation_6hr"] = xr.DataArray(
            np.zeros(shape, dtype=np.float32), dims=dims, coords=coords
        )

        # Add land sea mask and geo-potential at surface
        out_data["land_sea_mask"] = xr.DataArray(
            self.land_sea_mask, dims=("lat", "lon")
        )
        out_data["geopotential_at_surface"] = xr.DataArray(
            self.geopotential_at_surface, dims=("lat", "lon")
        )

        # change dtype
        for var in out_data.data_vars:
            out_data[var] = out_data[var].astype(np.float32)

        return out_data, target_lead_times

    def input_coords(self) -> CoordinateSystem:
        """Declare the pole-inclusive quarter-degree input grid and history."""
        return _jax_signature(VARIABLES, 6, (721, 1440))

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Plan six-hour output including accumulated precipitation."""
        return _jax_output_coords(self, input_coords, VARIABLES + ["tp06"], 6)

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
    @check_optional_dependencies()
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
            "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"
        )
        with open(params, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)

        sample_input = xr.load_dataset(
            package.resolve(
                "dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc"
            )
        )
        land_sea_mask = sample_input["land_sea_mask"].values
        geopotential_at_surface = sample_input["geopotential_at_surface"].values

        return cls(
            ckpt,
            diffs_stddev_by_level,
            mean_by_level,
            stddev_by_level,
            land_sea_mask,
            geopotential_at_surface,
        )
