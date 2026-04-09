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

import contextlib
import dataclasses
import functools
import io
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterator
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.lexicon.wb2 import WB2Lexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords
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
    import jax.numpy as jnp
    from graphcast import (
        checkpoint,
        data_utils,
        gencast,
        graphcast,
        nan_cleaning,
        normalization,
        rollout,
        xarray_jax,
    )
except ImportError:
    OptionalDependencyFailure("gencast")
    hk = None
    jax = None
    jnp = None
    chex = None
    checkpoint = None
    data_utils = None
    gencast = None
    graphcast = None
    nan_cleaning = None
    normalization = None
    rollout = None
    xarray_jax = None

# Input variables: 5 surface + 6x13 atmospheric (no precipitation in inputs)
INPUT_SURFACE_VARIABLES = [
    "t2m",
    "msl",
    "u10m",
    "v10m",
    "sst",
]

# Output adds tp12 to surface variables
OUTPUT_SURFACE_VARIABLES = [
    "t2m",
    "msl",
    "u10m",
    "v10m",
    "sst",
    "tp12",
]

ATMOS_VARIABLES = [
    "t",
    "z",
    "u",
    "v",
    "w",
    "q",
]

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

# Build full variable lists: surface + atmospheric (var + level)
INPUT_VARIABLES = INPUT_SURFACE_VARIABLES + [
    f"{var}{level}" for var in ATMOS_VARIABLES for level in PRESSURE_LEVELS
]

OUTPUT_VARIABLES = OUTPUT_SURFACE_VARIABLES + [
    f"{var}{level}" for var in ATMOS_VARIABLES for level in PRESSURE_LEVELS
]

# Forcing variables (GenCast does NOT use toa_incident_solar_radiation)
GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)

INV_VOCAB = {v: k for k, v in WB2Lexicon.VOCAB.items()}


@check_optional_dependencies()
class GenCastMini(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """GenCast Mini diffusion-based weather prediction model.

    A stochastic weather prediction model based on conditional diffusion that predicts
    in 12-hour time steps. This mini variant operates at 1.0-degree (181x360) resolution
    with 13 pressure levels. The model takes 2 input frames (t-12h and t) and predicts
    12 hours ahead.

    The mini variant trained on ERA5 reanalysis data (pre-2019), offering
    significantly lower memory requirements (~16 GB vRAM) compared to the full
    0.25-degree operational model. This wrapper runs the model with operational inputs
    which includes a zero 12hr total precipitation input.

    Note
    ----
    This model is provided by DeepMind.
    For more information see the following references:

    - https://arxiv.org/abs/2312.15796
    - https://github.com/google-deepmind/graphcast
    - https://www.nature.com/articles/s41586-024-08252-9

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    ckpt : gencast.CheckPoint
        Model checkpoint containing weights and configuration
    diffs_stddev_by_level : xr.Dataset
        Standard deviation of differences by level for normalization
    mean_by_level : xr.Dataset
        Mean values by level for normalization
    stddev_by_level : xr.Dataset
        Standard deviation by level for normalization
    min_by_level : xr.Dataset
        Minimum values by level for NaN cleaning
    land_sea_mask : np.ndarray
        Land-sea mask on lat-lon grid
    geopotential_at_surface : np.ndarray
        Geopotential at surface on lat-lon grid
    sst_nan_mask : np.ndarray
        Boolean mask indicating where SST values are NaN (ocean vs land)
    seed : int | None, optional
        Random seed for JAX PRNG key used in stochastic sampling. If None, a random seed
        is generated each time the model is called, producing stochastic forecasts. By
        default 0.
    jit_compile : bool, optional
        JIT-compile the model forward pass, requires 24GB of host RAM. JIT compilation
        adds a one-time cost (several minutes for the first call) but makes subsequent
        calls significantly faster, by default True.

    Badges
    ------
    region:global class:mrf product:wind product:precip product:temp product:atmos
    product:ocean year:2024 gpu:40gb
    """

    def __init__(
        self,
        ckpt: "gencast.CheckPoint",
        diffs_stddev_by_level: xr.Dataset,
        mean_by_level: xr.Dataset,
        stddev_by_level: xr.Dataset,
        min_by_level: xr.Dataset,
        land_sea_mask: np.ndarray,
        geopotential_at_surface: np.ndarray,
        sst_nan_mask: np.ndarray,
        seed: int | None = 0,
        jit_compile: bool = True,
    ):
        super().__init__()

        self.ckpt = ckpt
        self.diffs_stddev_by_level = diffs_stddev_by_level
        self.mean_by_level = mean_by_level
        self.stddev_by_level = stddev_by_level
        self.min_by_level = min_by_level
        self.land_sea_mask = land_sea_mask
        self.geopotential_at_surface = geopotential_at_surface
        self.sst_nan_mask = sst_nan_mask
        self.seed = seed

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
                    [
                        np.timedelta64(-12, "h"),
                        np.timedelta64(0, "h"),
                    ]
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
                "lead_time": np.array([np.timedelta64(12, "h")]),
                "variable": np.array(OUTPUT_VARIABLES),
                "lat": np.linspace(90, -90, n_lat, endpoint=True),
                "lon": np.linspace(0, 360, n_lon, endpoint=False),
            }
        )

        self.register_buffer("device_buffer", torch.empty(0))

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

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
        """Output coordinate system of the prognostic model.

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
        """Load default pre-trained GenCast Mini model package from Google Cloud.

        Returns
        -------
        Package
            Model package
        """
        return Package(
            "gs://dm_graphcast/gencast",
            cache_options={
                "cache_storage": Package.default_cache("gencast"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        jit_compile: bool = True,
        seed: int | None = 0,
    ) -> PrognosticModel:
        """Load prognostic model from package.

        Parameters
        ----------
        package : Package
            Package to load model from
        jit_compile : bool, optional
            JIT-compile the model forward pass with, by default True.
        seed : int | None, optional
            Random seed for JAX PRNG key used in stochastic sampling, by default 0.

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        # Load normalization stats
        diffs_stddev_by_level = xr.load_dataset(
            package.resolve("stats/diffs_stddev_by_level.nc")
        ).compute()
        mean_by_level = xr.load_dataset(
            package.resolve("stats/mean_by_level.nc")
        ).compute()
        stddev_by_level = xr.load_dataset(
            package.resolve("stats/stddev_by_level.nc")
        ).compute()
        min_by_level = xr.load_dataset(
            package.resolve("stats/min_by_level.nc")
        ).compute()

        # Load checkpoint
        params_path = package.resolve("params/GenCast 1p0deg Mini <2019.npz")
        with open(params_path, "rb") as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)

        # Load static fields from sample dataset (same pattern as GraphCast)
        sample_input = xr.load_dataset(
            package.resolve(
                "dataset/source-era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc"
            )
        )
        land_sea_mask = sample_input["land_sea_mask"].values
        geopotential_at_surface = sample_input["geopotential_at_surface"].values
        # SST NaN mask: True where ocean (valid SST), False where land (NaN)
        sst_nan_mask = ~np.isnan(sample_input["sea_surface_temperature"].values[0, 0])

        return cls(
            ckpt,
            diffs_stddev_by_level,
            mean_by_level,
            stddev_by_level,
            min_by_level,
            land_sea_mask,
            geopotential_at_surface,
            sst_nan_mask,
            seed=seed,
            jit_compile=jit_compile,
        )

    # -------------------------------------------------------------------------
    # Private / support methods
    # -------------------------------------------------------------------------

    def _load_run_forward_from_checkpoint(self, jit_compile: bool = True) -> Callable:
        """Build GenCast inference function from checkpoint.

        This function is based on the inference pipeline from:
        https://github.com/google-deepmind/graphcast

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

        Returns
        -------
        Callable
            Function that takes (rng, inputs, targets_template, forcings)
        """
        state: dict = {}
        params = self.ckpt.params
        task_config = self.ckpt.task_config
        sampler_config = self.ckpt.sampler_config
        noise_config = self.ckpt.noise_config
        noise_encoder_config = self.ckpt.noise_encoder_config

        # Replace SplashAttention with TriBlockDiag for GPU compatibility
        splash_spt_cfg = (
            self.ckpt.denoiser_architecture_config.sparse_transformer_config
        )
        tbd_spt_cfg = dataclasses.replace(
            splash_spt_cfg, attention_type="triblockdiag_mha", mask_type="full"
        )
        denoiser_architecture_config = dataclasses.replace(
            self.ckpt.denoiser_architecture_config,
            sparse_transformer_config=tbd_spt_cfg,
        )

        # Monkey-patch xarray_jax to handle JAX tracers inside jax.lax.fori_loop.
        # GenCast's diffusion sampler (DPM-Solver++2S) uses fori_loop which
        # turns loop variables into DynamicJaxprTracer. The upstream
        # JaxArrayWrapper.__array_ufunc__ rejects these tracers because
        # DynamicJaxprTracer is not jax.typing.ArrayLike.
        # This is an upstream bug: https://github.com/google-deepmind/graphcast/issues/203
        # We intercept NotImplemented and fall back to the JAX operation directly.
        _original_array_ufunc = xarray_jax.JaxArrayWrapper.__array_ufunc__

        # Map numpy ufuncs to their JAX equivalents so we can operate
        # on JAX tracers without triggering numpy's __array__ conversion.
        _UFUNC_TO_JAX: dict[np.ufunc, Any] = {
            np.multiply: jnp.multiply,
            np.add: jnp.add,
            np.subtract: jnp.subtract,
            np.divide: jnp.divide,
            np.true_divide: jnp.true_divide,
            np.negative: jnp.negative,
            np.sqrt: jnp.sqrt,
            np.square: jnp.square,
            np.abs: jnp.abs,
            np.log: jnp.log,
            np.exp: jnp.exp,
            np.power: jnp.power,
            np.maximum: jnp.maximum,
            np.minimum: jnp.minimum,
        }

        def _patched_array_ufunc(
            self: "xarray_jax.JaxArrayWrapper",
            ufunc: np.ufunc,
            method: str,
            *inputs: Any,
            **kwargs: Any,
        ) -> Any:
            result = _original_array_ufunc(self, ufunc, method, *inputs, **kwargs)
            if result is NotImplemented:
                jax_fn = _UFUNC_TO_JAX.get(ufunc)
                if jax_fn is None:
                    return NotImplemented
                jax_inputs = [
                    i.jax_array if isinstance(i, xarray_jax.JaxArrayWrapper) else i
                    for i in inputs
                ]
                return xarray_jax.JaxArrayWrapper(jax_fn(*jax_inputs, **kwargs))
            return result

        xarray_jax.JaxArrayWrapper.__array_ufunc__ = _patched_array_ufunc

        def construct_wrapped_gencast(
            task_config: "graphcast.TaskConfig",
            denoiser_architecture_config: "gencast.DenoiserArchitectureConfig",
            sampler_config: "gencast.SamplerConfig",
            noise_config: "gencast.NoiseConfig",
            noise_encoder_config: "gencast.NoiseEncoderConfig",
        ) -> "gencast.GenCast":
            """Construct GenCast predictor with normalization and NaN cleaning."""
            predictor = gencast.GenCast(
                task_config=task_config,
                denoiser_architecture_config=denoiser_architecture_config,
                sampler_config=sampler_config,
                noise_config=noise_config,
                noise_encoder_config=noise_encoder_config,
            )

            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=self.diffs_stddev_by_level,
                mean_by_level=self.mean_by_level,
                stddev_by_level=self.stddev_by_level,
            )
            # Applies ERA5 NaN masking to SST fields for consistency
            predictor = nan_cleaning.NaNCleaner(
                predictor=predictor,
                reintroduce_nans=True,
                fill_value=self.min_by_level,
                var_to_clean="sea_surface_temperature",
            )

            return predictor

        @hk.transform_with_state
        def run_forward(
            task_config: "graphcast.TaskConfig",
            denoiser_architecture_config: "gencast.DenoiserArchitectureConfig",
            sampler_config: "gencast.SamplerConfig",
            noise_config: "gencast.NoiseConfig",
            noise_encoder_config: "gencast.NoiseEncoderConfig",
            inputs: xr.Dataset,
            targets_template: xr.Dataset,
            forcings: xr.Dataset,
        ) -> xr.Dataset:
            predictor = construct_wrapped_gencast(
                task_config,
                denoiser_architecture_config,
                sampler_config,
                noise_config,
                noise_encoder_config,
            )
            return predictor(
                inputs, targets_template=targets_template, forcings=forcings
            )

        def with_configs(fn: Callable) -> Callable:
            return functools.partial(
                fn,
                task_config=task_config,
                denoiser_architecture_config=denoiser_architecture_config,
                sampler_config=sampler_config,
                noise_config=noise_config,
                noise_encoder_config=noise_encoder_config,
            )

        def with_params(fn: Callable) -> Callable:
            return functools.partial(fn, params=params, state=state)

        def drop_state(fn: Callable) -> Callable:
            return lambda **kw: fn(**kw)[0]

        fn = drop_state(with_params(with_configs(run_forward.apply)))
        if jit_compile:
            fn = jax.jit(fn)
        return fn

    def _chunked_prediction_generator(
        self,
        predictor_fn: Callable,
        rng: "chex.PRNGKey",
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
        init_datetime: np.ndarray,
    ) -> Generator[xr.Dataset, None, None]:
        """Autoregressive prediction generator for GenCast.

        This function is based on the rollout logic from:
        https://github.com/google-deepmind/graphcast

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

        Parameters
        ----------
        predictor_fn : Callable
            JIT-compiled GenCast forward function
        rng : chex.PRNGKey
            JAX PRNG key for stochastic sampling
        inputs : xr.Dataset
            Initial condition dataset (2 frames)
        targets_template : xr.Dataset
            Template for target predictions (NaN-filled)
        forcings : xr.Dataset
            Forcing variables dataset
        init_datetime : np.ndarray
            Absolute datetime(s) for the initial condition (t=0 reference).
            Used to compute correct forcing variables (day/year progress) at each step.

        Yields
        ------
        xr.Dataset
            Prediction for each 12-hour time step
        """
        inputs = xr.Dataset(inputs)
        targets_template = xr.Dataset(targets_template)
        forcings = xr.Dataset(forcings)

        targets_chunk_time = targets_template.time.isel(time=slice(0, 1))
        current_inputs = inputs

        def split_rng_fn(
            rng: "chex.PRNGKey",
        ) -> tuple["chex.PRNGKey", "chex.PRNGKey"]:
            rng1, rng2 = jax.random.split(rng)
            return rng1, rng2

        index = 0
        while True:
            # Update forcings time coordinate
            forcings = forcings.assign_coords(time=targets_chunk_time)
            forcings = forcings.compute()

            # Make prediction for this step (suppress graphcast debug prints)
            rng, this_rng = split_rng_fn(rng)
            with contextlib.redirect_stdout(io.StringIO()):
                predictions = predictor_fn(
                    rng=this_rng,
                    inputs=current_inputs,
                    targets_template=targets_template,
                    forcings=forcings,
                )
            next_frame = xr.merge([predictions, forcings])

            # Update inputs for next step
            next_inputs = rollout._get_next_inputs(current_inputs, next_frame)
            next_inputs = next_inputs.assign_coords(time=current_inputs.coords["time"])
            current_inputs = next_inputs

            # Assign actual time coordinates to predictions
            predictions = predictions.assign_coords(
                time=targets_template.coords["time"] + index * np.timedelta64(12, "h")
            )
            yield predictions
            del predictions

            # Compute forcings for the NEXT step directly from
            # datetime, matching the notebook's precomputed approach.
            # The next step (index+1) targets T0 + (index+2)*12h.
            next_target_dt = init_datetime + np.timedelta64((index + 2) * 12, "h")
            next_dt = np.atleast_1d(next_target_dt)
            seconds = next_dt.astype("datetime64[s]").astype(np.int64)
            lon = current_inputs.coords["lon"].data

            has_batch = "batch" in current_inputs.dims
            batch_dim = ("batch",) if has_batch else ()

            year_progress = data_utils.get_year_progress(seconds)
            day_progress = data_utils.get_day_progress(seconds, lon)

            if has_batch:
                year_progress = year_progress[np.newaxis]
                day_progress = day_progress[np.newaxis]

            year_feats = data_utils.featurize_progress(
                "year_progress", batch_dim + ("time",), year_progress
            )
            day_feats = data_utils.featurize_progress(
                "day_progress",
                batch_dim + ("time",) + current_inputs.coords["lon"].dims,
                day_progress,
            )

            forcings = xr.Dataset(
                {
                    k: v
                    for k, v in {**year_feats, **day_feats}.items()
                    if k in GENERATED_FORCING_VARS
                }
            )

            index += 1

    def iterator_result_to_tensor(self, dataset: xr.Dataset) -> torch.Tensor:
        """Convert an iterator result xarray Dataset to a torch Tensor.

        Parameters
        ----------
        dataset : xr.Dataset
            xarray Dataset from JAX GenCast prediction

        Returns
        -------
        torch.Tensor
            Output tensor with shape matching earth2studio conventions
        """
        for var in list(dataset.data_vars):
            if "level" in dataset[var].dims:
                for level in dataset[var].level:
                    dataset[f"{var}::{level.values}"] = dataset[var].sel(level=level)
                dataset = dataset.drop_vars(var)
            else:
                dataset = dataset.rename({var: f"{var}::"})
        dataset = dataset.drop_dims("level")

        if len(dataset.time) > 1:
            # Coming from __call__
            dataset = dataset.rename({"time": "lead_time"})
            dataset = dataset.expand_dims(dim="time")
        else:
            dataset = dataset.expand_dims(dim="lead_time")

        if "ensemble" in dataset.dims:
            dataset = dataset.squeeze("batch", drop=True)

        dataset = dataset.rename({key: INV_VOCAB[key] for key in dataset.data_vars})

        if "batch" in dataset.dims:
            dataarray = (
                dataset[self._output_coords["variable"]]
                .to_dataarray()
                .T.transpose(
                    ..., "batch", "time", "lead_time", "variable", "lat", "lon"
                )
            )
        else:
            dataarray = (
                dataset[self._output_coords["variable"]]
                .to_dataarray()
                .T.transpose(..., "time", "lead_time", "variable", "lat", "lon")
            )

        out = torch.from_numpy(dataarray.to_numpy().copy())
        # Flip lat from ascending (-90->90, JAX native) to (90->-90)
        out = out.flip(-2)
        return out

    @staticmethod
    def get_jax_device_from_tensor(x: torch.Tensor) -> "jax.Device":
        """From a tensor, get device and corresponding JAX device.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        jax.Device
            Corresponding JAX device
        """
        device_id = x.get_device()
        if device_id == -1:  # -1 is CPU
            device = jax.devices("cpu")[0]
        else:
            device = jax.devices("gpu")[device_id]
        return device

    def from_dataarray_to_dataset(
        self, data: xr.DataArray, lead_time: int = 12, hour_steps: int = 12
    ) -> tuple[xr.Dataset, list[str]]:
        """Convert earth2studio DataArray to xarray Dataset for JAX model.

        Parameters
        ----------
        data : xr.DataArray
            Input data array from earth2studio
        lead_time : int, optional
            Forecast lead time in hours, by default 12
        hour_steps : int, optional
            Time step size in hours, by default 12

        Returns
        -------
        tuple[xr.Dataset, list[str]]
            xarray Dataset suitable for GenCast and list of target lead times
        """
        if len(data.time.values) > 1:
            raise TypeError("GenCast model only supports 1 init_time.")

        # Convert lead_time dim to absolute time
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

        # 2nd date is center (t=0)
        start_date = data.time.values[1]
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

        # Reassemble pressure levels
        pressure_level_vars: dict[str, list[xr.DataArray]] = {}
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

        # Set up time coordinates for data_utils.extract_inputs_targets_forcings
        out_data = out_data.assign_coords(
            datetime=all_datetimes[: len(out_data.time.values)]
        )
        out_data = out_data.assign_coords(time=time_deltas[: len(out_data.time.values)])
        out_data["datetime"] = out_data.datetime.expand_dims(dict(batch=1))

        # Add batch dimension
        for var in out_data.data_vars:
            if "batch" not in out_data[var].dims:
                out_data[var] = out_data[var].expand_dims(dict(batch=1))

        # Pad times for target template
        out_data = out_data.pad(pad_width=dict(time=(0, len(lead_times))))
        out_data = out_data.assign_coords(
            coords=dict(
                time=time_deltas,
                datetime=(("batch", "time"), [all_datetimes]),
            )
        )

        # Reindex lat ascending (south-to-north) for JAX model
        out_data = out_data.reindex(lat=sorted(out_data.lat.values))
        out_data = out_data.transpose("batch", "time", "level", "lat", "lon", ...)

        # Add zero tp12 (GenCast does not need precipitation in inputs)
        shape = out_data["2m_temperature"].shape
        dims = out_data["2m_temperature"].dims
        tp_coords = {dim: out_data["2m_temperature"].coords[dim] for dim in dims}
        out_data["total_precipitation_12hr"] = xr.DataArray(
            np.zeros(shape, dtype=np.float32), dims=dims, coords=tp_coords
        )

        # Add static fields
        out_data["land_sea_mask"] = xr.DataArray(
            self.land_sea_mask, dims=("lat", "lon")
        )
        out_data["geopotential_at_surface"] = xr.DataArray(
            self.geopotential_at_surface, dims=("lat", "lon")
        )

        # Apply SST NaN mask
        sst_mask_da = xr.DataArray(
            self.sst_nan_mask,
            dims=("lat", "lon"),
            coords={"lat": out_data.lat, "lon": out_data.lon},
        )
        out_data["sea_surface_temperature"] = out_data["sea_surface_temperature"].where(
            sst_mask_da
        )

        # Cast all to float32
        for var in out_data.data_vars:
            out_data[var] = out_data[var].astype(np.float32)

        return out_data, target_lead_times

    # -------------------------------------------------------------------------
    # Forward pass and iteration
    # -------------------------------------------------------------------------

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
            Output tensor and coordinate system 12 hours in the future
        """
        device = x.device

        with jax.default_device(self.get_jax_device_from_tensor(x)):
            x, coords = map_coords(x, coords, self.input_coords())

            # Validate spatial dimensions match expected grid
            target_input_coords = self.input_coords()
            handshake_coords(coords, target_input_coords, "lat")
            handshake_coords(coords, target_input_coords, "lon")

            time_dim = list(coords.keys()).index("time")
            n_times = len(coords["time"])
            results = []
            for t in range(n_times):
                x_t = x.narrow(time_dim, t, 1)
                coords_t = coords.copy()
                coords_t["time"] = coords["time"][t : t + 1]

                data, target_lead_times = self.from_dataarray_to_dataset(
                    xr.DataArray(x_t.cpu(), coords=coords_t), 12
                )

                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    data,
                    target_lead_times=target_lead_times,
                    **dataclasses.asdict(self.ckpt.task_config),
                )

                # Create a fresh PRNG key for each time step
                if self.seed is not None:
                    step_rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), t)
                else:
                    step_rng = jax.random.PRNGKey(np.random.randint(0, 2**31))
                # Silence print out from graphcast package for this model
                with contextlib.redirect_stdout(io.StringIO()):
                    predictions = rollout.chunked_prediction(
                        self.run_forward,
                        rng=step_rng,
                        inputs=inputs,
                        targets_template=targets * np.nan,
                        forcings=forcings,
                    )
                results.append(self.iterator_result_to_tensor(predictions))

            out = torch.cat(results, dim=1) if n_times > 1 else results[0]
            output_coords = self.output_coords(coords)

            out = out.to(device)

            return out, output_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem]]:
        """Default generator for time-stepping through iterator results.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system at each time step
        """
        coords = coords.copy()
        coords_out = self.output_coords(coords)

        device = x.device

        # First yield: return last input frame with zeros for tp12
        coords_out["lead_time"] = coords["lead_time"][1:]
        # x shape: (batch, time, lead_time=2, n_input_vars, lat, lon)
        # Take last lead_time frame for input vars, add zero tp12 column
        out_input = x[:, :, 1:, ...]  # (batch, time, 1, n_input_vars, lat, lon)

        # Find SST index in input variables to insert tp12 after it
        # Output order: t2m, msl, u10m, v10m, sst, tp12, then atmos
        # Input order:  t2m, msl, u10m, v10m, sst, then atmos
        # Insert a zero slice at index 5 (after sst) for tp12
        tp12_zeros = torch.zeros_like(out_input[:, :, :, :1, ...])
        out = torch.cat(
            (out_input[:, :, :, :5, ...], tp12_zeros, out_input[:, :, :, 5:, ...]),
            dim=3,
        )
        yield out, coords_out

        while True:
            coords = self.output_coords(coords)

            # Get next prediction from all time iterators
            results = [
                self.iterator_result_to_tensor(next(it)) for it in self._iterators
            ]
            x = torch.cat(results, dim=1) if len(results) > 1 else results[0]

            x, coords = self.rear_hook(x, coords)

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
            time_dim = list(coords.keys()).index("time")
            n_times = len(coords["time"])
            self._iterators: list[Generator[xr.Dataset, None, None]] = []

            for t in range(n_times):
                x_t = x.narrow(time_dim, t, 1)
                coords_t = coords.copy()
                coords_t["time"] = coords["time"][t : t + 1]

                batch, target_lead_times = self.from_dataarray_to_dataset(
                    xr.DataArray(x_t.cpu(), coords=coords_t), 12
                )
                init_datetime = batch.coords["datetime"].values[0, 1]
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    batch,
                    target_lead_times=target_lead_times,
                    **dataclasses.asdict(self.ckpt.task_config),
                )

                # Create a fresh PRNG key for each time iterator.
                # If seed is set, use fold_in for reproducible per-iterator keys.
                if self.seed is not None:
                    iter_rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), t)
                else:
                    iter_rng = jax.random.PRNGKey(np.random.randint(0, 2**31))
                self._iterators.append(
                    self._chunked_prediction_generator(
                        predictor_fn=self.run_forward,
                        rng=iter_rng,
                        inputs=inputs,
                        targets_template=targets * np.nan,
                        forcings=forcings,
                        init_datetime=init_datetime,
                    )
                )

            yield from self._default_generator(x, coords)
