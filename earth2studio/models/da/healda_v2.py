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

import re
from collections import OrderedDict
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import xarray as xr
from loguru import logger

from earth2studio.data import NNJAObsConv, NNJAObsSat, NNJAObsSatwnd
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, FrameSchema, TimeArray, TimeTolerance

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment]

try:
    from healda import inference as healda_inference
    from healda.observations.adapters import e2s_nnja
except ImportError:
    OptionalDependencyFailure("da-healda-v2")
    healda_inference = None
    e2s_nnja = None

# 0.25 degree output grid of the lat/lon recipe.
NLAT, NLON = 721, 1440

# HealDA names that differ from Earth2Studio; pressure channels ``U1000`` become ``u1000``.
_SURFACE_TO_E2S: dict[str, str] = {
    "tas": "t2m",
    "uas": "u10m",
    "vas": "v10m",
    "100u": "u100m",
    "100v": "v100m",
    "pres_msl": "msl",
}
_LEVEL_CHANNEL = re.compile(r"([UVTZQW])(\d+)")


def channel_to_e2s(name: str) -> str:
    """HealDA channel name -> Earth2Studio variable name."""
    if name in _SURFACE_TO_E2S:
        return _SURFACE_TO_E2S[name]
    match = _LEVEL_CHANNEL.fullmatch(name)
    if match:
        return match.group(1).lower() + match.group(2)
    return name


@check_optional_dependencies()
class HealDAv2(torch.nn.Module, AutoModelMixin):
    """HealDA v2 data assimilation model: a 0.25 degree global analysis from NNJA
    observations.

    HealDA v2 is a stateless, deterministic, observation-only assimilation model. One
    analysis at time ``t`` reads an eight-frame, six-hourly window ending at ``t`` and
    every observation within three hours of each frame, i.e. observations spanning
    ``[t - 45h, t + 3h)``, and returns the analysis of the last frame on the 721 x 1440
    equiangular grid.

    The model takes the NNJA observing system it was trained on as three DataFrames:
    ``conv_obs`` (PrepBUFR ``u``, ``v``, ``q``, ``t``, ``pres`` and GPS-RO ``gps``,
    ``gps_refractivity``), ``satwnd_obs`` (``u``, ``v``) and ``sat_obs`` (radiances of
    ``atms``, ``amsua``, ``amsub``, ``mhs``, ``airs``, ``iasi`` and ``cris``). Fetch
    them with the sources ``data_sources`` returns: the PrepBUFR event, level placement
    and IR channels must match training.

    Any stream may be omitted; the analysis then uses the others alone. Operational
    channel denials (``healda.inference.DENIALS``) apply by analysis time.

    Parameters
    ----------
    model : healda.inference.DAModel
        The trained network with its observation pipeline, as returned by
        ``healda.inference.load_da_model``

    Note
    ----
    For more information see the following references:

    - https://github.com/NVlabs/HealDA
    - https://huggingface.co/nvidia/healda-v2

    Badges
    ------
    region:global class:data-assimilation product:wind product:temp product:atmos
    product:insitu product:sat year:2026 gpu:80gb provider:nvidia backend:pytorch
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self._model = model
        # Registered so state_dict/parameters see the network; device moves go through
        # ``to`` below, which also updates the pipeline's device.
        self.core_model = model.net
        self._variables = np.array(
            [channel_to_e2s(name) for name in model.channels], dtype=str
        )
        self._output_lat = np.linspace(90, -90, NLAT)
        self._output_lon = np.linspace(0, 360, NLON, endpoint=False)

    @property
    def sensor_indices(self) -> dict[str, list[int]]:
        """IR sounder channels the model reads, for ``NNJAObsSat(sensor_indices=...)``."""
        return self._model.ir_channels

    def data_sources(
        self, time_tolerance: TimeTolerance, **kwargs: Any
    ) -> tuple[NNJAObsConv, NNJAObsSatwnd, NNJAObsSat]:
        """The ``(conv_obs, satwnd_obs, sat_obs)`` sources, configured as trained.

        PrepBUFR as the archive the model trained on holds it: each observation's
        original event, every level of a report at the report's time and position.
        Radiances only for the IR sounder channels the model reads; the full CrIS
        spectrum is about 600M rows per six-hour cycle.

        Parameters
        ----------
        time_tolerance : TimeTolerance
            Window around each request time, ``[t - 45h, t + 3h)`` for one analysis
        **kwargs : Any
            Passed to every source, e.g. ``cache`` or ``decode_workers``
        """
        return (
            NNJAObsConv(
                time_tolerance=time_tolerance,
                event="original",
                balloon_drift=False,
                **kwargs,
            ),
            NNJAObsSatwnd(time_tolerance=time_tolerance, **kwargs),
            NNJAObsSat(
                time_tolerance=time_tolerance,
                sensor_indices=self.sensor_indices,
                **kwargs,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self._model.device

    def to(self, device: Any) -> "HealDAv2":  # type: ignore[override]
        """Move the network and the observation pipeline to ``device``."""
        device = torch.device(device)
        self._model.net.to(device)
        self._model.device = device
        return self

    def init_coords(self) -> None:
        """Initialization coords (not required)"""
        return None

    def input_coords(self) -> tuple[FrameSchema, FrameSchema, FrameSchema]:
        """Input coordinate system specifying required DataFrame fields.

        Returns
        -------
        tuple[FrameSchema, FrameSchema, FrameSchema]
            (conv_schema, satwnd_schema, sat_schema) for the three observation
            DataFrames; any may be ``None`` when calling the model, but not all
        """
        sat_variables = [
            e2s_nnja.SAT_VARIABLE.get(sensor, sensor)
            for sensor in self._model.satellite_sensors
        ]
        return (
            _frame_schema(NNJAObsConv.SCHEMA, CONV_VARIABLES),
            _frame_schema(NNJAObsSatwnd.SCHEMA, ["u", "v"]),
            _frame_schema(NNJAObsSat.SCHEMA, sat_variables),
        )

    def output_coords(
        self,
        input_coords: tuple[FrameSchema, FrameSchema, FrameSchema],
        request_time: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[CoordSystem]:
        """Output coordinate system for the HealDA v2 analysis.

        Parameters
        ----------
        input_coords : tuple[FrameSchema, FrameSchema, FrameSchema]
            Input coordinate system
        request_time : np.ndarray | None, optional
            Analysis valid time(s), by default None

        Returns
        -------
        tuple[CoordSystem]
            Coordinate system with time, variable, lat and lon dimensions
        """
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")
        return (
            CoordSystem(
                OrderedDict(
                    {
                        "time": request_time,
                        "variable": self._variables,
                        "lat": self._output_lat,
                        "lon": self._output_lon,
                    }
                )
            ),
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default HealDA v2 model package from HuggingFace.

        Returns
        -------
        Package
            Model package pointing to the HuggingFace repository
        """
        return Package(
            "hf://nvidia/healda-v2",
            cache_options={"same_names": True},
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        device: str | torch.device = "cuda",
        checkpoint_name: str = "healda_v2.checkpoint",
        loop_name: str = "v2-nnja-latlon-final",
    ) -> AssimilationModel:
        """Load HealDA v2 from package.

        The network runs on a CUDA device only, as a single unsharded process; the
        0.25 degree recipe also fetches ERA5 static fields into the healda cache the
        first time it is built.

        Parameters
        ----------
        package : Package
            Package containing the ``.checkpoint`` archive written by ``healda-train``
        device : str | torch.device, optional
            Device to build the model on, by default "cuda"
        checkpoint_name : str, optional
            Name of the checkpoint file inside the package, by default
            "healda_v2.checkpoint"
        loop_name : str, optional
            healda training preset to fall back to when the checkpoint carries no
            ``loop.json``, by default "v2-nnja-latlon-final"

        Returns
        -------
        AssimilationModel
            Loaded HealDA v2 assimilation model
        """
        if torch.device(device).type != "cuda":
            raise RuntimeError("HealDA v2 requires a CUDA device")
        path = package.resolve(checkpoint_name)
        logger.info(f"Building HealDA v2 from {checkpoint_name}")
        model = healda_inference.load_da_model(path, device, loop_name=loop_name)
        return cls(model)

    def __call__(
        self,
        conv_obs: pd.DataFrame | None = None,
        satwnd_obs: pd.DataFrame | None = None,
        sat_obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Run HealDA v2 from NNJA observations.

        At least one DataFrame must be provided. Each must carry a ``request_time``
        entry in its ``.attrs`` (``earth2studio.data.fetch_dataframe`` sets it) and
        should cover ``[t - 45h, t + 3h)`` around each analysis time.

        Parameters
        ----------
        conv_obs : pd.DataFrame | None, optional
            PrepBUFR and GPS-RO observations from
            ``NNJAObsConv(event="original", balloon_drift=False)``, by default None
        satwnd_obs : pd.DataFrame | None, optional
            Satellite winds from ``NNJAObsSatwnd``, by default None
        sat_obs : pd.DataFrame | None, optional
            Satellite radiances from ``NNJAObsSat``, by default None

        Returns
        -------
        xr.DataArray
            Global analysis with dimensions [time, variable, lat, lon]. Data is on the
            same device as the model (cupy array for GPU, numpy for CPU).

        Raises
        ------
        ValueError
            If every DataFrame is ``None`` or none carries ``request_time``
        """
        if conv_obs is None and satwnd_obs is None and sat_obs is None:
            raise ValueError(
                "At least one of conv_obs, satwnd_obs or sat_obs must be provided."
            )
        request_time = self._request_time(conv_obs, satwnd_obs, sat_obs)
        frames = [_to_pandas(df) for df in (conv_obs, satwnd_obs, sat_obs)]
        (output_coords,) = self.output_coords(
            self.input_coords(), request_time=request_time
        )
        if all(df is None or df.empty for df in frames):
            logger.warning("No observations provided, returning empty analysis")
            return self._empty_output(output_coords)

        tables = e2s_nnja.analysis_tables(
            *frames,
            sensors=self._model.satellite_sensors,
            ir_channels=self._model.loop.obs_config.nnja_ir_channels,
        )
        analysis = self._model.run_analysis(pd.DatetimeIndex(request_time), **tables)
        return self.build_output(analysis, output_coords)

    def create_generator(
        self,
    ) -> Generator[
        xr.DataArray,
        tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None],
        None,
    ]:
        """Creates a generator which accepts observations and yields the analysis.

        Yields
        ------
        xr.DataArray
            Global analysis on the 0.25 degree grid

        Receives
        --------
        tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]
            A ``(conv_obs, satwnd_obs, sat_obs)`` tuple sent via ``generator.send()``.
            Any element may be ``None`` but not all.
        """
        inputs = yield None  # type: ignore[misc]
        try:
            while True:
                da = self.__call__(*(inputs or (None, None, None)))
                inputs = yield da
        except GeneratorExit:
            logger.debug("HealDA v2 generator clean up complete.")

    @staticmethod
    def _request_time(*frames: pd.DataFrame | None) -> TimeArray:
        request_time = None
        for df in frames:
            if df is not None and df.attrs.get("request_time", None) is not None:
                request_time = df.attrs["request_time"]
                break
        if request_time is None:
            raise ValueError(
                "Observation DataFrame must have 'request_time' in attrs. "
                "This is typically set by earth2studio.data.fetch_dataframe."
            )
        if isinstance(request_time, np.ndarray):
            return request_time.astype("datetime64[ns]")
        return np.array([np.datetime64(request_time, "ns")], dtype="datetime64[ns]")

    def build_output(
        self, analysis: torch.Tensor, output_coords: CoordSystem
    ) -> xr.DataArray:
        """Physical-space analyses ``[time, variable, lat, lon]`` -> DataArray."""
        out = analysis.contiguous()
        if out.device.type == "cuda" and cp is not None:
            data = cp.asarray(out)
        else:
            data = out.cpu().numpy()
        return xr.DataArray(
            data=data, dims=["time", "variable", "lat", "lon"], coords=output_coords
        )

    def _empty_output(self, output_coords: CoordSystem) -> xr.DataArray:
        shape = tuple(len(values) for values in output_coords.values())
        data = torch.full(shape, float("nan"), dtype=torch.float32, device=self.device)
        return self.build_output(data, output_coords)


CONV_VARIABLES = ["u", "v", "q", "t", "pres", "gps", "gps_refractivity"]


def _frame_schema(schema: pa.Schema, variables: list[str]) -> FrameSchema:
    empty = schema.empty_table().to_pandas()
    columns = {name: empty[name].to_numpy() for name in schema.names}
    columns["variable"] = np.array(variables, dtype=str)
    return FrameSchema(columns)


def _to_pandas(df: Any) -> pd.DataFrame | None:
    # The adapters are pandas-only; cudf frames move to the host.
    if cudf is not None and isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df
