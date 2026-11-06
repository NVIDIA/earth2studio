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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import xarray as xr
from loguru import logger

from earth2studio.data import NNJAObsConv, NNJAObsSat, NNJAObsSatwnd
from earth2studio.data.utils_ncep import (
    _NCEP_GPSRO_WIDE_SCHEMA,
    _NCEP_PREPBUFR_WIDE_SCHEMA,
    _NCEP_SATWND_WIDE_SCHEMA,
    GPSRO_VARIABLES,
    PREPBUFR_VARIABLES,
    SAT_VARIABLE_SENSOR,
    _sat_wide_schema,
)
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.utils import coord_array
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem, FrameSchema, TimeArray

if TYPE_CHECKING:
    from earth2studio.data.base import DataFrameSource

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
    """HealDA v2 maps 48 hours of observations to a global, 104-channel atmospheric
    analysis on a 0.25 degree latitude-longitude grid (721 x 1440).

    This stateless, deterministic model uses eight six-hourly frames ending at
    analysis time ``t``, with observations spanning ``[t - 45h, t + 3h]``.

    Pass wide DataFrames as ``model(prepbufr, gpsro, satwnd, *sensors)``, following
    the schemas and sensor order in ``input_coords()``. ``default_sources()``
    provides matching NNJA sources for use with ``earth2studio.data.fetch_dataframes``.
    Streams may be ``None`` or empty, and trailing streams may be omitted.
    HealDA filters channels in dated denial windows and satellite footprints
    flagged by provider quality columns.

    Recommended environment, set before Python starts:

    - ``MALLOC_ARENA_MAX=16`` keeps host memory level over repeated analyses in one
      process.
    - ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` avoids GPU out-of-memory
      from fragmentation over repeated analyses.
    - ``OMP_NUM_THREADS=1``.

    Parameters
    ----------
    model : healda.inference.DAModel
        Trained network and observation pipeline from ``healda.inference.load_da_model``

    Note
    ----
    For more information see the following references:

    - https://github.com/NVlabs/HealDA
    - https://huggingface.co/nvidia/healda-v2

    Badges
    ------
    region:global class:data-assimilation product:wind product:temp product:atmos
    product:insitu product:sat year:2026 gpu:40gb provider:nvidia backend:pytorch
    """

    stochastic = False

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """No-op: HealDA v2 inference is deterministic.

        Parameters
        ----------
        seed : int
            Unused seed
        reset : bool, optional
            Unused reset flag, by default True
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
        # Satellite stream -> NNJAObsSat variable, in the model's sensor order
        self._sat_variables = {
            SAT_VARIABLE_SENSOR[variable]: variable
            for variable in (
                e2s_nnja.SAT_VARIABLE.get(sensor, sensor)
                for sensor in model.satellite_sensors
            )
        }
        self._streams = ("prepbufr", "gpsro", "satwnd", *self._sat_variables)

    @property
    def sensor_indices(self) -> dict[str, Sequence[int]]:
        """IR sounder channels the model reads, for ``NNJAObsSat(sensor_indices=...)``."""
        return self._model.ir_channels

    @property
    def device(self) -> torch.device:
        return self._model.device

    def to(self, device: Any) -> "HealDAv2":  # type: ignore[override]
        """Move the network and the observation pipeline to ``device``."""
        self._model.to(device)
        return self

    def input_coords(self) -> tuple[FrameSchema, ...]:
        """Return the required observation schemas.

        Returns
        -------
        tuple[FrameSchema, ...]
            Wide schemas in ``(prepbufr, gpsro, satwnd, *sensors)`` order.
            Sensors follow the model's order. Each schema lists request variables
            under ``variable`` and DataFrame columns under the remaining keys.
        """
        sat = tuple(
            _frame_schema(
                _sat_wide_schema(name, self.sensor_indices.get(name)), [variable]
            )
            for name, variable in self._sat_variables.items()
        )
        return (
            _frame_schema(_NCEP_PREPBUFR_WIDE_SCHEMA, PREPBUFR_VARIABLES),
            _frame_schema(_NCEP_GPSRO_WIDE_SCHEMA, GPSRO_VARIABLES),
            _frame_schema(_NCEP_SATWND_WIDE_SCHEMA, ["u", "v"]),
            *sat,
        )

    def output_coords(
        self,
        *input_coords: FrameSchema,
        request_time: np.ndarray | None = None,
    ) -> CoordinateSystem:
        """Return analysis coordinates on the fixed 0.25 degree grid.

        Parameters
        ----------
        *input_coords : FrameSchema
            Input schemas; unused for the fixed output grid
        request_time : np.ndarray | None, optional
            Analysis times, or a NaT placeholder if None, by default None

        Returns
        -------
        CoordinateSystem
            Analysis coordinates with dimensions [time, variable, lat, lon]
        """
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")
        return coord_array(
            ("time", "variable", "lat", "lon"),
            OrderedDict(
                {
                    "time": request_time,
                    "variable": self._variables,
                    "lat": self._output_lat,
                    "lon": self._output_lon,
                }
            ),
        )

    def default_sources(self) -> tuple["DataFrameSource", ...]:
        """Return wide-layout NNJA sources for the 48-hour observation window.

        Returns
        -------
        tuple[DataFrameSource, ...]
            Sources in ``input_coords()`` order: ``NNJAObsConv`` for PrepBUFR and
            GPS-RO, ``NNJAObsSatwnd`` for winds, and ``NNJAObsSat`` for sensors.
            All use a ``(-45h, +3h)`` time tolerance around each analysis time.
        """
        tolerance = (np.timedelta64(-45, "h"), np.timedelta64(3, "h"))
        conv = NNJAObsConv(
            time_tolerance=tolerance,
            original_event=True,
            exclude_message_types=("SATWND",),
            layout="wide",
        )
        satwnd = NNJAObsSatwnd(time_tolerance=tolerance, layout="wide")
        sat = NNJAObsSat(
            time_tolerance=tolerance,
            sensor_indices=self.sensor_indices,
            layout="wide",
        )
        return (conv, conv, satwnd, *([sat] * len(self._sat_variables)))

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
        checkpoint_name: str = "healda_v2.checkpoint",
    ) -> AssimilationModel:
        """Load HealDA v2 from a model package.

        Loads on CPU; use ``.to("cuda")`` for single-GPU inference. ERA5 static
        fields are downloaded to the HealDA cache on first use.

        Parameters
        ----------
        package : Package
            Package containing a ``healda-train`` checkpoint
        checkpoint_name : str, optional
            Checkpoint filename, by default "healda_v2.checkpoint"

        Returns
        -------
        AssimilationModel
            Loaded HealDA v2 assimilation model
        """
        path = package.resolve(checkpoint_name)
        logger.info(f"Building HealDA v2 from {checkpoint_name}")
        model = healda_inference.load_da_model(path, "cpu")
        return cls(model)

    def __call__(self, *obs: pd.DataFrame | None) -> xr.DataArray:
        """Run HealDA v2 from NNJA observations.

        Frames should cover ``[t - 45h, t + 3h]`` and share analysis times in
        ``attrs["request_time"]``, as set by ``earth2studio.data.fetch_dataframe``.

        Parameters
        ----------
        *obs : pd.DataFrame | None
            Wide frames in ``input_coords()`` order. Streams may be None or empty;
            trailing streams may be omitted. At least one frame is required.

        Returns
        -------
        xr.DataArray
            Global analysis with dimensions [time, variable, lat, lon], backed by
            CuPy on GPU or NumPy on CPU

        Raises
        ------
        ValueError
            If frames are missing, excessive, or long-layout, or analysis times
            are missing or inconsistent
        """
        if len(obs) > len(self._streams):
            raise ValueError(
                f"HealDAv2 takes at most {len(self._streams)} frames, in the order "
                f"{list(self._streams)}; got {len(obs)}"
            )
        frames = {
            stream: _to_pandas(df)
            for stream, df in zip(self._streams, obs)
            if df is not None
        }
        if not frames:
            raise ValueError("At least one observation stream must be provided.")
        request_time = self._request_time(*frames.values())
        for stream, df in frames.items():
            if "observation" in df.columns:
                raise ValueError(
                    f"{stream} is a long frame; HealDAv2 takes wide NNJA frames, "
                    'build the source with layout="wide"'
                )
        output_coords = self.output_coords(
            *self.input_coords(), request_time=request_time
        )
        # An empty frame (e.g. a sensor outside its archive period) is an omitted one;
        # with every stream omitted the model runs on no observations
        frames = {stream: df for stream, df in frames.items() if not df.empty}
        if not frames:
            logger.warning("No observations provided, running the analysis without any")

        sat = {s: df for s, df in frames.items() if s in self._sat_variables}
        tables = e2s_nnja.analysis_tables(
            prepbufr=frames.get("prepbufr"),
            gpsro=frames.get("gpsro"),
            satwnd=frames.get("satwnd"),
            sat=sat or None,
            ir_channels=self._model.ir_channels,
        )
        analysis = self._model.run_analysis(pd.DatetimeIndex(request_time), **tables)
        return self.build_output(analysis, output_coords)

    @staticmethod
    def _request_time(*frames: pd.DataFrame | None) -> TimeArray:
        stamps = [
            np.atleast_1d(np.asarray(df.attrs["request_time"], dtype="datetime64[ns]"))
            for df in frames
            if df is not None and df.attrs.get("request_time", None) is not None
        ]
        if not stamps:
            raise ValueError(
                "Observation DataFrame must have 'request_time' in attrs. "
                "This is typically set by earth2studio.data.fetch_dataframe."
            )
        if any(not np.array_equal(stamp, stamps[0]) for stamp in stamps[1:]):
            raise ValueError(
                "Observation DataFrames carry different 'request_time' values; fetch "
                f"them for the same analysis time, got {[list(s) for s in stamps]}"
            )
        return stamps[0]

    def build_output(
        self, analysis: torch.Tensor, output_coords: CoordinateSystem
    ) -> xr.DataArray:
        """Physical-space analyses ``[time, variable, lat, lon]`` -> DataArray."""
        out = analysis.contiguous()
        if out.device.type == "cuda" and cp is not None:
            data = cp.asarray(out)
        else:
            data = out.cpu().numpy()
        return xr.DataArray(
            data=data,
            dims=["time", "variable", "lat", "lon"],
            coords=output_coords.coords,
        )


# Columns the adapter does not read: it derives the look angle from scan_position
# and reads no satwnd quality indicators
_UNREAD_COLUMNS = frozenset(
    {"scan_angle", "quality_indicators", "amv_quality_indicators"}
)


def _frame_schema(schema: pa.Schema, variables: Sequence[str]) -> FrameSchema:
    """Wide columns read for ``variables``."""
    selected = {f"observation_{v}" for v in variables} | {
        f"quality_{v}" for v in variables
    }
    names = [
        name
        for name in schema.names
        if name not in _UNREAD_COLUMNS
        and (name in selected or not name.startswith(("observation_", "quality_")))
    ]
    empty = schema.empty_table().to_pandas()
    columns = {name: empty[name].to_numpy() for name in names}
    columns["variable"] = np.array(variables, dtype=str)
    return FrameSchema(columns)


def _to_pandas(df: Any) -> pd.DataFrame:
    # The adapters are pandas-only; cudf frames move to the host.
    if cudf is not None and isinstance(df, cudf.DataFrame):
        return df.to_pandas()
    return df
