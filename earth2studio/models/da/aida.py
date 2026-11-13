# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
import xarray as xr

from earth2studio.data import (
    IFS_ENS,
    NNJAObsConv,
    NNJAObsSat,
    NNJAObsSatwnd,
)
from earth2studio.data.oisst import OISST
from earth2studio.data.utils_ncep import RAW_CONV_VARIABLES, channel_axis
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.px.weathermesh5c import (
    STEP,
    Inference,
    WeatherMesh5C,
    _backgrounds,
    _decode,
    _inputs,
    _iterator,
    _layout,
    _load_core,
    _State,
    _step,
    _times,
    _validate,
    _variables,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import FrameSchema

if TYPE_CHECKING:
    import cudf

try:
    from weathermesh5c.inference import IASI_CHANNELS
except ImportError:
    OptionalDependencyFailure("da-aida")
    IASI_CHANNELS = ()

ObservationInput: TypeAlias = Union[pd.DataFrame, "cudf.DataFrame", None]

SOURCES = (
    "1bmhs",
    "1bamua",
    "atms",
    "mtiasi",
    "gpsro",
    "adpupa",
    "windborne",
    "satwnd",
    "adpsfc",
)
SATELLITES = {
    "metop-b": 3,
    "metop-a": 4,
    "metop-c": 5,
    "n15": 206,
    "n16": 207,
    "n17": 208,
    "n18": 209,
    "n19": 223,
    "npp": 224,
    "n20": 225,
    "n21": 226,
}
_MICROWAVE = {"1bmhs": "mhs", "1bamua": "amsua", "atms": "atms"}
_CONVENTIONAL = {**RAW_CONV_VARIABLES, "windborne": ("t", "u", "v", "q", "pres")}


def _schema(source: str) -> FrameSchema:
    fields: tuple[str, ...] = (
        "lat",
        "lon_native" if source in RAW_CONV_VARIABLES else "lon",
    )
    variables: tuple[str, ...]
    if source in _MICROWAVE:
        variables = (_MICROWAVE[source],)
        fields += ("satellite", "satellite_za", "solza")
    elif source == "mtiasi":
        variables = ("iasi_radiance",)
        fields += (
            "satellite_za",
            "satellite_aza",
            "solza",
            "solaza",
            "scan_position",
            "scan_line",
            "footprint_quality",
        )
    elif source == "gpsro":
        variables = ("gps", "gps_refractivity")
        fields += (
            "elev",
            "report",
            "profile_level",
            "confidence",
            "ionosphere_corrected",
        )
    elif source == "satwnd":
        variables = ("u", "v")
        fields += ("pres", "satellite_id", "satellite_za")
    else:
        variables = _CONVENTIONAL[source]
        if source == "adpupa":
            fields += ("station",)
        elif source == "adpsfc":
            fields += ("station_elev", "subset", "class")
    return OrderedDict(
        time=np.array([], dtype="datetime64[ns]"),
        variable=np.asarray(variables),
        **{field: np.array([]) for field in fields},
        **{f"observation_{variable}": np.array([]) for variable in variables},
    )


def _reports(source: str, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    fields = _schema(source).keys()
    missing = fields - {"variable"} - set(frame)
    if missing:
        raise ValueError(f"{source} observations lack {sorted(missing)}")
    if source == "gpsro":
        return _profiles(frame)
    if source == "adpsfc":
        frame = frame[
            frame["subset"].between(0, 99) & frame["class"].isin(("ADPSFC", "SFCSHP"))
        ]
    result = {"time": frame["time"].to_numpy(dtype="datetime64[ns]")}
    for name in fields - {
        "time",
        "variable",
        "station",
        "satellite",
        "class",
    }:
        if not (source in _MICROWAVE or source == "mtiasi") or not name.startswith(
            "observation_"
        ):
            key = "lon" if name == "lon_native" else name.removeprefix("observation_")
            result[key] = frame[name].to_numpy(dtype=np.float64, na_value=np.nan)
    if source in _MICROWAVE or source == "mtiasi":
        sensor = _MICROWAVE.get(source, "iasi")
        channel = channel_axis(frame, sensor)
        column = "iasi_radiance" if source == "mtiasi" else sensor
        array = pa.array(frame[f"observation_{column}"])
        lengths = pc.list_value_length(array)
        if array.null_count or pc.any(pc.not_equal(lengths, len(channel))).as_py():
            raise ValueError(
                f"{sensor} list lengths do not match the declared channel axis"
            )
        values = (
            pc.list_flatten(array)
            .to_numpy(zero_copy_only=False)
            .reshape(len(frame), len(channel))
        )
        if source == "mtiasi":
            if not np.array_equal(channel, IASI_CHANNELS):
                raise ValueError("IASI requires the native selected radiance channels")
            result.update(
                radiance=values,
                quality=result["footprint_quality"],
                bearing_azimuth=result["satellite_aza"],
                field_of_view=result["scan_position"],
            )
        else:
            result.update((f"chan_{n}", values[:, i]) for i, n in enumerate(channel))
            result["satellite_id"] = (
                frame["satellite"].map(SATELLITES).to_numpy(dtype=np.float64)
            )
    elif source == "adpupa":
        # BUFRLIB read RPID's eight bytes in a double slot, then truncated to int.
        encoded = b"".join(
            str(s).ljust(8).encode("ascii")[:8] for s in frame["station"]
        )
        result["station_id"] = np.trunc(np.frombuffer(encoded, dtype="=f8"))
    elif source == "adpsfc":
        result["is_ship"] = frame["class"].eq("SFCSHP").to_numpy()
    return result


def _profiles(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    # The two streams use different physical heights. Pair by report and the
    # original replication level, not by height or by a float station code.
    keys = ["time", "report", "profile_level"]
    if "cycle_time" in frame:
        keys.insert(0, "cycle_time")
    bending = frame.loc[
        frame["observation_gps"].notna() & frame["ionosphere_corrected"],
        [*keys, "lat", "lon", "confidence", "observation_gps"],
    ]
    refractivity = frame.loc[
        frame["observation_gps_refractivity"].notna(),
        [*keys, "elev", "observation_gps_refractivity"],
    ]
    pairs = bending.merge(refractivity, on=keys, sort=False, validate="one_to_one")
    return {
        "time": pairs["time"].to_numpy(dtype="datetime64[ns]"),
        "profile_id": pairs.groupby(keys[:-1], sort=False, dropna=False)
        .ngroup()
        .to_numpy(),
        **{
            name.removeprefix("observation_"): pairs[name].to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            for name in pairs
            if name not in keys
        },
    }


def _pack(
    runtime: Inference, source: str, frame: ObservationInput
) -> list[tuple[datetime, np.ndarray]]:
    if frame is None or frame.empty or source not in runtime.observation_sources:
        return []
    if not isinstance(frame, pd.DataFrame):
        attrs = frame.attrs
        frame = frame.to_pandas(arrow_type=True)
        frame.attrs = attrs
    return runtime.pack_raw_observations(source, _reports(source, frame))


def _prepare_packed(
    runtime: Inference,
    source: str,
    hours: list[tuple[datetime, np.ndarray]],
    valid_time: datetime,
    cap: int | None,
) -> torch.Tensor:
    if not hours:
        return torch.empty((1, 0, 0), device=runtime.device)
    # Keep complete profiles/reports when selecting the native four-hour file window.
    return runtime.prepare_observations(
        source,
        [
            (hour, values)
            for hour, values in hours
            if valid_time - timedelta(hours=2) < hour <= valid_time + timedelta(hours=2)
        ],
        valid_time,
        max_observations=cap,
    )


@check_optional_dependencies()
class AIDA(torch.nn.Module, AutoModelMixin):
    """WindBorne's AI data assimilation model.

    AIDA updates an ECMWF IFS ensemble background and NOAA OISST sea surface
    temperature every three hours using satellite radiances and winds, GPS radio
    occultation, radiosondes, and surface reports. WindBorne soundings are optional.
    The input time is the analysis valid time: the background precedes it by nine
    hours (six for two updates). The result is one analysis at lead zero, with
    instantaneous surface and pressure-level fields on a global 0.25°
    latitude-longitude grid (720 × 1440).

    Parameters
    ----------
    core_model : torch.nn.Module
        WeatherMesh-5c network with AIDA.
    assimilation_steps : int, optional
        Two or three three-hour updates, by default 3
    max_observations : int | None, optional
        Per-source, per-update report cap, by default None
    variables : tuple[str, ...] | None, optional
        Instantaneous analysis fields, by default all supported instantaneous fields

    Note
    ----
    For forecasts and learned ensemble branching, use
    [`AIDAWeatherMesh5C`][earth2studio.models.da.AIDAWeatherMesh5C] with
    [`CoupledRunner`](https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/coupling/runner.py) and set `source_member`
    to choose the IFS background for each member. See the [WindBorne AI DA article](
    https://windbornesystems.com/blog/windborne-ai-da), the
    [model card](https://huggingface.co/windborne/weathermesh-5c), and the
    [runtime](https://github.com/windborne/weathermesh-5c).

    Badges
    ------
    region:global class:data-assimilation product:atmos provider:windborne
    backend:pytorch gpu:40gb
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        assimilation_steps: int = 3,
        max_observations: int | None = None,
        variables: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if assimilation_steps not in (2, 3):
            raise ValueError("AIDA requires two or three updates")
        if max_observations is not None and max_observations < 1:
            raise ValueError("max_observations must be positive")
        self._runtime = Inference(core_model)
        self.assimilation_steps = assimilation_steps
        self.max_observations = max_observations
        self.variables = _variables(self._runtime.instantaneous_variables, variables)
        self.stochastic = max_observations is not None
        self._seed, self._position = 0, 0
        self._seeded = False

    @property
    def device(self) -> torch.device:
        """Device holding the numerical runtime."""
        return self._runtime.device

    def input_coords(self) -> tuple[xr.DataArray | FrameSchema, ...]:
        """Declare IFS, SST and nine optional raw-observation slots.

        Returns
        -------
        tuple[CoordinateSystem | FrameSchema, ...]
            Background fields at -9h (or -6h), followed by per-sensor frames.
        """
        return (
            *_inputs(self._runtime, -3 * self.assimilation_steps),
            *(_schema(s) for s in SOURCES),
        )

    def output_coords(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        mhs: FrameSchema | None = None,
        amsua: FrameSchema | None = None,
        atms: FrameSchema | None = None,
        iasi: FrameSchema | None = None,
        gpsro: FrameSchema | None = None,
        radiosonde: FrameSchema | None = None,
        windborne: FrameSchema | None = None,
        satwnd: FrameSchema | None = None,
        surface: FrameSchema | None = None,
    ) -> xr.DataArray:
        """Plan instantaneous physical fields at analysis time.

        Parameters
        ----------
        background, sst : CoordinateSystem
            IFS and SST background signatures.
        mhs, amsua, atms, iasi, gpsro, radiosonde, windborne, satwnd, surface : FrameSchema | None, optional
            Optional raw observation schemas, by default None

        Returns
        -------
        CoordinateSystem
            Instantaneous analysis at lead zero for the declared input offsets.
        """
        inputs = _inputs(self._runtime, -3 * self.assimilation_steps)
        _validate(background, inputs[0], allow_shift=False)
        _validate(sst, inputs[1], allow_shift=False)
        return _layout(
            self._runtime,
            background,
            self.variables,
            background.lead_time.values
            + np.timedelta64(3 * self.assimilation_steps, "h"),
        )

    def default_sources(self) -> tuple[Any, ...]:
        """Recommend standard Earth2 inputs in declared slot order.

        Returns
        -------
        tuple
            IFS, prior-day OISST, standard satellite and conventional sources.
            The WindBorne slot is empty unless the caller supplies a source.
        """
        tolerance = (
            -np.timedelta64(3 * (self.assimilation_steps - 1) * 60 + 90, "m"),
            np.timedelta64(90, "m"),
        )
        satellite = NNJAObsSat(time_tolerance=tolerance, layout="wide")
        iasi = NNJAObsSat(
            time_tolerance=tolerance,
            sensor_indices={"iasi": IASI_CHANNELS},
            footprint_float64=True,
            layout="wide",
        )
        conventional = NNJAObsConv(
            time_tolerance=tolerance, layout="wide", gpsro_profile_metadata=True
        )
        return (
            IFS_ENS(member=1),
            OISST(lag_days=1, lookback_days=7),
            satellite,
            satellite,
            satellite,
            iasi,
            conventional,
            NNJAObsConv(
                source="convbufr.adpupa",
                time_tolerance=tolerance,
                layout="wide",
            ),
            None,
            NNJAObsSatwnd(time_tolerance=tolerance, layout="wide"),
            NNJAObsConv(
                source="convbufr.adpsfc",
                time_tolerance=tolerance,
                layout="wide",
            ),
        )

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed new analyses without changing an existing continuation.

        Parameters
        ----------
        seed : int
            Model-local sampling seed.
        reset : bool, optional
            Reset the initialization stream, by default True
        """
        if reset or not self._seeded:
            self._seed, self._position = int(seed), 0
            self._seeded = True

    @contextmanager
    def _sampling(self, seed: int) -> Generator[None]:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            yield

    def _latent(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        frames: tuple[ObservationInput, ...],
    ) -> tuple[torch.Tensor, int]:
        values = []
        packed = [
            _pack(self._runtime, source, frame)
            for source, frame in zip(SOURCES, frames, strict=True)
        ]
        seed = self._seed + self._position
        with self._sampling(seed):
            for prepared, time in zip(
                _backgrounds(self._runtime, background, sst),
                _times(background),
                strict=True,
            ):
                analysis = time + timedelta(hours=3 * self.assimilation_steps)
                cycles = {
                    hour: {
                        source: _prepare_packed(
                            self._runtime,
                            source,
                            hours,
                            analysis + timedelta(hours=hour),
                            self.max_observations,
                        )
                        for source, hours in zip(SOURCES, packed, strict=True)
                    }
                    for hour in range(-3 * (self.assimilation_steps - 1), 1, 3)
                }
                values.append(self._runtime.assimilate(prepared, analysis, cycles))
        self._position += 1
        return torch.cat(values), seed

    @torch.inference_mode()
    def __call__(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        mhs: ObservationInput = None,
        amsua: ObservationInput = None,
        atms: ObservationInput = None,
        iasi: ObservationInput = None,
        gpsro: ObservationInput = None,
        radiosonde: ObservationInput = None,
        windborne: ObservationInput = None,
        satwnd: ObservationInput = None,
        surface: ObservationInput = None,
    ) -> xr.DataArray:
        """Compute an analysis from background fields and observation frames.

        Parameters
        ----------
        background, sst : xr.DataArray
            Background fields at the declared negative lead time.
        mhs, amsua, atms, iasi, gpsro, radiosonde, windborne, satwnd, surface : pd.DataFrame | cudf.DataFrame | None, optional
            Wide raw observations as declared by ``input_coords()``, by default
            None. Satellite frames carry the source channel axis in their attrs;
            IASI uses raw radiances in the model's selected channel order.

        Returns
        -------
        xr.DataArray
            Instantaneous analysis fields at analysis time.
        """
        frames = (mhs, amsua, atms, iasi, gpsro, radiosonde, windborne, satwnd, surface)
        layout = self.output_coords(background, sst)
        latent, seed = self._latent(background, sst, frames)
        return _decode(self._runtime, latent, layout, self.variables, seed)[0]

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default WeatherMesh-5c/AIDA package."""
        return WeatherMesh5C.load_default_package()

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        variables: tuple[str, ...] | None = None,
        *,
        assimilation_steps: int = 3,
        max_observations: int | None = None,
    ) -> AIDA:
        """Load AIDA on CPU in evaluation mode.

        Parameters
        ----------
        package : Package
            Checkpoint and assets resolved through the standard package cache.
        variables : tuple[str, ...] | None, optional
            Output fields, by default all supported fields for this model
        assimilation_steps : int, optional
            Number of three-hour AIDA updates, by default 3
        max_observations : int | None, optional
            Maximum reports per source per update, by default None

        Returns
        -------
        AIDA
            Loaded analysis model in evaluation mode.
        """
        return cls(
            _load_core(package),
            assimilation_steps=assimilation_steps,
            max_observations=max_observations,
            variables=variables,
        ).eval()


@check_optional_dependencies()
class AIDAWeatherMesh5C(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Global WeatherMesh-5c forecasts initialized by AIDA.

    AIDA uses an ECMWF IFS ensemble background, NOAA OISST, and satellite and
    conventional observations to initialize the forecast. The input time is the
    analysis valid time, nine hours after the background by default. Forecasts
    begin at +3h and provide surface and pressure-level fields every three hours
    on a global 0.25° latitude-longitude grid (720 × 1440). At +72h by default, a
    learned perturbation lets members that share a background diverge.

    Parameters
    ----------
    core_model : torch.nn.Module
        WeatherMesh-5c network with AIDA.
    assimilation_steps : int, optional
        Initial AIDA updates, by default 3
    max_observations : int | None, optional
        Maximum reports per source per update, by default None
    fork_hour : int | None, optional
        Learned branch time; None disables it, by default 72
    variables : tuple[str, ...] | None, optional
        Forecast fields, by default all fields

    Note
    ----
    Use [`CoupledRunner`](https://github.com/NVIDIA/earth2studio/blob/1.0.0-rc/earth2studio/coupling/runner.py) with `source_member`
    to assign several output members to the same IFS background. Batch those
    members together: use `member_batch` at least as large as the number of fork
    branches per IFS member to reuse observation preparation. With
    `member_batch=1`, preparation repeats for every member. Each branch still
    repeats AIDA and the pre-fork forecast. Decoded fields are converted to physical units on the CPU
    at every output step. See the [WeatherMesh-5c
    release notes](https://windbornesystems.com/blog/weathermesh-5c-release-notes),
    the [model card](https://huggingface.co/windborne/weathermesh-5c), and the
    [runtime](https://github.com/windborne/weathermesh-5c).

    Badges
    ------
    region:global class:medium-range product:atmos provider:windborne backend:pytorch
    gpu:40gb
    """

    stochastic = True

    def __init__(
        self,
        core_model: torch.nn.Module,
        assimilation_steps: int = 3,
        max_observations: int | None = None,
        fork_hour: int | None = 72,
        variables: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.analysis = AIDA(core_model, assimilation_steps, max_observations)
        if fork_hour is not None and (fork_hour < 12 or fork_hour % 12):
            raise ValueError("fork_hour must be a positive multiple of twelve hours")
        self.fork_hour = fork_hour
        self.variables = _variables(self._runtime.variables, variables)

    @property
    def _runtime(self) -> Inference:
        return self.analysis._runtime

    @property
    def device(self) -> torch.device:
        """Device holding the shared numerical runtime."""
        return self._runtime.device

    def input_coords(self) -> tuple[xr.DataArray | FrameSchema, ...]:
        """Declare the analysis component's background and observation inputs."""
        return self.analysis.input_coords()

    def default_sources(self) -> tuple[Any, ...]:
        """Recommend the analysis component's sources in input-slot order."""
        return self.analysis.default_sources()

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed new trajectories without changing existing continuation states."""
        self.analysis.set_rng(seed, reset)

    def forcing_coords(self) -> None:
        """No continuing assimilation is performed after initialization."""
        return None

    def output_coords(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        mhs: FrameSchema | None = None,
        amsua: FrameSchema | None = None,
        atms: FrameSchema | None = None,
        iasi: FrameSchema | None = None,
        gpsro: FrameSchema | None = None,
        radiosonde: FrameSchema | None = None,
        windborne: FrameSchema | None = None,
        satwnd: FrameSchema | None = None,
        surface: FrameSchema | None = None,
    ) -> xr.DataArray:
        """Plan the first forecast three hours after analysis time.

        See Also
        --------
        AIDA.output_coords : Shared physical and observation input signatures.

        Returns
        -------
        xr.DataArray
            Complete forecast signature at analysis lead +3h.
        """
        layout = self.analysis.output_coords(
            background,
            sst,
            mhs,
            amsua,
            atms,
            iasi,
            gpsro,
            radiosonde,
            windborne,
            satwnd,
            surface,
        )
        return _layout(
            self._runtime, layout, self.variables, layout.lead_time.values + STEP
        )

    def __call__(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        mhs: ObservationInput = None,
        amsua: ObservationInput = None,
        atms: ObservationInput = None,
        iasi: ObservationInput = None,
        gpsro: ObservationInput = None,
        radiosonde: ObservationInput = None,
        windborne: ObservationInput = None,
        satwnd: ObservationInput = None,
        surface: ObservationInput = None,
    ) -> xr.DataArray:
        """Return the first forecast without retaining the state.

        See Also
        --------
        AIDA.__call__ : Shared physical and raw-observation inputs.

        Returns
        -------
        xr.DataArray
            Complete physical forecast at analysis lead +3h.
        """
        return self.initialize(
            background,
            sst,
            mhs,
            amsua,
            atms,
            iasi,
            gpsro,
            radiosonde,
            windborne,
            satwnd,
            surface,
        )[0]

    @torch.inference_mode()
    def initialize(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        mhs: ObservationInput = None,
        amsua: ObservationInput = None,
        atms: ObservationInput = None,
        iasi: ObservationInput = None,
        gpsro: ObservationInput = None,
        radiosonde: ObservationInput = None,
        windborne: ObservationInput = None,
        satwnd: ObservationInput = None,
        surface: ObservationInput = None,
    ) -> tuple[xr.DataArray, _State]:
        """Assimilate observations, then compute the first +3h forecast.

        See Also
        --------
        AIDA.__call__ : Shared physical and raw-observation inputs.

        Returns
        -------
        tuple[xr.DataArray, _State]
            Forecast and private latent/RNG continuation state.
        """
        frames = (mhs, amsua, atms, iasi, gpsro, radiosonde, windborne, satwnd, surface)
        layout = self.output_coords(background, sst)
        latent, seed = self.analysis._latent(background, sst, frames)
        return _decode(
            self._runtime, self._runtime.step(latent), layout, self.variables, seed
        )

    def step(self, y: xr.DataArray, state: _State) -> tuple[xr.DataArray, _State]:
        """Advance three hours, applying the seeded fork at its configured hour.

        Parameters
        ----------
        y : xr.DataArray
            Unchanged physical forecast returned with state.
        state : _State
            Complete private continuation state.

        Returns
        -------
        tuple[xr.DataArray, _State]
            Next forecast and state, leaving both arguments unchanged.
        """
        return _step(self._runtime, y, state, self.variables, self.fork_hour)

    def create_iterator(
        self,
        background: xr.DataArray,
        sst: xr.DataArray,
        mhs: ObservationInput = None,
        amsua: ObservationInput = None,
        atms: ObservationInput = None,
        iasi: ObservationInput = None,
        gpsro: ObservationInput = None,
        radiosonde: ObservationInput = None,
        windborne: ObservationInput = None,
        satwnd: ObservationInput = None,
        surface: ObservationInput = None,
    ) -> Generator[xr.DataArray]:
        """Yield three-hour forecasts, beginning after initialization.

        See Also
        --------
        AIDA.__call__ : Shared physical and raw-observation inputs.

        Yields
        ------
        xr.DataArray
            Complete physical forecasts; continuing assimilation is not enabled.
        """
        yield from _iterator(
            self,
            background,
            sst,
            mhs,
            amsua,
            atms,
            iasi,
            gpsro,
            radiosonde,
            windborne,
            satwnd,
            surface,
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default WeatherMesh-5c/AIDA package."""
        return WeatherMesh5C.load_default_package()

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        variables: tuple[str, ...] | None = None,
        *,
        assimilation_steps: int = 3,
        max_observations: int | None = None,
        fork_hour: int | None = 72,
    ) -> AIDAWeatherMesh5C:
        """Load the coupled model on CPU in evaluation mode.

        Parameters
        ----------
        package : Package
            Checkpoint and assets resolved through the standard package cache.
        variables : tuple[str, ...] | None, optional
            Forecast fields, by default all supported fields
        assimilation_steps : int, optional
            Number of three-hour AIDA updates, by default 3
        max_observations : int | None, optional
            Maximum reports per source per update, by default None
        fork_hour : int | None, optional
            Hour when branches diverge, by default 72; None disables branching

        Returns
        -------
        AIDAWeatherMesh5C
            Loaded forecast model.
        """
        return cls(
            _load_core(package),
            assimilation_steps=assimilation_steps,
            max_observations=max_observations,
            fork_hour=fork_hour,
            variables=variables,
        ).eval()
