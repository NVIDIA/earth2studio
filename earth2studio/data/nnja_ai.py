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

# NOAA-NASA Joint Archive (NNJA) of Observations, read through the
# ``nnja-ai`` package's pre-decoded Parquet catalog (gs://gcp-nnja-ai)
# instead of raw BUFR files.
#
# Reference: https://www.brightband.com/data/nnja-ai/


from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from earth2studio.data.utils import prep_data_inputs
from earth2studio.data.utils_ncep import (
    _NCEP_SATELLITE_NAME_BY_SAID,
    _nominal_microwave_scan_angle,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import TimeArray, TimeTolerance, VariableArray

NNJA_AI_DEPENDENCY_KEY = "nnja-ai"

try:
    from nnja_ai import DataCatalog
except ImportError:
    OptionalDependencyFailure("data", NNJA_AI_DEPENDENCY_KEY)
    DataCatalog = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Satellite (microwave) sensor catalog wiring
# ---------------------------------------------------------------------------

# Per-sensor nnja-ai catalog dataset key, brightness-temperature parquet
# column prefix, and (optional) surface-elevation descriptor column. Only the
# NNJA-AI-hosted microwave sensors are covered here; the archive does not
# currently publish AMSU-B (retired well before the catalog's ~2015+
# coverage) or the hyperspectral IR sounders (AIRS/IASI/CrIS) in a schema
# this class handles.
_SAT_DATASETS: dict[str, dict[str, str | None]] = {
    "amsua": {
        "dataset": "amsua-1bamua-NC021023",
        "tmbr_prefix": "BRITCSTC.TMBR_",
        "elev_field": "HOLS",
    },
    "mhs": {
        "dataset": "mhs-1bmhs-NC021027",
        "tmbr_prefix": "BRITCSTC.TMBR_",
        "elev_field": "HOLS",
    },
    "atms": {
        "dataset": "atms-atms-NC021203",
        "tmbr_prefix": "ATMSCH.TMBR_",
        "elev_field": None,
    },
}


@check_optional_dependencies(NNJA_AI_DEPENDENCY_KEY)
class NNJAAIObsSat:
    """NNJA satellite microwave observations, read via the ``nnja-ai`` package.

    This is a drop-in alternative to
    [`earth2studio.data.UFSObsSat`][earth2studio.data.UFSObsSat] and
    [`earth2studio.data.NNJAObsSat`][earth2studio.data.NNJAObsSat]: it produces
    a DataFrame with the same column schema, but sources the underlying data
    from the ``nnja-ai`` package's pre-decoded Parquet catalog
    (``gs://gcp-nnja-ai``) rather than fetching and decoding raw BUFR files.
    Because the archive is already columnar and pre-decoded, fetches are
    substantially cheaper than the BUFR path -- no BUFR message parsing is
    performed at all, at the cost of coverage: only the microwave sounders
    published in the catalog are supported (AMSU-A, ATMS, MHS). AMSU-B and
    the hyperspectral IR sounders (AIRS, IASI, CrIS) are not available.

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric +/- window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(10, 'm').
    satellites : list[str] | None, optional
        Satellite platforms to include. ``None`` includes every platform
        present in the fetched data, by default None.
    mirror : str, optional
        ``nnja-ai`` catalog mirror to read from, by default ``"gcp_nodd"``.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.brightband.com/data/nnja-ai/
    - https://psl.noaa.gov/data/nnja_obs/

    Badges
    ------
    region:global dataclass:observation product:atmos product:sat
    """

    SOURCE_ID = "earth2studio.data.NNJAAIObsSat"
    VALID_SATELLITES = frozenset(_NCEP_SATELLITE_NAME_BY_SAID.values())
    VALID_VARIABLES = frozenset(_SAT_DATASETS.keys())
    MIN_DATE = datetime(1998, 1, 1)
    SCHEMA_COLUMNS = [
        "time",
        "elev",
        "class",
        "lat",
        "lon",
        "scan_angle",
        "sensor_index",
        "satellite",
        "satellite_za",
        "solza",
        "observation",
        "variable",
    ]

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(10, "m"),
        satellites: list[str] | None = None,
        mirror: str = "gcp_nodd",
    ) -> None:
        if satellites is not None:
            invalid = set(satellites) - self.VALID_SATELLITES
            if invalid:
                raise ValueError(
                    f"Invalid satellite(s): {sorted(invalid)}. "
                    f"Valid options: {sorted(self.VALID_SATELLITES)}"
                )
        self._satellites = satellites
        self._mirror = mirror
        self._catalog: DataCatalog | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    @property
    def catalog(self) -> "DataCatalog":
        """Lazily construct the nnja-ai data catalog."""
        if self._catalog is None:
            self._catalog = DataCatalog(mirror=self._mirror)
        return self._catalog

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> pd.DataFrame:
        """Fetch satellite observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamp(s) to fetch observations around.
        variable : str | list[str] | VariableArray
            Sensor name(s) to fetch. One of ``"amsua"``, ``"atms"``, ``"mhs"``.

        Returns
        -------
        pd.DataFrame
            Long-format observation DataFrame.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)

        sensors = []
        for v in variable_list:
            if v not in self.VALID_VARIABLES:
                raise ValueError(
                    f"Unsupported NNJAAIObsSat variable '{v}'. Supported "
                    f"sensors: {sorted(self.VALID_VARIABLES)}"
                )
            sensors.append(v)

        tmin = min(t for t in time_list) + self._tolerance_lower
        tmax = max(t for t in time_list) + self._tolerance_upper

        frames = [self._fetch_sensor(sensor, tmin, tmax) for sensor in sensors]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _fetch_sensor(
        self, sensor: str, tmin: datetime, tmax: datetime
    ) -> pd.DataFrame:
        """Fetch and reshape one sensor's aggregate observations to long format."""
        info = _SAT_DATASETS[sensor]
        ds = self.catalog[info["dataset"]]  # type: ignore[index]
        sub = ds.sel(time=slice(tmin, tmax))

        tmbr_prefix = info["tmbr_prefix"]
        elev_field = info["elev_field"]
        tmbr_cols = sorted(
            v.id for v in sub.variables.values() if v.id.startswith(tmbr_prefix)  # type: ignore[union-attr]
        )
        descriptor_cols = [
            "OBS_TIMESTAMP",
            "LAT",
            "LON",
            "SAID",
            "FOVN",
            "SAZA",
            "SOZA",
        ]
        if elev_field is not None:
            descriptor_cols.append(elev_field)
        cols = [c for c in descriptor_cols if c in sub.variables] + tmbr_cols
        wide = sub[cols].load_dataset(backend="pandas")

        if wide.empty:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        obs_time = pd.to_datetime(wide["OBS_TIMESTAMP"], utc=True).dt.tz_localize(None)
        in_window = (obs_time >= tmin) & (obs_time <= tmax)
        wide = wide.loc[in_window]
        obs_time = obs_time.loc[in_window]
        if wide.empty:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        satellite = wide["SAID"].astype("Int64").map(_NCEP_SATELLITE_NAME_BY_SAID)
        if self._satellites is not None:
            keep = satellite.isin(self._satellites)
            wide = wide.loc[keep]
            obs_time = obs_time.loc[keep]
            satellite = satellite.loc[keep]
            if wide.empty:
                return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        scan_position = wide["FOVN"].to_numpy(dtype=np.float64)
        scan_angle = np.array(
            [
                (
                    _nominal_microwave_scan_angle(sensor, int(pos))
                    if np.isfinite(pos)
                    else np.nan
                )
                for pos in scan_position
            ]
        )
        elev = (
            wide[elev_field].to_numpy(dtype=np.float32)
            if elev_field is not None and elev_field in wide.columns
            else np.full(len(wide), np.nan, dtype=np.float32)
        )

        rows = []
        for col in tmbr_cols:
            channel = int(col.rsplit("_", 1)[-1])
            observation = wide[col].to_numpy(dtype=np.float32)
            valid = np.isfinite(observation)
            if not valid.any():
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "time": obs_time.to_numpy(dtype="datetime64[ns]")[valid],
                        "elev": elev[valid],
                        "class": "rad",
                        "lat": wide["LAT"].to_numpy(dtype=np.float32)[valid],
                        "lon": (wide["LON"].to_numpy(dtype=np.float32)[valid] % 360.0),
                        "scan_angle": scan_angle[valid].astype(np.float32),
                        "sensor_index": np.uint16(channel),
                        "satellite": satellite.to_numpy()[valid],
                        "satellite_za": wide["SAZA"].to_numpy(dtype=np.float32)[valid],
                        "solza": wide["SOZA"].to_numpy(dtype=np.float32)[valid],
                        "observation": observation[valid],
                        "variable": sensor,
                    }
                )
            )
        if not rows:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        for t in times:
            if t < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than {cls.__name__}.MIN_DATE "
                    f"({cls.MIN_DATE.isoformat()})."
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True


# ---------------------------------------------------------------------------
# Conventional (in-situ) observation catalog wiring
# ---------------------------------------------------------------------------

_ADPUPA_DATASET = "conv-adpupa-NC002001"
_ADPSFC_DATASET = "conv-adpsfc-NC000001"

_ADPUPA_VARIABLES = frozenset(["t", "u", "v", "q"])
_ADPSFC_VARIABLES = frozenset(["pres"])
_UNSUPPORTED_CONV_VARIABLES = frozenset(["gps", "gps_t", "gps_q"])


def _specific_humidity_from_dewpoint(
    dewpoint_k: np.ndarray, pressure_pa: np.ndarray
) -> np.ndarray:
    """Derive specific humidity (kg/kg) from dewpoint temperature and pressure.

    Uses Bolton's (1980) approximation for saturation vapor pressure evaluated
    at the dewpoint (i.e. actual vapor pressure), since NNJA-AI's raw adpupa
    stream carries dewpoint rather than a moisture variable directly.
    """
    dewpoint_c = dewpoint_k - 273.15
    pressure_hpa = pressure_pa / 100.0
    vapor_pressure_hpa = 6.112 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))
    return (
        0.622 * vapor_pressure_hpa / (pressure_hpa - 0.378 * vapor_pressure_hpa)
    ).astype(np.float32)


@check_optional_dependencies(NNJA_AI_DEPENDENCY_KEY)
class NNJAAIObsConv:
    """NNJA conventional (in-situ) observations, read via the ``nnja-ai`` package.

    This is a partial, drop-in-compatible alternative to
    [`earth2studio.data.UFSObsConv`][earth2studio.data.UFSObsConv] and
    [`earth2studio.data.NNJAObsConv`][earth2studio.data.NNJAObsConv]: the
    output DataFrame uses the same column schema, but the underlying data
    comes from ``nnja-ai``'s pre-decoded Parquet catalog rather than fetching
    and decoding raw BUFR/PrepBUFR files.

    Coverage is narrower than the PrepBUFR-based sources:

    - ``t``, ``u``, ``v`` come from the raw ADPUPA (rawinsonde) dump stream,
      pivoted from its per-level wide columns into long rows. ``q`` (specific
      humidity) is not published directly; it is derived from dewpoint and
      pressure via Bolton's formula, so it will disagree slightly with a
      PrepBUFR-QC'd ``q`` observation.
    - ``pres`` comes from the raw ADPSFC (surface synoptic) dump stream's
      station-pressure field.
    - ``gps``/``gps_t``/``gps_q`` (GPS radio-occultation) are **not**
      available -- the catalog does not currently publish a GPSRO dataset.

    Unlike the merged PrepBUFR product, these are raw per-family dump
    streams: PrepBUFR report-type codes and quality marks are not present, so
    the output ``type`` column is always null.

    Parameters
    ----------
    time_tolerance : TimeTolerance, optional
        Time tolerance window for filtering observations. Accepts a single
        value (symmetric +/- window) or a tuple (lower, upper) for asymmetric
        windows, by default np.timedelta64(0, 'm').
    mirror : str, optional
        ``nnja-ai`` catalog mirror to read from, by default ``"gcp_nodd"``.

    Warning
    -------
    This is a remote data source and can potentially download a large amount of data
    to your local machine for large requests.

    Note
    ----
    Additional information on the data repository can be referenced here:

    - https://www.brightband.com/data/nnja-ai/
    - https://psl.noaa.gov/data/nnja_obs/

    Badges
    ------
    region:global dataclass:observation product:wind product:temp product:atmos product:insitu
    """

    SOURCE_ID = "earth2studio.data.NNJAAIObsConv"
    MIN_DATE = datetime(1979, 1, 1)
    VALID_VARIABLES = _ADPUPA_VARIABLES | _ADPSFC_VARIABLES
    SCHEMA_COLUMNS = [
        "time",
        "pres",
        "elev",
        "type",
        "class",
        "lat",
        "lon",
        "station",
        "station_elev",
        "observation",
        "variable",
    ]

    def __init__(
        self,
        time_tolerance: TimeTolerance = np.timedelta64(0, "m"),
        mirror: str = "gcp_nodd",
    ) -> None:
        self._mirror = mirror
        self._catalog: DataCatalog | None = None

        lower, upper = normalize_time_tolerance(time_tolerance)
        self._tolerance_lower = pd.to_timedelta(lower).to_pytimedelta()
        self._tolerance_upper = pd.to_timedelta(upper).to_pytimedelta()

    @property
    def catalog(self) -> "DataCatalog":
        """Lazily construct the nnja-ai data catalog."""
        if self._catalog is None:
            self._catalog = DataCatalog(mirror=self._mirror)
        return self._catalog

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> pd.DataFrame:
        """Fetch conventional observations for a set of timestamps.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamp(s) to fetch observations around.
        variable : str | list[str] | VariableArray
            Variable name(s) to fetch. Supported: ``"t"``, ``"u"``, ``"v"``,
            ``"q"``, ``"pres"``.

        Returns
        -------
        pd.DataFrame
            Long-format observation DataFrame.
        """
        time_list, variable_list = prep_data_inputs(time, variable)
        self._validate_time(time_list)

        unsupported = set(variable_list) & _UNSUPPORTED_CONV_VARIABLES
        if unsupported:
            warnings.warn(
                f"NNJAAIObsConv does not support GPS-RO variable(s) "
                f"{sorted(unsupported)}: the nnja-ai catalog does not "
                f"currently publish a GPSRO dataset. Skipping.",
                stacklevel=2,
            )
        unknown = (
            set(variable_list) - self.VALID_VARIABLES - _UNSUPPORTED_CONV_VARIABLES
        )
        if unknown:
            raise ValueError(
                f"Unsupported NNJAAIObsConv variable(s) {sorted(unknown)}. "
                f"Supported: {sorted(self.VALID_VARIABLES)}"
            )

        tmin = min(t for t in time_list) + self._tolerance_lower
        tmax = max(t for t in time_list) + self._tolerance_upper

        frames = []
        upper_air_vars = sorted(_ADPUPA_VARIABLES & set(variable_list))
        if upper_air_vars:
            frames.append(self._fetch_adpupa(upper_air_vars, tmin, tmax))
        if "pres" in variable_list:
            frames.append(self._fetch_adpsfc(tmin, tmax))

        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _station_id(self, wide: pd.DataFrame) -> pd.Series:
        wmob = wide["WMOB"].to_numpy(dtype=np.float64)
        wmos = wide["WMOS"].to_numpy(dtype=np.float64)
        station = np.array(
            [
                f"{int(b):02d}{int(s):03d}" if np.isfinite(b) and np.isfinite(s) else ""
                for b, s in zip(wmob, wmos)
            ]
        )
        return pd.Series(station, index=wide.index)

    def _fetch_adpupa(
        self, variables: list[str], tmin: datetime, tmax: datetime
    ) -> pd.DataFrame:
        """Fetch and pivot the ADPUPA rawinsonde profile stream to long format."""
        ds = self.catalog[_ADPUPA_DATASET]
        sub = ds.sel(time=slice(tmin, tmax))
        levels: list[int] = sub.dimensions["pressure"]["values"]

        need_wind = "u" in variables or "v" in variables
        need_temp = "t" in variables
        need_q = "q" in variables

        descriptor_cols = ["OBS_TIMESTAMP", "LAT", "LON", "WMOB", "WMOS"]
        level_cols: list[str] = []
        for lev in levels:
            if need_temp or need_q:
                level_cols.append(f"TMDB_PRLC{lev}")
            if need_q:
                level_cols.append(f"TMDP_PRLC{lev}")
            if need_wind:
                level_cols.append(f"WSPD_PRLC{lev}")
                level_cols.append(f"WDIR_PRLC{lev}")
        cols = descriptor_cols + level_cols
        wide = sub[cols].load_dataset(backend="pandas")
        if wide.empty:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        obs_time = pd.to_datetime(wide["OBS_TIMESTAMP"], utc=True).dt.tz_localize(None)
        in_window = (obs_time >= tmin) & (obs_time <= tmax)
        wide = wide.loc[in_window]
        obs_time = obs_time.loc[in_window]
        if wide.empty:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        station = self._station_id(wide)
        lat = wide["LAT"].to_numpy(dtype=np.float32)
        lon = wide["LON"].to_numpy(dtype=np.float32) % 360.0
        obs_time_ns = obs_time.to_numpy(dtype="datetime64[ns]")

        rows = []
        for lev in levels:
            pres_pa = np.float32(lev)
            tmdb = (
                wide[f"TMDB_PRLC{lev}"].to_numpy(dtype=np.float32)
                if f"TMDB_PRLC{lev}" in wide.columns
                else None
            )
            tmdp = (
                wide[f"TMDP_PRLC{lev}"].to_numpy(dtype=np.float32)
                if f"TMDP_PRLC{lev}" in wide.columns
                else None
            )
            wspd = (
                wide[f"WSPD_PRLC{lev}"].to_numpy(dtype=np.float32)
                if f"WSPD_PRLC{lev}" in wide.columns
                else None
            )
            wdir = (
                wide[f"WDIR_PRLC{lev}"].to_numpy(dtype=np.float32)
                if f"WDIR_PRLC{lev}" in wide.columns
                else None
            )

            if "t" in variables and tmdb is not None:
                valid = np.isfinite(tmdb)
                if valid.any():
                    rows.append(
                        self._level_frame(
                            lat,
                            lon,
                            obs_time_ns,
                            station,
                            pres_pa,
                            tmdb[valid],
                            "t",
                            valid,
                        )
                    )
            if "q" in variables and tmdb is not None and tmdp is not None:
                valid = np.isfinite(tmdb) & np.isfinite(tmdp)
                if valid.any():
                    q = _specific_humidity_from_dewpoint(
                        tmdp[valid].astype(np.float64),
                        np.full(valid.sum(), lev, dtype=np.float64),
                    )
                    rows.append(
                        self._level_frame(
                            lat, lon, obs_time_ns, station, pres_pa, q, "q", valid
                        )
                    )
            if need_wind and wspd is not None and wdir is not None:
                valid = np.isfinite(wspd) & np.isfinite(wdir)
                if valid.any():
                    wdir_rad = np.deg2rad(wdir[valid])
                    u = -wspd[valid] * np.sin(wdir_rad)
                    v = -wspd[valid] * np.cos(wdir_rad)
                    if "u" in variables:
                        rows.append(
                            self._level_frame(
                                lat, lon, obs_time_ns, station, pres_pa, u, "u", valid
                            )
                        )
                    if "v" in variables:
                        rows.append(
                            self._level_frame(
                                lat, lon, obs_time_ns, station, pres_pa, v, "v", valid
                            )
                        )
        if not rows:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)
        return pd.concat(rows, ignore_index=True)

    @staticmethod
    def _level_frame(
        lat: np.ndarray,
        lon: np.ndarray,
        obs_time_ns: np.ndarray,
        station: pd.Series,
        pres_pa: np.float32,
        observation: np.ndarray,
        variable: str,
        valid: np.ndarray,
    ) -> pd.DataFrame:
        n = int(valid.sum())
        return pd.DataFrame(
            {
                "time": obs_time_ns[valid],
                "pres": np.full(n, pres_pa, dtype=np.float32),
                "elev": np.full(n, np.nan, dtype=np.float32),
                "type": pd.array([None] * n, dtype="UInt16"),
                "class": None,
                "lat": lat[valid],
                "lon": lon[valid],
                "station": station.to_numpy()[valid],
                "station_elev": np.full(n, np.nan, dtype=np.float32),
                "observation": np.asarray(observation, dtype=np.float32),
                "variable": variable,
            }
        )

    def _fetch_adpsfc(self, tmin: datetime, tmax: datetime) -> pd.DataFrame:
        """Fetch surface station pressure from the ADPSFC dump stream."""
        ds = self.catalog[_ADPSFC_DATASET]
        sub = ds.sel(time=slice(tmin, tmax))
        cols = ["OBS_TIMESTAMP", "LAT", "LON", "SELV", "WMOB", "WMOS", "PRSSQ1.PRES"]
        wide = sub[cols].load_dataset(backend="pandas")
        if wide.empty:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        obs_time = pd.to_datetime(wide["OBS_TIMESTAMP"], utc=True).dt.tz_localize(None)
        in_window = (obs_time >= tmin) & (obs_time <= tmax)
        wide = wide.loc[in_window]
        obs_time = obs_time.loc[in_window]

        pres = wide["PRSSQ1.PRES"].to_numpy(dtype=np.float32)
        valid = np.isfinite(pres)
        wide = wide.loc[valid]
        obs_time = obs_time.loc[valid]
        pres = pres[valid]
        if wide.empty:
            return pd.DataFrame(columns=self.SCHEMA_COLUMNS)

        station = self._station_id(wide)
        n = len(wide)
        return pd.DataFrame(
            {
                "time": obs_time.to_numpy(dtype="datetime64[ns]"),
                "pres": pres,
                "elev": wide["SELV"].to_numpy(dtype=np.float32),
                "type": pd.array([None] * n, dtype="UInt16"),
                "class": None,
                "lat": wide["LAT"].to_numpy(dtype=np.float32),
                "lon": wide["LON"].to_numpy(dtype=np.float32) % 360.0,
                "station": station.to_numpy(),
                "station_elev": wide["SELV"].to_numpy(dtype=np.float32),
                "observation": pres,
                "variable": "pres",
            }
        )

    @classmethod
    def _validate_time(cls, times: list[datetime]) -> None:
        for t in times:
            if t < cls.MIN_DATE:
                raise ValueError(
                    f"Requested datetime {t} is earlier than {cls.__name__}.MIN_DATE "
                    f"({cls.MIN_DATE.isoformat()})."
                )

    @classmethod
    def available(cls, time: datetime | np.datetime64) -> bool:
        """Check if given date time is available.

        Parameters
        ----------
        time : datetime | np.datetime64
            Date time to check

        Returns
        -------
        bool
            If date time is available
        """
        if isinstance(time, np.datetime64):
            time = time.astype("datetime64[ns]").astype("datetime64[us]").item()
        try:
            cls._validate_time([time])
        except ValueError:
            return False
        return True
