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

import datetime as dt
import math
from collections import OrderedDict
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time import normalize_time_tolerance
from earth2studio.utils.type import CoordSystem, FrameSchema, TimeTolerance

try:
    import cupy as cp
except ImportError:
    cp = None  # type: ignore[assignment]

try:
    import cudf
except ImportError:
    cudf = None  # type: ignore[assignment, misc]

try:
    import earth2grid
    from physicsnemo.experimental.models.healda import HealDA as _HealDAModel
except ImportError:
    OptionalDependencyFailure("da-healda")
    earth2grid = None
    _HealDAModel = None


ERA5_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
ERA5_VARIABLES_3D = ["U", "V", "T", "Z", "Q"]
ERA5_VARIABLES_2D = [
    "tcwv",
    "tas",
    "uas",
    "vas",
    "100u",
    "100v",
    "pres_msl",
    "sst",
    "sic",
]
ERA5_CHANNELS = [
    *[f"{var}{lev}" for var in ERA5_VARIABLES_3D for lev in ERA5_LEVELS],
    *ERA5_VARIABLES_2D,
]


SAT_SENSORS = ("atms", "mhs", "amsua", "amsub")
ALL_SENSORS = (*SAT_SENSORS, "conv")
# E2studio conventional variable → 0-based local_channel_id.
# Channel ordering matches CONV_CHANNELS:
#   0: gps_angle, 1: gps_t, 2: gps_q, 3: ps, 4: q, 5: t, 6: u, 7: v
CONV_VAR_CHANNEL: dict[str, int] = {
    "gps": 0,
    "gps_t": 1,
    "gps_q": 2,
    "pres": 3,
    "q": 4,
    "t": 5,
    "u": 6,
    "v": 7,
}

# Conventional QC limits
_QC_HEIGHT_MIN = 0.0
_QC_HEIGHT_MAX = 60000.0
_QC_PRESSURE_MIN_GPS = 0.5
_QC_PRESSURE_MIN_DEFAULT = 200.0
_QC_PRESSURE_MAX = 1100.0

# Per-conv-channel valid ranges (name, min_valid, max_valid)
_CONV_CHANNEL_RANGES = [
    ("gps_angle", float("-inf"), float("inf")),
    ("gps_t", 150.0, 350.0),
    ("gps_q", 0.0, 1.0),
    ("ps", float("-inf"), float("inf")),
    ("q", 0.0, 1.0),
    ("t", 150.0, 350.0),
    ("u", -100.0, 100.0),
    ("v", -100.0, 100.0),
]


def _fourier_features(x_norm: torch.Tensor, num_freqs: int) -> torch.Tensor:
    device = x_norm.device
    freqs = torch.arange(1, num_freqs + 1, device=device, dtype=x_norm.dtype) * (
        2 * math.pi
    )
    x_expanded = x_norm.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)


def _compute_unified_metadata(
    target_time_sec: torch.Tensor,
    lon: torch.Tensor,
    time: torch.Tensor,
    height: torch.Tensor,
    pressure: torch.Tensor,
    scan_angle: torch.Tensor,
    sat_zenith_angle: torch.Tensor,
    sol_zenith_angle: torch.Tensor,
) -> torch.Tensor:
    """Compute 28-dim observation metadata features.

    Parameters
    ----------
    target_time_sec : torch.Tensor
        Target time as epoch seconds (int64), broadcast to all obs
    lon : torch.Tensor
        Longitude in degrees [n_obs]
    time : torch.Tensor
        Observation times as epoch nanoseconds (int64) [n_obs]
    height : torch.Tensor
        Height in meters [n_obs], NaN for satellite obs
    pressure : torch.Tensor
        Pressure in hPa [n_obs], NaN for satellite obs
    scan_angle : torch.Tensor
        Scan angle in degrees [n_obs], NaN for conventional obs
    sat_zenith_angle : torch.Tensor
        Satellite zenith angle in degrees [n_obs], NaN for conventional obs
    sol_zenith_angle : torch.Tensor
        Solar zenith angle in degrees [n_obs], NaN for conventional obs

    Returns
    -------
    torch.Tensor
        Metadata features [n_obs, 28]
    """
    features: list[torch.Tensor] = []

    # Local solar time (4 features)
    sod = (time // 1_000_000_000) % 86400
    utc_hours = sod.float() / 3600.0
    lst = (utc_hours + lon / 15.0) % 24.0
    features.append(_fourier_features(lst / 24.0, 2))

    # Relative time (2 features)
    target_ns = target_time_sec * 1_000_000_000
    dt_sec = (time - target_ns).float() * 1e-9
    dt_norm = dt_sec / 86400.0
    features.append(torch.stack([dt_norm, dt_norm**2], dim=-1))

    # Height (8 features)
    h_valid = ~torch.isnan(height)
    h_norm = torch.clamp(height / 60000.0, 0.0, 1.0)
    h_feat = _fourier_features(h_norm, 4)
    h_feat[~h_valid] = 0.0
    features.append(h_feat)

    # Pressure (8 features)
    p_valid = ~torch.isnan(pressure)
    p_norm = torch.clamp(pressure / 1100.0, 0.0, 1.0)
    p_feat = _fourier_features(p_norm, 4)
    p_feat[~p_valid] = 0.0
    features.append(p_feat)

    # Scan angle (2 features)
    s_valid = ~torch.isnan(scan_angle)
    xi = scan_angle / 50.0
    s_feat = torch.stack([xi, xi**2], dim=-1)
    s_feat[~s_valid] = 0.0
    features.append(s_feat)

    # Satellite zenith (2 features)
    sat_valid = ~torch.isnan(sat_zenith_angle)
    cos_sat = torch.cos(torch.deg2rad(sat_zenith_angle))
    sat_feat = torch.stack([cos_sat, cos_sat**2], dim=-1)
    sat_feat[~sat_valid] = 0.0
    features.append(sat_feat)

    # Solar zenith (2 features)
    sol_valid = ~torch.isnan(sol_zenith_angle)
    cos_sol = torch.cos(torch.deg2rad(sol_zenith_angle))
    sin_sol = torch.sin(torch.deg2rad(sol_zenith_angle))
    sol_feat = torch.stack([cos_sol, sin_sol], dim=-1)
    sol_feat[~sol_valid] = 0.0
    features.append(sol_feat)

    return torch.cat(features, dim=-1)  # [n_obs, 28]


def _load_sensor_stats(
    stats_dir: str, package: Package
) -> dict[str, dict[str, np.ndarray]]:
    """Load per-sensor normalization stats from package CSV files.

    Parameters
    ----------
    stats_dir : str
        Directory within the package containing CSV files
    package : Package
        The model package

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Mapping sensor_name -> {"means", "stds", "raw_to_local"}
    """
    result: dict[str, dict[str, np.ndarray]] = {}
    for sensor in ALL_SENSORS:
        try:
            csv_path = package.resolve(f"{stats_dir}/{sensor}_normalizations.csv")
        except FileNotFoundError:
            continue
        df = pd.read_csv(csv_path)
        df = df[df["Platform_ID"] == -1].sort_values("Raw_Channel_ID")
        means = df["obs_mean"].to_numpy(dtype=np.float32)
        stds = df["obs_std"].to_numpy(dtype=np.float32)
        raw_ids = df["Raw_Channel_ID"].to_numpy()
        max_raw = int(raw_ids.max())
        lut = np.full(max_raw + 1, 0, dtype=int)
        for local_idx, raw in enumerate(raw_ids, start=1):
            lut[int(raw)] = local_idx
        result[sensor] = {"means": means, "stds": stds, "raw_to_local": lut}
    return result


def _load_era5_denorm_stats(package: Package) -> tuple[np.ndarray, np.ndarray]:
    """Load ERA5 channel mean/std for output denormalization.

    Parameters
    ----------
    package : Package
        The model package

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mean, std) arrays of shape [n_channels] ordered by ERA5_CHANNELS
    """
    csv_path = package.resolve("stats/era5_13_levels_stats.csv")
    stats = pd.read_csv(csv_path)
    level = stats["level"].astype(int)
    channel = np.where(
        level.eq(-1), stats["variable"], stats["variable"] + level.astype(str)
    )
    ordered = stats.assign(channel=channel).set_index("channel").loc[ERA5_CHANNELS]
    return (
        ordered["mean"].to_numpy(dtype=np.float32),
        ordered["std"].to_numpy(dtype=np.float32),
    )


@check_optional_dependencies()
class HealDA(torch.nn.Module, AutoModelMixin):
    """HealDA data assimilation model for global weather analysis from sparse
    observations on a HEALPix grid.

    HealDA is a stateless assimilation model that produces a single global weather
    analysis from conventional and satellite observations. It operates on a HEALPix
    level-6 padded XY grid and outputs ERA5-compatible atmospheric variables.

    The model accepts pre-processed observation DataFrames (from
    :py:class:`earth2studio.data.UFSObsConv` and
    :py:class:`earth2studio.data.UFSObsSat`) and produces a global analysis field.

    Parameters
    ----------
    model : torch.nn.Module
        The underlying HealDA neural network
    condition : torch.Tensor
        Static conditioning fields (orography, land fraction) on the HEALPix grid
    era5_mean : torch.Tensor
        ERA5 per-channel mean for output denormalization [1, n_channels, 1, 1]
    era5_std : torch.Tensor
        ERA5 per-channel std for output denormalization [1, n_channels, 1, 1]
    sensor_stats : dict[str, dict[str, np.ndarray]]
        Per-sensor normalization statistics loaded from the package
    lat_lon : bool, optional
        If True the model output is regridded from the native HEALPix grid to a
        regular equiangular lat-lon grid using ``earth2grid``. If False the raw
        HEALPix output is returned with an ``npix`` dimension, by default False
    output_resolution : tuple[int, int], optional
        ``(nlat, nlon)`` size of the output lat-lon grid. Only used when
        ``lat_lon=True``, by default ``(181, 360)`` (1° resolution)
    time_tolerance : TimeTolerance, optional
        Time tolerance for filtering observations, by default np.timedelta64(12, "h")

    Warning
    -------
    This model requires the ``da-healda`` optional dependency group which includes
    earth2grid, physicsnemo, and GPU-accelerated DataFrame libraries.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        era5_mean: torch.Tensor,
        era5_std: torch.Tensor,
        sensor_stats: dict[str, dict[str, np.ndarray]],
        lat_lon: bool = False,
        output_resolution: tuple[int, int] = (181, 360),
        time_tolerance: TimeTolerance = np.timedelta64(12, "h"),
    ) -> None:
        super().__init__()
        self._model = model
        self.register_buffer("condition", condition)
        self.register_buffer("_era5_mean", era5_mean)
        self.register_buffer("_era5_std", era5_std)
        self._sensor_stats = sensor_stats
        self._lat_lon = lat_lon
        self._tolerance = normalize_time_tolerance(time_tolerance)
        self._grid = earth2grid.healpix.Grid(
            6, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
        )
        self._all_sat_platforms = sorted(
            {
                p
                for s in SAT_SENSORS
                if s in self._sensor_stats
                for p in self._sat_platforms(s)
            }
        )
        # Build the channel stats table once
        self._channel_stats = self._build_channel_stats()

        # Setup lat-lon regridder when requested
        if self._lat_lon:
            nlat, nlon = output_resolution
            self._output_lat = np.linspace(90, -90, nlat)
            self._output_lon = np.linspace(0, 360, nlon, endpoint=False)
            ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
            self._regridder = earth2grid.get_regridder(self._grid, ll_grid)
        else:
            self._output_lat = None
            self._output_lon = None
            self._regridder = None

    @staticmethod
    def _sat_platforms(sensor: str) -> list[str]:
        _SENSOR_PLATFORMS: dict[str, list[str]] = {
            "atms": ["npp", "n20"],
            "mhs": ["metop-a", "metop-b", "metop-c", "n18", "n19"],
            "amsua": [
                "metop-a",
                "metop-b",
                "metop-c",
                "n15",
                "n16",
                "n17",
                "n18",
                "n19",
            ],
            "amsub": ["n15", "n16", "n17"],
        }
        return _SENSOR_PLATFORMS.get(sensor, [])

    @staticmethod
    def _sensor_channels(sensor: str) -> int:
        _SENSOR_CHANNELS: dict[str, int] = {
            "atms": 22,
            "mhs": 5,
            "amsua": 15,
            "amsub": 5,
            "conv": 8,
        }
        return _SENSOR_CHANNELS.get(sensor, 0)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default HealDA model package from HuggingFace.

        Returns
        -------
        Package
            Model package pointing to the HuggingFace repository
        """
        return Package(
            "hf://nvidia/healda@21895a63705a770cde42c4d20d1d0543fefa954a",
            cache_options={"same_names": True},
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        lat_lon: bool = False,
        output_resolution: tuple[int, int] = (181, 360),
        time_tolerance: TimeTolerance = np.timedelta64(12, "h"),
        condition: torch.Tensor | None = None,
    ) -> AssimilationModel:
        """Load HealDA model from package.

        Parameters
        ----------
        package : Package
            Package containing model checkpoint and statistics
        lat_lon : bool, optional
            If True the output is regridded to a regular lat-lon grid,
            by default False
        output_resolution : tuple[int, int], optional
            ``(nlat, nlon)`` size of the output lat-lon grid. Only used when
            ``lat_lon=True``, by default ``(181, 360)``
        time_tolerance : TimeTolerance, optional
            Time tolerance for filtering observations, by default np.timedelta64(12, "h")
        condition : torch.Tensor | None, optional
            Static conditioning fields (orography, land fraction) on the HEALPix grid
            with shape [1, in_channels, time_length, npix]. If None, defaults to zeros.

        Returns
        -------
        AssimilationModel
            Loaded HealDA assimilation model
        """
        try:
            package.resolve("config.json")
        except FileNotFoundError:
            pass

        model = _HealDAModel.from_checkpoint(package.resolve("healda_ufs_era5.mdlus"))
        model.eval()

        # Static conditioning (orography, land fraction) on HEALPix grid
        # Shape: [1, in_channels, time_length, npix]
        if condition is None:
            n_in = model.in_channels
            npix = model.npix
            time_length = model.time_length
            condition = torch.zeros(1, n_in, time_length, npix, dtype=torch.float32)
            logger.info(
                "No static condition provided, using zeros. "
                "For best results, provide orography and land fraction fields."
            )

        # Load normalization statistics
        sensor_stats = _load_sensor_stats("stats", package)
        era5_mean, era5_std = _load_era5_denorm_stats(package)
        era5_mean_t = torch.from_numpy(era5_mean).view(1, -1, 1, 1)
        era5_std_t = torch.from_numpy(era5_std).view(1, -1, 1, 1)

        return cls(
            model=model,
            condition=condition,
            era5_mean=era5_mean_t,
            era5_std=era5_std_t,
            sensor_stats=sensor_stats,
            lat_lon=lat_lon,
            output_resolution=output_resolution,
            time_tolerance=time_tolerance,
        )

    def init_coords(self) -> None:
        """Initialzation coords (not required)"""
        return None

    def input_coords(self) -> tuple[FrameSchema, FrameSchema]:
        """Input coordinate system specifying required DataFrame fields.

        Returns two FrameSchemas: one for conventional observations and one for
        satellite observations. When calling the model, either may be ``None``
        but not both.

        Returns
        -------
        tuple[FrameSchema, FrameSchema]
            (conventional_schema, satellite_schema) describing the expected
            columns for each observation DataFrame
        """
        conv_schema = FrameSchema(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(
                    ["u", "v", "q", "t", "gps", "gps_t", "gps_q"], dtype=str
                ),
                "type": np.empty(0, dtype=np.uint16),
                "elev": np.empty(0, dtype=np.float32),
                "pres": np.empty(0, dtype=np.float32),
            }
        )
        sat_schema = FrameSchema(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(list(SAT_SENSORS), dtype=str),
                "channel_index": np.empty(0, dtype=np.uint16),
                "satellite": np.empty(0, dtype=str),
                "scan_angle": np.empty(0, dtype=np.float32),
                "satellite_za": np.empty(0, dtype=np.float32),
                "solza": np.empty(0, dtype=np.float32),
            }
        )
        return conv_schema, sat_schema

    def output_coords(
        self,
        input_coords: tuple[CoordSystem, CoordSystem],
        request_time: np.ndarray | None = None,
        **kwargs: Any,
    ) -> tuple[CoordSystem]:
        """Output coordinate system for the HealDA analysis.

        Parameters
        ----------
        input_coords : tuple[CoordSystem]
            Input coordinate system
        request_time : np.ndarray | None, optional
            Analysis valid time(s), by default None

        Returns
        -------
        tuple[CoordSystem]
            Coordinate system with time, variable, and lat/lon or npix dimensions
        """
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")

        if self._lat_lon:
            return (
                CoordSystem(
                    OrderedDict(
                        {
                            "time": request_time,
                            "variable": np.array(ERA5_CHANNELS, dtype=str),
                            "lat": self._output_lat,
                            "lon": self._output_lon,
                        }
                    )
                ),
            )

        return (
            CoordSystem(
                OrderedDict(
                    {
                        "time": request_time,
                        "variable": np.array(ERA5_CHANNELS, dtype=str),
                        "npix": np.arange(self._model.npix),
                    }
                )
            ),
        )

    def __call__(
        self,
        conv_obs: pd.DataFrame | None = None,
        sat_obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Run HealDA inference from conventional and/or satellite observations.

        At least one of the two observation DataFrames must be provided. Each
        DataFrame must carry a ``request_time`` entry in its ``.attrs``.

        Parameters
        ----------
        conv_obs : pd.DataFrame | None, optional
            Conventional observation DataFrame from
            :py:class:`earth2studio.data.UFSObsConv`, by default None
        sat_obs : pd.DataFrame | None, optional
            Satellite observation DataFrame from
            :py:class:`earth2studio.data.UFSObsSat`, by default None

        Returns
        -------
        xr.DataArray
            Global analysis on the HEALPix grid with dimensions
            [time, variable, npix]. Data is on the same device as the model
            (cupy array for GPU, numpy for CPU).

        Raises
        ------
        ValueError
            If both *conv_obs* and *sat_obs* are ``None``
        """
        if conv_obs is None and sat_obs is None:
            raise ValueError("At least one of conv_obs or sat_obs must be provided.")

        # Determine request_time from whichever DataFrame is present
        request_time = None
        for df in (conv_obs, sat_obs):
            if df is not None:
                request_time = df.attrs.get("request_time", None)
                if request_time is not None:
                    break
        if request_time is None:
            raise ValueError(
                "Observation DataFrame must have 'request_time' in attrs. "
                "This is typically set by earth2studio data sources."
            )

        device = self.condition.device

        # Pre-process each source into unified schema, then concatenate
        parts: list[pd.DataFrame] = []
        if conv_obs is not None and len(conv_obs) > 0:
            parts.append(self.prep_conv(conv_obs))
        if sat_obs is not None and len(sat_obs) > 0:
            for sensor in SAT_SENSORS:
                sensor_df = sat_obs[sat_obs["variable"] == sensor]
                if len(sensor_df) > 0:
                    parts.append(self.prep_sat_sensor(sensor_df, sensor))

        if len(parts) == 0:
            logger.warning("No observations provided, returning empty analysis")
            (output_coords,) = self.output_coords(
                self.input_coords(), request_time=request_time
            )
            return self._empty_output(output_coords)

        obs = pd.concat(parts, ignore_index=True)

        # Filter by time tolerance and normalize
        obs_filtered = self._filter_and_normalize(obs)
        if len(obs_filtered) == 0:
            logger.warning("No observations after filtering, returning empty analysis")
            (output_coords,) = self.output_coords(
                self.input_coords(), request_time=request_time
            )
            return self._empty_output(output_coords)

        # Sort by sensor order
        sensor_order = pd.CategoricalDtype(categories=ALL_SENSORS, ordered=True)
        obs_filtered["sensor"] = obs_filtered["sensor"].astype(sensor_order)
        obs_sorted = obs_filtered.sort_values("sensor", kind="stable").reset_index(
            drop=True
        )

        # Build model inputs
        valid_time = pd.Timestamp(request_time[0])
        inputs = self._build_model_inputs(obs_sorted, valid_time, device)

        # Run inference
        prediction = self._forward(inputs, device)

        # Build output DataArray
        (output_coords,) = self.output_coords(
            self.input_coords(), request_time=request_time
        )
        return self._to_output_dataarray(prediction, output_coords)

    def create_generator(self) -> Generator[
        xr.DataArray,
        tuple[pd.DataFrame | None, pd.DataFrame | None],
        None,
    ]:
        """Creates a generator for stateless iterative assimilation.

        The generator accepts a tuple of ``(conv_obs, sat_obs)`` DataFrames via
        ``send()`` and yields global analysis fields. Since HealDA is stateless,
        each call is independent.

        Yields
        ------
        xr.DataArray
            Global analysis on the HEALPix grid

        Receives
        --------
        tuple[pd.DataFrame | None, pd.DataFrame | None]
            A ``(conv_obs, sat_obs)`` tuple sent via ``generator.send()``.
            Either element may be ``None`` but not both.
        """
        inputs = yield None  # type: ignore[misc]
        try:
            while True:
                conv_obs, sat_obs = inputs if inputs is not None else (None, None)
                da = self.__call__(conv_obs, sat_obs)
                inputs = yield da
        except GeneratorExit:
            logger.debug("HealDA generator clean up complete.")

    # ------------------------------------------------------------------
    # Observation pre-processing
    # ------------------------------------------------------------------
    def _build_channel_stats(self) -> pd.DataFrame:
        """Build per-(sensor, local_channel) normalization stats table."""
        rows: list[dict] = []
        for sensor_name in ALL_SENSORS:
            if sensor_name not in self._sensor_stats:
                continue
            stats = self._sensor_stats[sensor_name]
            if sensor_name == "conv":
                rows.extend(
                    {
                        "sensor": sensor_name,
                        "local_channel": ch_id,
                        "mean": float(stats["means"][ch_id]),
                        "std": float(stats["stds"][ch_id]),
                        "min_valid": min_v,
                        "max_valid": max_v,
                    }
                    for ch_id, (_, min_v, max_v) in enumerate(_CONV_CHANNEL_RANGES)
                )
            else:
                rows.extend(
                    {
                        "sensor": sensor_name,
                        "local_channel": ch_id,
                        "mean": float(stats["means"][ch_id]),
                        "std": float(stats["stds"][ch_id]),
                        "min_valid": 0.0,
                        "max_valid": 400.0,
                    }
                    for ch_id in range(len(stats["means"]))
                )
        return pd.DataFrame(rows).astype({"local_channel": np.int32})

    def prep_sat_sensor(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """Standardize a satellite DataFrame into the unified obs schema.

        Parameters
        ----------
        df : pd.DataFrame
            Raw satellite observation DataFrame from UFSObsSat
        sensor : str
            Sensor name (atms, mhs, amsua, amsub)

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with unified column schema
        """
        stats = self._sensor_stats[sensor]
        raw_ch = df["channel_index"].values.astype(int)
        platforms = self._sat_platforms(sensor)
        platform_map = {name: i for i, name in enumerate(platforms)}
        return pd.DataFrame(
            {
                "lat": df["lat"].values.astype(np.float32),
                "lon": df["lon"].values.astype(np.float32),
                "obs_time_ns": df["time"]
                .values.astype("datetime64[ns]")
                .view(np.int64),
                "observation": df["observation"].values.astype(np.float64),
                "local_channel": (stats["raw_to_local"][raw_ch] - 1).astype(np.int32),
                "local_platform": df["satellite"]
                .map(platform_map)
                .values.astype(np.int64),
                "sensor": sensor,
                "obs_type": np.int32(0),
                "height": np.float32(np.nan),
                "pressure": np.float32(np.nan),
                "scan_angle": df["scan_angle"].values.astype(np.float32),
                "sat_zenith_angle": df["satellite_za"].values.astype(np.float32),
                "sol_zenith_angle": df["solza"].values.astype(np.float32),
            }
        )

    def prep_conv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize a conventional observation DataFrame into the unified obs schema.

        Parameters
        ----------
        df : pd.DataFrame
            Raw conventional observation DataFrame from UFSObsConv

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with unified column schema
        """
        return pd.DataFrame(
            {
                "lat": df["lat"].values.astype(np.float32),
                "lon": df["lon"].values.astype(np.float32),
                "obs_time_ns": df["time"]
                .values.astype("datetime64[ns]")
                .view(np.int64),
                "observation": df["observation"].values.astype(np.float64),
                "local_channel": df["variable"]
                .map(CONV_VAR_CHANNEL)
                .values.astype(np.int32),
                "local_platform": np.int64(0),
                "sensor": "conv",
                "obs_type": df["type"].fillna(0).values.astype(np.int32),
                "height": df["elev"].values.astype(np.float32),
                "pressure": df["pres"].values.astype(np.float32),
                "scan_angle": np.float32(np.nan),
                "sat_zenith_angle": np.float32(np.nan),
                "sol_zenith_angle": np.float32(np.nan),
            }
        )

    def _filter_and_normalize(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Join per-channel stats, apply QC filtering, and z-score normalize."""
        # Convert cudf to pandas if needed
        if cudf is not None and isinstance(obs, cudf.DataFrame):
            obs = obs.to_pandas()

        obs = obs.merge(self._channel_stats, on=["sensor", "local_channel"], how="left")

        # Base: obs must be finite and within valid range
        valid = obs["observation"].notna()
        valid &= (obs["observation"] >= obs["min_valid"]) & (
            obs["observation"] <= obs["max_valid"]
        )

        # Conv-specific: height and pressure physical bounds
        is_conv = obs["sensor"] == "conv"
        if is_conv.any():
            is_gps = obs["local_channel"] <= 2
            pres_min = np.where(
                is_conv & is_gps,
                _QC_PRESSURE_MIN_GPS,
                _QC_PRESSURE_MIN_DEFAULT,
            )
            height_ok = (
                obs["height"].notna()
                & (obs["height"] >= _QC_HEIGHT_MIN)
                & (obs["height"] <= _QC_HEIGHT_MAX)
            )
            pressure_ok = (
                obs["pressure"].notna()
                & (obs["pressure"] >= pres_min)
                & (obs["pressure"] <= _QC_PRESSURE_MAX)
            )
            valid &= ~is_conv | (height_ok & pressure_ok)

        obs = obs[valid].copy()
        obs["observation"] = ((obs["observation"] - obs["mean"]) / obs["std"]).astype(
            np.float32
        )
        return obs.drop(columns=["mean", "std", "min_valid", "max_valid"])

    # ------------------------------------------------------------------
    # Tensor assembly
    # ------------------------------------------------------------------
    def _build_model_inputs(
        self,
        obs: pd.DataFrame,
        valid_time: pd.Timestamp,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert processed DataFrame into model-ready tensors."""
        total_obs = len(obs)

        def to_dev(col: str) -> torch.Tensor:
            return torch.from_numpy(obs[col].values).to(device, non_blocking=True)

        lat = to_dev("lat")
        lon = to_dev("lon")

        # Cumulative per-sensor offsets
        counts = obs.groupby("sensor", observed=False).size()
        per_sensor = torch.tensor(
            [int(counts.get(s, 0)) for s in ALL_SENSORS], dtype=torch.int32
        )
        offsets = per_sensor.cumsum(0).reshape(-1, 1, 1)

        # Time encoding
        valid_dt = valid_time.to_pydatetime()
        valid_epoch = int(
            dt.datetime(
                valid_dt.year,
                valid_dt.month,
                valid_dt.day,
                valid_dt.hour,
                valid_dt.minute,
                valid_dt.second,
                tzinfo=dt.timezone.utc,
            ).timestamp()
        )
        target_time = torch.full(
            (total_obs,), valid_epoch, dtype=torch.int64, device=device
        )

        float_metadata = _compute_unified_metadata(
            target_time,
            lon=lon,
            time=to_dev("obs_time_ns"),
            height=to_dev("height"),
            pressure=to_dev("pressure"),
            scan_angle=to_dev("scan_angle"),
            sat_zenith_angle=to_dev("sat_zenith_angle"),
            sol_zenith_angle=to_dev("sol_zenith_angle"),
        )

        # Calendar features
        midnight = valid_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        second_of_day = (valid_dt - midnight).total_seconds()
        jan1 = valid_dt.replace(month=1, day=1, hour=0, minute=0, second=0)
        day_of_year = (valid_dt - jan1).total_seconds() / 86400.0

        return {
            "obs": to_dev("observation"),
            "float_metadata": float_metadata,
            "pix": self._grid.ang2pix(lon, lat).int(),
            "local_channel": to_dev("local_channel"),
            "local_platform": to_dev("local_platform"),
            "obs_type": to_dev("obs_type"),
            "offsets": offsets.to(device),
            "condition": self.condition,
            "second_of_day": torch.tensor(
                [[second_of_day]], dtype=torch.float32, device=device
            ),
            "day_of_year": torch.tensor(
                [[day_of_year]], dtype=torch.float32, device=device
            ),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _forward(
        self, inputs: dict[str, torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        """Run model forward pass and denormalize output."""
        batch_size = inputs["second_of_day"].shape[0]
        noise_labels = torch.zeros([batch_size], device=device)
        class_labels = torch.empty([batch_size, 0], device=device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = self._model(
                x=inputs["condition"],
                t=noise_labels,
                obs=inputs["obs"],
                float_metadata=inputs["float_metadata"],
                pix=inputs["pix"],
                local_channel=inputs["local_channel"],
                local_platform=inputs["local_platform"],
                obs_type=inputs["obs_type"],
                offsets=inputs["offsets"],
                second_of_day=inputs["second_of_day"],
                day_of_year=inputs["day_of_year"],
                class_labels=class_labels,
            )

        # Denormalize: prediction is [batch, channels, time, npix]
        return self._era5_std * prediction + self._era5_mean

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _to_output_dataarray(
        self,
        prediction: torch.Tensor,
        output_coords: CoordSystem,
    ) -> xr.DataArray:
        """Convert model output tensor to xarray DataArray.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output [batch, channels, time, npix]
        output_coords : CoordSystem
            Output coordinate system

        Returns
        -------
        xr.DataArray
            Analysis field, either on HEALPix (npix) or lat-lon grid
        """
        # Take last time step, shape -> [1, channels, npix]
        out = prediction[:, :, -1, :]

        if self._lat_lon and self._regridder is not None:
            # Regrid each batch×channel from HEALPix to lat-lon
            out = self._regridder(out.float())

        device = out.device
        if device.type == "cuda" and cp is not None:
            data = cp.asarray(out)
        else:
            data = out.cpu().numpy()

        if self._lat_lon:
            return xr.DataArray(
                data=data,
                dims=["time", "variable", "lat", "lon"],
                coords=output_coords,
            )

        return xr.DataArray(
            data=data,
            dims=["time", "variable", "npix"],
            coords=output_coords,
        )

    def _empty_output(self, output_coords: CoordSystem) -> xr.DataArray:
        """Return an empty (NaN-filled) DataArray."""
        n_time = len(output_coords["time"])
        n_var = len(output_coords["variable"])
        device = self.condition.device

        if self._lat_lon:
            n_lat = len(output_coords["lat"])
            n_lon = len(output_coords["lon"])
            shape = (n_time, n_var, n_lat, n_lon)
            dims = ["time", "variable", "lat", "lon"]
        else:
            n_pix = len(output_coords["npix"])
            shape = (n_time, n_var, n_pix)  # type: ignore[assignment]
            dims = ["time", "variable", "npix"]

        data = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
        if device.type == "cuda" and cp is not None:
            data_np = cp.asarray(data)
        else:
            data_np = data.cpu().numpy()

        return xr.DataArray(
            data=data_np,
            dims=dims,
            coords=output_coords,
        )
