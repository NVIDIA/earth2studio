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
from collections import OrderedDict
from collections.abc import Generator
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.da.base import AssimilationModel
from earth2studio.models.da.healda import E2S_CHANNELS, ERA5_CHANNELS
from earth2studio.models.da.healda_v2_utils import (
    CONV_CHANNELS,
    CONV_GPS_CHANNELS,
    CONV_GPS_LEVEL2_CHANNELS,
    CONV_UV_CHANNELS,
    CONV_UV_IN_SITU_TYPES,
    CONV_VAR_CHANNEL,
    IR_BT_MAX_VALID,
    IR_BT_MIN_VALID,
    PLATFORM_NAME_TO_ID,
    SENSOR_OFFSET,
    PCACodec,
    QCLimits,
    build_conv_plevel_channel_stats,
    build_raw_to_local_lut,
    compute_unified_metadata,
    conv_plevel_local_channel_lut,
    get_global_channel_id,
    nearest_pressure_level_index,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, FrameSchema

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
    from physicsnemo.experimental.models.healda import (
        VideoHealDA as _VideoHealDAModel,
    )
    from physicsnemo.experimental.models.healda import (
        prepare_obs_context,
    )
except ImportError:
    OptionalDependencyFailure("da-healda")
    earth2grid = None
    _VideoHealDAModel = None
    prepare_obs_context = None

# 8-frame video window at 6-hour spacing; each frame ingests its own
# (valid - 3h, valid + 3h] observation context, matching the training loader's
# end-aligned DA windows.
N_WINDOW = 8
WINDOW_STEP_HOURS = 6
FRAME_CONTEXT_HOURS = 3

# Microwave sounders consumed directly and infrared sounders consumed through
# PCA compression. UFSObsSat variable names for the raw IR sensors differ from
# the internal sensor names.
MW_SENSORS = ("atms", "mhs", "amsua", "amsub")
IR_PCA_SENSORS = ("iasi-pca", "cris-fsr-pca", "airs-pca")
IR_PCA_UFS_VARIABLE: dict[str, str] = {
    "iasi-pca": "iasi",
    "cris-fsr-pca": "crisfsr",
    "airs-pca": "airs",
}
SAT_UFS_VARIABLES = (*MW_SENSORS, *IR_PCA_UFS_VARIABLE.values())

SENSOR_PLATFORMS: dict[str, list[str]] = {
    "atms": ["npp", "n20"],
    "mhs": ["metop-a", "metop-b", "metop-c", "n18", "n19"],
    "amsua": ["metop-a", "metop-b", "metop-c", "n15", "n16", "n17", "n18", "n19"],
    "amsub": ["n15", "n16", "n17"],
    "iasi-pca": ["metop-a", "metop-b", "metop-c"],
    "cris-fsr-pca": ["npp", "n20"],
    "airs-pca": ["aqua"],
}

# Footprint identity columns for grouping long-format IR rows into soundings
_FP_COLS = ["satellite", "time", "lat", "lon", "scan_angle", "satellite_za", "solza"]

# Unified per-observation schema produced by the per-sensor prep methods
_OBS_FRAME_DTYPES = {
    "lat": "float32",
    "lon": "float32",
    "obs_time_ns": "datetime64[ns]",
    "observation": "float64",
    "global_channel": "int64",
    "global_platform": "int64",
    "obs_type": "int64",
    "height": "float32",
    "pressure": "float32",
    "scan_angle": "float32",
    "sat_zenith_angle": "float32",
    "sol_zenith_angle": "float32",
}


def _obs_frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data).astype(_OBS_FRAME_DTYPES, copy=False)


@check_optional_dependencies()
class HealDAv2(torch.nn.Module, AutoModelMixin):
    """HealDA-v2 video data assimilation model for global weather analysis from
    sparse observations on a HEALPix grid.

    HealDA-v2 is a stateless assimilation model that jointly produces an
    8-frame, 48-hour window of global weather analyses (6-hour spacing) from
    conventional and satellite observations. The final frame, valid at the
    request time, is the present-time analysis. Each frame ingests its own
    ±3-hour observation context, so the full observation window spans
    (request_time - 45h, request_time + 3h].

    Compared to v1, the model consumes infrared hyperspectral sounders (IASI,
    CrIS-FSR, AIRS) compressed to 32 PCA latents per footprint, and normalizes
    vertically structured conventional observations with pressure-level
    dependent statistics using an expanded per-level channel vocabulary
    ("conv-plevel").

    The model accepts observation DataFrames (from
    :py:class:`earth2studio.data.UFSObsConv` and
    :py:class:`earth2studio.data.UFSObsSat`) and produces a global analysis on
    the HEALPix level-6 padded XY grid with ERA5-compatible variables.

    Parameters
    ----------
    model : torch.nn.Module
        The underlying VideoHealDA neural network
    condition : torch.Tensor
        Static conditioning fields (orography, land fraction) on the HEALPix
        grid of size [1, n_static, 1, npix] or [1, n_static, time_length, npix]
    era5_mean : torch.Tensor
        ERA5 per-channel mean for output denormalization [1, out_variables, 1, 1]
    era5_std : torch.Tensor
        ERA5 per-channel std for output denormalization [1, out_variables, 1, 1]
    channel_stats : pd.DataFrame
        Per-global-channel statistics with columns ``Global_Channel_ID``,
        ``mean``, ``stddev``, ``min_valid``, ``max_valid`` covering the
        microwave, PCA and conv-plevel channels
    raw_to_local : dict[str, np.ndarray]
        Per-microwave-sensor lookup tables mapping raw (GSI) channel ids to
        local channels (see
        :func:`earth2studio.models.da.healda_v2_utils.build_raw_to_local_lut`)
    codecs : dict[str, PCACodec]
        PCA codecs keyed by PCA sensor name (``iasi-pca``, ``cris-fsr-pca``,
        ``airs-pca``)
    lat_lon : bool, optional
        If True the model output is regridded from the native HEALPix grid to a
        regular equiangular lat-lon grid using ``earth2grid``. If False the raw
        HEALPix output is returned with an ``npix`` dimension, by default False
    output_resolution : tuple[int, int], optional
        ``(nlat, nlon)`` size of the output lat-lon grid. Only used when
        ``lat_lon=True``, by default ``(181, 360)`` (1° resolution)

    Badges
    ------
    region:global class:da product:wind product:temp product:atmos product:sat
    product:insitu year:2026 gpu:40gb
    """

    def __init__(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        era5_mean: torch.Tensor,
        era5_std: torch.Tensor,
        channel_stats: pd.DataFrame,
        raw_to_local: dict[str, np.ndarray],
        codecs: dict[str, "PCACodec"],
        lat_lon: bool = False,
        output_resolution: tuple[int, int] = (181, 360),
    ) -> None:
        super().__init__()
        self._model = model
        self.register_buffer("condition", condition)
        self.register_buffer("_era5_mean", era5_mean)
        self.register_buffer("_era5_std", era5_std)
        self.register_buffer("device_buffer", torch.empty(0))
        self._channel_stats = channel_stats[
            ["Global_Channel_ID", "mean", "stddev", "min_valid", "max_valid"]
        ]
        self._raw_to_local = raw_to_local
        self._codecs = codecs
        self._lat_lon = lat_lon
        self._plevel_lut = conv_plevel_local_channel_lut()

        # Model geometry: observations are assigned to pixels on the backbone
        # (level_model) grid, output is on the finer level_in grid.
        self._time_length = int(model.time_length)
        self._npix_out = int(model.npix)
        self._npix_model = 12 * 4 ** int(model.level_model)
        self._grid = earth2grid.healpix.Grid(
            int(model.level_model), pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
        )

        # Setup lat-lon regridder when requested
        if self._lat_lon:
            nlat, nlon = output_resolution
            self._output_lat = np.linspace(90, -90, nlat)
            self._output_lon = np.linspace(0, 360, nlon, endpoint=False)
            out_grid = earth2grid.healpix.Grid(
                int(np.round(np.emath.logn(4, self._npix_out / 12))),
                pixel_order=earth2grid.healpix.HEALPIX_PAD_XY,
            )
            ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(nlat, nlon)
            self._regridder = earth2grid.get_regridder(out_grid, ll_grid)
        else:
            self._output_lat = None
            self._output_lon = None
            self._regridder = None

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    def init_coords(self) -> None:
        """Initialization coords (not required)"""
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
                "variable": np.array(list(CONV_VAR_CHANNEL.keys()), dtype=str),
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
                "variable": np.array(list(SAT_UFS_VARIABLES), dtype=str),
                "sensor_index": np.empty(0, dtype=np.uint16),
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
        """Output coordinate system for the HealDA-v2 analysis window.

        Parameters
        ----------
        input_coords : tuple[CoordSystem]
            Input coordinate system
        request_time : np.ndarray | None, optional
            Analysis valid time, by default None

        Returns
        -------
        tuple[CoordSystem]
            Coordinate system with time, lead_time, variable, and lat/lon or
            npix dimensions. The ``lead_time`` axis holds the 8 frame offsets
            relative to the analysis time (-42h ... 0h); the analysis frame is
            ``lead_time == 0``.
        """
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")

        lead_time = np.array(
            [
                np.timedelta64(-(self._time_length - 1 - g) * WINDOW_STEP_HOURS, "h")
                for g in range(self._time_length)
            ],
            dtype="timedelta64[ns]",
        )

        if self._lat_lon:
            return (
                CoordSystem(
                    OrderedDict(
                        {
                            "time": request_time,
                            "lead_time": lead_time,
                            "variable": np.array(E2S_CHANNELS, dtype=str),
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
                        "lead_time": lead_time,
                        "variable": np.array(E2S_CHANNELS, dtype=str),
                        "npix": np.arange(self._npix_out),
                    }
                )
            ),
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default HealDA-v2 model package.

        Raises
        ------
        NotImplementedError
            The HealDA-v2 model package has not been published yet
        """
        raise NotImplementedError(
            "The HealDA-v2 model package has not been published yet. Provide a "
            "local or remote Package with the expected layout to load_model."
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        lat_lon: bool = False,
        output_resolution: tuple[int, int] = (181, 360),
    ) -> AssimilationModel:
        """Load HealDA-v2 model from package.

        The package is expected to contain::

            healda_v2.mdlus                        # VideoHealDA checkpoint
            static/condition_hpx6_padxy.npy        # [1, 2, 1, npix] conditioning
            stats/channel_table.parquet            # global channel stats
            stats/conv_normalizations_by_level.csv # per-level conv stats
            stats/era5_13_levels_stats.csv         # output denormalization
            stats/normalizations/{sensor}_normalizations.csv  # MW raw ids
            codecs/{iasi,cris-fsr,airs}-pca.pt     # PCA codecs

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

        Returns
        -------
        AssimilationModel
            Loaded HealDA-v2 assimilation model
        """
        model = _VideoHealDAModel.from_checkpoint(package.resolve("healda_v2.mdlus"))
        model.eval()

        condition = torch.from_numpy(
            np.load(package.resolve("static/condition_hpx6_padxy.npy"))
        )

        # Global channel table: microwave + PCA sensor stats keyed by global id
        channel_table = pd.read_parquet(package.resolve("stats/channel_table.parquet"))
        channel_stats = channel_table[
            ["Global_Channel_ID", "mean", "stddev", "min_valid", "max_valid"]
        ]

        # Append the expanded conv-plevel stats built from the per-level CSV,
        # with base conv rows providing QC bounds and the surface fallback.
        level_stats = pd.read_csv(
            package.resolve("stats/conv_normalizations_by_level.csv")
        )
        conv_offset = SENSOR_OFFSET["conv"]
        base_conv = channel_stats[
            (channel_stats["Global_Channel_ID"] >= conv_offset)
            & (channel_stats["Global_Channel_ID"] < conv_offset + len(CONV_CHANNELS))
        ]
        plevel_offset = SENSOR_OFFSET["conv-plevel"]
        channel_stats = pd.concat(
            [
                channel_stats[channel_stats["Global_Channel_ID"] < plevel_offset],
                build_conv_plevel_channel_stats(level_stats, base_conv),
            ],
            ignore_index=True,
        )

        # Microwave raw (GSI) channel id -> local channel lookup tables
        raw_to_local: dict[str, np.ndarray] = {}
        for sensor in MW_SENSORS:
            df = pd.read_csv(
                package.resolve(f"stats/normalizations/{sensor}_normalizations.csv")
            )
            df = df[df["Platform_ID"] == -1]
            raw_to_local[sensor] = build_raw_to_local_lut(
                df["Raw_Channel_ID"].to_numpy()
            )

        # PCA codecs for the compressed infrared sounders
        codecs = {
            sensor: PCACodec.load(package.resolve(f"codecs/{sensor}.pt"))
            for sensor in IR_PCA_SENSORS
        }

        # ERA5 output normalization stats
        stats = pd.read_csv(package.resolve("stats/era5_13_levels_stats.csv"))
        level = stats["level"].astype(int)
        channel = np.where(
            level.eq(-1), stats["variable"], stats["variable"] + level.astype(str)
        )
        ordered = stats.assign(channel=channel).set_index("channel").loc[ERA5_CHANNELS]
        era5_mean = torch.from_numpy(ordered["mean"].to_numpy(dtype=np.float32))
        era5_std = torch.from_numpy(ordered["std"].to_numpy(dtype=np.float32))

        return cls(
            model=model,
            condition=condition,
            era5_mean=era5_mean.view(1, -1, 1, 1),
            era5_std=era5_std.view(1, -1, 1, 1),
            channel_stats=channel_stats,
            raw_to_local=raw_to_local,
            codecs=codecs,
            lat_lon=lat_lon,
            output_resolution=output_resolution,
        )

    # ------------------------------------------------------------------
    # Per-sensor preprocessing
    # ------------------------------------------------------------------

    def prep_conv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and QC a conventional DataFrame into the unified schema.

        Applies the HealDA-v2 conventional QC (height/pressure physical bounds,
        in-situ-only UV, GPS bending-angle only) and maps each observation to
        its pressure-level expanded ``conv-plevel`` global channel.

        Parameters
        ----------
        df : pd.DataFrame
            Raw conventional observation DataFrame from UFSObsConv

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with unified column schema
        """
        unknown_vars = set(df["variable"].unique()) - set(CONV_VAR_CHANNEL.keys())
        if unknown_vars:
            raise ValueError(f"Unknown conventional variable(s): {unknown_vars}")

        base_local = df["variable"].map(CONV_VAR_CHANNEL).to_numpy().astype(np.int64)
        observation = df["observation"].to_numpy().astype(np.float64)
        # Earth2Studio provides pressure-like values in Pa; the model was
        # trained on hPa.
        is_pres_obs = df["variable"].to_numpy() == "pres"
        observation[is_pres_obs] /= 100.0
        height = df["elev"].to_numpy().astype(np.float32)
        pressure = df["pres"].to_numpy().astype(np.float32) / np.float32(100.0)
        obs_type = df["type"].fillna(0).to_numpy().astype(np.int64)

        # Physical bounds QC. NaN height/pressure rows are dropped, which also
        # covers observation sources lacking those coordinates.
        is_gps = np.isin(base_local, CONV_GPS_CHANNELS)
        min_pressure = np.where(
            is_gps, QCLimits.PRESSURE_MIN_GPS, QCLimits.PRESSURE_MIN_DEFAULT
        )
        keep = (
            np.isfinite(height)
            & (height >= QCLimits.HEIGHT_MIN)
            & (height <= QCLimits.HEIGHT_MAX)
            & np.isfinite(pressure)
            & (pressure >= min_pressure)
            & (pressure <= QCLimits.PRESSURE_MAX)
        )
        # Satellite-derived winds are excluded; only in-situ UV types are kept
        is_uv = np.isin(base_local, CONV_UV_CHANNELS)
        keep &= ~is_uv | np.isin(obs_type, CONV_UV_IN_SITU_TYPES)
        # GPS level-2 retrievals (gps_t / gps_q) are excluded
        keep &= ~np.isin(base_local, CONV_GPS_LEVEL2_CHANNELS)

        base_local = base_local[keep]
        pressure = pressure[keep]

        # Pressure-level channel expansion: (base channel, level bin) ->
        # conv-plevel local channel -> global channel id
        level_idx = nearest_pressure_level_index(pressure)
        expanded_local = self._plevel_lut[base_local, level_idx].astype(np.int64)
        global_channel = SENSOR_OFFSET["conv-plevel"] + expanded_local

        platform = np.array(
            [
                PLATFORM_NAME_TO_ID[CONV_CHANNELS[channel].platform]
                for channel in base_local
            ],
            dtype=np.int64,
        )
        df = df.loc[keep]
        return _obs_frame(
            {
                "lat": df["lat"].to_numpy().astype(np.float32),
                "lon": df["lon"].to_numpy().astype(np.float32),
                "obs_time_ns": df["time"].to_numpy().astype("datetime64[ns]"),
                "observation": observation[keep],
                "global_channel": global_channel,
                "global_platform": platform,
                "obs_type": obs_type[keep],
                "height": height[keep],
                "pressure": pressure,
                "scan_angle": np.float32(np.nan),
                "sat_zenith_angle": np.float32(np.nan),
                "sol_zenith_angle": np.float32(np.nan),
            }
        )

    def prep_mw(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """Standardize a microwave sounder DataFrame into the unified schema.

        Parameters
        ----------
        df : pd.DataFrame
            Raw satellite observation DataFrame from UFSObsSat for one sensor
        sensor : str
            Sensor name (atms, mhs, amsua, amsub)

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with unified column schema
        """
        unknown = set(df["satellite"].unique()) - set(SENSOR_PLATFORMS[sensor])
        if unknown:
            raise ValueError(f"Unknown satellite platform(s) for {sensor}: {unknown}")

        global_channel = get_global_channel_id(
            sensor,
            df["sensor_index"].to_numpy().astype(np.int64),
            self._raw_to_local[sensor],
        )
        # Raw ids unknown to the LUT map below the sensor offset; drop them
        keep = global_channel >= SENSOR_OFFSET[sensor]
        df = df.loc[keep]
        return _obs_frame(
            {
                "lat": df["lat"].to_numpy().astype(np.float32),
                "lon": df["lon"].to_numpy().astype(np.float32),
                "obs_time_ns": df["time"].to_numpy().astype("datetime64[ns]"),
                "observation": df["observation"].to_numpy().astype(np.float64),
                "global_channel": global_channel[keep],
                "global_platform": df["satellite"]
                .map(PLATFORM_NAME_TO_ID)
                .to_numpy()
                .astype(np.int64),
                "obs_type": np.int64(0),
                "height": np.float32(np.nan),
                "pressure": np.float32(np.nan),
                "scan_angle": df["scan_angle"].to_numpy().astype(np.float32),
                "sat_zenith_angle": df["satellite_za"].to_numpy().astype(np.float32),
                "sol_zenith_angle": df["solza"].to_numpy().astype(np.float32),
            }
        )

    def prep_ir_pca(self, df: pd.DataFrame, sensor: str) -> pd.DataFrame:
        """Group long-format IR rows into footprints and PCA-encode them.

        Each footprint's brightness temperature spectrum is projected to 32
        latent observations. ``sensor_index`` is the GSI channel id; the
        codec's ``sensor_chan`` to global-channel mapping is the source of
        truth (AIRS has sparse channel ids).

        Parameters
        ----------
        df : pd.DataFrame
            Raw satellite observation DataFrame from UFSObsSat for one raw IR
            sensor
        sensor : str
            PCA sensor name (iasi-pca, cris-fsr-pca, airs-pca)

        Returns
        -------
        pd.DataFrame
            Standardized DataFrame with one row per (footprint, latent)
        """
        unknown = set(df["satellite"].unique()) - set(SENSOR_PLATFORMS[sensor])
        if unknown:
            raise ValueError(f"Unknown satellite platform(s) for {sensor}: {unknown}")

        codec = self._codecs[sensor]
        latent_width = codec.n_latent
        df = df.reset_index(drop=True)

        footprint_id = df.groupby(_FP_COLS, sort=False).ngroup().to_numpy()
        sensor_channel = df["sensor_index"].to_numpy().astype(np.int64)
        channel_lookup = {
            int(sensor_chan): int(gcid)
            for sensor_chan, gcid in zip(
                codec.sensor_chan.cpu().numpy(),
                codec.channel_gcids.cpu().numpy(),
            )
        }
        global_channel = np.array(
            [channel_lookup.get(int(ch), -1) for ch in sensor_channel],
            dtype=np.int64,
        )
        bt, first_row_idx = codec.preprocess(
            global_channel,
            df["observation"].to_numpy(),
            footprint_id,
            IR_BT_MIN_VALID,
            IR_BT_MAX_VALID,
        )
        latent_observation = codec.encode(torch.from_numpy(bt)).numpy()

        n_footprints = len(first_row_idx)
        footprint_rows = df.iloc[first_row_idx]

        def repeat_per_latent(values: Any) -> np.ndarray:
            return np.repeat(np.asarray(values), latent_width)

        return _obs_frame(
            {
                "lat": repeat_per_latent(footprint_rows["lat"]).astype(np.float32),
                "lon": repeat_per_latent(footprint_rows["lon"]).astype(np.float32),
                "obs_time_ns": repeat_per_latent(
                    footprint_rows["time"].to_numpy().astype("datetime64[ns]")
                ),
                "observation": latent_observation.ravel().astype(np.float64),
                "global_channel": np.tile(
                    np.arange(latent_width, dtype=np.int64) + SENSOR_OFFSET[sensor],
                    n_footprints,
                ),
                "global_platform": repeat_per_latent(
                    footprint_rows["satellite"]
                    .map(PLATFORM_NAME_TO_ID)
                    .to_numpy()
                    .astype(np.int64)
                ),
                "obs_type": np.int64(0),
                "height": np.float32(np.nan),
                "pressure": np.float32(np.nan),
                "scan_angle": repeat_per_latent(footprint_rows["scan_angle"]).astype(
                    np.float32
                ),
                "sat_zenith_angle": repeat_per_latent(
                    footprint_rows["satellite_za"]
                ).astype(np.float32),
                "sol_zenith_angle": repeat_per_latent(footprint_rows["solza"]).astype(
                    np.float32
                ),
            }
        )

    def _frame_parts(
        self, conv_df: pd.DataFrame | None, sat_df: pd.DataFrame | None
    ) -> list[pd.DataFrame]:
        """Per-sensor preprocessed DataFrames for a single 6-hour frame."""
        parts: list[pd.DataFrame] = []
        if conv_df is not None and len(conv_df) > 0:
            parts.append(self.prep_conv(conv_df))
        if sat_df is not None and len(sat_df) > 0:
            for sensor in MW_SENSORS:
                rows = sat_df[sat_df["variable"] == sensor]
                if len(rows) > 0:
                    parts.append(self.prep_mw(rows, sensor))
            for sensor in IR_PCA_SENSORS:
                if sensor not in self._codecs:
                    continue
                rows = sat_df[sat_df["variable"] == IR_PCA_UFS_VARIABLE[sensor]]
                if len(rows) > 0:
                    parts.append(self.prep_ir_pca(rows, sensor))
        return [part for part in parts if len(part) > 0]

    # ------------------------------------------------------------------
    # Window assembly and forward
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_valid_times(
        request_time: np.datetime64, time_length: int
    ) -> list[pd.Timestamp]:
        """Valid time of each window frame, ending at ``request_time``."""
        analysis_time = pd.Timestamp(request_time)
        return [
            analysis_time
            - pd.Timedelta(hours=WINDOW_STEP_HOURS * (time_length - 1 - g))
            for g in range(time_length)
        ]

    @staticmethod
    def _slice_frame(
        df: pd.DataFrame | None, valid_time: pd.Timestamp
    ) -> pd.DataFrame | None:
        """Observations in the frame's end-aligned (valid - 3h, valid + 3h] window."""
        if df is None or len(df) == 0:
            return None
        times = pd.to_datetime(df["time"])
        lo = valid_time - pd.Timedelta(hours=FRAME_CONTEXT_HOURS)
        hi = valid_time + pd.Timedelta(hours=FRAME_CONTEXT_HOURS)
        frame_df = df[(times > lo) & (times <= hi)]
        return frame_df if len(frame_df) > 0 else None

    def filter_and_normalize(
        self,
        conv_obs: pd.DataFrame | None,
        sat_obs: pd.DataFrame | None,
        request_time: np.datetime64,
    ) -> pd.DataFrame:
        """Preprocess, QC, and normalize observations for the full window.

        Buckets raw observations into the 8 window frames, converts each
        frame's rows into the unified per-observation schema, applies
        per-channel valid-range QC, and z-score normalizes with the global
        channel statistics.

        Parameters
        ----------
        conv_obs : pd.DataFrame | None
            Raw conventional observation DataFrame (or ``None``)
        sat_obs : pd.DataFrame | None
            Raw satellite observation DataFrame (or ``None``)
        request_time : np.datetime64
            Analysis valid time (final frame)

        Returns
        -------
        pd.DataFrame
            Normalized unified observation DataFrame with additional ``frame``
            and ``target_sec`` columns; may be empty
        """
        frame_times = self._frame_valid_times(request_time, self._time_length)
        parts: list[pd.DataFrame] = []
        for frame_idx, valid_time in enumerate(frame_times):
            frame_conv = self._slice_frame(conv_obs, valid_time)
            frame_sat = self._slice_frame(sat_obs, valid_time)
            for part in self._frame_parts(frame_conv, frame_sat):
                part = part.copy()
                part["frame"] = np.int64(frame_idx)
                part["target_sec"] = np.int64(self._datetime64_to_epoch_sec(valid_time))
                parts.append(part)

        if not parts:
            return pd.DataFrame(columns=[*_OBS_FRAME_DTYPES, "frame", "target_sec"])

        obs = pd.concat(parts, ignore_index=True)
        obs = obs.merge(
            self._channel_stats,
            left_on="global_channel",
            right_on="Global_Channel_ID",
            how="left",
        )
        observation = obs["observation"].to_numpy()
        valid = (
            np.isfinite(observation)
            & (observation >= obs["min_valid"].to_numpy())
            & (observation <= obs["max_valid"].to_numpy())
        )
        obs = obs[valid].reset_index(drop=True)
        obs["observation"] = (obs["observation"] - obs["mean"]) / obs["stddev"]
        return obs.drop(
            columns=["Global_Channel_ID", "mean", "stddev", "min_valid", "max_valid"]
        )

    @staticmethod
    def _datetime64_to_epoch_sec(t: np.datetime64 | pd.Timestamp) -> int:
        """Convert a time to integer UTC epoch seconds."""
        t_dt = pd.Timestamp(t).to_pydatetime()
        return int(
            dt.datetime(
                t_dt.year,
                t_dt.month,
                t_dt.day,
                t_dt.hour,
                t_dt.minute,
                t_dt.second,
                tzinfo=dt.timezone.utc,
            ).timestamp()
        )

    def build_input(
        self, obs: pd.DataFrame, request_time: np.datetime64
    ) -> dict[str, Any]:
        """Convert the normalized observation DataFrame into model inputs.

        Computes the 50-dim metadata features, assigns each observation to its
        (frame, pixel) bucket on the backbone HEALPix grid, and packs the
        observations for the ragged pixel cross-attention.

        Parameters
        ----------
        obs : pd.DataFrame
            Output of :meth:`filter_and_normalize`
        request_time : np.datetime64
            Analysis valid time (final frame)

        Returns
        -------
        dict[str, Any]
            Dictionary with ``condition``, ``second_of_day``, ``day_of_year``
            tensors and the packed ``obs_ctx``
        """
        device = self.device

        def col(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(obs[name].to_numpy(), dtype=dtype, device=device)

        lon = col("lon", torch.float32)
        lat = col("lat", torch.float32)
        time_ns = torch.as_tensor(
            obs["obs_time_ns"].to_numpy().astype("datetime64[ns]").astype(np.int64),
            device=device,
        )
        float_metadata = compute_unified_metadata(
            col("target_sec", torch.int64),
            time=time_ns,
            lon=lon,
            lat=lat,
            height=col("height", torch.float32),
            pressure=col("pressure", torch.float32),
            scan_angle=col("scan_angle", torch.float32),
            sat_zenith_angle=col("sat_zenith_angle", torch.float32),
            sol_zenith_angle=col("sol_zenith_angle", torch.float32),
        )

        pix = self._grid.ang2pix(lon, lat).long()
        frame_idx = col("frame", torch.long)
        flat_idx = (frame_idx * self._npix_model + pix).int()
        obs_ctx = prepare_obs_context(
            obs=col("observation", torch.float32),
            float_metadata=float_metadata,
            obs_type=col("obs_type", torch.long),
            channel=col("global_channel", torch.long),
            platform=col("global_platform", torch.long),
            flat_idx=flat_idx,
            total_pixels=self._time_length * self._npix_model,
        )

        # Per-frame calendar features, each shaped [1, time_length]
        frame_times = self._frame_valid_times(request_time, self._time_length)
        second_of_day = [
            float((t - t.normalize()).total_seconds()) for t in frame_times
        ]
        day_of_year = [
            float((t - t.normalize().replace(month=1, day=1)).total_seconds() / 86400.0)
            for t in frame_times
        ]

        condition = self.condition
        if condition.shape[2] == 1 and self._time_length > 1:
            condition = condition.expand(-1, -1, self._time_length, -1).contiguous()

        return {
            "condition": condition,
            "second_of_day": torch.tensor(
                [second_of_day], dtype=torch.float32, device=device
            ),
            "day_of_year": torch.tensor(
                [day_of_year], dtype=torch.float32, device=device
            ),
            "obs_ctx": obs_ctx,
        }

    @torch.inference_mode()
    def _forward(self, inputs: dict[str, Any]) -> torch.Tensor:
        noise_labels = torch.zeros(1, device=self.device)
        autocast = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=autocast):
            prediction = self._model(
                inputs["condition"],
                noise_labels,
                inputs["second_of_day"],
                inputs["day_of_year"],
                inputs["obs_ctx"],
            )
        # Denormalize: prediction is [batch, channels, time, npix]
        return self._era5_std * prediction.float() + self._era5_mean

    # ------------------------------------------------------------------
    # Call / generator
    # ------------------------------------------------------------------

    def __call__(
        self,
        conv_obs: pd.DataFrame | None = None,
        sat_obs: pd.DataFrame | None = None,
    ) -> xr.DataArray:
        """Run HealDA-v2 inference from conventional and/or satellite observations.

        At least one of the two observation DataFrames must be provided. Each
        DataFrame must carry a single ``request_time`` entry in its ``.attrs``;
        it is the valid time of the final window frame (the analysis). The
        DataFrames should cover the full observation window
        ``(request_time - 45h, request_time + 3h]``.

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
            Global analysis window with dimensions
            [time, lead_time, variable, npix] (or lat/lon). The analysis frame
            is ``lead_time == 0``. Data is on the same device as the model
            (cupy array for GPU, numpy for CPU).

        Raises
        ------
        ValueError
            If both *conv_obs* and *sat_obs* are ``None``, if ``request_time``
            is missing, or if more than one request time is provided
        """
        if conv_obs is None and sat_obs is None:
            raise ValueError("At least one of conv_obs or sat_obs must be provided.")

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

        if isinstance(request_time, np.ndarray):
            request_time = request_time.astype("datetime64[ns]")
        else:
            request_time = np.array(
                [np.datetime64(request_time, "ns")], dtype="datetime64[ns]"
            )
        if len(request_time) != 1:
            raise ValueError(
                "HealDAv2 assimilates a single analysis time per call; got "
                f"{len(request_time)} request times."
            )

        # Convert cudf to pandas if needed
        if cudf is not None:
            if isinstance(conv_obs, cudf.DataFrame):
                conv_obs = conv_obs.to_pandas()
            if isinstance(sat_obs, cudf.DataFrame):
                sat_obs = sat_obs.to_pandas()

        analysis_time = cast(np.datetime64, request_time[0])
        obs = self.filter_and_normalize(conv_obs, sat_obs, analysis_time)

        (output_coords,) = self.output_coords(
            self.input_coords(), request_time=request_time
        )
        if len(obs) == 0:
            logger.warning("No observations after filtering, returning empty analysis")
            return self._empty_output(output_coords)

        inputs = self.build_input(obs, analysis_time)
        prediction = self._forward(inputs)
        return self.build_output(prediction, output_coords)

    def create_generator(self) -> Generator[
        xr.DataArray,
        tuple[pd.DataFrame | None, pd.DataFrame | None],
        None,
    ]:
        """Creates a generator which accepts collection of input observations
        and yields the output global assimilated data.

        Yields
        ------
        xr.DataArray
            Global analysis window on the HEALPix grid

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
            logger.debug("HealDAv2 generator clean up complete.")

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def build_output(
        self,
        prediction: torch.Tensor,
        output_coords: CoordSystem,
    ) -> xr.DataArray:
        """Convert model output tensor to xarray DataArray.

        Parameters
        ----------
        prediction : torch.Tensor
            Model output [batch, variable, time_length, npix]
        output_coords : CoordSystem
            Output coordinate system

        Returns
        -------
        xr.DataArray
            Analysis window, either on HEALPix (npix) or lat-lon grid
        """
        # [1, variable, time_length, npix] -> [time, lead_time, variable, npix]
        out = prediction.permute(0, 2, 1, 3).contiguous()
        if self._lat_lon and self._regridder is not None:
            out = self._regridder(out.double())

        if self.device.type == "cuda" and cp is not None:
            data = cp.asarray(out)
        else:
            data = out.cpu().numpy()

        if self._lat_lon:
            return xr.DataArray(
                data=data,
                dims=["time", "lead_time", "variable", "lat", "lon"],
                coords=output_coords,
            )

        return xr.DataArray(
            data=data,
            dims=["time", "lead_time", "variable", "npix"],
            coords=output_coords,
        )

    def _empty_output(self, output_coords: CoordSystem) -> xr.DataArray:
        """Return an empty (NaN-filled) DataArray."""
        dims = list(output_coords.keys())
        shape = tuple(len(v) for v in output_coords.values())
        data = torch.full(shape, float("nan"), dtype=torch.float32, device=self.device)
        if self.device.type == "cuda" and cp is not None:
            data_np = cp.asarray(data)
        else:
            data_np = data.cpu().numpy()
        return xr.DataArray(data=data_np, dims=dims, coords=output_coords)
