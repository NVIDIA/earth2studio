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
"""Self-contained helpers for the HealDA-v2 (video-DA) assimilation model.

Ports the observation vocabulary, pressure-level conventional-obs
normalization, PCA infrared codec, and the 50-dim unified observation metadata
featurization from the HealDA training repository so the Earth2Studio wrapper
has no dependency on the internal ``healda`` package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Sensor vocabulary
#
# The global channel id space concatenates each sensor's local channels in this
# fixed order. Offsets must match the channel table the model was trained with.
# ---------------------------------------------------------------------------

# Sensor name -> number of local channels, in canonical (offset) order.
SENSOR_CHANNELS: dict[str, int] = {
    "atms": 22,
    "mhs": 5,
    "amsua": 15,
    "amsub": 5,
    "iasi": 175,
    "cris-fsr": 100,
    "conv": 8,
    "iasi-pca": 32,
    "cris-fsr-pca": 32,
    "airs": 117,
    "airs-pca": 32,
    "conv-plevel": 92,
}

SENSOR_OFFSET: dict[str, int] = {}
_offset = 0
for _name, _nchan in SENSOR_CHANNELS.items():
    SENSOR_OFFSET[_name] = _offset
    _offset += _nchan

SENSOR_NAME_TO_ID: dict[str, int] = {
    name: idx for idx, name in enumerate(SENSOR_CHANNELS.keys())
}

# Valid brightness-temperature range for the raw infrared sounders whose
# spectra are PCA-compressed (values outside are treated as missing).
IR_BT_MIN_VALID = 150.0
IR_BT_MAX_VALID = 350.0

PLATFORM_NAME_TO_ID: dict[str, int] = {
    "aqua": 0,
    "aura": 1,
    "f10": 2,
    "f11": 3,
    "f13": 4,
    "f14": 5,
    "f15": 6,
    "g08": 7,
    "g10": 8,
    "g11": 9,
    "g12": 10,
    "m08": 11,
    "m09": 12,
    "m10": 13,
    "metop-a": 14,
    "metop-b": 15,
    "metop-c": 16,
    "n11": 17,
    "n12": 18,
    "n14": 19,
    "n15": 20,
    "n16": 21,
    "n17": 22,
    "n18": 23,
    "n19": 24,
    "n20": 25,
    "npp": 26,
    "gps": 27,
    "ps": 28,
    "q": 29,
    "t": 30,
    "uv": 31,
}


# ---------------------------------------------------------------------------
# Conventional observation channels and QC limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvChannel:
    """Conventional sensor channel definition."""

    name: str
    platform: str
    min_valid: float
    max_valid: float


CONV_CHANNELS: list[ConvChannel] = [
    ConvChannel("gps_angle", "gps", 0.0, 0.1),
    ConvChannel("gps_t", "gps", 150.0, 350.0),
    ConvChannel("gps_q", "gps", 0.0, 1.0),
    ConvChannel("ps", "ps", 500.0, 1100.0),
    ConvChannel("q", "q", 0.0, 1.0),
    ConvChannel("t", "t", 150.0, 350.0),
    ConvChannel("u", "uv", -100.0, 100.0),
    ConvChannel("v", "uv", -100.0, 100.0),
]

CONV_CHANNEL_NAMES: list[str] = [c.name for c in CONV_CHANNELS]
CONV_GPS_CHANNELS: list[int] = [
    i for i, c in enumerate(CONV_CHANNELS) if c.platform == "gps"
]
CONV_GPS_LEVEL2_CHANNELS: list[int] = [
    i for i, c in enumerate(CONV_CHANNELS) if c.name in ("gps_t", "gps_q")
]
CONV_UV_CHANNELS: list[int] = [
    i for i, c in enumerate(CONV_CHANNELS) if c.platform == "uv"
]
CONV_UV_IN_SITU_TYPES: list[int] = [
    220,
    221,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    280,
    282,
]

# Earth2Studio conventional variable name -> base conv local channel id
CONV_VAR_CHANNEL: dict[str, int] = {
    "gps": CONV_CHANNEL_NAMES.index("gps_angle"),
    "gps_t": CONV_CHANNEL_NAMES.index("gps_t"),
    "gps_q": CONV_CHANNEL_NAMES.index("gps_q"),
    "pres": CONV_CHANNEL_NAMES.index("ps"),
    "q": CONV_CHANNEL_NAMES.index("q"),
    "t": CONV_CHANNEL_NAMES.index("t"),
    "u": CONV_CHANNEL_NAMES.index("u"),
    "v": CONV_CHANNEL_NAMES.index("v"),
}


class QCLimits:
    """Conventional observation QC filtering limits."""

    # Height limits (meters)
    HEIGHT_MIN = 0.0
    HEIGHT_MAX = 60000.0
    # Pressure limits (hPa). The HealDA-v2 training configuration lowers the
    # non-GPS floor from the historical 200 hPa to 1 hPa.
    PRESSURE_MIN_GPS = 0.5
    PRESSURE_MIN_DEFAULT = 1.0
    PRESSURE_MAX = 1100.0


# ---------------------------------------------------------------------------
# Pressure-level ("conv-plevel") channel expansion
#
# The vertically structured conventional channels (all except surface pressure)
# are expanded into one channel per ERA5 pressure level; normalization stats are
# then per (base channel, level). Surface pressure keeps a single channel with
# its base normalization.
# ---------------------------------------------------------------------------

PRESSURE_LEVELS_HPA = np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    dtype=np.int16,
)
CONV_PLEVEL_SURFACE_LOCAL_CHANNEL: int = CONV_CHANNEL_NAMES.index("ps")
CONV_PLEVEL_BASE_LOCAL_CHANNELS = np.array(
    [
        local_channel
        for local_channel in range(len(CONV_CHANNELS))
        if local_channel != CONV_PLEVEL_SURFACE_LOCAL_CHANNEL
    ],
    dtype=np.int16,
)
CONV_PLEVEL_N_LEVELS: int = PRESSURE_LEVELS_HPA.size
CONV_PLEVEL_N_VERTICAL_CHANNELS: int = CONV_PLEVEL_BASE_LOCAL_CHANNELS.size
CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL: int = (
    CONV_PLEVEL_N_VERTICAL_CHANNELS * CONV_PLEVEL_N_LEVELS
)
CONV_PLEVEL_N_CHANNELS: int = CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL + 1

_PRESSURE_LEVELS_ASC = np.sort(PRESSURE_LEVELS_HPA)
_PRESSURE_LEVEL_EDGES = (
    _PRESSURE_LEVELS_ASC[:-1].astype(np.float32)
    + _PRESSURE_LEVELS_ASC[1:].astype(np.float32)
) / 2.0


def nearest_pressure_level_index(pressure: np.ndarray) -> np.ndarray:
    """Return indices into :data:`PRESSURE_LEVELS_HPA` for nearest-level binning.

    Values outside the 50-1000 hPa span are clipped to the nearest endpoint by
    the fixed midpoint thresholds.

    Parameters
    ----------
    pressure : np.ndarray
        Observation pressures in hPa

    Returns
    -------
    np.ndarray
        Index of the nearest pressure level for each observation
    """
    pressure = np.asarray(pressure, dtype=np.float32)
    idx: np.ndarray = np.zeros(pressure.size, dtype=np.int8)
    for edge in _PRESSURE_LEVEL_EDGES:
        idx += pressure > edge
    return (PRESSURE_LEVELS_HPA.size - 1 - idx).astype(np.int8, copy=False)


def conv_plevel_local_channel_lut() -> np.ndarray:
    """Map (base conv local channel, pressure-level index) to expanded local id.

    The first axis is the original 8-channel conv vocabulary. The second axis is
    the index returned by :func:`nearest_pressure_level_index`. Surface pressure
    maps to the same final expanded channel for every pressure bin.

    Returns
    -------
    np.ndarray
        Lookup table of shape [len(CONV_CHANNELS), CONV_PLEVEL_N_LEVELS]
    """
    n_conv_channels = len(CONV_CHANNELS)
    base_to_group: np.ndarray = np.full(n_conv_channels, -1, dtype=np.int16)
    for group, local_channel in enumerate(CONV_PLEVEL_BASE_LOCAL_CHANNELS):
        base_to_group[local_channel] = group

    lut: np.ndarray = np.empty((n_conv_channels, CONV_PLEVEL_N_LEVELS), dtype=np.uint16)
    for local_channel in range(n_conv_channels):
        if local_channel == CONV_PLEVEL_SURFACE_LOCAL_CHANNEL:
            lut[local_channel, :] = CONV_PLEVEL_SURFACE_EXPANDED_CHANNEL
            continue
        lut[local_channel, :] = base_to_group[
            local_channel
        ] * CONV_PLEVEL_N_LEVELS + np.arange(CONV_PLEVEL_N_LEVELS, dtype=np.uint16)
    return lut


def build_conv_plevel_channel_stats(
    level_stats: pd.DataFrame,
    base_conv_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 92-row ``conv-plevel`` channel stats table.

    For pressure-expanded rows, matching level stats override the base conv
    normalization; missing (channel, level) rows fall back to base channel
    stats. The surface pressure row always uses the base ``ps`` normalization.
    QC ``min_valid``/``max_valid`` are inherited from the base channel.

    Parameters
    ----------
    level_stats : pd.DataFrame
        Per-level normalization stats with columns ``Global_Channel_ID`` (base
        conv global id), ``Level_hPa``, ``obs_mean``, ``obs_std``
    base_conv_stats : pd.DataFrame
        Base conv channel stats with columns ``Global_Channel_ID``, ``mean``,
        ``stddev`` for the 8 base conv channels

    Returns
    -------
    pd.DataFrame
        Stats table with columns ``Global_Channel_ID``, ``mean``, ``stddev``,
        ``min_valid``, ``max_valid`` for the 92 conv-plevel channels
    """
    conv_offset = SENSOR_OFFSET["conv"]
    conv_plevel_offset = SENSOR_OFFSET["conv-plevel"]

    base_norms = {
        int(row["Global_Channel_ID"]): (float(row["mean"]), float(row["stddev"]))
        for _, row in base_conv_stats.iterrows()
    }
    level_norms = {
        (int(row["Global_Channel_ID"]), int(row["Level_hPa"])): (
            float(row["obs_mean"]),
            float(row["obs_std"]),
        )
        for _, row in level_stats.iterrows()
    }

    rows = []
    expanded_local = 0
    for base_local in CONV_PLEVEL_BASE_LOCAL_CHANNELS:
        channel = CONV_CHANNELS[int(base_local)]
        base_gid = conv_offset + int(base_local)
        base_mean, base_stddev = base_norms.get(base_gid, (0.0, 1.0))
        for level_hpa in PRESSURE_LEVELS_HPA:
            mean, stddev = level_norms.get(
                (base_gid, int(level_hpa)), (base_mean, base_stddev)
            )
            rows.append(
                {
                    "Global_Channel_ID": conv_plevel_offset + expanded_local,
                    "mean": mean,
                    "stddev": stddev,
                    "min_valid": channel.min_valid,
                    "max_valid": channel.max_valid,
                }
            )
            expanded_local += 1

    surface = CONV_CHANNELS[CONV_PLEVEL_SURFACE_LOCAL_CHANNEL]
    surface_gid = conv_offset + CONV_PLEVEL_SURFACE_LOCAL_CHANNEL
    surface_mean, surface_stddev = base_norms.get(surface_gid, (0.0, 1.0))
    rows.append(
        {
            "Global_Channel_ID": conv_plevel_offset + expanded_local,
            "mean": surface_mean,
            "stddev": surface_stddev,
            "min_valid": surface.min_valid,
            "max_valid": surface.max_valid,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Raw channel id -> local channel lookup for microwave sounders
# ---------------------------------------------------------------------------


def build_raw_to_local_lut(raw_channel_ids: np.ndarray) -> np.ndarray:
    """Build a 1-indexed raw-channel-id to local-channel lookup table.

    Index 0 of the LUT (and any raw id not present) maps to 0, which callers
    interpret as local channel -1 after subtracting one.

    Parameters
    ----------
    raw_channel_ids : np.ndarray
        Sorted unique raw (GSI) channel ids for a sensor

    Returns
    -------
    np.ndarray
        Lookup table such that ``lut[raw_id] - 1`` is the local channel
    """
    raw_ids = np.unique(np.asarray(raw_channel_ids).ravel())
    lut: np.ndarray = np.zeros(int(raw_ids.max()) + 1, dtype=np.int64)
    for local_idx, raw in enumerate(raw_ids, start=1):
        lut[int(raw)] = local_idx
    return lut


def get_global_channel_id(
    sensor: str, raw_channel_ids: np.ndarray, raw_to_local: np.ndarray
) -> np.ndarray:
    """Map per-sensor raw channel ids to unified global channel ids.

    Parameters
    ----------
    sensor : str
        Sensor name (key of :data:`SENSOR_OFFSET`)
    raw_channel_ids : np.ndarray
        Raw (GSI) channel ids
    raw_to_local : np.ndarray
        Lookup table from :func:`build_raw_to_local_lut`

    Returns
    -------
    np.ndarray
        Global channel ids (int64); raw ids unknown to the LUT map to
        ``SENSOR_OFFSET[sensor] - 1`` and should be filtered by callers
    """
    raw_channel_ids = np.asarray(raw_channel_ids)
    safe_ids = np.minimum(raw_channel_ids, len(raw_to_local) - 1)
    local_channels = raw_to_local[safe_ids] - 1
    return (local_channels + SENSOR_OFFSET[sensor]).astype(np.int64)


# ---------------------------------------------------------------------------
# PCA codec for compressed infrared sounders (AIRS / IASI / CrIS-FSR)
# ---------------------------------------------------------------------------

_STD_FLOOR = 1e-6


class PCACodec(torch.nn.Module):
    """Deployable PCA codec for compressed IR sensors.

    The codec standardizes brightness temperatures per channel and projects
    onto a fixed set of PCA components::

        encode:  z = ((bt - bt_mean) / bt_std) @ components   (N, C) -> (N, k)

    Invalid channels (NaN) are zeroed after standardization so they contribute
    nothing to the latent projection. A saved codec is a single ``.pt`` state
    dict; channel count and latent dimension are recovered from buffer shapes.

    Parameters
    ----------
    bt_mean : torch.Tensor
        Per-channel brightness temperature mean [n_channels]
    bt_std : torch.Tensor
        Per-channel brightness temperature std [n_channels]
    components : torch.Tensor
        PCA projection matrix [n_channels, n_latent]
    channel_gcids : torch.Tensor | None, optional
        Global channel id of each codec column, by default None
    sensor_chan : torch.Tensor | None, optional
        Raw (GSI) sensor channel id of each codec column, by default None
    """

    def __init__(
        self,
        bt_mean: torch.Tensor,
        bt_std: torch.Tensor,
        components: torch.Tensor,
        channel_gcids: torch.Tensor | None = None,
        sensor_chan: torch.Tensor | None = None,
    ):
        super().__init__()
        bt_mean = torch.as_tensor(bt_mean, dtype=torch.float32)
        bt_std = torch.as_tensor(bt_std, dtype=torch.float32).clamp_min(_STD_FLOOR)
        components = torch.as_tensor(components, dtype=torch.float32)
        if bt_mean.ndim != 1 or bt_std.shape != bt_mean.shape:
            raise ValueError("bt_mean and bt_std must be 1D with matching length")
        if components.ndim != 2 or components.shape[0] != bt_mean.shape[0]:
            raise ValueError("components must have shape (n_channels, n_latent)")

        n_channels = bt_mean.shape[0]
        if channel_gcids is None:
            channel_gcids = torch.arange(n_channels, dtype=torch.int32)
        if sensor_chan is None:
            sensor_chan = torch.arange(n_channels, dtype=torch.int32)

        self.register_buffer("bt_mean", bt_mean)
        self.register_buffer("bt_std", bt_std)
        self.register_buffer("components", components)
        self.register_buffer(
            "channel_gcids", torch.as_tensor(channel_gcids, dtype=torch.int32)
        )
        self.register_buffer(
            "sensor_chan", torch.as_tensor(sensor_chan, dtype=torch.int32)
        )

    @property
    def n_channels(self) -> int:
        return int(self.bt_mean.shape[0])

    @property
    def n_latent(self) -> int:
        return int(self.components.shape[1])

    def standardize(
        self, bt: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        if valid is None:
            valid = ~torch.isnan(bt)
        z = (bt - self.bt_mean) / self.bt_std
        return z.masked_fill(~valid, 0.0)

    def encode(
        self, bt: torch.Tensor, valid: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.standardize(bt, valid) @ self.components

    def preprocess(
        self,
        channel_gcid: np.ndarray,
        obs: np.ndarray,
        footprint_id: np.ndarray,
        bt_min: float = float("-inf"),
        bt_max: float = float("inf"),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shape long per-(footprint, channel) BT rows into the wide
        ``(n_fp, n_channels)`` matrix that :meth:`encode` expects.

        Maps each row's global channel id to a codec column (dropping channels
        the codec does not model), groups rows by footprint, and marks missing
        or out-of-range BT as NaN (which :meth:`encode` masks to zero).

        Parameters
        ----------
        channel_gcid : np.ndarray
            Global channel ids matching ``self.channel_gcids`` [n_rows]
        obs : np.ndarray
            Brightness temperatures [n_rows]
        footprint_id : np.ndarray
            Group id identifying the sounding per row [n_rows]
        bt_min : float, optional
            Minimum valid BT; entries below are treated as missing
        bt_max : float, optional
            Maximum valid BT; entries above are treated as missing

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``bt`` wide BT matrix [n_fp, n_channels] (NaN for missing) and
            ``first_row_idx`` [n_fp] index of the first input row of each
            footprint, for gathering footprint-level metadata
        """
        gcid = np.asarray(channel_gcid)
        lut: np.ndarray = np.full(int(self.channel_gcids.max()) + 1, -1, dtype=np.int64)
        lut[self.channel_gcids.numpy()] = np.arange(self.n_channels)
        col = lut[np.minimum(gcid, len(lut) - 1)]
        keep_idx = np.where(col >= 0)[0]
        _, first_in_kept, inv = np.unique(
            np.asarray(footprint_id)[keep_idx], return_index=True, return_inverse=True
        )
        first_row_idx = keep_idx[first_in_kept]

        bt: np.ndarray = np.full(
            (len(first_row_idx), self.n_channels), np.nan, dtype=np.float32
        )
        bt[inv, col[keep_idx]] = np.asarray(obs)[keep_idx].astype(np.float32)
        bt[(bt < bt_min) | (bt > bt_max)] = np.nan
        return bt, first_row_idx

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> PCACodec:
        """Load a codec from a single ``.pt`` state-dict file.

        Parameters
        ----------
        path : str | Path
            Path to the saved codec
        map_location : str, optional
            Torch map location, by default "cpu"

        Returns
        -------
        PCACodec
            Loaded codec in eval mode
        """
        state = torch.load(path, map_location=map_location, weights_only=True)
        codec = cls(
            bt_mean=state["bt_mean"],
            bt_std=state["bt_std"],
            components=state["components"],
            channel_gcids=state.get("channel_gcids"),
            sensor_chan=state.get("sensor_chan"),
        )
        codec.load_state_dict(state)
        codec.eval()
        return codec


# ---------------------------------------------------------------------------
# 50-dim unified observation metadata (features v2)
# ---------------------------------------------------------------------------

N_METADATA_FEATURES = 50
# Conv/sat-split layout:
#   [0:10)  SHARED: LST fourier(2)[0:4) + dt [dt,dt2][4:6) + dt fourier(1)[6:8)
#                   + lat [sin,cos][8:10)
#   [10:30) SAT-private: scan fourier(3)[10:16) + sat_zen fourier(4)[16:24)
#                        + sol_zen fourier(3)[24:30)  -- zero for conv rows
#   [30:50) CONV-private: height fourier(5)[30:40) + pressure fourier(5)[40:50)
#                         -- zero for sat rows
# A row is conv iff height is not NaN; exactly one private block is non-zero.


def fourier_features(x_norm: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Sin/cos Fourier features for frequencies ``1..num_freqs``.

    Parameters
    ----------
    x_norm : torch.Tensor
        Input angles (radians when pre-multiplied by 2*pi) [n]
    num_freqs : int
        Number of frequencies

    Returns
    -------
    torch.Tensor
        Features [n, 2 * num_freqs] laid out as [sin(1x)..sin(kx), cos(1x)..cos(kx)]
    """
    freqs = torch.arange(1, num_freqs + 1, device=x_norm.device, dtype=x_norm.dtype)
    x_expanded = x_norm.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)


def local_solar_time(lon_deg: torch.Tensor, abs_time_ns: torch.Tensor) -> torch.Tensor:
    """Local solar time in hours from longitude and absolute time.

    Parameters
    ----------
    lon_deg : torch.Tensor
        Longitude in degrees [n]
    abs_time_ns : torch.Tensor
        Observation times as epoch nanoseconds (int64) [n]

    Returns
    -------
    torch.Tensor
        Local solar time in [0, 24) hours [n]
    """
    sec_of_day = (abs_time_ns // 1_000_000_000) % 86400
    utc_hours = sec_of_day.float() / 3600.0
    return (utc_hours + lon_deg / 15.0) % 24.0


def compute_unified_metadata(
    target_time_sec: torch.Tensor,
    time: torch.Tensor,
    lon: torch.Tensor,
    lat: torch.Tensor,
    height: torch.Tensor,
    pressure: torch.Tensor,
    scan_angle: torch.Tensor,
    sat_zenith_angle: torch.Tensor,
    sol_zenith_angle: torch.Tensor,
) -> torch.Tensor:
    """Compute the 50-dim unified observation metadata features (v2).

    Conv/sat specialization: height validity determines which private block
    fills feature slots 10-29 (satellite) or 30-49 (conventional).

    Parameters
    ----------
    target_time_sec : torch.Tensor
        Analysis (frame) time as epoch seconds (int64) [n]
    time : torch.Tensor
        Observation times as epoch nanoseconds (int64) [n]
    lon : torch.Tensor
        Longitude in degrees [n]
    lat : torch.Tensor
        Latitude in degrees [n]
    height : torch.Tensor
        Height in meters (NaN for satellite obs) [n]
    pressure : torch.Tensor
        Pressure in hPa (NaN for satellite obs) [n]
    scan_angle : torch.Tensor
        Scan angle in degrees (NaN for conventional obs) [n]
    sat_zenith_angle : torch.Tensor
        Satellite zenith angle in degrees (NaN for conventional obs) [n]
    sol_zenith_angle : torch.Tensor
        Solar zenith angle in degrees (NaN for conventional obs) [n]

    Returns
    -------
    torch.Tensor
        Metadata features [n, 50]
    """
    n_obs = lon.shape[0]
    for name, tensor in [
        ("target_time_sec", target_time_sec),
        ("time", time),
        ("lat", lat),
        ("height", height),
        ("pressure", pressure),
        ("scan_angle", scan_angle),
        ("sat_zenith_angle", sat_zenith_angle),
        ("sol_zenith_angle", sol_zenith_angle),
    ]:
        if tensor.shape[0] != n_obs:
            raise ValueError(f"{name} has length {tensor.shape[0]}, expected {n_obs}")

    out = torch.zeros(
        n_obs, N_METADATA_FEATURES, dtype=torch.float32, device=lon.device
    )
    if n_obs == 0:
        return out

    is_conv = ~torch.isnan(height)
    two_pi = 2 * math.pi

    # Shared: local solar time fourier(2) -> [0:4)
    lst = local_solar_time(lon, time)
    out[:, 0:4] = fourier_features(lst / 24.0 * two_pi, 2)

    # Shared: relative time polynomial -> [4:6)
    target_time_ns = target_time_sec * 1_000_000_000
    dt_days = (time - target_time_ns).float() * 1e-9 / 86400.0
    out[:, 4] = dt_days
    out[:, 5] = dt_days**2

    # Shared: relative time fourier(1) -> [6:8)
    out[:, 6:8] = fourier_features(dt_days, 1)

    # Shared: latitude -> [8:10)
    lat_rad = torch.deg2rad(lat)
    out[:, 8] = torch.sin(lat_rad)
    out[:, 9] = torch.cos(lat_rad)

    # Sat-private [10:30): scan fourier(3) + sat_zen fourier(4) + sol_zen fourier(3)
    is_sat = ~is_conv
    if is_sat.any():
        s = is_sat
        out[s, 10:16] = fourier_features(scan_angle[s] / 50.0 * two_pi, 3)
        out[s, 16:24] = fourier_features(sat_zenith_angle[s] / 90.0 * two_pi, 4)
        out[s, 24:30] = fourier_features(sol_zenith_angle[s] / 180.0 * two_pi, 3)

    # Conv-private [30:50): height fourier(5) + pressure fourier(5)
    if is_conv.any():
        c = is_conv
        h_norm = torch.clamp(height[c] / 60000.0, 0.0, 1.0)
        out[c, 30:40] = fourier_features(h_norm * two_pi, 5)
        p_norm = torch.clamp(pressure[c] / 1100.0, 0.0, 1.0)
        out[c, 40:50] = fourier_features(p_norm * two_pi, 5)

    return out
