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

import struct
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pytest

from earth2studio.data import MetOpAMSUA
from earth2studio.data.metop_amsua import (
    _GRH_SIZE,
    _MDR_RECORD_CLASS,
    _MDR_RECORD_SUBCLASS,
    _MDR_SIZE,
    _MPHR_RECORD_CLASS,
    _NUM_CHANNELS,
    _NUM_FOVS,
    _QUALITY_OFFSET,
    _parse_grh,
    _parse_mphr,
    _parse_native_amsua,
    _radiance_to_bt,
)
from earth2studio.lexicon import MetOpAMSUALexicon


# ---------------------------------------------------------------------------
# Helpers to build synthetic EPS native binary data
# ---------------------------------------------------------------------------
def _build_grh(
    record_class: int, instrument_group: int, record_subclass: int, record_size: int
) -> bytes:
    header = bytearray(20)
    header[0] = record_class
    header[1] = instrument_group
    header[2] = record_subclass
    struct.pack_into(">I", header, 4, record_size)
    return bytes(header)


def _build_mphr(fields: dict[str, str]) -> bytes:
    text = "\n".join(f"{k}={v}" for k, v in fields.items()) + "\n"
    payload = text.encode("ascii")
    grh = _build_grh(_MPHR_RECORD_CLASS, 0, 0, _GRH_SIZE + len(payload))
    return grh + payload


def _build_mdr(
    radiances: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    quality_val: int = 0,
) -> bytes:
    grh = _build_grh(_MDR_RECORD_CLASS, 0, _MDR_RECORD_SUBCLASS, _MDR_SIZE)
    payload = bytearray(_MDR_SIZE - _GRH_SIZE)

    # SCENE_RADIANCE at payload offset 2 (record offset 22)
    # Interleaved: (ch1_fov1, ch2_fov1, ..., ch15_fov1, ch1_fov2, ...)
    default_rad = int(0.01 * 1e7)
    if radiances is not None:
        # radiances should be (30, 15) — FOV-major, interleaved
        scaled = (radiances * 1e7).astype(np.int32)
        struct.pack_into(f">{_NUM_CHANNELS * _NUM_FOVS}i", payload, 2, *scaled.ravel())
    else:
        struct.pack_into(
            f">{_NUM_CHANNELS * _NUM_FOVS}i",
            payload,
            2,
            *([default_rad] * (_NUM_CHANNELS * _NUM_FOVS)),
        )

    # ANGULAR_RELATION at payload offset 1822
    # Interleaved: (solza0, satza0, solazi0, satazi0, solza1, ...)
    ang_vals = []
    for _ in range(_NUM_FOVS):
        ang_vals.extend([1000, 1000, 1000, 1000])  # 10.0 deg each
    struct.pack_into(f">{4 * _NUM_FOVS}h", payload, 1822, *ang_vals)

    # EARTH_LOCATION at payload offset 2062
    # Interleaved: (lat0, lon0, lat1, lon1, ..., lat29, lon29)
    if lat is not None and lon is not None:
        loc_interleaved = np.empty(2 * _NUM_FOVS, dtype=np.float64)
        loc_interleaved[0::2] = lat
        loc_interleaved[1::2] = lon
        scaled_loc = (loc_interleaved * 1e4).astype(np.int32)
        struct.pack_into(f">{2 * _NUM_FOVS}i", payload, 2062, *scaled_loc)
    else:
        loc_interleaved = []
        for i in range(_NUM_FOVS):
            loc_interleaved.extend([0, int(i * 1e4)])  # lat=0, lon=i
        struct.pack_into(f">{2 * _NUM_FOVS}i", payload, 2062, *loc_interleaved)

    # TERRAIN_ELEVATION at payload offset 2362
    struct.pack_into(f">{_NUM_FOVS}h", payload, 2362, *([100] * _NUM_FOVS))

    # QUALITY at payload offset = _QUALITY_OFFSET - _GRH_SIZE
    struct.pack_into(">I", payload, _QUALITY_OFFSET - _GRH_SIZE, quality_val)

    return grh + bytes(payload)


def _build_native_file(spacecraft_id: str = "M01", n_scans: int = 1) -> bytes:
    mphr = _build_mphr(
        {
            "SPACECRAFT_ID": spacecraft_id,
            "SENSING_START": "20250115100000Z",
            "SENSING_END": "20250115100800Z",
        }
    )
    mdrs = b"".join(_build_mdr() for _ in range(n_scans))
    return mphr + mdrs


# ---------------------------------------------------------------------------
# Mock test — exercises __call__ end-to-end without network
# ---------------------------------------------------------------------------
def test_metop_amsua_call_mock(tmp_path):
    nat_data = _build_native_file(spacecraft_id="M01", n_scans=3)
    nat_file = tmp_path / "mock_amsua.nat"
    nat_file.write_bytes(nat_data)

    with patch.object(MetOpAMSUA, "_download_products") as mock_dl:
        mock_dl.return_value = [str(nat_file)]
        ds = MetOpAMSUA(
            time_tolerance=timedelta(hours=24),
            cache=False,
            verbose=False,
        )
        df = ds(datetime(2025, 1, 15, 10), ["amsua"])

        assert list(df.columns) == ds.SCHEMA.names
        assert not df.empty
        assert set(df["variable"].unique()) == {"amsua"}
        assert set(df["class"].unique()) == {"rad"}
        assert set(df["satellite"].unique()) == {"Metop-B"}
        assert df["observation"].notna().all()
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        mock_dl.assert_called_once()


def test_metop_amsua_call_mock_fields_subset(tmp_path):
    nat_data = _build_native_file(n_scans=1)
    nat_file = tmp_path / "mock_amsua.nat"
    nat_file.write_bytes(nat_data)

    with patch.object(MetOpAMSUA, "_download_products") as mock_dl:
        mock_dl.return_value = [str(nat_file)]
        ds = MetOpAMSUA(time_tolerance=timedelta(hours=24), cache=False, verbose=False)

        subset = ["time", "lat", "lon", "observation", "variable"]
        df = ds(datetime(2025, 1, 15, 10), ["amsua"], fields=subset)
        assert list(df.columns) == subset
        assert not df.empty


def test_metop_amsua_call_mock_empty(tmp_path):
    with patch.object(MetOpAMSUA, "_download_products") as mock_dl:
        mock_dl.return_value = []
        ds = MetOpAMSUA(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
        df = ds(datetime(2025, 1, 15, 10), ["amsua"])
        assert df.empty
        assert list(df.columns) == ds.SCHEMA.names


# ---------------------------------------------------------------------------
# Network integration tests (slow, xfail)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time,variable,tol",
    [
        (
            datetime(2025, 1, 15, 11),
            ["amsua"],
            timedelta(hours=1),
        ),
        (
            [datetime(2025, 1, 15, 10), datetime(2025, 1, 15, 11)],
            ["amsua"],
            timedelta(hours=2),
        ),
    ],
)
def test_metop_amsua_fetch(time, variable, tol):
    ds = MetOpAMSUA(time_tolerance=tol, cache=False, verbose=False)
    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_metop_amsua_cache(cache):
    ds = MetOpAMSUA(time_tolerance=timedelta(hours=1), cache=cache, verbose=False)
    df = ds(datetime(2025, 1, 15, 11), ["amsua"])
    assert list(df.columns) == ds.SCHEMA.names


# ---------------------------------------------------------------------------
# Unit tests — exceptions, resolve_fields, binary parser, BT conversion
# ---------------------------------------------------------------------------
def test_metop_amsua_exceptions():
    ds = MetOpAMSUA(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
    with pytest.raises(KeyError):
        ds(datetime(2025, 1, 15, 11), ["invalid_variable"])

    with pytest.raises(KeyError):
        MetOpAMSUA.resolve_fields(["nonexistent"])

    with pytest.raises(TypeError):
        MetOpAMSUA.resolve_fields(pa.schema([pa.field("time", pa.string())]))


def test_metop_amsua_resolve_fields():
    assert MetOpAMSUA.resolve_fields(None) == MetOpAMSUA.SCHEMA
    assert MetOpAMSUA.resolve_fields("time").names == ["time"]
    assert MetOpAMSUA.resolve_fields(["time", "lat"]).names == ["time", "lat"]


def test_parse_grh():
    grh = _build_grh(8, 3, 2, 3464)
    rc, ig, sc, sz = _parse_grh(grh, 0)
    assert (rc, ig, sc, sz) == (8, 3, 2, 3464)


def test_parse_mphr():
    data = _build_mphr({"SPACECRAFT_ID": "M01", "SENSING_START": "20250115100000Z"})
    mphr = _parse_mphr(data)
    assert mphr["SPACECRAFT_ID"] == "M01"
    assert mphr["SENSING_START"] == "20250115100000Z"


def test_radiance_to_bt():
    radiance = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    bt = _radiance_to_bt(radiance, 0)
    assert np.all(np.isfinite(bt)) and np.all(bt > 0)
    assert bt[2] > bt[1] > bt[0]

    bt_zero = _radiance_to_bt(np.array([0.0, -1.0], dtype=np.float64), 5)
    assert np.all(np.isnan(bt_zero))


def test_parse_native_amsua():
    n_channels = MetOpAMSUALexicon.AMSUA_NUM_CHANNELS  # 14 (ch15 excluded)
    # Minimal: MPHR + 1 MDR = 30 FOVs × 14 channels = 420 rows
    data = _build_native_file(n_scans=1)
    df = _parse_native_amsua(data)
    assert len(df) == _NUM_FOVS * n_channels
    assert set(df["satellite"].unique()) == {"Metop-B"}
    assert (df["variable"] == "amsua").all()
    assert (df["class"] == "rad").all()
    assert "quality" in df.columns
    assert (df["quality"] == 0).all()  # default quality_val=0

    # Multiple scans
    data = _build_native_file(spacecraft_id="M03", n_scans=3)
    df = _parse_native_amsua(data)
    assert len(df) == 3 * _NUM_FOVS * n_channels
    assert set(df["satellite"].unique()) == {"Metop-C"}

    # Empty / malformed
    assert _parse_native_amsua(b"").empty
    assert _parse_native_amsua(
        _build_mphr(
            {
                "SPACECRAFT_ID": "M01",
                "SENSING_START": "20250115100000Z",
                "SENSING_END": "20250115100800Z",
            }
        )
    ).empty

    # Non-MDR records are skipped
    ipr = _build_grh(3, 0, 1, 27) + bytes(7)
    data = (
        _build_mphr(
            {
                "SPACECRAFT_ID": "M01",
                "SENSING_START": "20250115100000Z",
                "SENSING_END": "20250115100800Z",
            }
        )
        + ipr
        + _build_mdr()
    )
    df = _parse_native_amsua(data)
    assert len(df) == _NUM_FOVS * n_channels


def test_parse_native_amsua_quality():
    """Verify non-zero quality values propagate to all FOV × channel rows."""
    n_channels = MetOpAMSUALexicon.AMSUA_NUM_CHANNELS
    mphr = _build_mphr(
        {
            "SPACECRAFT_ID": "M01",
            "SENSING_START": "20250115100000Z",
            "SENSING_END": "20250115100800Z",
        }
    )
    # Build file with quality_val=42
    data = mphr + _build_mdr(quality_val=42)
    df = _parse_native_amsua(data)
    assert len(df) == _NUM_FOVS * n_channels
    assert (df["quality"] == 42).all()
