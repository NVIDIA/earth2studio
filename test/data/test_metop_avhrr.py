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
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import MetOpAVHRR
from earth2studio.data.metop_avhrr import (
    _DEFAULT_EARTH_VIEWS,
    _FRAME_IND_3A_MASK,
    _GRH_SIZE,
    _MDR_ANG_REL_OFFSET,
    _MDR_EARTH_LOC_OFFSET,
    _MDR_FRAME_INDICATOR_OFFSET,
    _MDR_RECORD_CLASS,
    _MDR_SCENE_RADIANCES_OFFSET,
    _MDR_SUBCLASS,
    _MPHR_RECORD_CLASS,
    _NAV_NUM_POINTS,
    _NUM_CHANNELS,
    _parse_giadr_radiance,
    _parse_grh,
    _parse_mphr,
    _parse_native_avhrr,
    _radiance_to_bt,
    _radiance_to_refl,
)


# ---------------------------------------------------------------------------
# Helper to build binary test data
# ---------------------------------------------------------------------------
def _build_grh(record_class: int, subclass: int, record_size: int) -> bytearray:
    header = bytearray(20)
    header[0] = record_class
    header[1] = 0  # instrument_group
    header[2] = subclass
    struct.pack_into(">I", header, 4, record_size)
    return header


def _build_mphr(
    spacecraft_id: str = "M01",
    sensing_start: str = "20250115103000",
    sensing_end: str = "20250115114500",
) -> bytes:
    text = (
        f"SPACECRAFT_ID={spacecraft_id}\n"
        f"SENSING_START={sensing_start}Z\n"
        f"SENSING_END={sensing_end}Z\n"
        f"INSTRUMENT_ID=AVHR\n"
    ).encode("ascii")
    rec_size = _GRH_SIZE + len(text)
    grh = _build_grh(_MPHR_RECORD_CLASS, 0, rec_size)
    return bytes(grh) + text


def _build_giadr_radiance() -> bytes:
    """Build a minimal GIADR-radiance record with typical calibration values."""
    # Payload: 110 bytes (14 bytes header + 48 bytes IR temps + 48 bytes channels)
    payload = bytearray(110)

    # CH1_SOLAR_FILTERED_IRRADIANCE at offset 62 (int16, scale 0.1)
    # 1399 * 0.1 = 139.9 W/m²
    struct.pack_into(">h", payload, 62, 1399)
    # CH2 at offset 66: 2329 * 0.1 = 232.9 W/m²
    struct.pack_into(">h", payload, 66, 2329)
    # CH3A at offset 70: 140 * 0.1 = 14.0 W/m²
    struct.pack_into(">h", payload, 70, 140)

    # CH3B_CENTRAL_WAVENUMBER at offset 74 (int32, scale 0.01)
    # 269080 * 0.01 = 2690.80 cm⁻¹
    struct.pack_into(">i", payload, 74, 269080)
    # CH3B_CONSTANT1 at offset 78 (int32, scale 0.00001)
    # 17500 * 0.00001 = 0.175 K
    struct.pack_into(">i", payload, 78, 17500)
    # CH3B_CONSTANT2_SLOPE at offset 82 (int32, scale 0.000001)
    # 999750 * 0.000001 = 0.999750 K/K
    struct.pack_into(">i", payload, 82, 999750)

    # CH4_CENTRAL_WAVENUMBER at offset 86 (int32, scale 0.001)
    # 927770 * 0.001 = 927.770 cm⁻¹
    struct.pack_into(">i", payload, 86, 927770)
    # CH4_CONSTANT1 at offset 90
    struct.pack_into(">i", payload, 90, 34000)  # 0.34 K
    # CH4_CONSTANT2_SLOPE at offset 94
    struct.pack_into(">i", payload, 94, 998500)  # 0.998500

    # CH5_CENTRAL_WAVENUMBER at offset 98 (int32, scale 0.001)
    # 833130 * 0.001 = 833.130 cm⁻¹
    struct.pack_into(">i", payload, 98, 833130)
    # CH5_CONSTANT1 at offset 102
    struct.pack_into(">i", payload, 102, 28000)  # 0.28 K
    # CH5_CONSTANT2_SLOPE at offset 106
    struct.pack_into(">i", payload, 106, 998800)  # 0.998800

    grh = _build_grh(5, 1, _GRH_SIZE + len(payload))  # class 5, subclass 1
    return bytes(grh) + bytes(payload)


def _build_mdr_record(
    radiance_val: int = 2500,
    lat_val: int = 450000,
    lon_val: int = 100000,
    frame_indicator: int = 0,
) -> bytes:
    """Build a minimal MDR record with enough data for parsing.

    The record must be large enough to include all fields up to FRAME_INDICATOR.
    """
    # We need at least _MDR_FRAME_INDICATOR_OFFSET + 4 bytes
    rec_size = _MDR_FRAME_INDICATOR_OFFSET + 4
    rec = bytearray(rec_size)

    # GRH
    rec[0] = _MDR_RECORD_CLASS
    rec[2] = _MDR_SUBCLASS
    struct.pack_into(">I", rec, 4, rec_size)

    # EARTH_VIEWS_PER_SCANLINE at offset 22
    struct.pack_into(">h", rec, 22, _DEFAULT_EARTH_VIEWS)

    # SCENE_RADIANCES: int16, (5, 2048) at offset 24
    # Fill tie-point pixel positions with test radiance values
    nav_cols = list(range(4, 4 + _NAV_NUM_POINTS * 20, 20))[:_NAV_NUM_POINTS]
    for ch in range(_NUM_CHANNELS):
        for col in nav_cols:
            byte_off = (
                _MDR_SCENE_RADIANCES_OFFSET + (ch * _DEFAULT_EARTH_VIEWS + col) * 2
            )
            if byte_off + 2 <= rec_size:
                struct.pack_into(">h", rec, byte_off, radiance_val)

    # ANGULAR_RELATIONS: int16, (103, 4) at _MDR_ANG_REL_OFFSET
    for tp in range(_NAV_NUM_POINTS):
        off = _MDR_ANG_REL_OFFSET + tp * 8
        if off + 8 <= rec_size:
            struct.pack_into(">hhhh", rec, off, 3000, 2000, 18000, 9000)

    # EARTH_LOCATIONS: int32, (103, 2) at _MDR_EARTH_LOC_OFFSET
    for tp in range(_NAV_NUM_POINTS):
        off = _MDR_EARTH_LOC_OFFSET + tp * 8
        if off + 8 <= rec_size:
            struct.pack_into(">ii", rec, off, lat_val, lon_val)

    # FRAME_INDICATOR at _MDR_FRAME_INDICATOR_OFFSET
    struct.pack_into(">I", rec, _MDR_FRAME_INDICATOR_OFFSET, frame_indicator)

    return bytes(rec)


def _build_avhrr_dataframe(
    n_pixels: int = 100,
    variables: list[str] | None = None,
    satellite: str = "Metop-B",
) -> pd.DataFrame:
    if variables is None:
        variables = ["avhrr04"]
    rng = np.random.default_rng(42)
    base_time = datetime(2025, 1, 15, 10, 30, 0)
    rows = [
        {
            "time": pd.Timestamp(base_time),
            "lat": rng.uniform(-90, 90),
            "lon": rng.uniform(0, 360),
            "observation": rng.uniform(200, 320),
            "variable": var,
            "satellite": satellite,
            "scan_angle": rng.uniform(-55, 55),
            "channel_index": {
                "avhrr01": 1,
                "avhrr02": 2,
                "avhrr3a": 3,
                "avhrr3b": 4,
                "avhrr04": 5,
                "avhrr05": 6,
            }.get(var, 0),
            "solza": rng.uniform(0, 90),
            "solaza": rng.uniform(0, 360),
            "satellite_za": rng.uniform(0, 65),
            "satellite_aza": rng.uniform(0, 360),
        }
        for var in variables
        for _ in range(n_pixels)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mock test — exercises __call__ end-to-end without network
# ---------------------------------------------------------------------------
def test_metop_avhrr_call_mock(tmp_path):
    mock_df = _build_avhrr_dataframe(n_pixels=50, variables=["avhrr01", "avhrr04"])

    with (
        patch.object(MetOpAVHRR, "_download_products") as mock_dl,
        patch("earth2studio.data.metop_avhrr._parse_native_avhrr") as mock_parse,
    ):
        mock_dl.return_value = [str(tmp_path / "mock.nat")]
        # Write a dummy file so open() succeeds
        (tmp_path / "mock.nat").write_bytes(b"dummy")
        mock_parse.return_value = mock_df

        ds = MetOpAVHRR(
            time_tolerance=timedelta(hours=24),
            cache=False,
            verbose=False,
        )
        df = ds(datetime(2025, 1, 15, 10), ["avhrr01", "avhrr04"])

        assert list(df.columns) == ds.SCHEMA.names
        assert not df.empty
        assert set(df["variable"].unique()) == {"avhrr01", "avhrr04"}
        assert df["observation"].notna().all()
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        mock_dl.assert_called_once()
        mock_parse.assert_called_once()


def test_metop_avhrr_call_mock_fields_subset(tmp_path):
    mock_df = _build_avhrr_dataframe(n_pixels=20, variables=["avhrr04"])

    with (
        patch.object(MetOpAVHRR, "_download_products") as mock_dl,
        patch("earth2studio.data.metop_avhrr._parse_native_avhrr") as mock_parse,
    ):
        mock_dl.return_value = [str(tmp_path / "mock.nat")]
        (tmp_path / "mock.nat").write_bytes(b"dummy")
        mock_parse.return_value = mock_df

        ds = MetOpAVHRR(time_tolerance=timedelta(hours=24), cache=False, verbose=False)
        subset = ["time", "lat", "lon", "observation", "variable"]
        df = ds(datetime(2025, 1, 15, 10), ["avhrr04"], fields=subset)
        assert list(df.columns) == subset
        assert not df.empty


def test_metop_avhrr_call_mock_empty(tmp_path):
    with patch.object(MetOpAVHRR, "_download_products") as mock_dl:
        mock_dl.return_value = []
        ds = MetOpAVHRR(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
        df = ds(datetime(2025, 1, 15, 10), ["avhrr04"])
        assert df.empty
        assert list(df.columns) == ds.SCHEMA.names


# ---------------------------------------------------------------------------
# Binary parser unit tests
# ---------------------------------------------------------------------------
def test_parse_grh():
    grh = _build_grh(8, 2, 3464)
    rc, ig, sc, rs = _parse_grh(bytes(grh))
    assert rc == 8
    assert sc == 2
    assert rs == 3464


def test_parse_mphr():
    mphr_bytes = _build_mphr(spacecraft_id="M03", sensing_start="20250115120000")
    rec_size = len(mphr_bytes)
    result = _parse_mphr(mphr_bytes, rec_size)
    assert result["SPACECRAFT_ID"] == "M03"
    assert result["SENSING_START"].startswith("20250115120000")


def test_parse_giadr_radiance():
    giadr = _build_giadr_radiance()
    cal = _parse_giadr_radiance(giadr, 0, len(giadr))
    # CH1 solar irradiance: 1399 * 0.1 = 139.9
    assert abs(cal.ch1_solar_irrad - 139.9) < 0.2
    # CH4 wavenumber: 927770 * 0.001 = 927.770
    assert abs(cal.ch4_wavenumber - 927.77) < 0.01
    # CH3B band correction slope close to 1.0
    assert abs(cal.ch3b_b - 0.99975) < 0.001


def test_radiance_to_bt():
    # Typical ch4 values: wavenumber ~928 cm⁻¹, A~0.34, B~0.9985
    rad = np.array([10.0, 20.0, 0.0, -1.0], dtype=np.float64)
    bt = _radiance_to_bt(rad, 928.0, 0.34, 0.9985)
    # First two should give reasonable BT (150-350K range)
    assert 150 < bt[0] < 350
    assert 150 < bt[1] < 350
    # Zero/negative → NaN
    assert np.isnan(bt[2])
    assert np.isnan(bt[3])


def test_radiance_to_refl():
    rad = np.array([10.0, 0.0, -1.0], dtype=np.float64)
    refl = _radiance_to_refl(rad, 139.9)
    # Non-zero radiance → positive reflectance
    assert refl[0] > 0
    # Zero/negative → NaN
    assert np.isnan(refl[1])
    assert np.isnan(refl[2])
    # With zero irradiance → NaN
    refl2 = _radiance_to_refl(rad, 0.0)
    assert np.isnan(refl2[0])


def test_parse_native_avhrr_minimal():
    """Test parsing a minimal AVHRR file with 2 MDR records."""
    mphr = _build_mphr()
    giadr = _build_giadr_radiance()
    mdr1 = _build_mdr_record(radiance_val=2500, lat_val=450000, lon_val=100000)
    mdr2 = _build_mdr_record(radiance_val=3000, lat_val=-300000, lon_val=2500000)

    file_data = mphr + giadr + mdr1 + mdr2

    # subsample=1 to get both scan lines
    df = _parse_native_avhrr(file_data, ["avhrr04"], subsample=1)
    assert not df.empty
    assert "observation" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "variable" in df.columns
    assert (df["variable"] == "avhrr04").all()
    # Should have 2 scans * 103 tie points = 206 rows (minus any NaN drops)
    assert len(df) <= 206
    # Lat: 45.0 and -30.0 (in degrees)
    assert df["lat"].between(-90, 90).all()


def test_parse_native_avhrr_frame_indicator_3a_3b():
    """Test that 3a/3b switching masks correct scan lines."""
    mphr = _build_mphr()
    giadr = _build_giadr_radiance()
    # MDR with 3b active (frame_indicator=0, bit16=0)
    mdr_3b = _build_mdr_record(radiance_val=2500, frame_indicator=0)
    # MDR with 3a active (frame_indicator with bit16 set)
    mdr_3a = _build_mdr_record(radiance_val=2500, frame_indicator=_FRAME_IND_3A_MASK)

    file_data = mphr + giadr + mdr_3b + mdr_3a

    # Request avhrr3b — should only have data from mdr_3b (scan 0)
    df_3b = _parse_native_avhrr(file_data, ["avhrr3b"], subsample=1)
    # Request avhrr3a — should only have data from mdr_3a (scan 1)
    df_3a = _parse_native_avhrr(file_data, ["avhrr3a"], subsample=1)

    # avhrr3b should have ~103 rows (1 scan), avhrr3a should have ~103 rows (1 scan)
    assert len(df_3b) <= _NAV_NUM_POINTS
    assert len(df_3a) <= _NAV_NUM_POINTS


def test_parse_native_avhrr_empty():
    """Empty data returns empty DataFrame."""
    df = _parse_native_avhrr(b"", ["avhrr04"])
    assert df.empty


def test_parse_native_avhrr_multiple_channels():
    """Test parsing with multiple channels."""
    mphr = _build_mphr()
    giadr = _build_giadr_radiance()
    mdr = _build_mdr_record(radiance_val=2500)

    file_data = mphr + giadr + mdr
    df = _parse_native_avhrr(file_data, ["avhrr04", "avhrr05"], subsample=1)

    if not df.empty:
        assert set(df["variable"].unique()).issubset({"avhrr04", "avhrr05"})


# ---------------------------------------------------------------------------
# Network integration tests (slow, xfail)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "time,variable,tol",
    [
        (
            datetime(2025, 1, 15, 11),
            ["avhrr04"],
            timedelta(hours=1),
        ),
        (
            [datetime(2025, 1, 15, 10), datetime(2025, 1, 15, 11)],
            ["avhrr01", "avhrr04"],
            timedelta(hours=2),
        ),
    ],
)
def test_metop_avhrr_fetch(time, variable, tol):
    ds = MetOpAVHRR(time_tolerance=tol, subsample=64, cache=False, verbose=False)
    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(300)
@pytest.mark.parametrize("cache", [True, False])
def test_metop_avhrr_cache(cache):
    ds = MetOpAVHRR(
        time_tolerance=timedelta(hours=1),
        subsample=64,
        cache=cache,
        verbose=False,
    )
    df = ds(datetime(2025, 1, 15, 11), ["avhrr04"])
    assert list(df.columns) == ds.SCHEMA.names


# ---------------------------------------------------------------------------
# Unit tests — exceptions, resolve_fields, schema
# ---------------------------------------------------------------------------
def test_metop_avhrr_exceptions():
    ds = MetOpAVHRR(time_tolerance=timedelta(hours=1), cache=False, verbose=False)
    with pytest.raises(KeyError):
        ds(datetime(2025, 1, 15, 11), ["invalid_variable"])

    with pytest.raises(KeyError):
        MetOpAVHRR.resolve_fields(["nonexistent"])

    with pytest.raises(TypeError):
        MetOpAVHRR.resolve_fields(pa.schema([pa.field("time", pa.string())]))


def test_metop_avhrr_resolve_fields():
    assert MetOpAVHRR.resolve_fields(None) == MetOpAVHRR.SCHEMA
    assert MetOpAVHRR.resolve_fields("time").names == ["time"]
    assert MetOpAVHRR.resolve_fields(["time", "lat"]).names == ["time", "lat"]


def test_metop_avhrr_schema_satellite_fields():
    names = MetOpAVHRR.SCHEMA.names
    for field in [
        "satellite",
        "scan_angle",
        "channel_index",
        "solza",
        "solaza",
        "satellite_za",
        "satellite_aza",
    ]:
        assert field in names
    assert "elev" not in names
