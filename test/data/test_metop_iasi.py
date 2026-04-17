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

from earth2studio.data import MetOpIASI
from earth2studio.data.metop_iasi import (
    _GIADR_RECORD_CLASS,
    _GIADR_SF_SUBCLASS,
    _GRH_SIZE,
    _MDR_INSTRUMENT_GROUP,
    _MDR_RECORD_CLASS,
    _MDR_RECORD_SUBCLASS,
    _MPHR_RECORD_CLASS,
    _NUM_CHANNELS,
    _NUM_CHANNELS_ALLOC,
    _NUM_EFOVS,
    _NUM_IFOVS,
    _build_channel_scale_array,
    _compute_mdr_field_offsets,
    _decode_eps_datetime,
    _parse_giadr_scale_factors,
    _parse_grh,
    _parse_mphr,
    _parse_native_iasi,
)
from earth2studio.data.utils import PLANCK_C1, PLANCK_C2, radiance_to_bt
from earth2studio.lexicon import MetOpIASILexicon


# ---------------------------------------------------------------------------
# Helpers to build synthetic EPS native binary data for IASI
# ---------------------------------------------------------------------------
def _build_grh(
    record_class: int,
    instrument_group: int,
    record_subclass: int,
    record_size: int,
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


def _build_giadr_sf(
    nb_bands: int = 3,
    first_ch: list[int] | None = None,
    last_ch: list[int] | None = None,
    scale_factors: list[int] | None = None,
) -> bytes:
    """Build a GIADR Scale Factors record.

    Default bands use absolute sample numbers (matching nsfirst=2581):
    Band 1: samples 2581-4577 (channels 0-1996), SF=5
    Band 2: samples 4578-6696 (channels 1997-4115), SF=5
    Band 3: samples 6697-11041 (channels 4116-8460), SF=5

    Scale factors are positive integers used as negative exponents:
    physical = raw × 10^(-sf).
    """
    if first_ch is None:
        first_ch = [2581, 4578, 6697]
    if last_ch is None:
        last_ch = [4577, 6696, 11041]
    if scale_factors is None:
        scale_factors = [5, 5, 5]

    # Payload: nb_bands(int16) + first[10](int16) + last[10](int16) + sf[10](int16) + iis_sf(int16)
    payload_size = 2 + 20 + 20 + 20 + 2
    payload = bytearray(payload_size)
    struct.pack_into(">h", payload, 0, nb_bands)
    for i in range(10):
        val = first_ch[i] if i < len(first_ch) else 0
        struct.pack_into(">h", payload, 2 + i * 2, val)
    for i in range(10):
        val = last_ch[i] if i < len(last_ch) else 0
        struct.pack_into(">h", payload, 22 + i * 2, val)
    for i in range(10):
        val = scale_factors[i] if i < len(scale_factors) else 0
        struct.pack_into(">h", payload, 42 + i * 2, val)
    # IDefScaleIISScaleFactor: int16
    struct.pack_into(">h", payload, 62, -5)

    grh = _build_grh(
        _GIADR_RECORD_CLASS,
        _MDR_INSTRUMENT_GROUP,
        _GIADR_SF_SUBCLASS,
        _GRH_SIZE + payload_size,
    )
    return grh + bytes(payload)


def _build_mdr(
    lat_deg: float = 45.0,
    lon_deg: float = 10.0,
    quality_val: int = 0,
    radiance_raw: int = 5000,
) -> bytes:
    """Build a synthetic IASI MDR record.

    The binary layout follows _compute_mdr_field_offsets exactly:
    GRH(20) + DEGRADED(2) + GEPSIasiMode(4) + GEPSOPSProcessingMode(4) +
    GEPSIdConf(32) + GEPSLocIasiAvhrr_IASI(1200) + GEPSLocIasiAvhrr_IIS(7500) +
    OBT(180) + OnboardUTC(180) + GEPSDatIasi(180) + GIsfLin(8) + GIsfCol(8) +
    GIsfPds1-4(32) + GEPS_CCD(30) + GEPS_SP(120) + GIrcImage(245760) +
    GQisFlagQual(360) + GQisFlagQualDetailed(240) + QualIndices(5*5+4+4=33) +
    GGeoSondLoc(960) + GGeoSondAnglesMETOP(960) + GGeoIISAnglesMETOP(6000) +
    GGeoSondAnglesSUN(960) + GGeoIISAnglesSUN(6000) + GGeoIISLoc(6000) +
    EARTH_SATELLITE_DISTANCE(4) + IDefSpectDWn1b(5) + IDefNsfirst1b(4) +
    IDefNslast1b(4) + GS1cSpect(2088000) + trailing(...)
    """
    # Compute payload size — must match _compute_mdr_field_offsets walk
    # Payload after GRH:
    payload_size = (
        2  # DEGRADED_INST + DEGRADED_PROC
        + 4  # GEPSIasiMode
        + 4  # GEPSOPSProcessingMode
        + 32  # GEPSIdConf
        + 1200  # GEPSLocIasiAvhrr_IASI
        + 7500  # GEPSLocIasiAvhrr_IIS
        + 180  # OBT
        + 180  # OnboardUTC
        + 180  # GEPSDatIasi
        + 8  # GIsfLinOrigin
        + 8  # GIsfColOrigin
        + 32  # GIsfPds1-4
        + 30  # GEPS_CCD
        + 120  # GEPS_SP
        + 245760  # GIrcImage
        + 360  # GQisFlagQual
        + 240  # GQisFlagQualDetailed
        + 33  # Quality indices (5*5+4+4)
        + 960  # GGeoSondLoc
        + 960  # GGeoSondAnglesMETOP
        + 6000  # GGeoIISAnglesMETOP
        + 960  # GGeoSondAnglesSUN
        + 6000  # GGeoIISAnglesSUN
        + 6000  # GGeoIISLoc
        + 4  # EARTH_SATELLITE_DISTANCE
        + 5  # IDefSpectDWn1b
        + 4  # IDefNsfirst1b
        + 4  # IDefNslast1b
        + _NUM_EFOVS * _NUM_IFOVS * _NUM_CHANNELS_ALLOC * 2  # GS1cSpect
    )

    record_size = _GRH_SIZE + payload_size
    grh = _build_grh(
        _MDR_RECORD_CLASS,
        _MDR_INSTRUMENT_GROUP,
        _MDR_RECORD_SUBCLASS,
        record_size,
    )
    payload = bytearray(payload_size)

    # Walk through payload filling key fields
    pos = 0

    # DEGRADED (2 bytes)
    pos += 2
    # GEPSIasiMode (4), GEPSOPSProcessingMode (4), GEPSIdConf (32)
    pos += 40
    # GEPSLocIasiAvhrr_IASI (1200), GEPSLocIasiAvhrr_IIS (7500)
    pos += 8700
    # OBT (180), OnboardUTC (180)
    pos += 360

    # GEPSDatIasi: EPSdatetime[30] — fill with 2025-01-15T10:00:00Z
    # days since 2000-01-01 = 9145 (2025-01-15), ms_of_day = 36_000_000 (10:00:00)
    days_since_epoch = (datetime(2025, 1, 15) - datetime(2000, 1, 1)).days
    ms_of_day = 10 * 3600 * 1000  # 10:00:00 UTC
    for efov in range(_NUM_EFOVS):
        off = pos + efov * 6
        struct.pack_into(">H", payload, off, days_since_epoch)
        struct.pack_into(">I", payload, off + 2, ms_of_day)
    pos += 180

    # GIsf* fields (8+8+32=48) + GEPS_CCD (30) + GEPS_SP (120) = 198
    pos += 198
    # GIrcImage (245760)
    pos += 245760
    # GQisFlagQual (360)
    pos += 360

    # GQisFlagQualDetailed: uint16[30][4]
    for i in range(_NUM_EFOVS * _NUM_IFOVS):
        struct.pack_into(">H", payload, pos + i * 2, quality_val)
    pos += 240

    # Quality indices (33 bytes)
    pos += 33

    # GGeoSondLoc: int32[30][4][2], SF=1e6 — [efov][ifov][lon=0, lat=1]
    lat_raw = int(lat_deg * 1e6)
    lon_raw = int(lon_deg * 1e6)
    for efov in range(_NUM_EFOVS):
        for ifov in range(_NUM_IFOVS):
            idx = (efov * _NUM_IFOVS + ifov) * 2
            struct.pack_into(">i", payload, pos + idx * 4, lon_raw)
            struct.pack_into(">i", payload, pos + (idx + 1) * 4, lat_raw)
    pos += 960

    # GGeoSondAnglesMETOP: int32[30][4][2], SF=1e6 — [zen=0, azi=1]
    ang_raw = int(20.0 * 1e6)  # 20 degrees
    for i in range(_NUM_EFOVS * _NUM_IFOVS * 2):
        struct.pack_into(">i", payload, pos + i * 4, ang_raw)
    pos += 960

    # GGeoIISAnglesMETOP (6000)
    pos += 6000

    # GGeoSondAnglesSUN: int32[30][4][2], SF=1e6
    for i in range(_NUM_EFOVS * _NUM_IFOVS * 2):
        struct.pack_into(">i", payload, pos + i * 4, ang_raw)
    pos += 960

    # GGeoIISAnglesSUN (6000), GGeoIISLoc (6000)
    pos += 12000
    # EARTH_SATELLITE_DISTANCE (4)
    pos += 4

    # IDefSpectDWn1b: VSFInt (1 byte scale + 4 byte value)
    # Spectral sampling: 25 m⁻¹ = 0.25 cm⁻¹
    # Encode as scale=0, value=25
    struct.pack_into(">b", payload, pos, 0)
    struct.pack_into(">i", payload, pos + 1, 25)
    pos += 5

    # IDefNsfirst1b: int32 = 2581 (645 cm⁻¹ / 0.25 cm⁻¹ ≈ 2580, rounded to 2581)
    struct.pack_into(">i", payload, pos, 2581)
    pos += 4

    # IDefNslast1b: int32 = 11041
    struct.pack_into(">i", payload, pos, 11041)
    pos += 4

    # GS1cSpect: int16[30][4][8700]
    # Fill selected channels with a known positive value
    for efov in range(_NUM_EFOVS):
        for ifov in range(_NUM_IFOVS):
            spect_base = pos + (efov * _NUM_IFOVS + ifov) * _NUM_CHANNELS_ALLOC * 2
            # Fill first 8461 channels with radiance_raw
            for ch in range(_NUM_CHANNELS):
                struct.pack_into(">h", payload, spect_base + ch * 2, radiance_raw)

    return grh + bytes(payload)


def _build_native_file(
    spacecraft_id: str = "M01",
    n_scans: int = 1,
    lat_deg: float = 45.0,
    lon_deg: float = 10.0,
    quality_val: int = 0,
    radiance_raw: int = 5000,
) -> bytes:
    mphr = _build_mphr(
        {
            "SPACECRAFT_ID": spacecraft_id,
            "SENSING_START": "20250115100000Z",
            "SENSING_END": "20250115100800Z",
        }
    )
    giadr = _build_giadr_sf()
    mdrs = b"".join(
        _build_mdr(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            quality_val=quality_val,
            radiance_raw=radiance_raw,
        )
        for _ in range(n_scans)
    )
    return mphr + giadr + mdrs


# ---------------------------------------------------------------------------
# Mock tests — exercises __call__ end-to-end without network
# ---------------------------------------------------------------------------
def test_metop_iasi_call_mock(tmp_path):
    nat_data = _build_native_file(spacecraft_id="M01", n_scans=1)
    nat_file = tmp_path / "mock_iasi.nat"
    nat_file.write_bytes(nat_data)

    with patch.object(MetOpIASI, "_download_products") as mock_dl:
        mock_dl.return_value = [str(nat_file)]
        ds = MetOpIASI(
            channel_indices=np.array([1, 101, 501]),
            time_tolerance=timedelta(hours=24),
            cache=False,
            verbose=False,
        )
        df = ds(datetime(2025, 1, 15, 10), ["iasi"])

        assert list(df.columns) == ds.SCHEMA.names
        assert not df.empty
        assert set(df["variable"].unique()) == {"iasi"}
        assert set(df["class"].unique()) == {"rad"}
        assert set(df["satellite"].unique()) == {"metop-b"}
        assert df["observation"].notna().all()
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        mock_dl.assert_called_once()


def test_metop_iasi_call_mock_fields_subset(tmp_path):
    nat_data = _build_native_file(n_scans=1)
    nat_file = tmp_path / "mock_iasi.nat"
    nat_file.write_bytes(nat_data)

    with patch.object(MetOpIASI, "_download_products") as mock_dl:
        mock_dl.return_value = [str(nat_file)]
        ds = MetOpIASI(
            channel_indices=np.array([1, 1001]),
            time_tolerance=timedelta(hours=24),
            cache=False,
            verbose=False,
        )

        subset = ["time", "lat", "lon", "observation", "variable"]
        df = ds(datetime(2025, 1, 15, 10), ["iasi"], fields=subset)
        assert list(df.columns) == subset
        assert not df.empty


def test_metop_iasi_call_mock_empty(tmp_path):
    with patch.object(MetOpIASI, "_download_products") as mock_dl:
        mock_dl.return_value = []
        ds = MetOpIASI(time_tolerance=timedelta(minutes=5), cache=False, verbose=False)
        df = ds(datetime(2025, 1, 15, 10), ["iasi"])
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
            ["iasi"],
            timedelta(minutes=5),
        ),
        (
            [datetime(2025, 1, 15, 10), datetime(2025, 1, 15, 11)],
            ["iasi"],
            timedelta(minutes=5),
        ),
    ],
)
def test_metop_iasi_fetch(time, variable, tol):
    ds = MetOpIASI(satellite="metop-c", time_tolerance=tol, cache=False, verbose=False)
    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_metop_iasi_cache(cache):
    ds = MetOpIASI(
        satellite="metop-c",
        time_tolerance=timedelta(minutes=5),
        cache=cache,
        verbose=False,
    )
    df = ds(datetime(2025, 1, 15, 11), ["iasi"])
    assert list(df.columns) == ds.SCHEMA.names


# ---------------------------------------------------------------------------
# Unit tests — exceptions, resolve_fields, binary parsers
# ---------------------------------------------------------------------------
def test_metop_iasi_exceptions():
    ds = MetOpIASI(time_tolerance=timedelta(minutes=5), cache=False, verbose=False)
    with pytest.raises(KeyError):
        ds(datetime(2025, 1, 15, 11), ["invalid_variable"])

    with pytest.raises(KeyError):
        MetOpIASI.resolve_fields(["nonexistent"])

    with pytest.raises(TypeError):
        MetOpIASI.resolve_fields(pa.schema([pa.field("time", pa.string())]))


def test_metop_iasi_resolve_fields():
    assert MetOpIASI.resolve_fields(None) == MetOpIASI.SCHEMA
    assert MetOpIASI.resolve_fields("time").names == ["time"]
    assert MetOpIASI.resolve_fields(["time", "lat"]).names == ["time", "lat"]


def test_parse_grh():
    grh = _build_grh(8, 8, 2, 2723000)
    rc, ig, sc, sz = _parse_grh(grh, 0)
    assert (rc, ig, sc, sz) == (8, 8, 2, 2723000)


def test_parse_mphr():
    data = _build_mphr({"SPACECRAFT_ID": "M03", "SENSING_START": "20250115100000Z"})
    mphr = _parse_mphr(data)
    assert mphr["SPACECRAFT_ID"] == "M03"
    assert mphr["SENSING_START"] == "20250115100000Z"


def test_decode_eps_datetime():
    buf = bytearray(6)
    # 2025-01-15 = 9145 days since 2000-01-01
    days = (datetime(2025, 1, 15) - datetime(2000, 1, 1)).days
    struct.pack_into(">H", buf, 0, days)
    # 10:00:00 = 36_000_000 ms
    struct.pack_into(">I", buf, 2, 36_000_000)
    dt = _decode_eps_datetime(bytes(buf), 0)
    assert dt == datetime(2025, 1, 15, 10, 0, 0)


def test_radiance_to_bt():
    # Known physical values: wavenumber 900 cm⁻¹ (~11 µm, window channel)
    wn = np.array([900.0], dtype=np.float64)
    # A typical radiance for 280K at 900 cm⁻¹
    # BT = C2 * nu / ln(1 + C1 * nu^3 / L) → solve for L
    bt_expected = 280.0
    l_expected = PLANCK_C1 * 900.0**3 / (np.exp(PLANCK_C2 * 900.0 / bt_expected) - 1.0)
    rad = np.array([[l_expected]], dtype=np.float64)
    bt = radiance_to_bt(rad, wn)
    assert np.isfinite(bt[0, 0])
    np.testing.assert_allclose(bt[0, 0], bt_expected, rtol=1e-6)

    # Zero/negative radiance → NaN
    bt_zero = radiance_to_bt(np.array([[0.0, -1.0]], dtype=np.float64), wn)
    assert np.all(np.isnan(bt_zero))

    # Higher radiance → higher BT
    rads = np.array([[0.01, 0.05, 0.1]], dtype=np.float64)
    bt_multi = radiance_to_bt(rads, wn)
    assert np.all(np.isfinite(bt_multi))
    assert bt_multi[0, 0] < bt_multi[0, 1] < bt_multi[0, 2]


def test_parse_giadr_scale_factors():
    giadr = _build_giadr_sf(
        nb_bands=3,
        first_ch=[2581, 4578, 6697],
        last_ch=[4577, 6696, 11041],
        scale_factors=[5, 5, 5],
    )
    # Parse starting at offset 0 (this is a standalone record)
    first, last, sf, nb = _parse_giadr_scale_factors(giadr, 0)
    assert nb == 3
    np.testing.assert_array_equal(first, [2581, 4578, 6697])
    np.testing.assert_array_equal(last, [4577, 6696, 11041])
    np.testing.assert_array_equal(sf, [5, 5, 5])


def test_build_channel_scale_array():
    # Absolute sample numbers with nsfirst=2581
    first = np.array([2581, 4578, 6697])
    last = np.array([4577, 6696, 11041])
    sf = np.array([5, 5, 5])  # Positive: used as negative exponent → 10^(-5)
    scales = _build_channel_scale_array(first, last, sf, nsfirst=2581)
    assert scales.shape == (_NUM_CHANNELS,)
    assert scales[0] == 1e-5  # Channel 0 → sample 2581 → Band 1
    assert scales[1996] == 1e-5  # Last channel of band 1
    assert scales[1997] == 1e-5  # First channel of band 2
    assert scales[8460] == 1e-5  # Last valid channel


def test_parse_native_iasi():
    # 1 scan → 30 EFOVs × 4 IFOVs = 120 obs, extract 3 channels → 360 rows
    ch_idx = np.array([1, 101, 501])
    data = _build_native_file(n_scans=1)
    df = _parse_native_iasi(data, channel_indices=ch_idx)
    expected_rows = _NUM_EFOVS * _NUM_IFOVS * len(ch_idx)
    assert len(df) == expected_rows
    assert set(df["satellite"].unique()) == {"metop-b"}
    assert (df["variable"] == "iasi").all()
    assert (df["class"] == "rad").all()
    assert "quality" in df.columns
    assert (df["quality"] == 0).all()

    # Channel indices should be 1-based
    assert set(df["channel_index"].unique()) == {1, 101, 501}


def test_parse_native_iasi_multiple_scans():
    ch_idx = np.array([1, 1001, 4001])
    data = _build_native_file(spacecraft_id="M03", n_scans=3)
    df = _parse_native_iasi(data, channel_indices=ch_idx)
    expected_rows = 3 * _NUM_EFOVS * _NUM_IFOVS * len(ch_idx)
    assert len(df) == expected_rows
    assert set(df["satellite"].unique()) == {"metop-c"}


def test_parse_native_iasi_empty():
    assert _parse_native_iasi(b"").empty
    # MPHR only (no GIADR or MDR)
    assert _parse_native_iasi(
        _build_mphr(
            {
                "SPACECRAFT_ID": "M01",
                "SENSING_START": "20250115100000Z",
                "SENSING_END": "20250115100800Z",
            }
        )
    ).empty


def test_parse_native_iasi_quality():
    ch_idx = np.array([1])
    data = _build_native_file(quality_val=42)
    df = _parse_native_iasi(data, channel_indices=ch_idx)
    assert not df.empty
    assert (df["quality"] == 42).all()


def test_parse_native_iasi_geolocation():
    ch_idx = np.array([1])
    data = _build_native_file(lat_deg=45.0, lon_deg=10.0)
    df = _parse_native_iasi(data, channel_indices=ch_idx)
    # All observations should have lat=45, lon=10
    np.testing.assert_allclose(df["lat"].values, 45.0, atol=0.01)
    np.testing.assert_allclose(df["lon"].values, 10.0, atol=0.01)


def test_parse_native_iasi_negative_longitude():
    ch_idx = np.array([1])
    # Negative longitude should be converted to [0, 360]
    data = _build_native_file(lat_deg=45.0, lon_deg=-10.0)
    df = _parse_native_iasi(data, channel_indices=ch_idx)
    np.testing.assert_allclose(df["lon"].values, 350.0, atol=0.01)


def test_compute_mdr_field_offsets():
    data = _build_native_file(n_scans=1)
    # Find MDR offset by scanning records
    offset = 0
    mdr_offset = None
    while offset + _GRH_SIZE <= len(data):
        rc, ig, sc, sz = _parse_grh(data, offset)
        if rc == _MDR_RECORD_CLASS and ig == _MDR_INSTRUMENT_GROUP:
            mdr_offset = offset
            break
        offset += sz
    assert mdr_offset is not None

    offsets = _compute_mdr_field_offsets(data, mdr_offset)
    assert "GEPSDatIasi" in offsets
    assert "GGeoSondLoc" in offsets
    assert "GGeoSondAnglesMETOP" in offsets
    assert "GGeoSondAnglesSUN" in offsets
    assert "GQisFlagQualDetailed" in offsets
    assert "IDefSpectDWn1b" in offsets
    assert "IDefNsfirst1b" in offsets
    assert "GS1cSpect" in offsets
    # Offsets should be increasing
    assert offsets["GEPSDatIasi"] < offsets["GGeoSondLoc"]
    assert offsets["GGeoSondLoc"] < offsets["GS1cSpect"]


def test_parse_native_iasi_skip_non_mdr():
    # Insert an IPR record between GIADR and MDR — it should be skipped
    mphr = _build_mphr(
        {
            "SPACECRAFT_ID": "M01",
            "SENSING_START": "20250115100000Z",
            "SENSING_END": "20250115100800Z",
        }
    )
    giadr = _build_giadr_sf()
    ipr = _build_grh(3, 0, 1, 27) + bytes(7)  # IPR record
    mdr = _build_mdr()
    data = mphr + giadr + ipr + mdr
    ch_idx = np.array([1])
    df = _parse_native_iasi(data, channel_indices=ch_idx)
    assert len(df) == _NUM_EFOVS * _NUM_IFOVS


def test_metop_iasi_default_channels():
    assert hasattr(MetOpIASI, "DEFAULT_CHANNELS")
    assert len(MetOpIASI.DEFAULT_CHANNELS) == 174
    assert all(1 <= ch <= _NUM_CHANNELS for ch in MetOpIASI.DEFAULT_CHANNELS)


def test_lexicon_vocab():
    assert "iasi" in MetOpIASILexicon.VOCAB
    key, modifier = MetOpIASILexicon.get_item("iasi")
    assert key == "brightnessTemperature"
    # Identity modifier
    val = np.array([1.0, 2.0])
    np.testing.assert_array_equal(modifier(val), val)


def test_lexicon_invalid_variable():
    with pytest.raises(KeyError):
        MetOpIASILexicon["nonexistent"]
