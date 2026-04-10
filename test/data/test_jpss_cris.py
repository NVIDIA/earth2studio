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

import pathlib
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pyarrow as pa
import pytest

from earth2studio.data import JPSS_CRIS
from earth2studio.data.jpss_cris import (
    _CRIS_GSI_SENSOR_CHAN,
    _CRIS_WAVENUMBER,
    _hamming_apodize,
)


# ---------------------------------------------------------------------------
# Unit tests for channel mapping
# ---------------------------------------------------------------------------
def test_guard_channel_layout():
    """Verify the 2+2 guard channel layout per band."""
    # Each band has 2 low-end guards (sensor_chan=0) and 2 high-end guards
    # LWIR: indices 0,1 = guard; 2..714 = science (1..713); 715,716 = guard
    assert _CRIS_GSI_SENSOR_CHAN[0] == 0
    assert _CRIS_GSI_SENSOR_CHAN[1] == 0
    assert _CRIS_GSI_SENSOR_CHAN[2] == 1
    assert _CRIS_GSI_SENSOR_CHAN[714] == 713
    assert _CRIS_GSI_SENSOR_CHAN[715] == 0
    assert _CRIS_GSI_SENSOR_CHAN[716] == 0

    # MWIR: indices 717,718 = guard; 719..1583 = science (714..1578); 1584,1585 = guard
    assert _CRIS_GSI_SENSOR_CHAN[717] == 0
    assert _CRIS_GSI_SENSOR_CHAN[718] == 0
    assert _CRIS_GSI_SENSOR_CHAN[719] == 714
    assert _CRIS_GSI_SENSOR_CHAN[1583] == 1578
    assert _CRIS_GSI_SENSOR_CHAN[1584] == 0
    assert _CRIS_GSI_SENSOR_CHAN[1585] == 0

    # SWIR: indices 1586,1587 = guard; 1588..2220 = science (1579..2211); 2221,2222 = guard
    assert _CRIS_GSI_SENSOR_CHAN[1586] == 0
    assert _CRIS_GSI_SENSOR_CHAN[1587] == 0
    assert _CRIS_GSI_SENSOR_CHAN[1588] == 1579
    assert _CRIS_GSI_SENSOR_CHAN[2220] == 2211
    assert _CRIS_GSI_SENSOR_CHAN[2221] == 0
    assert _CRIS_GSI_SENSOR_CHAN[2222] == 0

    # Total guard channels
    assert (_CRIS_GSI_SENSOR_CHAN == 0).sum() == 12
    # Total science channels
    assert (_CRIS_GSI_SENSOR_CHAN > 0).sum() == 2211


def test_wavenumber_grid_starts():
    """Verify wavenumber grid starts at correct values including guard channels."""
    # LWIR starts at 648.75 (2 guard channels below 650.0)
    np.testing.assert_allclose(_CRIS_WAVENUMBER[0], 648.75)
    np.testing.assert_allclose(_CRIS_WAVENUMBER[1], 649.375)
    np.testing.assert_allclose(_CRIS_WAVENUMBER[2], 650.0)  # first science channel

    # MWIR starts at 1208.75
    np.testing.assert_allclose(_CRIS_WAVENUMBER[717], 1208.75)
    np.testing.assert_allclose(_CRIS_WAVENUMBER[719], 1210.0)  # first science channel

    # SWIR starts at 2153.75
    np.testing.assert_allclose(_CRIS_WAVENUMBER[1586], 2153.75)
    np.testing.assert_allclose(_CRIS_WAVENUMBER[1588], 2155.0)  # first science channel


# ---------------------------------------------------------------------------
# Unit tests for apodization
# ---------------------------------------------------------------------------
def test_hamming_apodize_shape():
    """Verify _hamming_apodize trims guard channels correctly."""
    rng = np.random.default_rng(0)
    # Single spectrum
    spec_1d = rng.uniform(1, 50, 2223).astype(np.float32)
    apod_1d = _hamming_apodize(spec_1d)
    assert apod_1d.shape == (2211,)

    # Batch of spectra
    spec_2d = rng.uniform(1, 50, (5, 2223)).astype(np.float32)
    apod_2d = _hamming_apodize(spec_2d)
    assert apod_2d.shape == (5, 2211)


def test_hamming_apodize_constant_spectrum():
    """For a spatially-constant spectrum the Hamming kernel is an identity
    (0.23+0.54+0.23=1.0), so the output should equal the input at science
    channels."""
    val = 42.0
    spec = np.full(2223, val, dtype=np.float64)
    apod = _hamming_apodize(spec)
    np.testing.assert_allclose(apod, val, atol=1e-12)


def test_hamming_apodize_kernel_weights():
    """Verify the 3-tap [0.23, 0.54, 0.23] kernel is applied correctly on
    an interior LWIR channel."""
    spec = np.zeros(2223, dtype=np.float64)
    # Set a single spike at LWIR channel 100 (interior, far from edge/guard).
    # With 2 guard channels at the low end trimmed, input index 100 maps to
    # apodized output index 98.
    spec[100] = 1.0
    apod = _hamming_apodize(spec)
    # The spike should spread to neighbours (output indices 97, 98, 99)
    assert abs(apod[97] - 0.23) < 1e-14
    assert abs(apod[98] - 0.54) < 1e-14
    assert abs(apod[99] - 0.23) < 1e-14
    # All other science channels should be zero
    mask = np.ones(2211, dtype=bool)
    mask[97:100] = False
    np.testing.assert_allclose(apod[mask], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Helper – generate a synthetic HDF5 file pair (SDR + GEO)
# ---------------------------------------------------------------------------
def _make_mock_hdf5_pair(sdr_path, geo_path, n_scan=2, n_for=30, n_fov=9, seed=42):
    """Write minimal synthetic CrIS SDR and GEO HDF5 files."""
    n_lw, n_mw, n_sw = 717, 869, 637

    # SDR file
    with h5py.File(sdr_path, "w") as f:
        grp = f.create_group("All_Data/CrIS-FS-SDR_All")
        rng = np.random.default_rng(seed)
        grp.create_dataset(
            "ES_RealLW",
            data=rng.uniform(5, 80, (n_scan, n_for, n_fov, n_lw)).astype(np.float32),
        )
        grp.create_dataset(
            "ES_RealMW",
            data=rng.uniform(1, 30, (n_scan, n_for, n_fov, n_mw)).astype(np.float32),
        )
        grp.create_dataset(
            "ES_RealSW",
            data=rng.uniform(0.1, 10, (n_scan, n_for, n_fov, n_sw)).astype(np.float32),
        )
        grp.create_dataset(
            "QF1_SCAN_CRISSDR",
            data=np.zeros(n_scan, dtype=np.uint8),
        )
        grp.create_dataset(
            "QF3_CRISSDR",
            data=np.zeros((n_scan, n_for, n_fov, 3), dtype=np.uint8),
        )

    # GEO file
    # IET epoch: 1958-01-01T00:00:00. We need microseconds from there.
    # 2024-06-01 12:00:00 UTC in IET microseconds:
    base_dt = datetime(2024, 6, 1, 12, 0, 0)
    iet_epoch = datetime(1958, 1, 1)
    base_iet = int((base_dt - iet_epoch).total_seconds() * 1_000_000)
    # Offset by seed so each granule gets distinct times/coords
    time_offset = (seed - 42) * 32_000_000  # ~32s apart per seed step

    with h5py.File(geo_path, "w") as f:
        grp = f.create_group("All_Data/CrIS-SDR-GEO_All")
        lat_base = 30.0 + (seed - 42) * 5.0  # shift lat per granule
        grp.create_dataset(
            "Latitude",
            data=np.linspace(lat_base, lat_base + 20, n_scan * n_for * n_fov)
            .reshape(n_scan, n_for, n_fov)
            .astype(np.float32),
        )
        lon_base = 250.0 + (seed - 42) * 10.0  # shift lon per granule
        grp.create_dataset(
            "Longitude",
            data=np.linspace(lon_base, lon_base + 40, n_scan * n_for * n_fov)
            .reshape(n_scan, n_for, n_fov)
            .astype(np.float32),
        )
        grp.create_dataset(
            "Height",
            data=np.zeros((n_scan, n_for, n_fov), dtype=np.float32),
        )
        grp.create_dataset(
            "SatelliteZenithAngle",
            data=np.full((n_scan, n_for, n_fov), 25.0, dtype=np.float32),
        )
        grp.create_dataset(
            "SatelliteAzimuthAngle",
            data=np.full((n_scan, n_for, n_fov), 90.0, dtype=np.float32),
        )
        grp.create_dataset(
            "SolarZenithAngle",
            data=np.full((n_scan, n_for, n_fov), 45.0, dtype=np.float32),
        )
        grp.create_dataset(
            "SolarAzimuthAngle",
            data=np.full((n_scan, n_for, n_fov), 180.0, dtype=np.float32),
        )
        # FORTime: (n_scan, n_for) – slight offset per FOR to simulate real data
        for_time = np.full((n_scan, n_for), base_iet + time_offset, dtype=np.int64)
        for s in range(n_scan):
            for fi in range(n_for):
                for_time[s, fi] = (
                    base_iet + time_offset + (s * n_for + fi) * 200_000
                )  # 200 ms step
        grp.create_dataset("FORTime", data=for_time)


# ---------------------------------------------------------------------------
# Network / slow tests
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2025, month=1, day=1, hour=0),
        [datetime(year=2025, month=1, day=1, hour=0)],
    ],
)
@pytest.mark.parametrize("variable", ["crisfsr", ["crisfsr"]])
def test_jpss_cris_fetch(time, variable):
    ds = JPSS_CRIS(
        satellites=["n20"],
        time_tolerance=timedelta(seconds=30),
        cache=False,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset({"crisfsr"})
    assert "observation" in df.columns
    assert "satellite" in df.columns
    assert "channel_index" in df.columns

    if not df.empty:
        # Apodized (default): contiguous sensor_chan 1..2211
        assert df["channel_index"].between(1, 2211).all()
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        # Brightness temperature values should be finite and in a
        # physically reasonable range (K).
        assert df["observation"].notna().all()
        assert (df["observation"] > 100).all()  # > 100 K
        assert (df["observation"] < 400).all()  # < 400 K


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_jpss_cris_schema_fields():
    ds = JPSS_CRIS(
        satellites=["n20"],
        time_tolerance=timedelta(seconds=30),
        cache=False,
        verbose=False,
    )
    time = datetime(2025, 1, 1, 0)

    df_full = ds(time, ["crisfsr"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["crisfsr"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_jpss_cris_cache(cache):
    ds = JPSS_CRIS(
        satellites=["n20"],
        time_tolerance=timedelta(seconds=30),
        cache=cache,
        verbose=False,
    )
    df = ds(datetime(2025, 1, 1, 0), ["crisfsr"])
    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Mock / offline tests (no network required)
# ---------------------------------------------------------------------------
def test_jpss_cris_call_mock(tmp_path):
    """Exercise the full __call__ path with synthetic HDF5 files (apodized)."""
    n_scan, n_for, n_fov = 2, 4, 9  # reduced to keep test fast
    n_channels_apod = 713 + 865 + 633  # 2211 science channels after apodization

    sdr_file = tmp_path / "SCRIF_j01.h5"
    geo_file = tmp_path / "GCRSO_j01.h5"
    _make_mock_hdf5_pair(str(sdr_file), str(geo_file), n_scan, n_for, n_fov)

    sdr_uri = "s3://noaa-nesdis-n20-pds/CrIS-FS-SDR/2024/06/01/SCRIF_j01.h5"
    geo_uri = "s3://noaa-nesdis-n20-pds/CrIS-SDR-GEO/2024/06/01/GCRSO_j01.h5"

    def fake_cache_path(self, s3_uri):
        if "SCRIF" in s3_uri or "SDR" in s3_uri and "GEO" not in s3_uri:
            return str(sdr_file)
        return str(geo_file)

    mock_task = MagicMock(
        sdr_uri=sdr_uri,
        geo_uri=geo_uri,
        datetime_min=datetime(2024, 6, 1, 11, 0),
        datetime_max=datetime(2024, 6, 1, 13, 0),
        satellite="n20",
        variable="crisfsr",
        modifier=lambda x: x,
    )

    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=[mock_task]),
    ):
        ds = JPSS_CRIS(satellites=["n20"], cache=False, verbose=False)
        df = ds(datetime(2024, 6, 1, 12), ["crisfsr"])

    assert not df.empty
    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()) == {"crisfsr"}
    # Apodized: contiguous sensor_chan 1..2211 (no guard channel sentinels)
    assert df["channel_index"].between(1, 2211).all()
    expected_rows = n_scan * n_for * n_fov * n_channels_apod
    assert len(df) == expected_rows
    assert df["satellite"].iloc[0] == "n20"
    assert "quality" in df.columns
    assert (df["quality"] == 0).all()
    # Verify observation values are positive (our mock data uses positive values)
    assert (df["observation"] > 0).all()


def test_jpss_cris_call_mock_unapodized(tmp_path):
    """Exercise the full __call__ path with apodize=False (unapodized)."""
    n_scan, n_for, n_fov = 2, 4, 9
    n_channels_raw = 717 + 869 + 637  # 2223 including guard channels

    sdr_file = tmp_path / "SCRIF_j01.h5"
    geo_file = tmp_path / "GCRSO_j01.h5"
    _make_mock_hdf5_pair(str(sdr_file), str(geo_file), n_scan, n_for, n_fov)

    sdr_uri = "s3://noaa-nesdis-n20-pds/CrIS-FS-SDR/2024/06/01/SCRIF_j01.h5"
    geo_uri = "s3://noaa-nesdis-n20-pds/CrIS-SDR-GEO/2024/06/01/GCRSO_j01.h5"

    def fake_cache_path(self, s3_uri):
        if "SCRIF" in s3_uri or "SDR" in s3_uri and "GEO" not in s3_uri:
            return str(sdr_file)
        return str(geo_file)

    mock_task = MagicMock(
        sdr_uri=sdr_uri,
        geo_uri=geo_uri,
        datetime_min=datetime(2024, 6, 1, 11, 0),
        datetime_max=datetime(2024, 6, 1, 13, 0),
        satellite="n20",
        variable="crisfsr",
        modifier=lambda x: x,
    )

    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=[mock_task]),
    ):
        ds = JPSS_CRIS(satellites=["n20"], apodize=False, cache=False, verbose=False)
        df = ds(datetime(2024, 6, 1, 12), ["crisfsr"])

    assert not df.empty
    assert list(df.columns) == ds.SCHEMA.names
    # Unapodized: sensor_chan includes 0 (guard) and 1..2211
    assert df["channel_index"].between(0, 2211).all()
    expected_rows = n_scan * n_for * n_fov * n_channels_raw
    assert len(df) == expected_rows
    # Guard channels should have sensor_chan 0
    guard_rows = df[df["channel_index"] == 0]
    assert len(guard_rows) > 0  # 12 guard channels per FOV


def test_jpss_cris_subsample_mock(tmp_path):
    """Verify granule-level sub-sampling reduces data by the expected factor."""
    n_scan, n_for, n_fov = 2, 4, 9
    n_channels_apod = 713 + 865 + 633  # 2211 (default: apodized)

    # Create 3 distinct granule file pairs (different seeds → different coords)
    granule_files = []
    for i in range(3):
        sdr_file = tmp_path / f"SCRIF_j01_{i}.h5"
        geo_file = tmp_path / f"GCRSO_j01_{i}.h5"
        _make_mock_hdf5_pair(
            str(sdr_file), str(geo_file), n_scan, n_for, n_fov, seed=42 + i
        )
        granule_files.append((str(sdr_file), str(geo_file)))

    sdr_uris = [f"s3://bucket/SDR/granule_{i}.h5" for i in range(3)]
    geo_uris = [f"s3://bucket/GEO/granule_{i}.h5" for i in range(3)]

    def fake_cache_path(self, s3_uri):
        for i in range(3):
            if s3_uri == sdr_uris[i]:
                return granule_files[i][0]
            if s3_uri == geo_uris[i]:
                return granule_files[i][1]
        return str(tmp_path / "missing.h5")

    mock_tasks = [
        MagicMock(
            sdr_uri=sdr_uris[i],
            geo_uri=geo_uris[i],
            datetime_min=datetime(2024, 6, 1, 11, 0),
            datetime_max=datetime(2024, 6, 1, 13, 0),
            satellite="n20",
            variable="crisfsr",
            modifier=lambda x: x,
        )
        for i in range(3)
    ]

    rows_per_granule = n_scan * n_for * n_fov * n_channels_apod

    # subsample=1 → all 3 granules
    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=list(mock_tasks)),
    ):
        ds_full = JPSS_CRIS(satellites=["n20"], subsample=1, cache=False, verbose=False)
        df_full = ds_full(datetime(2024, 6, 1, 12), ["crisfsr"])

    assert len(df_full) == 3 * rows_per_granule

    # subsample=2 → granules [0, 2] (every 2nd), so 2 granules
    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=list(mock_tasks)),
    ):
        ds_sub = JPSS_CRIS(satellites=["n20"], subsample=2, cache=False, verbose=False)
        df_sub = ds_sub(datetime(2024, 6, 1, 12), ["crisfsr"])

    assert len(df_sub) == 2 * rows_per_granule

    # subsample=3 → only granule [0], so 1 granule
    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=list(mock_tasks)),
    ):
        ds_sub3 = JPSS_CRIS(satellites=["n20"], subsample=3, cache=False, verbose=False)
        df_sub3 = ds_sub3(datetime(2024, 6, 1, 12), ["crisfsr"])

    assert len(df_sub3) == 1 * rows_per_granule


# ---------------------------------------------------------------------------
# Validation / error tests (no network)
# ---------------------------------------------------------------------------
@pytest.mark.timeout(15)
def test_jpss_cris_available():
    assert JPSS_CRIS.available(datetime(2024, 6, 1, 12))
    assert not JPSS_CRIS.available(datetime(2015, 1, 1))
    assert JPSS_CRIS.available(np.datetime64("2024-06-01T12:00"))
    assert not JPSS_CRIS.available(np.datetime64("2015-01-01T00:00"))


@pytest.mark.timeout(15)
def test_jpss_cris_validate_time():
    with pytest.raises(ValueError):
        ds = JPSS_CRIS(satellites=["n20"], cache=False)
        ds(datetime(2015, 1, 1), ["crisfsr"])


@pytest.mark.timeout(15)
def test_jpss_cris_invalid_satellite():
    with pytest.raises(ValueError, match="Invalid satellite"):
        JPSS_CRIS(satellites=["invalid"])


def test_jpss_cris_exceptions():
    ds = JPSS_CRIS(satellites=["n20"], cache=False, verbose=False)

    with pytest.raises(KeyError):
        ds(datetime(2024, 6, 1, 12), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 6, 1, 12),
            ["crisfsr"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(datetime(2024, 6, 1, 12), ["crisfsr"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 6, 1, 12), ["crisfsr"], fields=wrong_type_schema)


def test_jpss_cris_tolerance_conversion():
    ds_td = JPSS_CRIS(time_tolerance=timedelta(minutes=30), cache=False, verbose=False)
    assert ds_td._tolerance_lower == timedelta(minutes=-30)
    assert ds_td._tolerance_upper == timedelta(minutes=30)

    ds_np = JPSS_CRIS(
        time_tolerance=np.timedelta64(30, "m"), cache=False, verbose=False
    )
    assert ds_np._tolerance_lower == timedelta(minutes=-30)
    assert ds_np._tolerance_upper == timedelta(minutes=30)

    ds_asym = JPSS_CRIS(
        time_tolerance=(np.timedelta64(-10, "m"), np.timedelta64(60, "m")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(minutes=-10)
    assert ds_asym._tolerance_upper == timedelta(minutes=60)


def test_jpss_cris_parse_filename():
    # Valid CrIS SDR filename
    t = JPSS_CRIS._parse_filename_time(
        "SCRIF_j01_d20250101_t0000391_e0000392_b12345_c20250101010000_oebc_ops.h5"
    )
    assert t == datetime(2025, 1, 1, 0, 0, 39)

    # Invalid filename
    assert JPSS_CRIS._parse_filename_time("random_file.h5") is None

    # Truncated date
    assert JPSS_CRIS._parse_filename_time("SCRIF_j01_d202.h5") is None


def test_jpss_cris_granule_key():
    sdr = "SCRIF_j01_d20250101_t0000391_e0000392_b12345_c20250101010000_oebc_ops.h5"
    geo = "GCRSO_j01_d20250101_t0000391_e0000392_b12345_c20250101010099_oebc_ops.h5"
    # Same granule → same key despite different creation timestamps
    assert JPSS_CRIS._granule_key(sdr) == JPSS_CRIS._granule_key(geo)
    assert JPSS_CRIS._granule_key(sdr) == "j01_d20250101_t0000391_e0000392_b12345"

    # Different orbit → different key
    sdr2 = "SCRIF_j01_d20250101_t0000391_e0000392_b99999_c20250101010000_oebc_ops.h5"
    assert JPSS_CRIS._granule_key(sdr) != JPSS_CRIS._granule_key(sdr2)
