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


# ---------------------------------------------------------------------------
# Helper – generate a synthetic HDF5 file pair (SDR + GEO)
# ---------------------------------------------------------------------------
def _make_mock_hdf5_pair(sdr_path, geo_path, n_scan=2, n_for=30, n_fov=9):
    """Write minimal synthetic CrIS SDR and GEO HDF5 files."""
    n_lw, n_mw, n_sw = 717, 869, 637

    # SDR file
    with h5py.File(sdr_path, "w") as f:
        grp = f.create_group("All_Data/CrIS-FS-SDR_All")
        rng = np.random.default_rng(42)
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

    with h5py.File(geo_path, "w") as f:
        grp = f.create_group("All_Data/CrIS-SDR-GEO_All")
        grp.create_dataset(
            "Latitude",
            data=np.linspace(30, 50, n_scan * n_for * n_fov)
            .reshape(n_scan, n_for, n_fov)
            .astype(np.float32),
        )
        grp.create_dataset(
            "Longitude",
            data=np.linspace(250, 290, n_scan * n_for * n_fov)
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
        for_time = np.full((n_scan, n_for), base_iet, dtype=np.int64)
        for s in range(n_scan):
            for fi in range(n_for):
                for_time[s, fi] = base_iet + (s * n_for + fi) * 200_000  # 200 ms step
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
        assert df["channel_index"].between(0, 2222).all()
        assert df["lat"].between(-90, 90).all()
        assert df["lon"].between(0, 360).all()
        # Radiance values should be finite and mostly positive; some
        # channels may have near-zero or slightly negative values due
        # to instrument noise (especially SWIR).
        assert df["observation"].notna().all()
        assert (df["observation"] < 200).all()  # mW/(m^2 sr cm^-1)


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
    """Exercise the full __call__ path with synthetic HDF5 files."""
    n_scan, n_for, n_fov = 2, 4, 9  # reduced to keep test fast
    n_channels = 717 + 869 + 637  # 2223

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
    assert df["channel_index"].between(0, 2222).all()
    expected_rows = n_scan * n_for * n_fov * n_channels
    assert len(df) == expected_rows
    assert df["satellite"].iloc[0] == "n20"
    assert "quality" in df.columns
    assert (df["quality"] == 0).all()
    # Verify observation values are positive (our mock data uses positive values)
    assert (df["observation"] > 0).all()


def test_jpss_cris_subsample_mock(tmp_path):
    """Verify sub-sampling reduces spatial points by the expected factor."""
    n_scan, n_for, n_fov = 2, 30, 9  # use 30 FORs to test sub-sampling
    n_channels = 717 + 869 + 637  # 2223

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

    # subsample=1 → all 30 FORs
    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=[mock_task]),
    ):
        ds_full = JPSS_CRIS(satellites=["n20"], subsample=1, cache=False, verbose=False)
        df_full = ds_full(datetime(2024, 6, 1, 12), ["crisfsr"])

    assert len(df_full) == n_scan * n_for * n_fov * n_channels

    # subsample=3 → every 3rd FOR: ceil(30/3) = 10 FORs
    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=[mock_task]),
    ):
        ds_sub = JPSS_CRIS(satellites=["n20"], subsample=3, cache=False, verbose=False)
        df_sub = ds_sub(datetime(2024, 6, 1, 12), ["crisfsr"])

    n_for_sub = len(range(0, n_for, 3))  # 0,3,6,9,...,27 → 10
    expected_rows = n_scan * n_for_sub * n_fov * n_channels
    assert len(df_sub) == expected_rows

    # subsample=5 → every 5th FOR: 0,5,10,15,20,25 → 6 FORs
    with (
        patch.object(JPSS_CRIS, "_fetch_remote_file", return_value=None),
        patch.object(JPSS_CRIS, "_cache_path", fake_cache_path),
        patch.object(JPSS_CRIS, "_create_tasks", return_value=[mock_task]),
    ):
        ds_sub5 = JPSS_CRIS(satellites=["n20"], subsample=5, cache=False, verbose=False)
        df_sub5 = ds_sub5(datetime(2024, 6, 1, 12), ["crisfsr"])

    n_for_sub5 = len(range(0, n_for, 5))  # 0,5,10,15,20,25 → 6
    expected_rows5 = n_scan * n_for_sub5 * n_fov * n_channels
    assert len(df_sub5) == expected_rows5


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
