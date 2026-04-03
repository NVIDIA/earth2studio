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

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import MetOpAVHRR


# ---------------------------------------------------------------------------
# Helper to build a synthetic AVHRR DataFrame (as _parse_avhrr_with_satpy
# would return)
# ---------------------------------------------------------------------------
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
            "channel_index": int(
                var.replace("avhrr", "").replace("3a", "3").replace("3b", "3")
            ),
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
        patch("earth2studio.data.metop_avhrr._parse_avhrr_with_satpy") as mock_parse,
    ):
        mock_dl.return_value = [str(tmp_path / "mock.nat")]
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
        patch("earth2studio.data.metop_avhrr._parse_avhrr_with_satpy") as mock_parse,
    ):
        mock_dl.return_value = [str(tmp_path / "mock.nat")]
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
