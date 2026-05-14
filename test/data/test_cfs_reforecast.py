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
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from earth2studio.data import CFS_Reforecast_FX, CFS_Reforecast_FX_Flux
from earth2studio.data.cfs_reforecast import (
    _NCEP_TO_PARAM_NAME,
    _resolve_pygrib_filter,
)
from earth2studio.lexicon import CFSFluxLexicon, CFSLexicon

# Cycle inside the reforecast archive used by --slow network tests.
_TEST_CYCLE = datetime(year=2010, month=6, day=15, hour=0)


# ----------------------------------------------------------------------
# Pure-offline translator tests (no network)
# ----------------------------------------------------------------------


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "level_desc,expected",
    [
        ("500 mb", ("isobaricInhPa", 500)),
        ("1000 mb", ("isobaricInhPa", 1000)),
        ("2 m above ground", ("heightAboveGround", 2)),
        ("10 m above ground", ("heightAboveGround", 10)),
        ("mean sea level", ("meanSea", 0)),
        ("surface", ("surface", 0)),
        (
            "entire atmosphere (considered as a single layer)",
            ("atmosphereSingleLayer", 0),
        ),
        ("high cloud layer", ("highCloudLayer", 0)),
        ("middle cloud layer", ("middleCloudLayer", 0)),
        ("low cloud layer", ("lowCloudLayer", 0)),
    ],
)
def test_resolve_pygrib_filter(level_desc, expected):
    assert _resolve_pygrib_filter(level_desc) == expected


@pytest.mark.timeout(5)
def test_resolve_pygrib_filter_unknown():
    with pytest.raises(KeyError):
        _resolve_pygrib_filter("100 mb above mean sea level")


@pytest.mark.timeout(5)
def test_translator_covers_lexicons():
    # Every NCEP param name referenced by either lexicon must have an
    # entry in the pygrib-name translator, and every level description
    # must resolve.  Catches drift if a lexicon is extended without also
    # extending the translator.
    for vocab in (CFSLexicon.VOCAB, CFSFluxLexicon.VOCAB):
        for entry in vocab.values():
            _, param_ncep, level_desc = entry.split("::", 2)
            assert (
                param_ncep in _NCEP_TO_PARAM_NAME
            ), f"Missing pygrib translation for NCEP param {param_ncep!r}"
            # Will raise KeyError if level description is unparseable.
            _resolve_pygrib_filter(level_desc)


# ----------------------------------------------------------------------
# Construction / validation (no network)
# ----------------------------------------------------------------------


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=1981, month=12, day=11),  # one day before archive start
        datetime(year=2011, month=4, day=1),  # one cycle after archive end
        datetime(year=2010, month=1, day=1, hour=3),  # off 6-h cycle
    ],
)
def test_cfs_reforecast_invalid_time(time):
    ds = CFS_Reforecast_FX(cache=False)
    with pytest.raises(ValueError):
        ds(time, timedelta(hours=6), "msl")


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=-6),
        timedelta(hours=3),  # not on 6-h cycle
        timedelta(days=300),  # past the 9-month cap
    ],
)
def test_cfs_reforecast_invalid_leadtime(lead_time):
    ds = CFS_Reforecast_FX(cache=False)
    with pytest.raises(ValueError):
        ds(_TEST_CYCLE, lead_time, "msl")


@pytest.mark.timeout(5)
def test_cfs_reforecast_invalid_variable():
    ds = CFS_Reforecast_FX(cache=False)
    with pytest.raises(KeyError):
        ds(_TEST_CYCLE, timedelta(hours=6), "definitely_not_a_real_variable")


@pytest.mark.timeout(5)
def test_cfs_reforecast_uri_construction():
    ds_pgbf = CFS_Reforecast_FX()
    ds_flxf = CFS_Reforecast_FX_Flux()
    t = datetime(2010, 6, 15, 6)
    lt = timedelta(hours=12)
    # Forecast issued at 06z, valid at 18z, same cycle directory.
    assert ds_pgbf._grib_uri(t, lt).endswith(
        "/2010/201006/20100615/pgbf2010061518.01.2010061506.grb2"
    )
    assert ds_flxf._grib_uri(t, lt).endswith(
        "/2010/201006/20100615/flxf2010061518.01.2010061506.grb2"
    )
    # pgbf and flxf live under different subdirs.
    assert "6-hourly-by-pressure-level-9-month-runs" in ds_pgbf._grib_uri(t, lt)
    assert "6-hourly-flux-9-month-runs" in ds_flxf._grib_uri(t, lt)


@pytest.mark.timeout(5)
def test_cfs_reforecast_available():
    assert CFS_Reforecast_FX.available(_TEST_CYCLE)
    # Before archive start.
    assert not CFS_Reforecast_FX.available(datetime(year=1980, month=1, day=1))
    # After archive end.
    assert not CFS_Reforecast_FX.available(datetime(year=2012, month=1, day=1))
    # Off 6-h cycle.
    assert not CFS_Reforecast_FX.available(datetime(year=2010, month=6, day=15, hour=3))
    # np.datetime64 input path.
    assert CFS_Reforecast_FX.available(np.datetime64("2010-06-15T00:00"))


# ----------------------------------------------------------------------
# Mock end-to-end (no network, exercises full __call__ path)
# ----------------------------------------------------------------------


@pytest.mark.timeout(15)
def test_cfs_reforecast_call_mock(tmp_path, monkeypatch):
    """Exercise __call__ without network by mocking download + decode."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_grid = np.full((181, 360), 50000.0, dtype=np.float32)

    async def _fake_fetch_remote_file(self, uri):
        return str(tmp_path / "ignored.grb2")

    def _fake_decode(grib_file, variables):
        # Return the same fake grid for every requested variable; the modifier
        # in the lexicon (e.g. *9.81 for HGT) is still applied by fetch_array
        # so we get to test the modifier path too.
        return [
            (var_idx, modifier(fake_grid))
            for var_idx, _param, _tol, _lvl, modifier in variables
        ]

    ds = CFS_Reforecast_FX(cache=True)

    with (
        patch.object(
            CFS_Reforecast_FX, "_async_init", new=AsyncMock(return_value=None)
        ),
        patch.object(
            CFS_Reforecast_FX,
            "_fetch_remote_file",
            new=_fake_fetch_remote_file,
        ),
        patch(
            "earth2studio.data.cfs_reforecast._decode_cfs_reforecast_grib",
            side_effect=_fake_decode,
        ),
    ):
        ds.fs = object()  # type: ignore[assignment]
        data = ds(_TEST_CYCLE, [timedelta(hours=6)], ["msl", "z500"])

    assert data.shape == (1, 1, 2, 181, 360)
    np.testing.assert_allclose(data.sel(variable="msl").values[0, 0], fake_grid)
    # z500 picks up the HGT geopotential modifier (* 9.81).
    np.testing.assert_allclose(data.sel(variable="z500").values[0, 0], fake_grid * 9.81)


@pytest.mark.timeout(15)
def test_cfs_reforecast_missing_variable_returns_nan(tmp_path, monkeypatch):
    # If _decode_cfs_reforecast_grib drops a variable (no matching grib
    # record), the corresponding output slot must be NaN, not uninitialised
    # memory.
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    fake_grid = np.full((181, 360), 50000.0, dtype=np.float32)

    async def _fake_fetch_remote_file(self, uri):
        return str(tmp_path / "ignored.grb2")

    def _fake_decode_drop_second(grib_file, variables):
        # Decode only the first requested variable; the second is reported
        # missing.  This mirrors the in-product behaviour when a grib record
        # is absent from the file.
        return [
            (variables[0][0], variables[0][4](fake_grid)),
        ]

    ds = CFS_Reforecast_FX(cache=True)
    with (
        patch.object(
            CFS_Reforecast_FX, "_async_init", new=AsyncMock(return_value=None)
        ),
        patch.object(
            CFS_Reforecast_FX, "_fetch_remote_file", new=_fake_fetch_remote_file
        ),
        patch(
            "earth2studio.data.cfs_reforecast._decode_cfs_reforecast_grib",
            side_effect=_fake_decode_drop_second,
        ),
    ):
        ds.fs = object()  # type: ignore[assignment]
        data = ds(_TEST_CYCLE, [timedelta(hours=6)], ["msl", "z500"])

    np.testing.assert_allclose(data.sel(variable="msl").values[0, 0], fake_grid)
    assert np.isnan(data.sel(variable="z500").values).all()


# ----------------------------------------------------------------------
# Live HTTPS fetch tests (slow)
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "time,lead_time,variable",
    [
        (_TEST_CYCLE, timedelta(hours=6), "msl"),
        (
            _TEST_CYCLE,
            [timedelta(hours=0), timedelta(hours=6)],
            ["z500", "u500", "v500"],
        ),
        (
            np.array([np.datetime64(_TEST_CYCLE.isoformat())]),
            np.array([np.timedelta64(12, "h")]),
            np.array(["t850", "q850", "d2m"]),
        ),
    ],
)
def test_cfs_reforecast_pgbf_fetch(time, lead_time, variable):
    ds = CFS_Reforecast_FX(cache=False)
    data = ds(time, lead_time, variable)

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]
    if isinstance(time, datetime):
        time = [time]

    assert data.shape == (len(time), len(lead_time), len(variable), 181, 360)
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert float(data.lat.min()) == -90.0
    assert float(data.lat.max()) == 90.0


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(180)
def test_cfs_reforecast_flxf_fetch():
    ds = CFS_Reforecast_FX_Flux(cache=False)
    data = ds(_TEST_CYCLE, timedelta(hours=6), ["t2m", "u10m", "v10m", "tpf"])

    assert data.shape == (1, 1, 4, 190, 384)
    assert not np.isnan(data.values).any()
    # T126 Gaussian latitudes are non-uniform; the poleward node sits at
    # ~89.28 deg, never exactly 90.
    assert 89.0 < float(data.lat.max()) < 90.0
    assert -90.0 < float(data.lat.min()) < -89.0
    t2m = data.sel(variable="t2m").values
    assert 180.0 < t2m.min() and t2m.max() < 340.0
