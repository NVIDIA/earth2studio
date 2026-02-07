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
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from earth2studio.data import (
    PlanetaryComputerECMWFOpenDataIFS,
    PlanetaryComputerMODISFire,
    PlanetaryComputerOISST,
    PlanetaryComputerSentinel3AOD,
)
from earth2studio.data.planetary_computer import _PlanetaryComputerData
from earth2studio.lexicon.planetary_computer import OISSTLexicon


def test_planetary_computer_base_init() -> None:
    class DummyLexicon(OISSTLexicon):
        pass

    ds = _PlanetaryComputerData(
        collection_id="dummy-collection",
        lexicon=DummyLexicon,
        asset_key="netcdf",
        search_kwargs={"limit": 1},
        spatial_dims={"lat": np.linspace(-1, 1, 2), "lon": np.linspace(0, 1, 2)},
    )

    assert ds._collection_id == "dummy-collection"
    assert ds._asset_key == "netcdf"
    assert ds._spatial_shape == (2, 2)


def test_planetary_computer_base_invalid_spatial_dims() -> None:
    class DummyLexicon(OISSTLexicon):
        pass

    with pytest.raises(ValueError):
        _PlanetaryComputerData(
            collection_id="dummy-collection",
            lexicon=DummyLexicon,
            spatial_dims={},
        )


def test_planetary_computer_base_extract_not_implemented() -> None:
    class DummyLexicon(OISSTLexicon):
        pass

    ds = _PlanetaryComputerData(
        collection_id="dummy-collection",
        lexicon=DummyLexicon,
        spatial_dims={"lat": np.linspace(-1, 1, 2), "lon": np.linspace(0, 1, 2)},
    )

    with pytest.raises(NotImplementedError):
        ds.extract_variable_numpy(None, None, datetime.now(timezone.utc))


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time,variable",
    [
        (
            datetime(2024, 6, 19, tzinfo=timezone.utc),
            ["sst", "ssta"],
        ),
        (
            [
                datetime(2024, 6, 18, tzinfo=timezone.utc),
                datetime(2024, 6, 19, tzinfo=timezone.utc),
            ],
            "sst",
        ),
    ],
)
def test_planetary_computer_oisst_fetch(time, variable) -> None:
    """Fetch OISST sample and verify shapes/coordinates."""
    ds = PlanetaryComputerOISST(cache=False, verbose=False)
    data = ds(time=time, variable=variable)

    times = list(time) if isinstance(time, (list, tuple)) else [time]
    variables = list(variable) if isinstance(variable, (list, tuple)) else [variable]

    assert data.shape == (len(times), len(variables), 720, 1440)
    assert np.array_equal(data.coords["variable"].values, np.array(variables))
    assert data.coords["lat"].size == 720
    assert data.coords["lon"].size == 1440
    assert np.isfinite(data.values).any()


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_planetary_computer_oisst_cache(cache: bool) -> None:
    """Ensure cache directory behavior matches flag settings."""
    ds = PlanetaryComputerOISST(cache=cache, verbose=False)
    time = datetime(2024, 6, 19, tzinfo=timezone.utc)
    variable = ["sst", "ssta"]

    data = ds(time=time, variable=variable)
    assert data.shape == (1, len(variable), 720, 1440)

    cache_path = pathlib.Path(ds.cache)
    assert cache_path.is_dir() == cache

    try:
        shutil.rmtree(cache_path)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time,variable",
    [
        (
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            ["s3sy02aod", "s3sy01sr"],
        ),
        (
            [
                datetime(2024, 6, 1, tzinfo=timezone.utc),
                datetime(2024, 6, 1, tzinfo=timezone.utc) + timedelta(hours=6),
            ],
            "s3sy02aod",
        ),
    ],
)
def test_planetary_computer_sentinel3_fetch(time, variable) -> None:
    """Fetch Sentinel-3 sample and verify shapes/coordinates."""
    ds = PlanetaryComputerSentinel3AOD(cache=False, verbose=False)
    data = ds(time=time, variable=variable)

    times = list(time) if isinstance(time, (list, tuple)) else [time]
    variables = list(variable) if isinstance(variable, (list, tuple)) else [variable]

    assert data.shape == (len(times), len(variables), 4040, 324)
    assert np.array_equal(data.coords["variable"].values, np.array(variables))
    assert np.isfinite(data.values).any()


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_planetary_computer_sentinel3_cache(cache: bool) -> None:
    """Ensure Sentinel-3 cache behavior aligns with flag."""
    ds = PlanetaryComputerSentinel3AOD(cache=cache, verbose=False)
    time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    variable = ["s3sy02aod"]

    data = ds(time=time, variable=variable)
    assert data.shape == (1, len(variable), 4040, 324)

    cache_path = pathlib.Path(ds.cache)
    assert cache_path.is_dir() == cache

    try:
        shutil.rmtree(cache_path)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time,variable,tile",
    [
        (
            datetime(2023, 7, 28, tzinfo=timezone.utc),
            ["fmask", "mfrp"],
            "h35v10",
        ),
        (
            [
                datetime(2023, 7, 27, tzinfo=timezone.utc),
                datetime(2023, 7, 28, tzinfo=timezone.utc),
            ],
            "fmask",
            "h34v10",
        ),
    ],
)
def test_planetary_computer_modis_fire_fetch(time, variable, tile) -> None:
    ds = PlanetaryComputerMODISFire(tile=tile, cache=False, verbose=False)
    data = ds(time=time, variable=variable)

    times = list(time) if isinstance(time, (list, tuple)) else [time]
    variables = list(variable) if isinstance(variable, (list, tuple)) else [variable]

    assert data.shape == (len(times), len(variables), 1200, 1200)
    assert np.array_equal(data.coords["variable"].values, np.array(variables))
    assert np.isfinite(data.values).any()
    # Validate that grid() produces lat/lon arrays of expected shape
    lat, lon = PlanetaryComputerMODISFire.grid(tile)
    assert lat.shape == (1200, 1200)
    assert lon.shape == (1200, 1200)


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_planetary_computer_modis_fire_cache(cache: bool) -> None:
    ds = PlanetaryComputerMODISFire(tile="h35v10", cache=cache, verbose=False)
    time = datetime(2023, 7, 28, tzinfo=timezone.utc)
    variable = ["fmask"]

    data = ds(time=time, variable=variable)
    assert data.shape == (1, len(variable), 1200, 1200)

    cache_path = pathlib.Path(ds.cache)
    assert cache_path.is_dir() == cache

    try:
        shutil.rmtree(cache_path)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail()
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time,variable",
    [
        (
            datetime(2025, 7, 28, tzinfo=timezone.utc),
            ["u10m", "z500"],
        ),
        (
            [
                datetime(2025, 7, 27, tzinfo=timezone.utc),
                datetime(2025, 7, 28, tzinfo=timezone.utc),
            ],
            "t2m",
        ),
    ],
)
def test_planetary_computer_ifs_fetch(time, variable) -> None:
    ds = PlanetaryComputerECMWFOpenDataIFS(cache=False, verbose=False)
    data = ds(time=time, variable=variable)

    times = list(time) if isinstance(time, (list, tuple)) else [time]
    variables = list(variable) if isinstance(variable, (list, tuple)) else [variable]

    assert data.shape == (len(times), len(variables), 721, 1440)
    assert np.array_equal(data.coords["variable"].values, np.array(variables))
