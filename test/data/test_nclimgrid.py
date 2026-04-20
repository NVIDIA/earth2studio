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
from datetime import datetime

import numpy as np
import pytest

from earth2studio.data import NClimGrid
from earth2studio.lexicon.nclimgrid import NClimGridLexicon


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2010, month=7, day=1),
        [
            datetime(year=2010, month=7, day=1),
            datetime(year=2010, month=7, day=2),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["t2m_max", ["t2m_max", "tp"]])
def test_nclimgrid_fetch(time, variable):
    ds = NClimGrid(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] > 100  # lat grid points (CONUS ~596)
    assert shape[3] > 100  # lon grid points (CONUS ~1385)
    assert not np.isnan(data.values).all()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("2010-07-01T00:00")])],
)
@pytest.mark.parametrize("variable", [["t2m_max", "tp"]])
@pytest.mark.parametrize("cache", [True, False])
def test_nclimgrid_cache(time, variable, cache):
    ds = NClimGrid(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert not np.isnan(data.values).all()

    if cache:
        assert pathlib.Path(ds.cache).is_dir()
    else:
        assert not pathlib.Path(ds.cache).exists()

    # Reload from cache
    data = ds(time, variable[0])
    assert data.shape[1] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=1940, month=1, day=1),
    ],
)
@pytest.mark.parametrize("variable", ["not_available"])
def test_nclimgrid_valid_time(time, variable):
    with pytest.raises(ValueError):
        ds = NClimGrid()
        ds(time, variable)


def test_nclimgrid_available():
    assert NClimGrid.available(datetime(2010, 7, 1))
    assert not NClimGrid.available(datetime(1940, 1, 1))
    assert NClimGrid.available(np.datetime64("2010-07-01"))
    assert not NClimGrid.available(np.datetime64("1940-01-01"))


def test_nclimgrid_lexicon():
    for v in ["t2m_max", "t2m_min", "tp", "spi"]:
        native, mod = NClimGridLexicon[v]
        assert isinstance(native, str)
        assert callable(mod)

    # Unit conversions
    _, mod = NClimGridLexicon["t2m_max"]
    np.testing.assert_allclose(mod(np.array([25.0])), [298.15])

    _, mod = NClimGridLexicon["tp"]
    np.testing.assert_allclose(mod(np.array([100.0])), [0.1])

    _, mod = NClimGridLexicon["spi"]
    np.testing.assert_allclose(mod(np.array([1.2])), [1.2])
