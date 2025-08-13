# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

from earth2studio.data import CMIP6


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "table_id, variable",
    [
        pytest.param("Amon", ["u10m", "v10m"], id="atmos_multi"),
        pytest.param("Omon", ["sst"], id="ocean_sst"),
    ],
)
@pytest.mark.parametrize(
    "time",
    [
        pytest.param(datetime(2015, 1, 16), id="single_stamp"),
        pytest.param(
            [datetime(2015, 1, 15), datetime(2015, 1, 17)],
            id="two_stamps",
        ),
        pytest.param(
            datetime(2014, 1, 15),
            id="non_matching_timespan",
        ),
    ],
)
def test_cmip6_fetch(table_id, variable, time):
    """Generic CMIP6 fetch test covering both atmospheric and ocean tables."""
    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id=table_id,
        variant_label="r1i1p2f1",
    )

    data = ds(time, variable)
    shape = data.shape

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "table_id, variable",
    [
        pytest.param("Omon", ["sst"]),
    ],
)
@pytest.mark.parametrize(
    "time",
    [
        pytest.param([datetime(2015, 1, 16)]),
    ],
)
@pytest.mark.parametrize("cache", [True, False])
def test_cmip6_cache(table_id, variable, time, cache):

    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id=table_id,
        variant_label="r1i1p2f1",
        cache=cache,
    )
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 291
    assert shape[3] == 360
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 291
    assert shape[3] == 360

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_cmip6_init_valid():
    """CMIP6 initialisation with typical parameters should succeed."""

    ds = CMIP6(
        experiment_id="historical",
        source_id="MPI-ESM1-2-LR",
        table_id="Amon",
        variant_label="r1i1p1f1",
    )

    # Attributes should be stored as-is
    assert ds.experiment_id == "historical"
    assert ds.source_id == "MPI-ESM1-2-LR"
    assert ds.table_id == "Amon"
    assert ds.variant_label == "r1i1p1f1"


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "variable, expected_exc",
    [
        pytest.param("not_a_variable", KeyError, id="invalid_name"),
        pytest.param("sst", ValueError, id="ocean_var_on_day_table"),
        pytest.param("t550", KeyError, id="pressure_level_missing"),
    ],
)
def test_cmip6_input(variable, expected_exc):
    """Trigger specific validation errors inside the CMIP6 datasource."""

    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="day",  # daily atmospheric table
        variant_label="r1i1p2f1",
    )

    with pytest.raises(expected_exc):
        _ = ds(datetime(2015, 1, 16), variable)


@pytest.mark.parametrize(
    "time, source_id, expected",
    [
        pytest.param(datetime(2015, 1, 16), "CanESM5", True, id="exists"),
        pytest.param(datetime(1800, 1, 1), "CanESM5", False, id="too_early"),
        pytest.param(datetime(2015, 1, 16), "NonExistentModel", False, id="bad_model"),
    ],
)
def test_cmip6_available(time, source_id, expected):
    """Test the CMIP6.available class-method."""

    result = CMIP6.available(
        time,
        experiment_id="ssp585",
        source_id=source_id,
        table_id="day",
        variant_label="r1i1p2f1",
    )

    assert result is expected
