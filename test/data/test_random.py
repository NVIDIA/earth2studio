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

import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from earth2studio.data import Random, Random_FX, RandomDataFrame


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime.now(),
        [datetime.datetime.now(), datetime.datetime.now() - datetime.timedelta(days=1)],
    ],
)
@pytest.mark.parametrize(
    "variable", ["t2m", ["msl"], ["u10m", "v10m", "t2m", "z500", "t850", "r500"]]
)
@pytest.mark.parametrize("lat", [[-1, 0, -1], np.linspace(-90, 90, 181)])
@pytest.mark.parametrize("lon", [[0, 1, 2, 3], np.linspace(0, 359, 360)])
def test_random(time, variable, lat, lon):

    coords = OrderedDict({"lat": lat, "lon": lon})

    data_source = Random(coords)

    data = data_source(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == len(coords["lat"])
    assert shape[3] == len(coords["lon"])
    assert not np.isnan(data.values).any()


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime.now(),
        [datetime.datetime.now(), datetime.datetime.now() - datetime.timedelta(days=1)],
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize(
    "variable", ["t2m", ["msl"], ["u10m", "v10m", "t2m", "z500", "t850", "r500"]]
)
@pytest.mark.parametrize("lat", [[-1, 0, -1], np.linspace(-90, 90, 181)])
@pytest.mark.parametrize("lon", [[0, 1, 2, 3], np.linspace(0, 359, 360)])
def test_random_forecast(time, lead_time, variable, lat, lon):

    coords = OrderedDict({"lat": lat, "lon": lon})

    data_source = Random_FX(coords)

    data = data_source(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert shape[3] == len(coords["lat"])
    assert shape[4] == len(coords["lon"])
    assert not np.isnan(data.values).any()


@pytest.mark.parametrize(
    "time, variable, n_obs",
    [
        (datetime.datetime.now(), "t2m", 5),
        (
            [
                datetime.datetime.now(),
                datetime.datetime.now() - datetime.timedelta(days=1),
            ],
            ["u10m"],
            1,
        ),
        (np.array([np.datetime64("2024-01-01T12:00")]), ["t2m", "u10m", "v10m"], 10),
    ],
)
@pytest.mark.parametrize(
    "field_generators",
    [
        {},
        {
            "lat": lambda: np.random.uniform(25.0, 50.0),
            "lon": lambda: np.random.uniform(235.0, 295.0),
        },
    ],
)
@pytest.mark.parametrize(
    "tolerance,fields",
    [
        (datetime.timedelta(0), None),
        (datetime.timedelta(hours=6), ["time", "observation", "variable"]),
    ],
)
def test_random_dataframe(time, variable, n_obs, field_generators, tolerance, fields):
    np.random.seed(0)
    data_source = RandomDataFrame(
        n_obs=n_obs,
        field_generators=field_generators,
        tolerance=tolerance,
    )

    df = data_source(time, variable, fields=fields)

    # Check it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Normalize inputs for checking
    variable_list = [variable] if isinstance(variable, str) else variable
    if isinstance(time, (datetime.datetime, np.datetime64)):
        time_list = [pd.to_datetime(time)]
    elif isinstance(time, np.ndarray):
        time_list = [pd.to_datetime(t) for t in time]
    else:
        time_list = [pd.to_datetime(t) for t in time]

    # Check number of rows
    expected_rows = len(time_list) * len(variable_list) * n_obs
    assert len(df) == expected_rows

    # Check that all requested variables are present
    assert set(df["variable"].unique()).issubset(set(variable_list))

    # Check times are within tolerance
    if "time" in df.columns:
        df_times = pd.to_datetime(df["time"])
        for requested_time in time_list:
            # Find observations for this requested time
            time_mask = (df_times >= requested_time - tolerance) & (
                df_times <= requested_time + tolerance
            )
            # Check that we have observations for this time
            # (they should be distributed across variables)
            assert (
                time_mask.sum() > 0
            ), f"No observations found for time {requested_time}"
            # Verify all times are within tolerance
            obs_times = df_times[time_mask]
            assert (obs_times >= requested_time - tolerance).all()
            assert (obs_times <= requested_time + tolerance).all()

    # Check lat/lon ranges based on field_generators
    if "lat" in df.columns and "lon" in df.columns:
        if "lat" in field_generators:
            # Custom lat range - check values are reasonable (within -90 to 90)
            assert df["lat"].min() >= -90.0
            assert df["lat"].max() <= 90.0
        else:
            # Default lat range
            assert df["lat"].min() >= -90.0
            assert df["lat"].max() <= 90.0

        if "lon" in field_generators:
            # Custom lon range - check values are reasonable (within 0 to 360)
            assert df["lon"].min() >= 0.0
            assert df["lon"].max() <= 360.0
        else:
            # Default lon range
            assert df["lon"].min() >= 0.0
            assert df["lon"].max() <= 360.0

    # Check fields parameter
    if fields is None:
        # All fields should be present
        assert "time" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "observation" in df.columns
        assert "variable" in df.columns
    elif isinstance(fields, str):
        # Single field
        assert fields in df.columns
        assert len(df.columns) == 1
    else:
        # List of fields
        for field in fields:
            assert field in df.columns
        # Should only have requested fields
        assert set(df.columns).issubset(set(fields))

    # Check no NaN values
    assert not df.isnull().any().any()
