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
import pytest

from earth2studio.data import Random, Random_FX


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
