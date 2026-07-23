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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.data import Random, Random_FX, fetch_data
from earth2studio.models.px.datareplay import DataReplay

LAT = np.linspace(90, -90, 8)
LON = np.linspace(0, 360, 16, endpoint=False)
DOMAIN = OrderedDict(lat=LAT, lon=LON)
TIME = np.array([np.datetime64("2020-01-01T00:00:00")])
VARIABLE = np.array(["t2m", "u10m", "z500"])
STEP = np.timedelta64(6, "h")


def _initial_condition(source: Random | Random_FX):
    return fetch_data(
        source,
        time=TIME,
        variable=VARIABLE,
        lead_time=np.array([np.timedelta64(0, "h")]),
    )


@pytest.mark.parametrize("source_type", [Random, Random_FX])
def test_datareplay_call(source_type):
    source = source_type(DOMAIN)
    x, coords = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN, step=STEP)

    output, output_coords = replay(x, coords)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    np.testing.assert_array_equal(output_coords["time"], TIME)
    np.testing.assert_array_equal(output_coords["lead_time"], np.array([STEP]))


@pytest.mark.parametrize("source_type", [Random, Random_FX])
def test_datareplay_iter(source_type):
    source = source_type(DOMAIN)
    x, coords = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN, step=STEP)
    hook_calls = {"front": 0, "rear": 0}

    def front_hook(data, data_coords):
        hook_calls["front"] += 1
        return data, data_coords

    def rear_hook(data, data_coords):
        hook_calls["rear"] += 1
        return data, data_coords

    replay.front_hook = front_hook
    replay.rear_hook = rear_hook
    iterator = replay.create_iterator(x, coords)

    initial, initial_coords = next(iterator)
    torch.testing.assert_close(initial, x)
    np.testing.assert_array_equal(
        initial_coords["lead_time"], np.array([np.timedelta64(0, "h")])
    )
    assert hook_calls == {"front": 0, "rear": 0}

    _, first_coords = next(iterator)
    np.testing.assert_array_equal(first_coords["lead_time"], np.array([STEP]))
    assert hook_calls == {"front": 1, "rear": 1}

    _, second_coords = next(iterator)
    np.testing.assert_array_equal(second_coords["lead_time"], np.array([2 * STEP]))
    assert hook_calls == {"front": 2, "rear": 2}


def test_datareplay_input_coords_copy():
    replay = DataReplay(Random(DOMAIN), "t2m", DOMAIN)
    coords = replay.input_coords()
    coords["variable"][0] = "msl"

    assert str(replay) == "DataReplay()"
    assert replay.input_coords()["variable"][0] == "t2m"


def test_datareplay_output_coords_copy():
    source = Random(DOMAIN)
    _, coords = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN)
    original_lead_time = coords["lead_time"].copy()

    output_coords = replay.output_coords(coords)

    np.testing.assert_array_equal(coords["lead_time"], original_lead_time)
    assert output_coords is not coords


def test_datareplay_grid_mismatch_raises():
    source = Random(OrderedDict(lat=np.linspace(90, -90, 9), lon=LON))
    x, coords = _initial_condition(Random(DOMAIN))
    replay = DataReplay(source, VARIABLE, DOMAIN)

    with pytest.raises(ValueError, match="not the same"):
        replay(x, coords)


def test_datareplay_nonfinite_raises(monkeypatch):
    source = Random(DOMAIN)
    x, coords = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN)
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.full(shape, np.nan))

    with pytest.raises(ValueError, match="non-finite"):
        replay(x, coords)


@pytest.mark.parametrize(
    "coords_update, match",
    [
        ({"time": np.empty(0, dtype="datetime64[ns]")}, "non-empty time"),
        ({"variable": VARIABLE[::-1]}, "not the same"),
    ],
)
def test_datareplay_invalid_coords(coords_update, match):
    source = Random(DOMAIN)
    x, coords = _initial_condition(source)
    coords.update(coords_update)
    replay = DataReplay(source, VARIABLE, DOMAIN)

    with pytest.raises(ValueError, match=match):
        replay(x, coords)


@pytest.mark.parametrize(
    "step, error",
    [
        (6, TypeError),
        (np.timedelta64(0, "h"), ValueError),
        (np.timedelta64(-1, "h"), ValueError),
        (np.timedelta64("NaT"), ValueError),
    ],
)
def test_datareplay_invalid_step(step, error):
    with pytest.raises(error):
        DataReplay(Random(DOMAIN), VARIABLE, DOMAIN, step=step)
