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

import numpy as np
import pytest
import xarray as xr

import earth2studio.utils.time_statistics as e2s


def test_time_statistic_declarations():
    assert e2s.time_statistic_metadata("mean:24h") == {
        "modifier": "mean:24h",
        "method": "mean",
        "window": "PT24H",
        "start_offset": "-PT24H",
        "end_offset": "PT0S",
        "closed": "left",
    }
    assert e2s.time_statistic_metadata("max:-12h:+12h")["modifier"] == ("max:-12h:+12h")
    assert e2s._group_time_statistics(
        ["u10m", "v10m", "t2m"],
        {"u10m": "mean:24h", "v10m": "mean:24h", "t2m": "max:24h"},
    ) == {"mean:24h": ("u10m", "v10m"), "max:24h": ("t2m",)}
    assert e2s._group_time_statistics(["u10m", "t2m"], "sum:6h") == {
        "sum:6h": ("u10m", "t2m")
    }

    for modifier, message in (
        ("median:24h", "Unknown"),
        ("mean:0h", "positive"),
        ("mean:12h:-12h", "after"),
        ("mean", "must be"),
    ):
        with pytest.raises(ValueError, match=message):
            e2s.time_statistic_metadata(modifier)
    with pytest.raises(ValueError, match="unknown variables"):
        e2s._group_time_statistics(["u10m"], {"t2m": "mean:24h"})


def test_time_statistic_source_coordinates():
    valid_time = np.datetime64("2026-08-28T06:00")
    delta_t = np.timedelta64(6, "h")
    np.testing.assert_array_equal(
        e2s.source_times("mean:24h", valid_time, delta_t),
        np.arange(
            np.datetime64("2026-08-27T06:00"), valid_time, np.timedelta64(6, "h")
        ),
    )
    np.testing.assert_array_equal(
        e2s.source_lead_times("mean:24h", np.timedelta64(6, "h"), delta_t),
        np.array([-18, -12, -6, 0], dtype="timedelta64[h]"),
    )
    assert e2s.source_times(
        "mean:24h",
        np.array([valid_time, valid_time + np.timedelta64(6, "h")]),
        delta_t,
    ).shape == (2, 4)

    for invalid_delta, message in (
        (np.timedelta64(0, "h"), "positive"),
        (np.timedelta64(5, "h"), "divisible"),
    ):
        with pytest.raises(ValueError, match=message):
            e2s.source_times("mean:24h", valid_time, invalid_delta)
    with pytest.raises(TypeError, match="numpy.timedelta64"):
        e2s.source_times("mean:24h", valid_time, "6h")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="datetime"):
        e2s.source_times("mean:24h", [0], delta_t)
    with pytest.raises(TypeError, match="timedelta"):
        e2s.source_lead_times("mean:24h", [0], delta_t)


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("mean", [2.0, 5.0]),
        ("sum", [6.0, 15.0]),
        ("min", [1.0, 4.0]),
        ("max", [3.0, 6.0]),
    ),
)
def test_time_statistic_block_reductions(method, expected):
    array = xr.DataArray(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        dims=("time", "variable"),
        coords={"variable": ["u10m", "v10m"]},
        attrs={"units": "m/s"},
    )
    result = e2s.apply_time_statistic(array, f"{method}:3h")
    np.testing.assert_array_equal(result, expected)
    assert result.attrs == array.attrs

    with pytest.raises(ValueError, match="not present"):
        e2s.apply_time_statistic(array, f"{method}:3h", "lead_time")


def test_custom_time_statistic_and_dimension_inference():
    def value_range(data, dimension):
        return data.max(dim=dimension) - data.min(dim=dimension)

    e2s.register_time_statistic("range", value_range)
    assert "range" in e2s.list_time_statistics()
    array = xr.DataArray(np.arange(6).reshape(2, 3), dims=("time", "lead_time"))
    np.testing.assert_array_equal(e2s.apply_time_statistic(array, "range:3h"), [2, 2])
    with pytest.raises(ValueError, match="no 'time'"):
        e2s.apply_time_statistic(xr.DataArray([1, 2], dims="sample"), "mean:2h")
