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

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from earth2studio.models.da import CorrDiffCosmoEra5SDA, HealDA
from earth2studio.utils import coord_array, coord_array_like


@pytest.mark.parametrize("lat_lon", [False, True])
def test_healda_planning_without_weights(lat_lon):
    model = object.__new__(HealDA)
    model._lat_lon = lat_lon
    model._model = SimpleNamespace(npix=12)
    model._output_lat = np.array([30.0, 40.0])
    model._output_lon = np.array([240.0, 250.0])
    assert model.input_coords() is None
    conv, sat = model.forcing_coords()
    assert "pres" in conv
    assert "sensor_index" in sat
    times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    output = model.output_coords(request_time=times)
    assert isinstance(output, xr.DataArray)
    assert output.dims == (
        ("time", "variable", "lat", "lon") if lat_lon else ("time", "variable", "npix")
    )
    np.testing.assert_array_equal(output.time, times)


def test_corrdiff_planning_without_weights():
    model = object.__new__(CorrDiffCosmoEra5SDA)
    native = coord_array(
        ("batch", "time", "variable", "lat", "lon"),
        {"variable": ["u10m"], "lat": [30.0, 40.0], "lon": [240.0, 250.0]},
        dynamic=("batch", "time"),
    )
    target = coord_array(
        ("time", "sample", "variable", "y", "x"),
        {
            "sample": [0],
            "variable": ["u10m"],
            "y": [0],
            "x": [0],
            "lat": (("y", "x"), [[35.0]]),
            "lon": (("y", "x"), [[245.0]]),
        },
        dynamic=("time",),
    )
    model.model = SimpleNamespace(
        input_coords=Mock(return_value=native), output_coords=Mock(return_value=target)
    )
    model.assimilate_variables = ["u10m"]
    model.number_of_samples = 2
    state = model.input_coords()
    assert state.dims == ("time", "variable", "lat", "lon")
    assert isinstance(model.forcing_coords(), dict)
    output = model.output_coords(state)
    assert output.attrs["earth2studio_dynamic_dims"] == ("time",)
    times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    output = model.output_coords(coord_array_like(state, {"time": times}))
    assert output.shape == (2, 2, 1, 1, 1)
    np.testing.assert_array_equal(output.time, times)
    assert output.lat.dims == ("y", "x")
