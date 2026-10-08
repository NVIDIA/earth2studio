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
import xarray as xr

from earth2studio.grids import E2S_CRS, LatLonGrid
from earth2studio.io.utils import output_template, plan_arrays, plan_read, plan_write
from earth2studio.models.px import Persistence
from earth2studio.run import _output_dimensions
from earth2studio.utils.coords import (
    E2S_DYNAMIC_DIMS,
    E2S_STATISTICS,
    coord_array,
)

GRID = LatLonGrid(np.array([1.0, 0.0, -1.0]), np.arange(4.0))
TIMES = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[s]")
LEADS = np.arange(3) * np.timedelta64(6, "h")


@pytest.fixture
def signature() -> xr.DataArray:
    model = Persistence(["t2m", "tp:sum:6h"], GRID)
    return model.output_coords(model.input_coords())


def test_output_template(signature: xr.DataArray) -> None:
    assert signature.attrs[E2S_DYNAMIC_DIMS] == ("batch",)
    template = output_template(
        signature, {"ensemble": [0, 1], "time": TIMES}, {"lead_time": LEADS}
    )
    assert template.dims == ("ensemble", "time", "lead_time", "variable", "lat", "lon")
    assert dict(template.sizes) == {
        "ensemble": 2,
        "time": 2,
        "lead_time": 3,
        "variable": 2,
        "lat": 3,
        "lon": 4,
    }
    assert template.attrs[E2S_DYNAMIC_DIMS] == ()
    assert template.attrs[E2S_CRS] == signature.attrs[E2S_CRS]
    assert set(template.attrs[E2S_STATISTICS]) == {"tp:sum:6h"}
    np.testing.assert_array_equal(template.lat, GRID.coords()["lat"])
    with pytest.raises(TypeError, match="field values"):
        template.values  # allocation-free


def test_output_template_without_dynamic_dims() -> None:
    concrete = coord_array(("variable",), {"variable": ["t2m"]})
    template = output_template(concrete, {"time": TIMES})
    assert template.dims == ("time", "variable")


def test_output_template_rejects(signature: xr.DataArray) -> None:
    with pytest.raises(ValueError, match="already fixed"):
        output_template(signature, {"lead_time": LEADS})
    with pytest.raises(ValueError, match="unknown"):
        output_template(signature, {"time": TIMES}, {"level": [500]})
    with pytest.raises(ValueError, match="spatial"):
        output_template(signature, {"time": TIMES}, {"lat": [0.0]})
    with pytest.raises(ValueError, match="nonempty"):
        output_template(signature, {"time": TIMES[:0]})


def test_output_dimensions_matches_legacy_plan() -> None:
    model = Persistence(["t2m"], GRID)
    coords = _output_dimensions(model, TIMES, 2)
    assert isinstance(coords, OrderedDict)
    assert list(coords) == ["time", "lead_time", "variable", "lat", "lon"]
    np.testing.assert_array_equal(coords["time"], TIMES)
    np.testing.assert_array_equal(coords["lead_time"], LEADS)


def test_plan_arrays(signature: xr.DataArray) -> None:
    template = output_template(signature, {"time": TIMES}, {"lead_time": LEADS})
    plan = plan_arrays(template)
    assert plan.names == ("t2m", "tp:sum:6h")
    assert plan.dims == ("time", "lead_time", "lat", "lon")
    assert plan.shape == (2, 3, 3, 4)
    assert "variable" not in plan.coords
    assert plan.coords["time"].dtype == np.dtype("datetime64[ns]")
    assert plan.dtype == np.float32
    assert np.isnan(plan.fill_value)
    assert E2S_STATISTICS not in plan.attrs
    assert E2S_DYNAMIC_DIMS not in plan.attrs
    assert plan.attrs[E2S_CRS] == signature.attrs[E2S_CRS]


def test_plan_arrays_existing_coords() -> None:
    template = coord_array(("lat", "lon"), {"lat": [0.0, np.nan], "lon": [1]}, name="a")
    existing = {"lat": xr.Variable("lat", [0.0, np.nan])}
    assert list(plan_arrays(template, existing).coords) == ["lon"]
    with pytest.raises(ValueError, match="lat"):
        plan_arrays(template, {"lat": xr.Variable("lat", [0.0, 1.0])})


def test_plan_arrays_rejects() -> None:
    with pytest.raises(ValueError, match="collide"):
        plan_arrays(coord_array(("sample",), {"sample": [0, 1]}, name="sample"))
    with pytest.raises(ValueError, match="collide"):
        plan_arrays(
            coord_array(("x",), {"x": [0]}, name="lat"),
            {"lat": xr.Variable("lat", [0.0])},
        )
    repeated = np.array(["2024-01-01"] * 2, dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="unique"):
        plan_arrays(coord_array(("time",), {"time": repeated}, name="a"))


def test_plan_write_indexers() -> None:
    arrays = {"t2m": {"time": 2, "lat": 3}}
    coords = {"time": TIMES.astype("datetime64[ns]"), "lat": np.array([1.0, 0.0, -1.0])}

    contiguous = xr.DataArray(
        np.zeros((1, 1, 2)),
        dims=("time", "variable", "lat"),
        coords={"time": TIMES[1:], "variable": ["t2m"], "lat": [0.0, -1.0]},
    )
    [(name, indexers, field)] = plan_write(contiguous, arrays, coords)
    assert name == "t2m"
    assert indexers == {"time": slice(1, 2), "lat": slice(1, 3)}
    assert field.dims == ("time", "lat")

    gapped = xr.DataArray(
        np.arange(2.0).reshape(1, 2),
        dims=("time", "lat"),
        coords={"time": TIMES[:1], "lat": [-1.0, 1.0]},
        name="t2m",
    )
    [(_, indexers, field)] = plan_write(gapped, arrays, coords)
    np.testing.assert_array_equal(indexers["lat"], [0, 2])
    np.testing.assert_array_equal(field.lat, [1.0, -1.0])
    np.testing.assert_array_equal(field.values, [[1.0, 0.0]])


def test_plan_write_unlabelled_dimensions() -> None:
    arrays = {"t2m": {"sample": 3}}
    full = xr.DataArray(np.zeros(3), dims="sample", name="t2m")
    [(_, indexers, _)] = plan_write(full, arrays, {})
    assert indexers == {"sample": slice(None)}
    with pytest.raises(ValueError, match="no labels"):
        plan_write(full.assign_coords(sample=[0, 1, 2]), arrays, {})
    with pytest.raises(ValueError, match="span all 3"):
        plan_write(full.isel(sample=[0]), arrays, {})
    with pytest.raises(ValueError, match="span all 3"):
        plan_write(full.isel(sample=[0]), arrays, {"sample": np.arange(3)})


def test_plan_read() -> None:
    arrays = {"t2m": {"time": 2, "lat": 3}, "u10m": {"time": 2, "lat": 3}}
    coords = {"time": TIMES.astype("datetime64[ns]"), "lat": np.array([1.0, 0.0, -1.0])}
    selection = {"lat": [-1.0, 1.0], "variable": ["u10m"], "time": TIMES}
    selection, names, indexers = plan_read(selection, arrays, coords)
    assert selection.dims == ("lat", "variable", "time")
    assert names == ("u10m",)
    assert indexers["time"] == slice(0, 2)
    np.testing.assert_array_equal(indexers["lat"], [2, 0])  # requested order
    with pytest.raises(ValueError, match="differ"):
        plan_read({"variable": ["t2m"], "lat": [0.0]}, arrays, coords)
