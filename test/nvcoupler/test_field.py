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

from earth2studio.nvcoupler import (
    DEFAULT_DICTIONARY,
    CouplingError,
    Field,
    State,
    UnknownFieldError,
)


def _grid_coords(nlat=8, nlon=16):
    return OrderedDict(
        {
            "lat": np.linspace(90, -90, nlat),
            "lon": np.linspace(0, 360, nlon, endpoint=False),
        }
    )


def _field(name="sea_surface_temperature", units="K", value=1.0, nlat=8, nlon=16):
    return Field(
        data=torch.full((nlat, nlon), value),
        coords=_grid_coords(nlat, nlon),
        standard_name=name,
        units=units,
    )


def test_field_validation():
    # dims mismatch
    with pytest.raises(CouplingError):
        Field(torch.zeros(4), _grid_coords(), "sea_surface_temperature", "K")
    # variable dim forbidden
    coords = OrderedDict({"variable": np.array(["sst"]), **_grid_coords()})
    with pytest.raises(CouplingError):
        Field(torch.zeros(1, 8, 16), coords, "sea_surface_temperature", "K")


def test_field_clone_and_grid_signature():
    f = _field()
    g = f.clone()
    g.data += 1.0
    assert torch.all(f.data == 1.0)  # clone is independent
    assert f.grid_signature() == g.grid_signature()
    assert f.grid_signature() != _field(nlat=4, nlon=8).grid_signature()


def test_state_mapping_and_subset():
    s = State("imports", [_field(), _field("air_temperature_2m", "K", 2.0)])
    assert len(s) == 2
    assert s["sea_surface_temperature"].data.mean() == 1.0
    sub = s.subset(["air_temperature_2m"])
    assert list(sub) == ["air_temperature_2m"]
    with pytest.raises(KeyError) as err:
        s["geopotential_at_500hpa"]
    assert "imports" in str(err.value)
    # key must equal standard_name
    with pytest.raises(CouplingError):
        s["wrong_key"] = _field()


def test_as_tensor_from_tensor_roundtrip():
    s = State(
        "exports",
        [_field("sea_surface_temperature", "K", 1.0), _field("air_temperature_2m", "K", 2.0)],
    )
    x, coords = s.as_tensor(["sea_surface_temperature", "air_temperature_2m"])
    # variable inserted before spatial dims
    assert list(coords) == ["variable", "lat", "lon"]
    assert x.shape == (2, 8, 16)
    assert torch.all(x[0] == 1.0) and torch.all(x[1] == 2.0)

    s2 = State.from_tensor("roundtrip", x, coords, DEFAULT_DICTIONARY)
    assert sorted(s2) == ["air_temperature_2m", "sea_surface_temperature"]
    assert torch.equal(
        s2["sea_surface_temperature"].data, s["sea_surface_temperature"].data
    )
    assert list(s2["air_temperature_2m"].coords) == ["lat", "lon"]


def test_as_tensor_grid_mismatch_raises():
    s = State("bad", [_field(), _field("air_temperature_2m", nlat=4, nlon=8)])
    with pytest.raises(ValueError):
        s.as_tensor(["sea_surface_temperature", "air_temperature_2m"])


def test_from_tensor_alias_resolution_and_strict():
    coords = OrderedDict({"variable": np.array(["sst", "mystery_var"]), **_grid_coords()})
    x = torch.zeros(2, 8, 16)
    with pytest.raises(UnknownFieldError):
        State.from_tensor("s", x, coords, DEFAULT_DICTIONARY)
    s = State.from_tensor("s", x, coords, DEFAULT_DICTIONARY, strict=False)
    assert list(s) == ["sea_surface_temperature"]  # alias resolved, unknown skipped


def test_from_tensor_with_leading_dims():
    coords = OrderedDict(
        {
            "time": np.array([np.datetime64("2024-01-01")]),
            "variable": np.array(["z1000", "t2m"]),
            **_grid_coords(),
        }
    )
    x = torch.arange(2 * 8 * 16, dtype=torch.float32).reshape(1, 2, 8, 16)
    s = State.from_tensor("s", x, coords, DEFAULT_DICTIONARY)
    f = s["geopotential_at_1000hpa"]
    assert list(f.coords) == ["time", "lat", "lon"]
    assert f.data.shape == (1, 8, 16)
    # stacking back inserts variable before lat (after time)
    y, ycoords = s.as_tensor()
    assert list(ycoords) == ["time", "variable", "lat", "lon"]
