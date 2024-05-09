# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import xarray as xr

from earth2studio.io import XarrayBackend
from earth2studio.utils.coords import split_coords


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_xarray_fields(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test Memory Store
    z = XarrayBackend(total_coords)
    assert isinstance(z.root, xr.Dataset)

    array_name = "fields"
    z.add_array(total_coords, array_name)
    for dim in total_coords:
        assert dim in z.coords
        assert z.coords[dim].shape == total_coords[dim].shape

    # Test __contains__
    assert array_name in z

    # Test __getitem__
    shape = tuple([len(dim) for dim in total_coords.values()])
    assert z[array_name].shape == shape

    # Test __len__
    assert len(z) == 5

    # Test __iter__
    for array in z:
        assert array in ["fields", "time", "variable", "lat", "lon"]

    # Test add_array with torch.Tensor
    z.add_array(
        total_coords,
        "dummy_1",
        data=torch.randn(shape, device=device, dtype=torch.float32),
    )

    assert "dummy_1" in z
    assert z["dummy_1"].shape == shape

    # Test writing

    # Test full write
    x = torch.randn(shape, device=device, dtype=torch.float32)
    z.write(x, total_coords, "fields_1")
    assert "fields_1" in z
    assert z["fields_1"].shape == x.shape

    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)
    z.write(partial_data, partial_coords, array_name)
    assert np.allclose(z[array_name][0, 0, :, :180], partial_data.cpu().numpy())

    xx, _ = z.read(partial_coords, array_name, device=device)
    assert torch.allclose(partial_data, xx)


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_xarray_variable(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Remove var names
    coords = total_coords.copy()
    var_names = coords.pop("variable")

    # Test Memory Store
    z = XarrayBackend(coords)
    assert isinstance(z.root, xr.Dataset)

    z.add_array(coords, var_names)
    for dim in coords:
        assert z.coords[dim].shape == coords[dim].shape

    for var_name in var_names:
        assert var_name in z
        assert z[var_name].shape == tuple([len(values) for values in coords.values()])

    # Test writing
    partial_coords = OrderedDict(
        {
            "time": np.asarray(time)[:1],
            "variable": np.asarray(variable)[:1],
            "lat": total_coords["lat"],
            "lon": total_coords["lon"][:180],
        }
    )
    partial_data = torch.randn((1, 1, 180, 180), device=device)

    z.write(*split_coords(partial_data, partial_coords, "variable"))
    assert np.allclose(z[variable[0]][0, :, :180], partial_data.cpu().numpy())


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_xarray_exceptions(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    # Test Memory Store
    z = XarrayBackend(total_coords)
    assert isinstance(z.root, xr.Dataset)

    # Test mismatch between len(array_names) and len(data)
    shape = tuple([len(values) for values in total_coords.values()])
    array_name = "fields"
    dummy = torch.randn(shape, device=device, dtype=torch.float32)
    with pytest.raises(ValueError):
        z.add_array(total_coords, array_name, data=[dummy] * 2)

    # Test trying to add the same array twice.
    z.add_array(
        total_coords,
        ["dummy_1"],
        data=[dummy],
    )
    with pytest.raises(AssertionError):
        z.add_array(
            total_coords,
            ["dummy_1"],
            data=[dummy],
        )

    # Try to write with bad coords
    bad_coords = {"ensemble": np.arange(0)} | total_coords
    bad_shape = (1,) + shape
    dummy = torch.randn(bad_shape, device=device, dtype=torch.float32)
    with pytest.raises(AssertionError):
        z.write(dummy, bad_coords, "dummy_1")

    # Try to write with too many array names
    with pytest.raises(ValueError):
        z.write([dummy, dummy], bad_coords, "dummy_1")
