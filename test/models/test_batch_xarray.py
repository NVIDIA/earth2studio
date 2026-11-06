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

from earth2studio.models.batch import batch_func
from earth2studio.utils import coord_array


class ArrayModel:
    def input_coords(self):
        return coord_array(
            ("batch", "variable", "lat"),
            {"variable": ["u", "v"], "lat": [1, 2, 3]},
            dynamic=("batch",),
        )

    @batch_func()
    def __call__(self, x, scale=2):
        assert x.dims == ("batch", "variable", "lat")
        result = x.isel(variable=[0]) * scale
        result.attrs = {"output": True}
        return result

    @batch_func()
    def iterate(self, x):
        try:
            scale = yield x
            try:
                yield x * scale
            except RuntimeError:
                yield x * 3
        finally:
            self.closed = True


def array_input(leading):
    dims = (*leading, "variable", "lat")
    coords = {dim: np.arange(2) + 10 for dim in leading}
    if "time" in leading:
        coords["time"] = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    coords.update(variable=["u", "v"], lat=[1, 2, 3], elevation=("lat", [7, 8, 9]))
    if leading:
        coords["valid"] = (leading, np.ones((2,) * len(leading)))
    return xr.DataArray(
        np.arange(2 ** len(leading) * 6, dtype=np.float32).reshape(
            (2,) * len(leading) + (2, 3)
        ),
        dims=dims,
        coords=coords,
        name="weather",
        attrs={"user": "metadata"},
    )


@pytest.mark.parametrize(
    "leading", [(), ("batch",), ("member", "time"), ("batch", "time")]
)
def test_array_batch_restores_coordinates(leading):
    x = array_input(leading)
    original = x.copy(deep=True)
    result = ArrayModel()(x, scale=4)
    expected = x.isel(variable=[0]) * 4
    expected.attrs = {"output": True}
    xr.testing.assert_identical(result, expected)
    xr.testing.assert_identical(x, original)


def test_array_batch_generator_protocol():
    x = array_input(("batch", "time"))
    model = ArrayModel()
    gen = model.iterate(x)
    xr.testing.assert_identical(next(gen), x)
    xr.testing.assert_equal(gen.send(2), x * 2)
    xr.testing.assert_equal(gen.throw(RuntimeError), x * 3)
    gen.close()
    assert model.closed


def test_array_batch_validates_dimension_order():
    x = array_input(("time",)).transpose("variable", "time", "lat")
    with pytest.raises(ValueError, match="dimension"):
        ArrayModel()(x)


def test_array_batch_rejects_changed_batch_size():
    class BadModel(ArrayModel):
        @batch_func()
        def __call__(self, x):
            return x.isel(batch=slice(1))

    with pytest.raises(ValueError, match="batch.*size"):
        BadModel()(array_input(("time",)))


def test_array_batch_keyword_call():
    x = array_input(("time",))
    xr.testing.assert_identical(ArrayModel()(x=x), ArrayModel()(x))


@pytest.mark.parametrize("leading", [(), ("time",)])
def test_array_batch_scalar_coordinate(leading):
    x = array_input(leading).assign_coords(batch=42)
    result = ArrayModel()(x)
    expected = x.isel(variable=[0]) * 2
    expected.attrs = {"output": True}
    xr.testing.assert_identical(result, expected)


def test_array_batch_rejects_reordering():
    class Reversed(ArrayModel):
        @batch_func()
        def __call__(self, x):
            return x.isel(batch=slice(None, None, -1))

    with pytest.raises(ValueError, match="batch"):
        Reversed()(array_input(("time",)))


def test_array_batch_cuda():
    import torch

    cp = pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = array_input(("member", "time")).e2s.as_cupy()
    result = ArrayModel()(x)
    assert isinstance(result.data, cp.ndarray)
    xr.testing.assert_equal(
        result.e2s.as_numpy(), (x.e2s.as_numpy() * 2).isel(variable=[0])
    )
