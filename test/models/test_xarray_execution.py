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

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.dx.precipitation_afno import PrecipitationAFNO
from earth2studio.models.px.fcn import FCN
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.checkpoint import Checkpoint
from earth2studio.utils.cupy import from_torch


def small_signature(prognostic: bool = True) -> xr.DataArray:
    coords = {"variable": ["u", "v"], "lat": [1, 2], "lon": [3, 4, 5]}
    if prognostic:
        coords = {"lead_time": np.array([0], dtype="timedelta64[h]"), **coords}
    return coord_array(("batch", *coords), coords, dynamic=("batch",))


def make_fcn() -> FCN:
    model = FCN(torch.nn.Identity(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))
    model.input_coords = small_signature
    return model


def input_array(signature: xr.DataArray) -> xr.DataArray:
    dims = ("member", "time", *signature.dims[1:])
    coords = {key: value for key, value in signature.coords.items()}
    coords.update(member=[2, 5], time=[10, 20], height=("lat", [8, 9]))
    return xr.DataArray(
        np.ones((2, 2, *signature.shape[1:]), dtype=np.float32),
        dims=dims,
        coords=coords,
        name="weather",
        attrs={"experiment": "test"},
    )


def test_from_torch_signature_metadata() -> None:
    x = input_array(small_signature())
    sig = coord_array(x.dims, x.coords, attrs=x.attrs)
    result = from_torch(torch.from_numpy(x.data), sig, name=x.name)
    xr.testing.assert_identical(result, x)
    assert result.data.__array_interface__["data"] == x.data.__array_interface__["data"]


def test_fcn_step_iterator_and_hooks() -> None:
    model = make_fcn()
    x = input_array(model.input_coords()).assign_coords(
        lead_time=np.array([12], dtype="timedelta64[h]")
    )
    planned = model.output_coords(x)
    assert planned.data.nbytes == 0
    out = model(x)
    xr.testing.assert_identical(
        out, x.assign_coords(lead_time=x.lead_time + np.timedelta64(6, "h"))
    )
    calls = []

    def hook(y: xr.DataArray) -> xr.DataArray:
        calls.append(y.dims)
        return y.copy(data=y.data + 1)

    model.front_hook = hook
    model.rear_hook = hook
    xr.testing.assert_identical(model(x), out)
    assert not calls
    gen = model.create_iterator(x)
    xr.testing.assert_identical(next(gen), x)
    second = next(gen)
    np.testing.assert_array_equal(second.data, x.data + 2)
    third = next(gen)
    np.testing.assert_array_equal(
        third.lead_time, np.array([24], dtype="timedelta64[h]")
    )
    model.clear_hooks()
    gen.close()
    assert len(calls) == 4


@pytest.mark.parametrize(
    "lead",
    [
        [1],
        np.array(["2020-01-01"], dtype="datetime64[D]"),
        np.array(["NaT"], dtype="timedelta64[h]"),
    ],
)
def test_fcn_rejects_invalid_lead_time(lead: list | np.ndarray) -> None:
    model = make_fcn()
    x = input_array(model.input_coords()).assign_coords(lead_time=lead)
    with pytest.raises(ValueError, match="lead_time"):
        model.output_coords(x)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_precipitation_array_execution(device: str) -> None:
    if device.startswith("cuda"):
        pytest.importorskip("cupy")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")

    class Core(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1]

    model = PrecipitationAFNO.__new__(PrecipitationAFNO)
    torch.nn.Module.__init__(model)
    model.core_model = Core()
    model.register_buffer("center", torch.zeros(2, 1, 1))
    model.register_buffer("scale", torch.ones(2, 1, 1))
    model.eps = 1e-5
    model.input_coords = lambda: small_signature(False)
    model.to(device)
    x = input_array(model.input_coords())
    if device.startswith("cuda"):
        x = x.e2s.as_cupy(device=0)
    out = model(x)
    assert out.e2s.is_cupy == x.e2s.is_cupy
    assert out.dims == x.dims
    assert out["variable"].values.tolist() == ["tp"]
    np.testing.assert_allclose(out.e2s.as_numpy().data, 1e-5 * np.expm1(1), rtol=1e-6)
    xr.testing.assert_identical(out.height, x.height)
    assert out.attrs["experiment"] == "test"
    assert "earth2studio_kind" not in out.attrs
    assert (
        model.output_coords(x).attrs["earth2studio_statistics"]
        == out.attrs["earth2studio_statistics"]
    )


def test_fcn_checkpoint_metadata_round_trip(tmp_path: Path) -> None:
    class Increment(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    x = input_array(small_signature())
    x.coords["height"].attrs["units"] = "m"
    x.encoding["test"] = "encoding"
    x.attrs.update(
        {
            key: value
            for key, value in coord_array_like(
                x, statistics={"u": "mean:6h"}
            ).attrs.items()
            if key == "earth2studio_statistics"
        }
    )
    with Checkpoint("fcn-array", path=tmp_path, flush_interval=1, level=2) as ckpt:
        model = make_fcn()
        model.model = Increment()
        iterator = model.create_iterator(x)
        next(iterator)
        saved = next(iterator)
        ckpt.write(lead_time=saved.lead_time.values[-1])
        expected = next(iterator)
        iterator.close()

    with Checkpoint("fcn-array", path=tmp_path, level=2).select(-1):
        model = make_fcn()
        model.model = Increment()
        assert model.checkpoint.checkpoint_state_loaded
        iterator = model.create_iterator(x.copy(data=x.data - 10))
        resumed = next(iterator)
        iterator.close()
    xr.testing.assert_identical(resumed, expected)
    assert resumed.encoding == x.encoding


def test_fcn_cuda_execution() -> None:
    cp = pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = make_fcn().cuda()
    x = input_array(model.input_coords()).e2s.as_cupy()
    out = model(x)
    assert isinstance(out.data, cp.ndarray)
    np.testing.assert_array_equal(out.data.get(), x.data.get())
