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
from earth2studio.models.px.fuxi_s2s import DAILY_VARIABLES, FuXiS2S
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
    assert out["variable"].values.tolist() == ["tp:sum:6h"]
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
                x, {"variable": ["u:mean:6h", "v"]}
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


def make_fuxi() -> FuXiS2S:
    model = FuXiS2S.__new__(FuXiS2S)
    torch.nn.Module.__init__(model)
    model.register_buffer("device_buffer", torch.empty(0))
    model._time_step = np.timedelta64(1, "D")
    signature = model.input_coords()
    model.input_coords = lambda: signature.isel(lat=slice(2), lon=slice(3))
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fuxi_array_rollout(device: str) -> None:
    if device.startswith("cuda"):
        pytest.importorskip("cupy")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    model = make_fuxi()
    model.device_buffer = model.device_buffer.to(device)
    signature = coord_array_like(
        model.input_coords(),
        {
            "batch": [2, 7],
            "time": np.array(["2020-01-01", "2020-06-01"], dtype="datetime64[ns]"),
        },
    ).rename(batch="member")
    x = from_torch(torch.ones(signature.shape, device=device), signature, name="daily")
    x = x.assign_coords(lead_time=x.lead_time + np.timedelta64(10, "D"), source="test")
    x.encoding["test"] = "retained"
    steps = []

    def forward(tensor: torch.Tensor, coords: xr.DataArray, step: int) -> torch.Tensor:
        steps.append(step)
        return torch.cat((tensor[:, :, -1:], tensor[:, :, -1:] + 1), dim=2)

    model._forward = forward
    output = model(x)
    assert output.dims == x.dims
    assert output.e2s.to_torch()[0].device == torch.device(device)
    assert output.encoding == x.encoding
    assert (
        output.attrs["earth2studio_statistics"]
        == signature.attrs["earth2studio_statistics"]
    )
    assert "earth2studio_kind" not in output.attrs
    np.testing.assert_array_equal(
        output.lead_time, np.array([11], dtype="timedelta64[D]")
    )
    hooks = []

    def rear(y: xr.DataArray) -> xr.DataArray:
        hooks.append(y.dims)
        return y.copy(data=y.data + 2)

    model.rear_hook = rear
    iterator = model.create_iterator(x)
    xr.testing.assert_identical(next(iterator), x.isel(lead_time=slice(-1, None)))
    first = next(iterator)
    second = next(iterator)
    torch.testing.assert_close(
        first.e2s.to_torch()[0], torch.full(first.shape, 4.0, device=device)
    )
    torch.testing.assert_close(
        second.e2s.to_torch()[0], torch.full(second.shape, 7.0, device=device)
    )
    assert steps == [10, 10, 11]
    assert hooks == [x.dims, x.dims]
    assert second.encoding == x.encoding
    torch.testing.assert_close(x.e2s.to_torch()[0], torch.ones(x.shape, device=device))
    iterator.close()


def test_fuxi_signature_statistics() -> None:
    model = make_fuxi()
    signature = model.input_coords()
    output = model.output_coords(signature)
    assert signature.data.nbytes == output.data.nbytes == 0
    assert signature["variable"].values.tolist() == DAILY_VARIABLES
    assert (
        output.attrs["earth2studio_statistics"]["tp:mean:1h:25h"]["modifier"]
        == "mean:+1h:+25h"
    )
    with pytest.raises(ValueError):
        model.output_coords(signature.assign_coords(lead_time=[0, 1]))
    with pytest.raises(ValueError):
        model.output_coords(
            signature.assign_coords(
                lead_time=np.array(["NaT", "NaT"], dtype="timedelta64[D]")
            )
        )
