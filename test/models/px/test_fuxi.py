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

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.auto import Package
from earth2studio.models.px import FuXi
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def test_fuxi_call_resets_short_session(monkeypatch):
    monkeypatch.setattr(
        "earth2studio.models.px.fuxi.create_ort_session",
        lambda path, device: SimpleNamespace(_model_path=path),
    )
    model = FuXi.__new__(FuXi)
    FuXi.__init__.__wrapped__(model, "short", "medium", "long")
    signature = model.input_coords()
    assert signature.data.nbytes == 0
    assert signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        signature.dims,
        {"lead_time": signature.lead_time, "variable": signature.coords["variable"]},
        dynamic=("batch", "time"),
        grid=LatLonGrid([45, -45], [0, 120, 240]),
    )
    monkeypatch.setattr(model, "input_coords", lambda: signature.copy())
    monkeypatch.setattr(
        model,
        "_forward",
        lambda x, coords, session: x
        + {"short": 1, "medium": 2, "long": 3}[session._model_path],
    )
    coords = coord_array_like(
        signature,
        {"batch": [0, 1], "time": np.array(["2000-01-01"], dtype="datetime64[ns]")},
    )
    x = from_torch(torch.randn(coords.shape), coords, name="weather").rename(
        batch="member"
    )
    x.encoding = {"source": "fixture"}
    original = x.copy(deep=True)
    iterator = model.create_iterator(x)
    initial = next(iterator)
    total = 0
    for step in range(42):
        out = next(iterator)
        total += 1 if step < 20 else 2 if step < 40 else 3
        np.testing.assert_allclose(out.data, initial.data + total, rtol=1e-5)
        assert out.lead_time.values[0] == np.timedelta64((step + 1) * 6, "h")
        assert out.dims == x.dims and out.encoding == x.encoding
    assert model.ort._model_path == "long"
    out = model(x)
    assert model.ort._model_path == "short"
    np.testing.assert_allclose(out.data, x.isel(lead_time=slice(-1, None)).data + 1)
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(initial, original.isel(lead_time=slice(-1, None)))


class PhooFuXiModel(torch.nn.Module):
    def __init__(self, model_type="short"):
        super().__init__()
        self.delta_t = {"short": 1, "medium": 2, "long": 3}[model_type]

    def forward(self, x, y):
        return x + self.delta_t + 0 * y[0, 0]


@pytest.fixture(scope="module")
def fuxi_test_package(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tmp_path = tmp_path_factory.mktemp("fuxi-onnx")
    for model_type in ("short", "medium", "long"):
        torch.onnx.export(
            PhooFuXiModel(model_type),
            (torch.rand(1, 2, 70, 721, 1440), torch.rand(1, 12)),
            str(tmp_path / f"{model_type}.onnx"),
            export_params=True,
            opset_version=10,
            dynamo=False,
            input_names=["input", "temb"],
            output_names=["output"],
        )
        (tmp_path / model_type).touch()
    return Package(str(tmp_path))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_fuxi_onnx_integration(device, fuxi_test_package):
    model = FuXi.load_model(fuxi_test_package).to(device)
    signature = coord_array_like(
        model.input_coords(),
        {"batch": [0], "time": np.array(["1999-10-11T12:00"], dtype="datetime64[ns]")},
    )
    tensor = torch.rand(signature.shape, device=device)
    tensor[..., -1, :, :] *= 0.001
    x = from_torch(tensor, signature).rename(batch="ensemble")
    strided = from_torch(
        tensor.transpose(2, 3).contiguous().transpose(2, 3), signature
    ).rename(batch="ensemble")
    out = model(x)
    assert torch.allclose(out.e2s.to_torch()[0], model(strided).e2s.to_torch()[0])
    assert out.shape == (1, 1, 1, 70, 721, 1440)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
    np.testing.assert_array_equal(out.time, x.time)
    iterator = model.create_iterator(x)
    initial = next(iterator).e2s.to_torch()[0].clone()
    total = 0
    for step in range(43):
        out = next(iterator)
        total += 1 if step < 20 else 2 if step < 40 else 3
        actual = out.e2s.to_torch()[0]
        assert torch.allclose(actual[..., :-1, :, :], initial[..., :-1, :, :] + total)
        assert torch.allclose(
            actual[..., -1, :, :], initial[..., -1, :, :] + total / 1000, atol=1e-6
        )
        assert out.lead_time.values[0] == np.timedelta64((step + 1) * 6, "h")
        assert out.dims == x.dims
    out = model(x)
    assert torch.allclose(
        out.e2s.to_torch()[0][..., :-1, :, :], initial[..., :-1, :, :] + 1
    )


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fuxi_package():
    model = FuXi.load_model(FuXi.load_default_package()).to("cuda:0")
    signature = coord_array_like(
        model.input_coords(),
        {"batch": [0], "time": np.array(["2000-01-01"], dtype="datetime64[ns]")},
    )
    x = from_torch(torch.zeros(signature.shape, device="cuda:0"), signature)
    out = model(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 1, 70, 721, 1440)
    np.testing.assert_array_equal(out.lead_time, model.output_coords(x).lead_time)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
