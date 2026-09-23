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

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.auto import Package
from earth2studio.models.px import FengWu
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def test_fengwu_normalization(monkeypatch):
    class Session:
        def __init__(self):
            self.inputs, self.outputs = {}, {}

        def io_binding(self):
            return self

        def bind_input(self, name, shape, buffer_ptr, **kwargs):
            self.inputs[name] = np.ctypeslib.as_array(
                (ctypes.c_float * int(np.prod(shape))).from_address(buffer_ptr)
            ).reshape(shape)

        def bind_output(self, name, shape, buffer_ptr, **kwargs):
            self.outputs[name] = np.ctypeslib.as_array(
                (ctypes.c_float * int(np.prod(shape))).from_address(buffer_ptr)
            ).reshape(shape)

        def synchronize_inputs(self):
            pass

        def synchronize_outputs(self):
            pass

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def run_with_iobinding(self, binding):
            self.outputs["output"][...] = np.concatenate(
                (self.inputs["input"][:, 69:] + 6, self.inputs["input"][:, :69]), axis=1
            )

    monkeypatch.setattr(
        "earth2studio.models.px.fengwu.create_ort_session",
        lambda path, device: Session(),
    )
    model = FengWu.__new__(FengWu)
    FengWu.__init__.__wrapped__(
        model, "6", torch.full((69,), 3.0), torch.full((69,), 2.0)
    )
    declared = model.input_coords()
    assert declared.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        declared.dims,
        {"lead_time": declared.lead_time, "variable": declared.coords["variable"]},
        dynamic=("batch",),
        grid=LatLonGrid([45, -45], [0, 120, 240]),
    )
    monkeypatch.setattr(model, "input_coords", lambda: signature.copy())
    coords = coord_array_like(signature, {"batch": [0, 1]})
    x = from_torch(torch.randn(coords.shape), coords, name="weather").rename(
        batch="member"
    )
    x.data = np.ascontiguousarray(x.data.swapaxes(-1, -2)).swapaxes(-1, -2)
    assert not x.data.flags.c_contiguous
    x.encoding = {"source": "fixture"}
    original = x.copy(deep=True)
    out = model(x)
    np.testing.assert_allclose(
        out.data, x.isel(lead_time=slice(-1, None)).data + 12, rtol=1e-5
    )
    iterator = model.create_iterator(x)
    initial = next(iterator)
    for step in range(1, 4):
        out = next(iterator)
        np.testing.assert_allclose(out.data, initial.data + 12 * step, rtol=1e-5)
        assert out.dims == x.dims and out.encoding == x.encoding
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(initial, original.isel(lead_time=slice(-1, None)))


class PhooFengWuModel(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x[:, 69:], torch.empty_like(x[:, 69:])], dim=1) + 6


@pytest.fixture(scope="module")
def fengwu_test_package(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tmp_path = tmp_path_factory.mktemp("fengwu-onnx")
    torch.onnx.export(
        PhooFengWuModel(),
        torch.rand(2, 138, 721, 1440),
        str(tmp_path / "fengwu_v1.onnx"),
        export_params=True,
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    np.save(tmp_path / "global_means.npy", np.zeros(69))
    np.save(tmp_path / "global_stds.npy", np.ones(69))
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
def test_fengwu_onnx_integration(device, fengwu_test_package):
    model = FengWu.load_model(fengwu_test_package).to(device)
    signature = coord_array_like(model.input_coords(), {"batch": [0, 1]})
    tensor = torch.rand(signature.shape, device=device)
    x = from_torch(tensor, signature).rename(batch="ensemble")
    strided = from_torch(
        tensor.transpose(1, 2).contiguous().transpose(1, 2), signature
    ).rename(batch="ensemble")
    out = model(x)
    assert torch.allclose(out.e2s.to_torch()[0], model(strided).e2s.to_torch()[0])
    assert out.shape == (2, 1, 69, 721, 1440)
    assert out.dims == x.dims
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
    iterator = model.create_iterator(x)
    initial = next(iterator)
    for step in range(1, 6):
        out = next(iterator)
        assert torch.allclose(
            out.e2s.to_torch()[0], initial.e2s.to_torch()[0] + 6 * step
        )
        assert out.lead_time.values[0] == np.timedelta64(step * 6, "h")


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fengwu_package():
    model = FengWu.load_model(FengWu.load_default_package()).to("cuda:0")
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    x = from_torch(torch.zeros(signature.shape, device="cuda:0"), signature)
    out = model(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 69, 721, 1440)
    np.testing.assert_array_equal(out.lead_time, model.output_coords(x).lead_time)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
