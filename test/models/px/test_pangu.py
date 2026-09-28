# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.auto import Package
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import Pangu3, Pangu6, Pangu24
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def make_model(name, monkeypatch):
    monkeypatch.setattr(
        "earth2studio.models.px.pangu.create_ort_session",
        lambda path, device: SimpleNamespace(_model_path=path),
    )
    cls = {"Pangu3": Pangu3, "Pangu6": Pangu6, "Pangu24": Pangu24}[name]
    model = cls.__new__(cls)
    cls.__init__.__wrapped__(
        model,
        *{"Pangu3": ("24", "6", "3"), "Pangu6": ("24", "6"), "Pangu24": ("24",)}[name],
    )
    signature = model.input_coords()
    assert signature.data.nbytes == 0
    assert signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        signature.dims,
        {"lead_time": signature.lead_time, "variable": signature.coords["variable"]},
        dynamic=("batch",),
        grid=LatLonGrid([45, -45], [0, 120, 240]),
    )
    monkeypatch.setattr(model, "input_coords", lambda: signature.copy())
    monkeypatch.setattr(
        model, "_forward", lambda x, session: x + int(session._model_path)
    )
    return model


def make_input(model):
    coords = coord_array_like(
        model.input_coords(),
        {"batch": [0, 1], "lead_time": np.array([12], dtype="timedelta64[h]")},
    )
    x = from_torch(torch.randn(coords.shape), coords, name="weather").rename(
        batch="member"
    )
    x.encoding = {"source": "fixture"}
    return x


@pytest.mark.parametrize("name", ["Pangu3", "Pangu6", "Pangu24"])
def test_pangu_sessions_cached(name, monkeypatch):
    model = make_model(name, monkeypatch)
    x = make_input(model)
    original = x.copy(deep=True)
    iterator = model.create_iterator(x)
    initial = next(iterator)
    for step in range(1, 10):
        out = next(iterator)
        np.testing.assert_allclose(
            out.data, initial.data + step * int(name[5:]), rtol=1e-5
        )
        assert out.lead_time.values[0] == np.timedelta64(12 + step * int(name[5:]), "h")
        assert out.name == x.name and out.encoding == x.encoding
    session = model.ort if name == "Pangu24" else model._ort24_session
    assert session is not None
    for _ in range(10):
        next(iterator)
    assert (model.ort if name == "Pangu24" else model._ort24_session) is session
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(initial, original)
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]


def test_pangu_call_does_not_copy_rollout_anchors(monkeypatch):
    import xarray as xr

    model = make_model("Pangu3", monkeypatch)
    x = make_input(model)
    original = xr.DataArray.copy

    def copy(field, deep=True, data=None):
        if field is x and deep:
            raise AssertionError("single call allocated rollout anchors")
        return original(field, deep=deep, data=data)

    monkeypatch.setattr(xr.DataArray, "copy", copy)
    model(x)


class PhooPanguModel(torch.nn.Module):
    def __init__(self, delta_t=24):
        super().__init__()
        self.delta_t = delta_t

    def forward(self, x, x_surface):
        return x + self.delta_t, x_surface + self.delta_t


@pytest.fixture(scope="module")
def onnx_test_package(tmp_path_factory):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    input_tensors = (torch.rand(5, 13, 721, 1440), torch.rand(4, 721, 1440))
    tmp_path = tmp_path_factory.mktemp("pangu-onnx")
    for delta_t in (24, 6, 3):
        torch.onnx.export(
            PhooPanguModel(delta_t),
            input_tensors,
            str(tmp_path / f"pangu_weather_{delta_t}.onnx"),
            export_params=True,
            opset_version=10,
            dynamo=False,
            input_names=["input", "input_surface"],
            output_names=["output", "output_surface"],
        )
    return Package(str(tmp_path))


@pytest.mark.parametrize("cls, delta_t", [(Pangu3, 3), (Pangu6, 6), (Pangu24, 24)])
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
def test_pangu_onnx_integration(cls, delta_t, device, onnx_test_package):
    model = cls.load_model(onnx_test_package).to(device)
    signature = coord_array_like(model.input_coords(), {"batch": [4]})
    tensor = torch.rand(signature.shape, device=device)
    x = from_torch(tensor, signature).rename(batch="ensemble")
    x = x.expand_dims(time=np.array(["1999-10-11T12:00"], dtype="datetime64[ns]"))
    out = model(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 1, 69, 721, 1440)
    assert torch.allclose(out.e2s.to_torch()[0], x.e2s.to_torch()[0] + delta_t)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
    np.testing.assert_array_equal(out.time, x.time)
    iterator = model.create_iterator(x)
    initial = next(iterator)
    for step in range(1, 10):
        out = next(iterator)
        assert torch.allclose(
            out.e2s.to_torch()[0],
            initial.e2s.to_torch()[0] + step * delta_t,
            atol=1e-3,
            rtol=1e-3,
        )
        assert out.lead_time.values[0] == np.timedelta64(step * delta_t, "h")
        assert out.dims == x.dims


@pytest.mark.package
@pytest.mark.parametrize("cls", [Pangu3, Pangu6, Pangu24])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_pangu_package(cls):
    model = cls.load_model(cls.load_default_package()).to("cuda:0")
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    x = from_torch(torch.zeros(signature.shape, device="cuda:0"), signature)
    out = model(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 69, 721, 1440)
    np.testing.assert_array_equal(out.lead_time, model.output_coords(x).lead_time)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
