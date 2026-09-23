# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import DLWP
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def make_model(name, monkeypatch):
    model = DLWP.__new__(DLWP)
    static = torch.zeros(6, 2, 2)
    DLWP.__init__.__wrapped__(
        model,
        torch.nn.Identity(),
        static,
        static,
        static,
        static,
        torch.eye(24),
        torch.eye(24),
        torch.zeros(1, 7, 1, 1),
        torch.ones(1, 7, 1, 1),
    )
    signature = model.input_coords()
    assert signature.data.nbytes == 0
    assert signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        signature.dims,
        {"lead_time": signature.lead_time, "variable": signature.coords["variable"]},
        dynamic=("batch", "time"),
        grid=LatLonGrid(np.linspace(90, -90, 4), np.arange(6) * 60),
    )
    monkeypatch.setattr(model, "input_coords", lambda: signature.copy())
    monkeypatch.setattr(
        model, "to_cubedsphere", lambda x: x.reshape(*x.shape[:-2], 6, 2, 2)
    )
    monkeypatch.setattr(
        model, "to_equirectangular", lambda x: x.reshape(*x.shape[:-3], 4, 6)
    )
    monkeypatch.setattr(
        model,
        "_forward",
        lambda x, coords: torch.cat((x[:, :, -1:] + 6, x[:, :, -1:] + 12), dim=2),
    )
    return model


def make_input(model):
    coords = coord_array_like(
        model.input_coords(),
        {
            "batch": [0, 1],
            "time": np.array(["2000-01-01", "2001-02-03"], dtype="datetime64[ns]"),
            "lead_time": np.array([6, 12], dtype="timedelta64[h]"),
        },
    )
    x = from_torch(torch.randn(coords.shape), coords, name="weather").rename(
        batch="member"
    )
    x = x.expand_dims(sample=1, axis=1).assign_coords(
        experiment=("member", [7, 8]),
        terrain=(("lat", "lon"), np.arange(24).reshape(4, 6)),
    )
    x.attrs["source"] = "fixture"
    x.encoding = {"source": "fixture"}
    return x


def test_dlwp_declares_core_hook_cadence(monkeypatch):
    model = make_model("DLWP", monkeypatch)
    assert model.front_hook_interval == 2
    calls = []
    model.front_hook = lambda x: calls.append("front") or x
    model.rear_hook = lambda x: calls.append("rear") or x
    iterator = model.create_iterator(make_input(model))
    next(iterator)
    for _ in range(4):
        next(iterator)
    assert calls == ["front", "rear", "rear", "front", "rear", "rear"]
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]


def test_dlwp_core_prescriptive_fields_and_history(monkeypatch):
    model = make_model("DLWP", monkeypatch)
    monkeypatch.delattr(model, "_forward")
    times = []

    def zenith(time_array, lead_time, device):
        times.append(time_array + np.timedelta64(lead_time))
        return torch.zeros(len(time_array), 6, 2, 2, device=device)

    monkeypatch.setattr(model, "get_cosine_zenith_fields", zenith)

    class Net(torch.nn.Module):
        def forward(self, x):
            assert x.shape[1:] == (18, 6, 2, 2)
            return torch.cat((x[:, 8:15] + 6, x[:, 8:15] + 12), dim=1)

    model.model = Net()
    x = make_input(model)
    iterator = model.create_iterator(x)
    initial = next(iterator)
    for i in range(1, 5):
        out = next(iterator)
        np.testing.assert_allclose(out.data, initial.data + 6 * i, rtol=1e-5)
    np.testing.assert_array_equal(
        times[0], np.tile(x.time.values + np.timedelta64(6, "h"), 2)
    )
    np.testing.assert_array_equal(
        times[1], np.tile(x.time.values + np.timedelta64(12, "h"), 2)
    )
    np.testing.assert_array_equal(times[2], times[0] + np.timedelta64(12, "h"))


def test_dlwp_leading_x(monkeypatch):
    model = make_model("DLWP", monkeypatch)
    x = make_input(model).rename(member="x")
    out = model(x)
    assert out.dims == x.dims
    assert "sample" not in out.coords and out.sizes["sample"] == 1
    xr.testing.assert_identical(out.x, x.x)
    iterator = model.create_iterator(x)
    next(iterator)
    assert next(iterator).dims == x.dims


def test_dlwp_initial_yield_does_not_transform(monkeypatch):
    model = make_model("DLWP", monkeypatch)
    x = make_input(model)

    def fail(*args):
        raise AssertionError("initial yield must not process fields")

    monkeypatch.setattr(model, "to_cubedsphere", fail)
    xr.testing.assert_identical(
        next(model.create_iterator(x)), x.isel(lead_time=slice(-1, None))
    )


@pytest.mark.parametrize("hook_slot", ["front_hook", "rear_hook"])
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
def test_dlwp_latest_hook_metadata(monkeypatch, hook_slot, device):
    model = make_model("DLWP", monkeypatch).to(device)
    x = make_input(model)
    if device != "cpu":
        x = x.e2s.as_cupy()
    count = 0
    front_seen = []
    original = x.copy(deep=True)

    def front(field):
        front_seen.append(field.copy(deep=True))
        return rear(field) if hook_slot == "front_hook" else field

    def rear(field):
        nonlocal count
        count += 1
        field.data += 1
        field = field.rename(None).drop_vars("experiment", errors="ignore")
        field.attrs.pop("source", None)
        field.encoding.clear()
        return field

    model.front_hook = front
    if hook_slot == "rear_hook":
        model.rear_hook = rear
    direct = model(x)
    assert count == 0 and direct.name == x.name
    iterator = model.create_iterator(x)
    initial = next(iterator)
    retained = []
    for i in range(1, 4):
        out = next(iterator)
        retained.append((out, out.copy(deep=True)))
        assert out.name is None and "source" not in out.attrs
        assert out.encoding == {} and "experiment" not in out.coords
        assert "sample" not in out.coords and out.dims == x.dims
        assert out.e2s.to_torch()[0].device == torch.device(device)
        xr.testing.assert_identical(out.terrain, x.terrain)
    assert count == (2 if hook_slot == "front_hook" else 3)
    assert front_seen[1].name is None
    assert "experiment" not in front_seen[1].coords
    for out, saved in retained:
        xr.testing.assert_identical(out, saved)
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(initial, original.isel(lead_time=slice(-1, None)))


class PhooDLWPModel(torch.nn.Module):
    def forward(self, x):
        # Preserve the original mock's two predictions without overlapping writes.
        latest = x[:, 8:15].clone()
        return torch.cat((latest + 6, latest + 12), dim=1)


@pytest.fixture
def dlwp_phoo_cs_transform():
    cs_num = 6 * 64 * 64
    indices = np.stack([np.arange(cs_num), np.arange(cs_num)], axis=0)
    return torch.sparse_coo_tensor(
        indices, np.ones(cs_num), size=(cs_num, 721 * 1440), dtype=torch.float32
    )


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
def test_dlwp_sparse_integration(device, dlwp_phoo_cs_transform):
    pytest.importorskip("physicsnemo.utils.zenith_angle")
    static = torch.ones(6, 64, 64)
    model = DLWP(
        PhooDLWPModel(),
        static,
        static,
        static,
        static,
        dlwp_phoo_cs_transform,
        dlwp_phoo_cs_transform.T,
        torch.zeros(1, 7, 1, 1),
        torch.ones(1, 7, 1, 1),
    ).to(device)
    signature = coord_array_like(
        model.input_coords(),
        {"batch": [0], "time": np.array(["1993-04-05"], dtype="datetime64[ns]")},
    )
    x = from_torch(torch.rand(signature.shape, device=device), signature)
    out = model(x)
    initial = x.e2s.to_torch()[0][:, :, -1:]
    expected = model.to_equirectangular(model.to_cubedsphere(initial + 6))
    assert torch.allclose(out.e2s.to_torch()[0], expected)
    assert out.shape == (1, 1, 1, 7, 721, 1440)
    iterator = model.create_iterator(x)
    xr.testing.assert_identical(
        next(iterator).e2s.as_numpy(), x.isel(lead_time=slice(-1, None)).e2s.as_numpy()
    )
    for step in range(1, 8):
        out = next(iterator)
        expected = model.to_equirectangular(model.to_cubedsphere(initial + 6 * step))
        assert torch.allclose(out.e2s.to_torch()[0], expected)
        assert out.lead_time.values[0] == np.timedelta64(6 * step, "h")
        assert out.dims == x.dims


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dlwp_package():
    model = DLWP.load_model(DLWP.load_default_package()).to("cuda:0")
    signature = coord_array_like(
        model.input_coords(),
        {"batch": [0], "time": np.array(["2000-01-01"], dtype="datetime64[ns]")},
    )
    x = from_torch(torch.zeros(signature.shape, device="cuda:0"), signature)
    out = model(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 1, 7, 721, 1440)
    np.testing.assert_array_equal(out.lead_time, model.output_coords(x).lead_time)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
