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

import earth2studio.models.px.aurora as aurora_module
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import Aurora
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


class PhooAuroraModel(torch.nn.Module):
    def forward(self, x):
        return aurora_module.Batch(
            surf_vars={k: v[:, -1:] for k, v in x.surf_vars.items()},
            static_vars=x.static_vars,
            atmos_vars={k: v[:, -1:] for k, v in x.atmos_vars.items()},
            metadata=x.metadata,
        )


@pytest.fixture
def aurora(monkeypatch):
    if aurora_module.Batch is None:
        monkeypatch.setattr(aurora_module, "Batch", SimpleNamespace)
        monkeypatch.setattr(aurora_module, "Metadata", SimpleNamespace)
    p = Aurora.__new__(Aurora)
    Aurora.__init__.__wrapped__(
        p, PhooAuroraModel(), *(torch.ones(4, 8) for _ in range(3))
    )
    native = p.input_coords()
    small = coord_array(
        native.dims,
        {
            "lead_time": native.lead_time.values,
            "variable": native.coords["variable"].values,
            "lat": np.linspace(90, -90, 4, endpoint=False),
            "lon": np.arange(8) * 45.0,
        },
        dynamic=("batch", "time"),
    )
    monkeypatch.setattr(p, "input_coords", lambda: small)
    return p


def _input(p, device="cpu"):
    signature = coord_array_like(
        p.input_coords(), {"time": np.array(["2001-06-04"], dtype="datetime64[ns]")}
    )
    signature = coord_array_like(signature, {"batch": [0]})
    x = from_torch(torch.randn(signature.shape, device=device), signature).isel(
        batch=0, drop=True
    )
    x.name = "weather"
    x.attrs["source"] = "test"
    x.encoding["note"] = "kept"
    return x.assign_coords(marker=7)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora_call(aurora, device):
    p = aurora.to(device)
    x = _input(p)  # CPU input must also work with a CUDA model.
    original = x.copy(deep=True)
    out = p(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 69, 4, 8)
    torch.testing.assert_close(out.e2s.to_torch()[0].cpu(), x.e2s.to_torch()[0][:, -1:])
    assert out.lead_time.values == np.timedelta64(6, "h")
    assert out.name == x.name and out.encoding == x.encoding
    assert out.attrs == x.attrs and out.marker == 7
    xr.testing.assert_identical(x, original)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora_iter(aurora, device):
    p = aurora.to(device)
    x = _input(p).expand_dims(member=2)  # Deliberately unlabelled leading dimension.
    original = x.copy(deep=True)
    calls = []

    def hook(field):
        assert field.dims == x.dims and "member" not in field.coords
        calls.append(field.sizes["lead_time"])
        field.data += 1
        return field

    p.front_hook = p.rear_hook = hook
    it = p.create_iterator(x)
    initial = next(it)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))
    first = next(it)
    saved = first.copy(deep=True)
    second = next(it)
    assert calls == [2, 1, 2, 1]
    assert second.lead_time.values == np.timedelta64(12, "h")
    torch.testing.assert_close(second.e2s.to_torch()[0], first.e2s.to_torch()[0] + 2)
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(first, saved)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))


def test_aurora_exceptions(aurora):
    x = _input(aurora)
    for bad in (
        x.transpose(..., "lon", "lat"),
        x.assign_coords(lead_time=[0, 6]),
        x.drop_vars("lead_time"),
    ):
        with pytest.raises(ValueError):
            aurora(bad)


def test_aurora_conformance(aurora):
    p = Aurora.__new__(Aurora)
    torch.nn.Module.__init__(p)
    signature = p.input_coords()
    assert (
        signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg-south-pole-excluded"
    )
    assert signature.shape == (0, 0, 2, 69, 720, 1440)
    shifted = coord_array_like(
        signature, {"lead_time": np.array([6, 12], dtype="timedelta64[h]")}
    )
    assert p.output_coords(shifted).lead_time.values == np.timedelta64(18, "h")
    check_prognostic_contract(aurora)


@pytest.fixture
def model():
    pytest.importorskip("aurora")
    return Aurora.load_model(Aurora.load_default_package())


@pytest.mark.package
def test_aurora_package(model):
    p = model.to("cuda:0")
    x = _input(p, "cuda:0")
    out = p(x)
    assert out.shape == (1, 1, 69, 720, 1440)
    np.testing.assert_array_equal(
        out.coords["variable"], p.output_coords(x).coords["variable"]
    )
