# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import SFNO
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def test_sfno_valid_times(monkeypatch):
    calls = []

    class Net(torch.nn.Module):
        def forward(self, x, times, normalized_data=False):
            assert not normalized_data
            calls.extend(times)
            return x + 6

    model = SFNO.__new__(SFNO)
    SFNO.__init__.__wrapped__(model, Net())
    declared = model.input_coords()
    assert declared.data.nbytes == 0
    assert declared.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        declared.dims,
        {"lead_time": declared.lead_time, "variable": declared.coords["variable"]},
        dynamic=("batch", "time"),
        grid=LatLonGrid([45, -45], [0, 120, 240]),
    )
    monkeypatch.setattr(model, "input_coords", lambda: signature.copy())
    coords = coord_array_like(
        signature,
        {
            "batch": [0, 1],
            "time": np.array(["2000-01-01", "2001-02-03"], dtype="datetime64[ns]"),
            "lead_time": np.array([12], dtype="timedelta64[h]"),
        },
    )
    x = from_torch(torch.randn(coords.shape), coords, name="weather").rename(
        batch="member"
    )
    x.attrs["source"] = "fixture"
    x.encoding = {"source": "fixture"}
    original = x.copy(deep=True)
    out = model(x)
    np.testing.assert_allclose(out.data, x.data + 6)
    expected = np.tile(x.time.values + x.lead_time.values[-1], 2)
    np.testing.assert_array_equal(
        np.array([t.replace(tzinfo=None) for t in calls], dtype="datetime64[ns]"),
        expected,
    )
    iterator = model.create_iterator(x)
    initial = next(iterator)
    for step in range(1, 4):
        out = next(iterator)
        np.testing.assert_allclose(out.data, x.data + 6 * step, rtol=1e-5)
        assert out.name == x.name and out.encoding == x.encoding
        assert out.attrs == x.attrs
        assert out.lead_time.values[0] == np.timedelta64(12 + 6 * step, "h")
    xr.testing.assert_identical(initial, original)
    xr.testing.assert_identical(x, original)
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_sfno_package():
    model = SFNO.load_model(SFNO.load_default_package()).to("cuda:0")
    signature = coord_array_like(
        model.input_coords(),
        {"batch": [0], "time": np.array(["2000-01-01"], dtype="datetime64[ns]")},
    )
    x = from_torch(torch.zeros(signature.shape, device="cuda:0"), signature)
    out = model(x)
    assert out.dims == x.dims
    assert out.shape == (1, 1, 1, 73, 721, 1440)
    np.testing.assert_array_equal(out.lead_time, model.output_coords(x).lead_time)
    np.testing.assert_array_equal(out.coords["variable"], x.coords["variable"])
