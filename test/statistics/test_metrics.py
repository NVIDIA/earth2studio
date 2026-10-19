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

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.statistics import (
    Metric,
    lat_weight,
    mae,
    rmse,
    skill_spread,
    spread_skill_ratio,
)
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.cupy import backend, from_torch

lat_weights = lat_weight(torch.as_tensor(np.linspace(-90.0, 90.0, 361)))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("metric", [rmse, mae])
def test_error_metric_torch_autograd(device, metric) -> None:
    tensor = torch.randn((3, 4, 6), device=device, requires_grad=True)
    truth = torch.randn((4, 6), device=device, requires_grad=True)
    coords = {"time": np.arange(4), "station": np.arange(6)}
    with backend("torch"):
        x = from_torch(tensor, {"ensemble": np.arange(3), **coords})
        y = from_torch(truth, coords)
        result = metric(["station"], ensemble_dimension="ensemble")(x, y)
        result_t, _ = result.e2s.to_torch()
        assert result_t.requires_grad
        assert result_t.device == tensor.device

    diff = tensor.mean(dim=0) - truth
    expected = (
        diff.square().mean(dim=1).sqrt() if metric is rmse else diff.abs().mean(dim=1)
    )
    torch.testing.assert_close(result_t, expected)
    actual_grad = torch.autograd.grad(result_t.sum(), (tensor, truth))
    expected_grad = torch.autograd.grad(expected.sum(), (tensor, truth))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference)


def _check_output(metric: Metric, x: xr.DataArray, z: xr.DataArray) -> None:
    """Check output DataArray against the allocation-free output signature"""
    out = metric.output_coords(x)
    assert z.dims == out.dims
    assert z.shape == out.shape
    for i, ci in enumerate(z.dims):
        handshake_dim(out, ci, i)
        handshake_coords(out, z, ci)


@pytest.mark.parametrize("statistic", [rmse, mae])
@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["ensemble"], None),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 720)),
        (["lat"], lat_weights),
        (["ensemble", "lat"], lat_weights.repeat(10, 1)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rmse(
    statistic: Metric, reduction_weights: tuple[list[str], np.ndarray], device: str
) -> None:

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )
    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), x_coords)

    y_coords = copy.deepcopy(x_coords)
    y = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), y_coords)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    RMSE = statistic(reduction_dimensions, weights=weights)

    z = RMSE(x, y)
    assert not any([ri in z.dims for ri in reduction_dimensions])
    assert z.dims == tuple(d for d in x.dims if d not in reduction_dimensions)
    _check_output(RMSE, x, z)

    ## Test broadcasting
    y_coords["ensemble"] = np.arange(1)
    y = from_torch(torch.randn((1, 1, 2, 361, 720), device=device), y_coords)

    # Non-matching ensemble coordinates are no longer broadcastable
    with pytest.raises(ValueError):
        RMSE(x, y)


@pytest.mark.parametrize("statistic", [rmse, mae])
@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rmse_ensemble(
    statistic: Metric, ensemble_dimension: str | None, device: str
) -> None:

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )
    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), x_coords)

    y_coords = copy.deepcopy(x_coords)
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)

    RMSE = statistic(
        reduction_dimensions=["time", "lat", "lon"],
        ensemble_dimension=ensemble_dimension,
    )

    z = RMSE(x, y)
    assert ensemble_dimension not in z.dims
    _check_output(RMSE, x, z)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_mean(device: str) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    y_t = torch.randn((10, 10), device=device)
    y_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})

    RMSE = rmse(["ensemble"], batch_update=True)
    assert str(RMSE) == "ensemble_rmse"
    for inds in range(0, 100, 10):
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]
        x = from_torch(big_x[inds : inds + 10], coords)
        # y must share the dimensions of x, so broadcast it along ensemble
        y = from_torch(y_t[None].expand(10, -1, -1).contiguous(), coords)

        if inds == 0:
            # y without the ensemble dimension is not compatible with x
            with pytest.raises(ValueError):
                RMSE(x, from_torch(y_t, y_coords))

        z = RMSE(x, y)
        assert torch.allclose(
            torch.as_tensor(z.e2s.to_torch()[0], device=device),
            torch.sqrt(torch.mean((big_x[: inds + 10] - y_t[None]) ** 2, dim=0)),
            atol=1e-3,
        )
        assert z.dims == tuple(y_coords.keys())
        for d, v in y_coords.items():
            np.testing.assert_array_equal(z.coords[d].values, v)

        _check_output(RMSE, x, z)


@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["time", "lat", "lon"], lat_weights.unsqueeze(1).repeat(2, 1, 90)),
        (["lat"], lat_weights),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 90)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_spread_skill(
    reduction_weights: tuple[list[str], np.ndarray], device: str
) -> None:

    n_ensemble = 2800
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(n_ensemble),
            "time": np.array(
                [np.datetime64("1993-04-05T00:00"), np.datetime64("1993-04-06T00:00")]
            ),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 90, endpoint=False),
        }
    )
    x = from_torch(
        3.0 + 2.0 * torch.randn((n_ensemble, 2, 2, 361, 90), device=device), x_coords
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")
    y = from_torch(3.0 + 2.0 * torch.randn((2, 2, 361, 90), device=device), y_coords)

    ensemble_dimension = "ensemble"
    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    SSR = spread_skill_ratio(
        ensemble_dimension, reduction_dimensions, reduction_weights=weights
    )

    z = SSR(x, y)
    assert not any([ri in z.dims for ri in ["ensemble"] + reduction_dimensions])
    z_t = torch.as_tensor(z.e2s.to_torch()[0])
    assert list(z_t.shape) == list(z.shape)
    assert torch.allclose(z_t, torch.ones_like(z_t), rtol=1e-1, atol=1e-1)

    _check_output(SSR, x, z)


@pytest.mark.parametrize(
    "reduction_weights",
    [
        (["time", "lat", "lon"], lat_weights.unsqueeze(1).repeat(2, 1, 720)),
        (["lat"], lat_weights),
        (["lat", "lon"], lat_weights.unsqueeze(1).repeat(1, 720)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_skill_spread(
    reduction_weights: tuple[list[str], np.ndarray], device: str
) -> None:

    n_ensemble = 600
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(n_ensemble),
            "time": np.array(
                [np.datetime64("1993-04-05T00:00"), np.datetime64("1993-04-06T00:00")]
            ),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )
    x = from_torch(
        3.0 + 2.0 * torch.randn((n_ensemble, 2, 2, 361, 720), device=device), x_coords
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")
    y = from_torch(3.0 + 2.0 * torch.randn((2, 2, 361, 720), device=device), y_coords)

    ensemble_dimension = "ensemble"
    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    SSR = skill_spread(
        ensemble_dimension, reduction_dimensions, reduction_weights=weights
    )

    z = SSR(x, y)
    assert not any([ri in z.dims for ri in ["ensemble"] + reduction_dimensions])
    assert z.dims[0] == "metric"
    np.testing.assert_array_equal(z.coords["metric"].values, ["mse", "variance"])
    assert z.sizes["metric"] == 2

    _check_output(SSR, x, z)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_ensemble_batch_spread_skill(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    y_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})
    y = from_torch(torch.randn((10, 10), device=device), y_coords)

    ensemble_dimension = "ensemble"
    reduction_dimensions = ["lat", "lon"]
    SSR1 = spread_skill_ratio(
        ensemble_dimension, reduction_dimensions, ensemble_batch_update=True
    )
    SSR2 = spread_skill_ratio(ensemble_dimension, reduction_dimensions)

    assert str(SSR1) == "ensemble_lat_lon_spread_skill"
    for inds in range(0, 100, 10):
        coords1 = big_coords.copy()
        coords1["ensemble"] = coords1["ensemble"][inds : inds + 10]
        x1 = from_torch(big_x[inds : inds + 10], coords1)

        ss1 = SSR1(x1, y)

        coords2 = big_coords.copy()
        coords2["ensemble"] = coords2["ensemble"][: inds + 10]
        x2 = from_torch(big_x[: inds + 10], coords2)
        ss2 = SSR2(x2, y)

        assert torch.allclose(
            torch.as_tensor(ss1.e2s.to_torch()[0], device=device),
            torch.as_tensor(ss2.e2s.to_torch()[0], device=device),
            atol=1e-3,
        )

        assert ss1.dims == ss2.dims
        assert ss1.shape == ss2.shape
        xr.testing.assert_equal(ss1.coords.to_dataset(), ss2.coords.to_dataset())

        _check_output(SSR1, x1, ss1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_time_batch_spread_skill(device) -> None:

    big_x = torch.randn((10, 100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    y_t = torch.randn((100, 10, 10), device=device)
    y_coords = OrderedDict(
        {"time": big_coords["time"], "lat": big_coords["lat"], "lon": big_coords["lon"]}
    )

    ensemble_dimension = "ensemble"
    reduction_dimensions = ["time", "lat", "lon"]
    SSR1 = spread_skill_ratio(
        ensemble_dimension, reduction_dimensions, reduction_batch_update=True
    )
    SSR2 = spread_skill_ratio(ensemble_dimension, reduction_dimensions)

    assert str(SSR1) == "ensemble_time_lat_lon_spread_skill"
    for inds in range(0, 100, 10):
        coords1 = big_coords.copy()
        coords1["time"] = coords1["time"][inds : inds + 10]
        x1 = from_torch(big_x[:, inds : inds + 10], coords1)

        y_coords1 = y_coords.copy()
        y_coords1["time"] = y_coords1["time"][inds : inds + 10]
        y1 = from_torch(y_t[inds : inds + 10], y_coords1)

        ss1 = SSR1(x1, y1)

        coords2 = big_coords.copy()
        coords2["time"] = coords2["time"][: inds + 10]
        x2 = from_torch(big_x[:, : inds + 10], coords2)

        y_coords2 = y_coords.copy()
        y_coords2["time"] = y_coords2["time"][: inds + 10]
        y2 = from_torch(y_t[: inds + 10], y_coords2)

        ss2 = SSR2(x2, y2)

        assert torch.allclose(
            torch.as_tensor(ss1.e2s.to_torch()[0], device=device),
            torch.as_tensor(ss2.e2s.to_torch()[0], device=device),
            atol=1e-3,
        )

        assert ss1.dims == ss2.dims
        assert ss1.shape == ss2.shape
        xr.testing.assert_equal(ss1.coords.to_dataset(), ss2.coords.to_dataset())

        _check_output(SSR1, x1, ss1)
