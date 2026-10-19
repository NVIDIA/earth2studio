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

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.statistics import lat_weight, moments
from earth2studio.utils.coords import handshake_coords
from earth2studio.utils.cupy import backend, from_torch

lat_weights = lat_weight(torch.as_tensor(np.linspace(-90.0, 90.0, 361)))


def _check_output_coords(stat, x: xr.DataArray, y: xr.DataArray) -> None:
    """Check that the a priori output signature matches the computed output"""
    out_test = stat.output_coords(x)
    assert out_test.dims == y.dims
    assert out_test.shape == y.shape
    assert list(out_test.coords) == list(y.coords)
    for d in y.dims:
        handshake_coords(out_test, y, d)


def _to_np(y: xr.DataArray) -> np.ndarray:
    return y.e2s.to_torch()[0].cpu().numpy()


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
def test_mean(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), coords)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    mean = moments.mean(reduction_dimensions, weights=weights)

    y = mean(x)
    assert not any([ri in y.dims for ri in reduction_dimensions])
    assert list(y.shape) == [x.sizes[d] for d in y.dims]
    assert y.dims == tuple(d for d in x.dims if d not in reduction_dimensions)

    _check_output_coords(mean, x, y)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_mean(device: str) -> None:

    coords = OrderedDict(
        {"ensemble": np.arange(10), "lat": np.linspace(-90.0, 90.0, 361)}
    )

    x_t = torch.randn((10, 361), device=device)
    x = from_torch(x_t, coords)

    reduction_dimensions, weights = ["lat"], lat_weights
    mean = moments.mean(reduction_dimensions, weights=weights)

    assert str(mean) == "lat_mean"
    y = mean(x)

    # Compute numpy weighted average
    y_np = np.average(x_t.cpu().numpy(), axis=1, weights=lat_weights.cpu().numpy())

    assert y.dims == ("ensemble",)
    assert np.allclose(_to_np(y), y_np)

    _check_output_coords(mean, x, y)


def test_mean_without_reduction_dimensions() -> None:
    coords = OrderedDict(
        {
            "time": np.arange(2),
            "station": np.arange(5),
        }
    )
    x = from_torch(torch.randn((2, 5)), coords)

    y = moments.mean([])(x)

    assert np.array_equal(_to_np(y), _to_np(x))
    assert y.dims == x.dims
    assert list(y.coords) == list(coords)
    for dimension in coords:
        assert np.array_equal(y.coords[dimension].values, coords[dimension])


def test_mean_reversed_numpy() -> None:
    x = xr.DataArray(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        dims=("lat", "lon"),
        coords={"lat": np.arange(3), "lon": np.arange(4)},
    ).isel(lat=slice(None, None, -1))
    result = moments.mean(["lat"])(x)
    np.testing.assert_allclose(_to_np(result), x.mean("lat").values)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("statistic", [moments.mean, moments.variance, moments.std])
def test_moments_float16_large_reduction(device, weighted, statistic) -> None:
    tensor = torch.ones((361, 720), dtype=torch.float16, device=device)
    tensor[::2] = 3
    weights = torch.ones_like(tensor) if weighted else None
    x = from_torch(tensor, {"lat": np.arange(361), "lon": np.arange(720)})

    result, _ = statistic(["lat", "lon"], weights=weights)(x).e2s.to_torch()
    expected = getattr(
        torch, {moments.variance: "var"}.get(statistic, statistic.__name__)
    )(tensor.float())
    assert result.device == tensor.device
    assert torch.isfinite(result)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("statistic", [moments.mean, moments.variance, moments.std])
def test_moments_torch_autograd(device, statistic) -> None:
    tensor = torch.randn((4, 6), dtype=torch.float64, device=device, requires_grad=True)
    weights = torch.rand(6, dtype=torch.float64, device=device, requires_grad=True)
    with backend("torch"):
        x = from_torch(tensor, {"time": np.arange(4), "station": np.arange(6)})
        result = statistic(["station"], weights=weights)(x)
        result_t, _ = result.e2s.to_torch()
        assert result_t.requires_grad
        assert result_t.device == tensor.device

    expected = (tensor * weights).sum(dim=1) / weights.sum()
    if statistic is not moments.mean:
        expected = (weights * (tensor - expected[:, None]).square()).sum(dim=1) / (
            weights.sum() - weights.square().sum() / weights.sum()
        )
        if statistic is moments.std:
            expected = expected.sqrt()
    torch.testing.assert_close(result_t, expected)
    actual_grad = torch.autograd.grad(result_t.sum(), (tensor, weights))
    expected_grad = torch.autograd.grad(expected.sum(), (tensor, weights))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("statistic", [moments.variance, moments.std])
def test_variance_and_std_require_reduction_dimensions(statistic) -> None:
    with pytest.raises(ValueError, match="at least one reduction dimension"):
        statistic([])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_mean(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    reduced_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})
    mean = moments.mean(["ensemble"], batch_update=True)
    for inds in range(0, 100, 10):
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]
        x = from_torch(big_x[inds : inds + 10], coords)

        y = mean(x)
        expected = torch.mean(big_x[: inds + 10], dim=0).cpu().numpy()
        assert np.allclose(_to_np(y), expected, atol=1e-3)
        assert y.dims == tuple(reduced_coords)
        for d, v in reduced_coords.items():
            assert np.array_equal(y.coords[d].values, v)

        _check_output_coords(mean, x, y)


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
def test_var(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), coords)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    var = moments.variance(reduction_dimensions, weights=weights)

    y = var(x)
    assert not any([ri in y.dims for ri in reduction_dimensions])
    assert list(y.shape) == [x.sizes[d] for d in y.dims]
    assert y.dims == tuple(d for d in x.dims if d not in reduction_dimensions)
    assert (_to_np(y) >= 0.0).all()

    _check_output_coords(var, x, y)


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
def test_std(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    x = from_torch(torch.randn((10, 1, 2, 361, 720), device=device), coords)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    std = moments.std(reduction_dimensions, weights=weights)

    y = std(x)
    assert not any([ri in y.dims for ri in reduction_dimensions])
    assert list(y.shape) == [x.sizes[d] for d in y.dims]
    assert y.dims == tuple(d for d in x.dims if d not in reduction_dimensions)
    assert (_to_np(y) >= 0.0).all()

    _check_output_coords(std, x, y)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_var(device: str) -> None:

    coords = OrderedDict(
        {"ensemble": np.arange(1), "lat": np.linspace(-90.0, 90.0, 361)}
    )

    x_t = torch.randn((1, 361), device=device)
    x = from_torch(x_t, coords)

    reduction_dimensions, weights = ["lat"], lat_weights
    var = moments.variance(reduction_dimensions, weights=weights)

    y = var(x)

    assert str(var) == "lat_variance"

    # Compute numpy weighted average
    y_np = np.cov(x_t.cpu().numpy(), aweights=lat_weights.cpu().numpy())

    assert y.dims == ("ensemble",)
    assert np.allclose(_to_np(y), y_np)

    _check_output_coords(var, x, y)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_var_std(device) -> None:

    big_x = torch.randn((100, 10, 10), device=device)
    big_coords = OrderedDict(
        {
            "ensemble": np.arange(100),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 10, endpoint=False),
        }
    )

    reduced_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})

    std = moments.std(["ensemble"], batch_update=True)
    var = moments.variance(["ensemble"], batch_update=True)

    assert str(std) == "ensemble_std"
    assert str(var) == "ensemble_variance"

    for inds in range(0, 100, 10):
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]
        x = from_torch(big_x[inds : inds + 10], coords)

        y = std(x)
        expected = torch.std(big_x[: inds + 10], dim=0).cpu().numpy()
        assert np.allclose(_to_np(y), expected, rtol=1e-4, atol=1e-6)
        assert y.dims == tuple(reduced_coords)
        for d, v in reduced_coords.items():
            assert np.array_equal(y.coords[d].values, v)
        _check_output_coords(std, x, y)

        y = var(x)
        expected = torch.var(big_x[: inds + 10], dim=0).cpu().numpy()
        assert np.allclose(_to_np(y), expected, rtol=1e-4, atol=1e-6)
        assert y.dims == tuple(reduced_coords)
        for d, v in reduced_coords.items():
            assert np.array_equal(y.coords[d].values, v)
        _check_output_coords(var, x, y)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_moments_failures(device) -> None:
    # Test weights not the same shape as reduction dimensions
    reduction_dimensions = ["lat", "lon"]
    weights = torch.as_tensor([10], device=device)

    with pytest.raises(ValueError):
        moments.mean(reduction_dimensions, weights=weights)

    with pytest.raises(ValueError):
        moments.variance(reduction_dimensions, weights=weights)

    with pytest.raises(ValueError):
        moments.std(reduction_dimensions, weights=weights)

    x = from_torch(
        torch.randn((10,), device=device), OrderedDict({"lat": np.arange(10)})
    )
    with pytest.raises(ValueError):
        m = moments.mean(reduction_dimensions)

        m(x)

    with pytest.raises(ValueError):
        var = moments.variance(reduction_dimensions)

        var(x)

    with pytest.raises(ValueError):
        std = moments.std(reduction_dimensions)

        std(x)
