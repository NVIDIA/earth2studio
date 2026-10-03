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

from earth2studio.statistics import lat_weight, moments
from earth2studio.utils.coords import handshake_coords, handshake_dim

lat_weights = lat_weight(torch.as_tensor(np.linspace(-90.0, 90.0, 361)))


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

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    mean = moments.mean(reduction_dimensions, weights=weights)

    y, c = mean(x, coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(y.shape) == [len(val) for val in c.values()]

    out_test_coords = mean.output_coords(coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_mean(device: str) -> None:

    coords = OrderedDict(
        {"ensemble": np.arange(10), "lat": np.linspace(-90.0, 90.0, 361)}
    )

    x = torch.randn((10, 361), device=device)

    reduction_dimensions, weights = ["lat"], lat_weights
    mean = moments.mean(reduction_dimensions, weights=weights)

    assert str(mean) == "lat_mean"
    y, c = mean(x, coords)

    # Compute numpy weighted average
    y_np = np.average(x.cpu().numpy(), axis=1, weights=lat_weights.cpu().numpy())

    assert torch.allclose(y, torch.as_tensor(y_np, device=device))

    out_test_coords = mean.output_coords(coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


def test_mean_without_reduction_dimensions() -> None:
    coords = OrderedDict(
        {
            "time": np.arange(2),
            "station": np.arange(5),
        }
    )
    x = torch.randn((2, 5))

    y, output_coords = moments.mean([])(x, coords)

    assert torch.equal(y, x)
    assert list(output_coords) == list(coords)
    for dimension in coords:
        assert np.array_equal(output_coords[dimension], coords[dimension])


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
            "lat": np.linspace(-90.0, 90.0, 1),
            "lon": np.linspace(0.0, 360.0, 1, endpoint=False),
        }
    )

    reduced_coords = OrderedDict({"lat": big_coords["lat"], "lon": big_coords["lon"]})
    mean = moments.mean(["ensemble"], batch_update=True)
    for inds in range(0, 100, 10):
        x = big_x[inds : inds + 10]
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]

        y, c = mean(x, coords)
        assert torch.allclose(y, torch.mean(big_x[: inds + 10], dim=0), atol=1e-3)
        assert c == reduced_coords

        out_test_coords = mean.output_coords(coords)
        for i, ci in enumerate(c):
            handshake_dim(out_test_coords, ci, i)
            handshake_coords(out_test_coords, c, ci)


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

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    var = moments.variance(reduction_dimensions, weights=weights)

    y, c = var(x, coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(y.shape) == [len(val) for val in c.values()]
    assert torch.all(y >= 0.0)

    out_test_coords = var.output_coords(coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


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

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)
    std = moments.std(reduction_dimensions, weights=weights)

    y, c = std(x, coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(y.shape) == [len(val) for val in c.values()]
    assert torch.all(y >= 0.0)

    out_test_coords = std.output_coords(coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weighted_var(device: str) -> None:

    coords = OrderedDict(
        {"ensemble": np.arange(1), "lat": np.linspace(-90.0, 90.0, 361)}
    )

    x = torch.randn((1, 361), device=device)

    reduction_dimensions, weights = ["lat"], lat_weights
    var = moments.variance(reduction_dimensions, weights=weights)

    y, c = var(x, coords)

    assert str(var) == "lat_variance"

    # Compute numpy weighted average
    y_np = np.cov(x.cpu().numpy(), aweights=lat_weights.cpu().numpy())

    assert torch.allclose(y, torch.as_tensor(y_np, device=device))

    out_test_coords = var.output_coords(coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


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
        x = big_x[inds : inds + 10]
        coords = big_coords.copy()
        coords["ensemble"] = coords["ensemble"][inds : inds + 10]

        y, c = std(x, coords)
        assert torch.allclose(y, torch.std(big_x[: inds + 10], dim=0))
        assert c == reduced_coords

        y, c = var(x, coords)
        assert torch.allclose(y, torch.var(big_x[: inds + 10], dim=0))
        assert c == reduced_coords

        out_test_coords = var.output_coords(coords)
        for i, ci in enumerate(c):
            handshake_dim(out_test_coords, ci, i)
            handshake_coords(out_test_coords, c, ci)


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

    x = torch.randn((10,), device=device)
    coords = OrderedDict({"lat": np.arange(10)})
    with pytest.raises(ValueError):
        m = moments.mean(reduction_dimensions)

        m(x, coords)

    with pytest.raises(ValueError):
        var = moments.variance(reduction_dimensions)

        var(x, coords)

    with pytest.raises(ValueError):
        std = moments.std(reduction_dimensions)

        std(x, coords)


@pytest.mark.parametrize("shape", [(2, 3), (3, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("statistic", ["variance", "std"])
def test_running_moment_nonleading_axis(
    shape: tuple[int, int], dtype: torch.dtype, statistic: str
) -> None:
    """A dropped reduction axis must not broadcast the batch mean onto sites."""
    values = torch.arange(np.prod(shape), dtype=dtype).reshape(shape)
    coords = OrderedDict(site=np.arange(shape[0]), time=np.arange(shape[1]))
    before = values.clone()
    moment = getattr(moments, statistic)(["time"], batch_update=True)
    result, output_coords = moment(values, coords)
    reference = values.var(dim=1) if statistic == "variance" else values.std(dim=1)
    torch.testing.assert_close(result, reference)
    assert list(output_coords) == ["site"]
    np.testing.assert_array_equal(output_coords["site"], coords["site"])
    state = moment if statistic == "variance" else moment.var
    assert state.sum.shape == reference.shape
    assert state.sum2.shape == reference.shape
    torch.testing.assert_close(values, before, rtol=0, atol=0)


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (1, 2, 3, 0), (1, 0, 3, 2)])
@pytest.mark.parametrize("statistic", ["variance", "std"])
def test_running_moment_multiple_axes_and_unequal_batches(
    order: tuple[int, ...], statistic: str
) -> None:
    """Streaming updates match direct reductions across arbitrary axis positions."""
    raw = torch.arange(6 * 2 * 3 * 4, dtype=torch.float64).reshape(6, 2, 3, 4)
    raw = torch.sin(raw / 13.0) + raw / 17.0
    before = raw.clone()
    names = ("time", "site", "level", "feature")
    coord_names = [names[index] for index in order]
    reduced_dims = tuple(coord_names.index(name) for name in ("time", "level"))
    output_names = [name for name in coord_names if name not in ("time", "level")]
    moment = getattr(moments, statistic)(["time", "level"], batch_update=True)
    start = 0
    for stop in (2, 3, 6):
        chunk = raw[start:stop].permute(order)
        coords = OrderedDict(
            (name, np.arange(size)) for name, size in zip(coord_names, chunk.shape)
        )
        value, output_coords = moment(chunk, coords)
        accumulated = raw[:stop].permute(order)
        reference = (
            accumulated.var(dim=reduced_dims)
            if statistic == "variance"
            else accumulated.std(dim=reduced_dims)
        )
        # Default batch-count ratios retain their existing float32 arithmetic.
        torch.testing.assert_close(value, reference, rtol=1e-7, atol=1e-9)
        assert list(output_coords) == output_names
        for name in output_names:
            np.testing.assert_array_equal(output_coords[name], coords[name])
        state = moment if statistic == "variance" else moment.var
        assert state.sum.shape == reference.shape
        start = stop
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


def test_running_moment_weighted_centering_keeps_existing_normalization() -> None:
    """Correct axis placement without changing the running-weight denominator."""
    values = torch.tensor([[1.0, 4.0, 8.0], [11.0, 12.0, 19.0]], dtype=torch.float64)
    weights = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    coords = OrderedDict(site=np.arange(2), time=np.arange(3))
    result, _ = moments.variance(["time"], weights=weights, batch_update=True)(
        values, coords
    )
    reference_mean = np.average(values.numpy(), axis=1, weights=weights.numpy())
    reference = ((values.numpy() - reference_mean[:, None]) ** 2 * weights.numpy()).sum(
        axis=1
    ) / 3.0
    torch.testing.assert_close(result, torch.from_numpy(reference))
    torch.testing.assert_close(
        weights, torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    )


def test_running_moment_single_batch_gradients_match_direct_variance() -> None:
    """The corrected moment retains differentiability for a non-leading axis."""
    values = torch.tensor(
        [[1.0, 4.0, 8.0], [2.0, 7.0, 11.0]], dtype=torch.float64, requires_grad=True
    )
    coords = OrderedDict(site=np.arange(2), time=np.arange(3))

    def evaluate(x: torch.Tensor) -> torch.Tensor:
        return moments.variance(["time"], batch_update=True)(x, coords)[0]

    actual = evaluate(values)
    (gradient,) = torch.autograd.grad(actual.sum(), values)
    (reference,) = torch.autograd.grad(values.var(dim=1).sum(), values)
    torch.testing.assert_close(gradient, reference, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(evaluate, (values,))
