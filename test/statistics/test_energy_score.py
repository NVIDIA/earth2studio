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

from earth2studio.statistics import energy_score, lat_weight
from earth2studio.statistics.energy_score import _energy_score_compute
from earth2studio.utils.coords import handshake_coords, handshake_dim

lat_weights = lat_weight(torch.as_tensor(np.linspace(-90.0, 90.0, 10)))


@pytest.mark.parametrize(
    "ensemble_dimension",
    [
        "ensemble",
    ],
)
@pytest.mark.parametrize(
    "reduction_weights",
    [
        (None, None),
        (["lat"], lat_weights),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score(
    ensemble_dimension: str,
    reduction_weights: tuple[list[str], np.ndarray],
    device: str,
) -> None:

    x = torch.randn((8, 1, 2, 10, 20), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(8),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 20, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)

    # Use lon as the multivariate dimension for testing
    ES = energy_score(
        ensemble_dimension,
        multivariate_dimensions=["lon"],
        reduction_dimensions=reduction_dimensions,
        weights=weights,
    )

    z, c = ES(x, x_coords, y, y_coords)

    # Ensemble dim and multivariate dims should be removed
    assert ensemble_dimension not in c
    assert "lon" not in c
    if reduction_dimensions is not None:
        assert all(rd not in c for rd in reduction_dimensions)
    assert list(z.shape) == [len(val) for val in c.values()]

    # Check output coords match
    out_test_coords = ES.output_coords(x_coords)
    for i, ci in enumerate(c):
        handshake_dim(out_test_coords, ci, i)
        handshake_coords(out_test_coords, c, ci)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_fair(device: str) -> None:
    """Test fair Energy Score with > 1 ensemble members."""
    x = torch.randn((8, 1, 2, 10, 20), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(8),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 20, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = torch.randn(y_shape, device=device)

    ES = energy_score(
        "ensemble",
        multivariate_dimensions=["lon"],
        fair=True,
    )

    z, c = ES(x, x_coords, y, y_coords)

    assert "ensemble" not in c
    assert "lon" not in c
    assert list(z.shape) == [len(val) for val in c.values()]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_failures(device: str) -> None:
    x = torch.randn((8, 1, 2, 10, 20), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(8),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 20, endpoint=False),
        }
    )

    # Error: ensemble_dimension not a string
    with pytest.raises(ValueError):
        energy_score(["ensemble"], multivariate_dimensions=["lon"])

    # Error: multivariate_dimensions is empty
    with pytest.raises(ValueError):
        energy_score("ensemble", multivariate_dimensions=[])

    ES = energy_score("ensemble", multivariate_dimensions=["lon"])

    # Error: ensemble_dimension in y_coords
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        ES(x, x_coords, y, y_coords)

    # Error: x and y shapes not broadcastable
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            if c != "ensemble":
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        ES(x, x_coords, y, y_coords)

    # Error: reduction_dimension not in x_coords
    with pytest.raises(ValueError):
        y_coords = OrderedDict()
        for c in x_coords:
            if c != "ensemble":
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)

        bad_x_coords = copy.deepcopy(x_coords)
        bad_x_coords.pop("ensemble")
        ES(x, bad_x_coords, y, y_coords)

    # Error: fair with < 2 ensemble members
    with pytest.raises(ValueError):
        ES_fair = energy_score("ensemble", multivariate_dimensions=["lon"], fair=True)
        x1 = torch.randn((1, 1, 2, 10, 20), device=device)
        x1_coords = copy.deepcopy(x_coords)
        x1_coords["ensemble"] = np.arange(1)
        y_coords = OrderedDict()
        for c in x1_coords:
            if c != "ensemble":
                y_coords[c] = x1_coords[c]
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = torch.randn(y_shape, device=device)
        ES_fair(x1, x1_coords, y, y_coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_accuracy(
    device: str, rtol: float = 1e-2, atol: float = 1e-2
) -> None:
    """Test Energy Score accuracy against brute-force computation."""
    torch.manual_seed(42)
    M = 50
    D = 5

    # Generate ensemble and truth
    x = torch.randn((M, D), device=device, dtype=torch.float64)
    y = torch.randn((D,), device=device, dtype=torch.float64)

    # Manual brute-force computation
    # Term 1: (1/M) * sum_m ||x_m - y||
    term1 = torch.norm(x - y.unsqueeze(0), dim=1).mean()

    # Term 2: 1/(2*M^2) * sum_m sum_m' ||x_m - x_m'||
    dists = torch.cdist(x.unsqueeze(0), x.unsqueeze(0), p=2).squeeze(0)
    term2 = dists.sum() / (2.0 * M * M)

    expected_es = term1 - term2

    # Compute via our function
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(M),
            "variable": np.array([f"v{i}" for i in range(D)]),
        }
    )

    computed_es = _energy_score_compute(
        x,
        y,
        x_coords,
        ensemble_dimension="ensemble",
        multivariate_dimensions=["variable"],
        fair=False,
    )

    assert torch.allclose(
        computed_es,
        expected_es,
        rtol=rtol,
        atol=atol,
    ), f"Expected {expected_es.item()}, got {computed_es.item()}"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_nonnegativity(device: str) -> None:
    """Energy Score should always be non-negative."""
    torch.manual_seed(123)
    M = 20
    x = torch.randn((M, 3, 10), device=device)
    y = torch.randn((3, 10), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(M),
            "variable": np.array(["a", "b", "c"]),
            "lon": np.linspace(0, 360, 10),
        }
    )

    es = _energy_score_compute(
        x,
        y,
        x_coords,
        ensemble_dimension="ensemble",
        multivariate_dimensions=["variable", "lon"],
        fair=False,
    )

    assert es.item() >= 0.0, f"Energy Score should be non-negative, got {es.item()}"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_perfect_ensemble(device: str) -> None:
    """Energy score should be zero for a perfect deterministic ensemble."""
    torch.manual_seed(0)
    M = 10
    D = 8

    y = torch.randn((D,), device=device, dtype=torch.float64)
    x = y.unsqueeze(0).repeat(M, 1)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(M),
            "variable": np.array([f"v{i}" for i in range(D)]),
        }
    )

    es = _energy_score_compute(
        x,
        y,
        x_coords,
        ensemble_dimension="ensemble",
        multivariate_dimensions=["variable"],
        fair=False,
    )

    assert torch.allclose(
        es,
        torch.zeros_like(es),
        atol=1e-6,
    ), f"Perfect ensemble should have ES ~= 0, got {es.item()}"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_str(device: str) -> None:
    """Test string representation."""
    ES = energy_score(
        "ensemble",
        multivariate_dimensions=["lon"],
        reduction_dimensions=["lat"],
    )
    assert str(ES) == "lat_energy_score"

    ES2 = energy_score(
        "ensemble",
        multivariate_dimensions=["lon"],
    )
    assert str(ES2) == "energy_score"


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_energy_score_multiple_mv_dims(device: str) -> None:
    """Test with multiple multivariate dimensions."""
    torch.manual_seed(99)
    M = 5
    x = torch.randn((M, 1, 2, 10, 20), device=device)
    y = torch.randn((1, 2, 10, 20), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(M),
            "time": np.array([np.datetime64("2024-01-01")]),
            "variable": np.array(["t2m", "u10m"]),
            "lat": np.linspace(-90.0, 90.0, 10),
            "lon": np.linspace(0.0, 360.0, 20, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y_coords.pop("ensemble")

    ES = energy_score(
        "ensemble",
        multivariate_dimensions=["variable", "lat", "lon"],
    )

    z, c = ES(x, x_coords, y, y_coords)

    assert "ensemble" not in c
    assert "variable" not in c
    assert "lat" not in c
    assert "lon" not in c
    assert "time" in c
    assert list(z.shape) == [len(val) for val in c.values()]
