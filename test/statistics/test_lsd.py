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

from earth2studio.statistics import log_spectral_distance
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.cupy import from_torch


def _check_output(LSD, x, z) -> None:
    """Check output DataArray against the allocation-free output signature"""
    out = LSD.output_coords(x)
    assert z.dims == out.dims
    assert z.shape == out.shape
    for i, ci in enumerate(z.dims):
        handshake_dim(out, ci, i)
        handshake_coords(out, z, ci)


@pytest.mark.parametrize("wavenumber_cutoff", [-10, 10, None])
@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_lsd(
    ensemble_dimension: str | None, device: str, wavenumber_cutoff: int
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

    LSD = log_spectral_distance(
        reduction_dimensions=["time"],
        wavenumber_cutoff=wavenumber_cutoff,
        ensemble_dimension=ensemble_dimension,
    )

    z = LSD(x, y)
    assert "time" not in z.dims
    assert list(z.shape) == [z.sizes[d] for d in z.dims]
    assert "lat" not in z.dims and "lon" not in z.dims

    _check_output(LSD, x, z)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_lsd_failures(device: str) -> None:
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

    LSD = log_spectral_distance(
        reduction_dimensions=["time"],
        spatial_dimensions=("lat", "lon"),
        ensemble_dimension="ensemble",
    )

    # Test ensemble_dimension in y error
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = from_torch(torch.randn(y_shape, device=device), y_coords)
        LSD(x, y)

    # Test x and y don't have broadcastable shapes
    with pytest.raises(ValueError):
        y_coords = OrderedDict({"phony": np.arange(1)})
        for c in x_coords:
            if c != "ensemble":
                y_coords[c] = x_coords[c]

        y_shape = [len(y_coords[c]) for c in y_coords]
        y = from_torch(torch.randn(y_shape, device=device), y_coords)
        LSD(x, y)

    # Test ensemble_dimension not in x
    with pytest.raises(ValueError):
        y_coords = copy.deepcopy(x_coords)
        y_shape = [len(y_coords[c]) for c in y_coords]
        y = from_torch(torch.randn(y_shape, device=device), y_coords)

        xc = copy.deepcopy(x_coords)
        xc.pop("ensemble")
        xs = from_torch(torch.randn((1, 2, 361, 720), device=device), xc)
        LSD(xs, y)

    # Test incompatible coordinate values
    yc = copy.deepcopy(x_coords)
    yc.pop("ensemble")
    yc["lat"] = np.linspace(-80.0, 80.0, 361)
    y = from_torch(torch.randn((1, 2, 361, 720), device=device), yc)
    with pytest.raises(ValueError):
        LSD(x, y)

    # Test reduction_dimension not in x_coords (output signature)
    xc = copy.deepcopy(x_coords)
    xc.pop("time")
    xs = from_torch(torch.randn((10, 2, 361, 720), device=device), xc)
    with pytest.raises(KeyError):
        LSD.output_coords(xs)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_lsd_accuracy(device: str, atol: float = 1e-5) -> None:
    x_t = torch.randn((1, 2, 128, 128), device=device)

    x_coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["t2m", "tcwv"]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
        }
    )
    x = from_torch(x_t, x_coords)

    # y is a constant factor times x
    factor = 2.0
    y = from_torch(x_t * factor, copy.deepcopy(x_coords))

    LSD = log_spectral_distance(reduction_dimensions=["time"])

    z = LSD(x, y).e2s.to_torch()[0]

    # LSD should be this value
    ref_value = 10 * np.log10(factor)
    assert abs(z - ref_value).all()

    # test that LSD to self is 0
    z = LSD(x, x).e2s.to_torch()[0]
    assert (abs(z) < atol).all()


@pytest.mark.parametrize("wavenumber_cutoff", [-10, 10, None])
@pytest.mark.parametrize("ensemble_dimension", ["ensemble", None])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fss_spatial_dims(
    ensemble_dimension: str | None, device: str, wavenumber_cutoff: int | None
) -> None:
    # test that we can use spatial dimensions that are not the last 2 dims
    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "lat": np.linspace(-90.0, 90.0, 128),
            "lon": np.linspace(0.0, 360.0, 128, endpoint=False),
            "variable": np.array(["t2m", "tcwv"]),
        }
    )
    x = from_torch(torch.zeros((10, 1, 128, 128, 2), device=device), x_coords)

    y_coords = copy.deepcopy(x_coords)
    if ensemble_dimension is not None:
        y_coords.pop(ensemble_dimension)
    y_shape = [len(y_coords[c]) for c in y_coords]
    y = from_torch(torch.randn(y_shape, device=device), y_coords)

    LSD = log_spectral_distance(
        reduction_dimensions=["time"],
        spatial_dimensions=("lat", "lon"),
        wavenumber_cutoff=wavenumber_cutoff,
        ensemble_dimension=ensemble_dimension,
    )

    z = LSD(x, y)
    _check_output(LSD, x, z)
