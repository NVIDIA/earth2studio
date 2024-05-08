# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
from datetime import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.statistics import acc, lat_weight
from earth2studio.utils.type import TimeArray, VariableArray

lat_weights = lat_weight(torch.as_tensor(np.linspace(-90.0, 90.0, 361)))


class PhooClimatology:
    """Fake Climatology"""

    def __init__(self, data):
        self.data = data

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self.data.sel(time=time, variable=variable)


@pytest.mark.parametrize("domain_shape", [(721, 1440), (360, 720)])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_acc_correctness(
    domain_shape, device, rtol: float = 1e-1, atol: float = 1e-1
):

    # Set lat/lon in terms of degrees (for use with _compute_lat_weights)
    lon = torch.linspace(-180, 180, domain_shape[1], device=device, dtype=torch.float32)
    lat = torch.linspace(-90, 90, domain_shape[0], device=device, dtype=torch.float32)
    LON, LAT = torch.meshgrid(lon, lat, indexing="xy")

    weights = lat_weight(lat).unsqueeze(1).repeat(1, domain_shape[1])

    x = torch.cos(2 * torch.pi * LAT / (180)).unsqueeze(0).unsqueeze(0)
    x_coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": np.array(["dummy"]),
            "lat": lat.cpu().numpy(),
            "lon": lon.cpu().numpy(),
        }
    )

    y = torch.cos(torch.pi * LAT / (180)).unsqueeze(0).unsqueeze(0)
    y_coords = x_coords

    # Independent of the time means, the ACC score for cos(2*x) and cos(x) is 1/8 π sqrt(15/(32 - 3 π^2))
    # or about 0.98355. For derivation, note that the lat weight gives an extra factor of cos(x)/2 and
    # p1 = int[ (cos(x) -y - E[cos(x)-y]) * (cos(2x) - y - E[cos(2x)-y])] = pi/24
    # p2 = int[ (cos(2x) - y - E[cos(x) - y])^2 cos(x)/2 ] = 16/45
    # p3 = int[ (cos(x) - y - E[cos(x) - y])^2 cos(x)/2 ] = 2/3 - pi^2/16 (here E[.] denotes mean)
    # and acc = p / sqrt(p2 * p3) = 1/8 π sqrt(15/(32 - 3 π^2))

    # Create fake Climatology DataSource
    mean = (
        np.pi / 2 * np.ones((1, 1, domain_shape[0], domain_shape[1]), dtype=np.float32)
    )
    mean = xr.DataArray(
        data=mean, dims=["time", "variable", "lat", "lon"], coords=x_coords
    )
    climatology = PhooClimatology(mean)
    ACC = acc(["lat", "lon"], climatology=climatology, weights=weights)
    acc_, _ = ACC(x, x_coords, y, y_coords)

    assert torch.allclose(
        acc_,
        torch.as_tensor(0.9836, device=device),
        rtol=rtol,
        atol=atol,
    )


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
def test_acc(reduction_weights: tuple[list[str], np.ndarray], device: str) -> None:

    x = torch.randn((10, 1, 2, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y = torch.randn((10, 1, 2, 361, 720), device=device)

    # Create fake Climatology DataSource
    mean = torch.randn_like(y)
    mean = xr.DataArray(data=mean.cpu().numpy(), dims=list(x_coords), coords=x_coords)
    climatology = PhooClimatology(mean)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)

    ACC = acc(reduction_dimensions, climatology=climatology, weights=weights)

    z, c = ACC(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]

    # Test with no provided climatology
    ACC = acc(reduction_dimensions, weights=weights)

    z, c = ACC(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]


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
def test_acc_leadtime(
    reduction_weights: tuple[list[str], np.ndarray], device: str
) -> None:

    x = torch.randn((10, 1, 1, 2, 361, 720), device=device)

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "lead_time": np.array([np.timedelta64(6, "h")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    y_coords = copy.deepcopy(x_coords)
    y = torch.randn((10, 1, 1, 2, 361, 720), device=device)

    # Create fake Climatology DataSource
    mean = torch.randn((10, 2, 2, 361, 720), device=device)
    mean_coords = copy.deepcopy(x_coords)
    mean_coords.pop("lead_time")
    mean_coords["time"] = np.array(
        [np.datetime64("1993-04-05T00:00"), np.datetime64("1993-04-05T06:00:00")]
    )
    mean = xr.DataArray(
        data=mean.cpu().numpy(), dims=list(mean_coords), coords=mean_coords
    )
    climatology = PhooClimatology(mean)

    reduction_dimensions, weights = reduction_weights
    if weights is not None:
        weights = weights.to(device)

    ACC = acc(reduction_dimensions, climatology=climatology, weights=weights)

    z, c = ACC(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]

    # Test with no provided climatology
    ACC = acc(reduction_dimensions, weights=weights)

    z, c = ACC(x, x_coords, y, y_coords)
    assert not any([ri in c for ri in reduction_dimensions])
    assert list(z.shape) == [len(val) for val in c.values()]


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
def test_acc_failures(
    reduction_weights: tuple[list[str], np.ndarray], device: str
) -> None:

    coords = OrderedDict(
        {
            "ensemble": np.arange(10),
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "lead_time": np.array([np.timedelta64(6, "h")]),
            "variable": ["t2m", "tcwv"],
            "lat": np.linspace(-90.0, 90.0, 361),
            "lon": np.linspace(0.0, 360.0, 720, endpoint=False),
        }
    )

    # Create fake Climatology DataSource
    mean = torch.randn((10, 1, 1, 2, 361, 720), device=device)
    mean = xr.DataArray(data=mean.cpu().numpy(), dims=list(coords), coords=coords)
    climatology = PhooClimatology(mean)

    reduction_dimensions, weights = reduction_weights

    # Test with wrong # dimension of weights
    if weights is not None:
        weights = weights.to(device)
        with pytest.raises(ValueError):
            acc(
                reduction_dimensions,
                climatology=climatology,
                weights=weights.unsqueeze(0),
            )

    # Test if x has "lead_time" but "y" does not
    x = torch.randn((10, 1, 1, 2, 361, 720), device=device)
    y = torch.randn((10, 1, 2, 361, 720), device=device)
    y_coords = coords.copy()
    y_coords.pop("lead_time")

    ACC = acc(reduction_dimensions)
    with pytest.raises(KeyError):
        ACC(x, coords, y, y_coords)

    # Test if variables do not match
    x = torch.randn((10, 1, 1, 2, 361, 720), device=device)
    y = torch.randn((10, 1, 1, 1, 361, 720), device=device)
    y_coords = coords.copy()
    y_coords["variable"] = np.array(["t2m"])

    ACC = acc(reduction_dimensions)
    with pytest.raises(ValueError):
        ACC(x, coords, y, y_coords)
