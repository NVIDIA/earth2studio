# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import DLWP
from earth2studio.utils import handshake_dim


class PhooDLWPModel(torch.nn.Module):
    """Dummy DLWP model, adds time-step"""

    def __init__(self, delta_t: int = 6):
        super().__init__()
        self.delta_t = delta_t

    def forward(self, x):
        # 8:15 because field 7 is cosine zenith of first timestep
        x[:, :7] = x[:, 8:15] + self.delta_t
        x[:, 7:14] = x[:, 8:15] + 2 * self.delta_t
        return x[:, :14].contiguous()


@pytest.fixture()
def dlwp_phoo_cs_transform():
    er_num = 721 * 1440
    cs_num = 6 * 64 * 64
    values = np.ones(cs_num)
    indices = np.stack([np.arange(cs_num), np.arange(cs_num)], axis=0)
    return torch.sparse_coo_tensor(
        indices, values, size=(cs_num, er_num), dtype=torch.float
    )


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_dlwp_call(time, dlwp_phoo_cs_transform, device):
    model = PhooDLWPModel()
    landsea_mask = torch.ones(6, 64, 64)
    orography = torch.ones(6, 64, 64)
    latgrid = torch.ones(6, 64, 64)
    longrid = torch.ones(6, 64, 64)
    center = torch.zeros(1, 7, 1, 1)
    scale = torch.ones(1, 7, 1, 1)
    p = DLWP(
        model,
        landsea_mask=landsea_mask,
        orography=orography,
        latgrid=latgrid,
        longrid=longrid,
        cubed_sphere_transform=dlwp_phoo_cs_transform,
        cubed_sphere_inverse=dlwp_phoo_cs_transform.T,
        center=center,
        scale=scale,
    ).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size(
        [len(time), 1, len(p.output_coords(p.input_coords())["variable"]), 721, 1440]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert torch.allclose(
        out, p.to_equirectangular(p.to_cubedsphere(x[:, 1:] + 6))
    )  # Need to cs transform here to get right values
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dlwp_iter(ensemble, dlwp_phoo_cs_transform, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Use dummy model
    model = PhooDLWPModel()
    landsea_mask = torch.ones(6, 64, 64)
    orography = torch.ones(6, 64, 64)
    latgrid = torch.ones(6, 64, 64)
    longrid = torch.ones(6, 64, 64)
    center = torch.zeros(1, 7, 1, 1)
    scale = torch.ones(1, 7, 1, 1)
    p = DLWP(
        model,
        landsea_mask=landsea_mask,
        orography=orography,
        latgrid=latgrid,
        longrid=longrid,
        cubed_sphere_transform=dlwp_phoo_cs_transform,
        cubed_sphere_inverse=dlwp_phoo_cs_transform.T,
        center=center,
        scale=scale,
    ).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    out, out_coords = next(p_iter)  # Skip first which should return the input
    assert torch.allclose(out, x[:, :, 1:])

    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape[0] == ensemble
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        assert torch.allclose(
            out, p.to_equirectangular(p.to_cubedsphere(x[:, :, 1:] + 6 * (i + 1)))
        )  # Need to cs transform here to get right values

        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "ensemble", 0)

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlwp_exceptions(dc, dlwp_phoo_cs_transform, device):
    # Test invalid coordinates error
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Use dummy model
    model = PhooDLWPModel()
    landsea_mask = torch.ones(6, 64, 64)
    orography = torch.ones(6, 64, 64)
    latgrid = torch.ones(6, 64, 64)
    longrid = torch.ones(6, 64, 64)
    center = torch.zeros(1, 7, 1, 1)
    scale = torch.ones(1, 7, 1, 1)
    p = DLWP(
        model,
        landsea_mask=landsea_mask,
        orography=orography,
        latgrid=latgrid,
        longrid=longrid,
        cubed_sphere_transform=dlwp_phoo_cs_transform,
        cubed_sphere_inverse=dlwp_phoo_cs_transform.T,
        center=center,
        scale=scale,
    ).to(device)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model(model_cache_context) -> DLWP:
    # Test only on cuda device
    with model_cache_context():
        package = DLWP.load_default_package()
        p = DLWP.load_model(package)
        return p


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlwp_package(device, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package DLWP
    p = model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 7, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
