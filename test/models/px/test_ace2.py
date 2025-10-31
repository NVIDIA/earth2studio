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

from collections.abc import Iterable

import numpy as np
import pytest
import torch

# pytest.importorskip("fme")

from earth2studio.data import Random, fetch_data  # noqa: E402
from earth2studio.data.ace import ACE2ERA5Data  # noqa: E402
from earth2studio.models.px.ace2 import ACE2ERA5  # noqa: E402
from earth2studio.utils import handshake_dim  # noqa: E402


@pytest.fixture(scope="function")
def model(model_cache_context) -> ACE2ERA5:
    with model_cache_context():
        package = ACE2ERA5.load_default_package()
        p = ACE2ERA5.load_model(package, forcing_data_source=ACE2ERA5Data(mode="forcing"))
        return p


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_ace2era5_call(model, device):
    torch.cuda.empty_cache()
    # Use a single timestamp; forcing source will handle needed adjustments
    time = np.array([np.datetime64("2001-01-01T00:00")])

    p = model.to(device)

    # Build a Random data source over the model grid
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Validate shape and coords handshake
    assert out.shape[0] == len(time)
    assert out.shape[1] == 1  # one lead time step
    assert out.shape[3] == len(p.lat)
    assert out.shape[4] == len(p.lon)
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_ace2era5_iter(batch, device, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2001-01-01T00:00")])

    p = model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to the front
    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    p_iter = p.create_iterator(x, coords)

    # First yield returns the first forecast step
    out, out_coords = next(p_iter)
    assert len(out.shape) == 6
    assert out.shape[0] == batch
    assert out_coords["lead_time"][0] == np.timedelta64(0, "h")

    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["batch"] == np.arange(batch)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        if i > 2:
            break


