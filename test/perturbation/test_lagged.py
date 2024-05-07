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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.data import GFS, Random, fetch_data
from earth2studio.perturbation import LaggedEnsemble


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00:00")]),
        np.array(
            [np.datetime64("1999-10-11T12:00"), np.datetime64("2001-06-04T00:00")]
        ),
    ],
)
@pytest.mark.parametrize(
    "lags",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-12, "h"), np.timedelta64(6, "h")]),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_lagged_api(time, lags, lead_time, device):

    # Construct Domain Coordinates
    dc = OrderedDict(
        {
            "lat": np.linspace(-90, 90, 360),
            "lon": np.linspace(0, 360, 720, endpoint=False),
        }
    )

    coords = OrderedDict(
        {
            "ensemble": np.arange(len(lags)),
            "time": time,
            "lead_time": lead_time,
            "variable": ["t2m", "tcwv"],
            "lat": dc["lat"],
            "lon": dc["lon"],
        }
    )

    # Initialize Data Source
    r = Random(dc)

    # Initialize Lagged Ensemble
    le = LaggedEnsemble(r, lags)

    x = torch.randn(
        [len(c) for c in coords.values()], device=device, dtype=torch.float32
    )

    y, y_coords = le(x, coords)

    with pytest.raises(ValueError):
        # Test with wrong size of ensemble
        coords["ensemble"] = np.arange(len(lags) + 1)
        x = torch.randn(
            [len(c) for c in coords.values()], device=device, dtype=torch.float32
        )
        y, y_coords = le(x, coords)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_lagged_accuracy(device):

    time = np.array([np.datetime64("2024-01-01T00:00:00")])
    lead_time = np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])
    lags = np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])
    variable = np.array(["t2m", "tcwv"])

    # Construct Domain Coordinates
    dc = OrderedDict(
        {
            "lat": np.linspace(-90, 90, 360),
            "lon": np.linspace(0, 360, 720, endpoint=False),
        }
    )

    coords = OrderedDict(
        {
            "ensemble": np.arange(len(lags)),
            "time": time,
            "lead_time": lead_time,
            "variable": variable,
            "lat": dc["lat"],
            "lon": dc["lon"],
        }
    )

    # Initialize Data Source
    gfs = GFS()
    x0, c = fetch_data(
        source=GFS(), time=time, variable=variable, lead_time=lead_time, device=device
    )
    x = x0.clone().unsqueeze(0).repeat(len(lags), 1, 1, 1, 1, 1)

    # Initialize Lagged Ensemble
    le = LaggedEnsemble(gfs, lags)

    y, y_coords = le(x, coords)

    assert torch.allclose(y[1], x0), torch.linalg.norm(y[1] - x0) / torch.linalg.norm(
        x0
    )
    assert torch.allclose(y[0, 0, 1], x0[0, 0]), torch.linalg.norm(
        y[0, 0, 1] - x0[0, 0]
    ) / torch.linalg.norm(x0[0, 0])
