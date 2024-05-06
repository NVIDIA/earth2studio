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

import earth2studio.run as run
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import Identity
from earth2studio.models.px import Persistence
from earth2studio.utils.type import CoordSystem


class PhooDiagnostic(torch.nn.Module):
    """Dummy diagnostic that just sub-selects variables"""

    def __init__(self, in_variable: list[str], out_variable: list[str]):
        super().__init__()
        self.in_variable = np.array(in_variable)
        self.out_variable = np.array(out_variable)

    @property
    def input_coords(self) -> CoordSystem:
        return OrderedDict({"batch": np.empty(0), "variable": self.in_variable})

    @property
    def output_coords(self) -> CoordSystem:
        return OrderedDict({"batch": np.empty(0), "variable": self.out_variable})

    def __call__(self, x, coords):
        dim = list(coords).index("variable")
        indexes = np.argwhere(
            (coords["variable"][:, None] == self.out_variable[None]).sum(axis=1)
        )[:, 0]
        out = torch.index_select(x, dim, torch.IntTensor(indexes).to(x.device))
        out_coords = coords.copy()
        out_coords["variable"] = self.out_variable
        return out, out_coords


# This class is used to verify the workflow moved the model onto the right device
class TestPersistence(Persistence):
    def __init__(self, *args, target_device="cpu"):
        super().__init__(*args)
        self.target_device = torch.device(target_device)

    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        assert x.device == self.target_device
        return super()._forward(x, coords)


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))]),
        OrderedDict([("c1", np.arange(10))]),
        OrderedDict([("c1", np.arange(5)), ("c2", np.arange(5)), ("c3", np.arange(5))]),
    ],
)
@pytest.mark.parametrize(
    "variable", [["t2m"], ["u10m", "v10m"], ["u10m", "u100", "nvidia"]]
)
@pytest.mark.parametrize("nsteps", [5, 10])
@pytest.mark.parametrize("time", [["2024-01-01"]])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_diagnostic(coords, variable, nsteps, time, device):

    data = Random(domain_coords=coords)
    model = TestPersistence(variable, coords, target_device=device)
    diagnostic = Identity()

    io = ZarrBackend()

    io = run.diagnostic(time, nsteps, model, diagnostic, data, io, device=device)

    for var in variable:
        assert io[var].shape[0] == len(time)
        assert io[var].shape[1] == nsteps + 1
        for i, (key, value) in enumerate(coords.items()):
            assert io[var].shape[i + 2] == value.shape[0]


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("c1", np.arange(5)), ("c2", np.arange(5)), ("c3", np.arange(5))]),
        OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))]),
    ],
)
@pytest.mark.parametrize(
    "in_variable, out_variable",
    [
        (["u10m", "u100", "z500", "t2m"], ["t2m"]),
        (["u10m", "v10m"], ["u10m", "v10m"]),
        (["u10m", "u100", "nvidia"], ["nvidia"]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_diagnostic_mapping(coords, in_variable, out_variable, device):

    nsteps = 5
    time = np.array([np.datetime64("1993-04-05T00:00")])
    data = Random(domain_coords=coords)
    model = Persistence(
        ["u10m", "v10m", "u100", "z500", "t2m", "r700", "msl", "nvidia"], coords
    )
    diagnostic = PhooDiagnostic(in_variable, out_variable)

    io = ZarrBackend()

    io = run.diagnostic(time, nsteps, model, diagnostic, data, io, device=device)
    # Check zarr keys are expected
    for i in io:
        assert i in ["time", "lead_time"] + list(coords) + out_variable

    for var in out_variable:
        assert io[var].shape[0] == len(time)
        assert io[var].shape[1] == nsteps + 1
        for i, (key, value) in enumerate(coords.items()):
            assert io[var].shape[i + 2] == value.shape[0]


@pytest.mark.parametrize(
    "output_coords",
    [
        OrderedDict({"variable": np.array(["u10m"])}),
        OrderedDict(
            {"variable": np.array(["v10m", "t2m"]), "lat": np.array([0, 1, 2, 3])}
        ),
        OrderedDict(
            {
                "variable": np.array(["nvidia"]),
                "lon": np.array([0, 1, 2, 3]),
                "lat": np.array([4, 5, 6]),
            }
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_diagnostic_output_coords(output_coords, device):
    output_coords = output_coords.copy()
    coords = OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])
    variable = ["u10m", "v10m", "u100", "t2m", "nvidia"]
    nsteps = 2
    time = ["1993-04-05T12:00:00"]

    data = Random(domain_coords=coords)
    model = TestPersistence(variable, coords, target_device=device)
    diagnostic = Identity()
    io = ZarrBackend()

    io = run.diagnostic(
        time, nsteps, model, diagnostic, data, io, output_coords, device=device
    )

    for name in variable:
        if name not in output_coords["variable"]:
            assert name not in list(io.root.array_keys())
        else:
            assert name in list(io.root.array_keys())

    del output_coords["variable"]
    for key, value in output_coords.items():
        assert np.array_equal(io[key], value)
