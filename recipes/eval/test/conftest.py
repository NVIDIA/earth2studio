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

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from earth2studio.data import Random
from earth2studio.models.px import Persistence
from earth2studio.utils.type import CoordSystem

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]


class FakeDiagnostic(torch.nn.Module):
    """Minimal diagnostic model for testing.

    Accepts a set of input variables on a lat/lon grid and produces a set
    of output variables on the same grid.  The output is simply a copy of
    the matching input channels (or zeros for variables not in the input).
    """

    def __init__(
        self,
        input_variables: list[str],
        output_variables: list[str],
        domain_coords: OrderedDict | None = None,
    ) -> None:
        super().__init__()
        self._input_variables = input_variables
        self._output_variables = output_variables
        self._domain = domain_coords or OrderedDict(
            {"lat": SMALL_LAT, "lon": SMALL_LON}
        )

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(self._input_variables),
                "lat": self._domain["lat"],
                "lon": self._domain["lon"],
            }
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        output = input_coords.copy()
        output["variable"] = np.array(self._output_variables)
        return output

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        # Produce output with the correct number of channels.
        n_out = len(self._output_variables)
        out_shape = list(x.shape)
        # Variable dimension is typically index 2 in (batch, time, var, lat, lon)
        # but for diagnostics it depends on the coord layout.  Find it:
        var_idx = list(coords.keys()).index("variable")
        out_shape[var_idx] = n_out
        y = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
        out_coords = self.output_coords(coords)
        return y, out_coords


@pytest.fixture()
def small_domain() -> OrderedDict:
    return OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})


@pytest.fixture()
def prognostic(small_domain) -> Persistence:
    return Persistence(variable=VARIABLES, domain_coords=small_domain)


@pytest.fixture()
def data_source(small_domain) -> Random:
    return Random(domain_coords=small_domain)


@pytest.fixture()
def fake_diagnostic(small_domain) -> FakeDiagnostic:
    """A diagnostic that takes t2m+z500 input and produces 'diag_a' output."""
    return FakeDiagnostic(
        input_variables=["t2m", "z500"],
        output_variables=["diag_a"],
        domain_coords=small_domain,
    )


@pytest.fixture()
def base_cfg(tmp_path) -> OmegaConf:
    return OmegaConf.create(
        {
            "project": "test_eval",
            "run_id": "unit",
            "start_times": ["2024-01-01 00:00:00"],
            "nsteps": 2,
            "ensemble_size": 1,
            "random_seed": 42,
            "pipeline": "forecast",
            "model": {"architecture": "earth2studio.models.px.Persistence"},
            "data_source": {"_target_": "earth2studio.data.Random"},
            "output": {
                "path": str(tmp_path / "outputs"),
                "variables": list(VARIABLES),
                "overwrite": True,
                "thread_writers": 0,
                "chunks": {"time": 1, "lead_time": 1},
            },
        }
    )
