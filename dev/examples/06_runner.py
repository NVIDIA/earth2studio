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

"""Runners: one work item at a time
=====================================

A runner holds models and bound sources and executes one work item. ``Pipeline``
will drive any runner the same way: distribute items, filter and write each
yielded stream, and resume at item granularity. This example covers both built-in
runners and a hand-written one.

Uses FCN's real execution path with an add-one core and a tiny synthetic
signature, fed by a constant zero source, so step ``k`` holds the value ``k``.
Needs no model weights, downloads, or GPU.
"""

from collections import OrderedDict
from collections.abc import Iterator, Mapping

import numpy as np
import torch
import xarray as xr

from earth2studio.data import Constant
from earth2studio.models.dx import DerivedWS, Identity
from earth2studio.models.px.fcn import FCN
from earth2studio.run import (
    DataRequest,
    DiagnosticRunner,
    PrognosticRunner,
    Runner,
    WorkItem,
)
from earth2studio.utils import coord_array
from earth2studio.utils.coords import CoordSystem


class AddOne(torch.nn.Module):
    """Model core that makes every step observable."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Advance by adding one."""
        return x + 1


class TinyFCN(FCN):
    """Small synthetic signature for demonstrating the real FCN execution path."""

    def input_coords(self) -> xr.DataArray:
        """Return a two-variable, 2 by 3 input signature."""
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": ["u10m", "v10m"],
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
            dynamic=("batch",),
        )


model = TinyFCN(AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))
source = Constant(OrderedDict(lat=np.array([10, 0]), lon=np.array([0, 10, 20])), 0)
item = WorkItem(np.datetime64("2026-01-01"), np.timedelta64(18, "h"))

# %%
# Built-in runner
# ---------------
# Built once from a model and a source, run once per item. Each step yields one
# output per stream; diagnostics add streams next to ``forecast``.

runner = PrognosticRunner(model, source, diagnostics={"copy": Identity()})
steps = list(runner.run_item(item))
np.testing.assert_equal(len(steps), 4)  # initial condition plus three steps
np.testing.assert_array_equal(steps[-1]["forecast"].values, 3)

# %%
# Output schemas and data requests are known before running, so a pipeline can
# lay out stores and predownload without executing anything.

print(
    {name: list(coords) for name, coords in runner.output_coords(item.horizon).items()}
)
(request,) = runner.data_requests(item)
print(request.variable, request.lead_time)

# %%
# Diagnostics on source data
# --------------------------
# ``DiagnosticRunner`` skips the prognostic model and applies diagnostics to
# fetched data. Step ``k`` reads the source at the item's time plus ``k * step``.

global_source = Constant(
    OrderedDict(lat=np.linspace(90, -90, 721), lon=np.linspace(0, 359.75, 1440)), 1
)
wind = DiagnosticRunner(
    {"ws10m": DerivedWS(levels=["10m"])}, global_source, step=np.timedelta64(6, "h")
)
wind_steps = list(wind.run_item(item))
np.testing.assert_equal(len(wind_steps), 4)
np.testing.assert_allclose(wind_steps[-1]["ws10m"].values, np.hypot(1, 1), rtol=1e-6)

# %%
# A hand-written runner
# ---------------------
# Anything with these four members is a runner: no base class, no session or
# plan types. This one stops once every value reaches a threshold, so it cannot
# declare its requests upfront and returns ``None``; that disables complete
# predownload but not execution.


class StopAtThreshold:
    """Roll out until every value reaches ``threshold``."""

    supports_member_batching = False

    def __init__(self, model: TinyFCN, source: Constant, threshold: float) -> None:
        self.inner = PrognosticRunner(model, source)
        self.threshold = threshold

    def to(self, device: torch.device) -> "StopAtThreshold":
        """Move the wrapped model."""
        self.inner.to(device)
        return self

    def output_coords(self, horizon: np.timedelta64) -> Mapping[str, CoordSystem]:
        """Return the wrapped schema; an upper bound."""
        return self.inner.output_coords(horizon)

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...] | None:
        """Declare requests unknown upfront."""
        return None

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Yield steps until the threshold."""
        for step in self.inner.run_item(item):
            yield step
            if (step["forecast"] >= self.threshold).all():
                return


custom: Runner = StopAtThreshold(model, source, threshold=2)
np.testing.assert_equal(len(list(custom.run_item(item))), 3)

# %%
# A coupled runner has the same shape: it builds the coupler's driver per item,
# fetches each component's initial condition, and yields each component's
# exports as streams. ``Pipeline`` cannot tell the difference.
