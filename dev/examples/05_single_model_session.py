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

"""Single-model execution through pipelines
==============================================

A coupled workflow and a one-model forecast are meant to run down the same
``ExecutionPlan`` / ``RunSession`` path. The risk is that the coupled vocabulary
leaks into the simple case. This example measures what the single-model path
costs a user, at three levels of customization:

- **Default:** a model and a source. No classes to write.
- **Tier one:** change how steps are post-processed, keeping inputs and outputs
  unchanged. Override one method on the default session: 2 lines here.
- **Tier two:** change what is published. Override that method *and* the
  matching declaration, on the same session class: 13 lines here, 6 of them the
  declaration.

For comparison, writing the plan and session from scratch took 94 lines, 61 of
them boilerplate the plan factory now derives from the model.

The plan is never subclassed at any tier, and ``earth2studio.coupling`` is never
imported; the last section asserts that mechanically.

Uses FCN's real execution path with an add-one core and a tiny synthetic
signature, fed by a constant zero source, so step ``k`` holds the value ``k``.
Needs no model weights, downloads, or GPU.
"""

import sys
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch
import xarray as xr

from earth2studio.data import Constant
from earth2studio.models.px.fcn import FCN
from earth2studio.run.schedules import FixedCadence
from earth2studio.run.session import OutputPort, PortRef, WorkItem
from earth2studio.run.single import SingleModelPlan, SingleModelSession
from earth2studio.utils import coord_array, coord_array_like


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

# %%
# Default
# -------
# A plan is built once from a model and a source, and opened once per work item.
# Identity, cadence, output ports, source requests, resume capability, and the
# snapshot codec are all derived from the model's coordinate contract. The plan
# holds the *source*; the session fetches each item's initial condition.

plan = SingleModelPlan(model, source)
print(plan.describe())

item = WorkItem(np.datetime64("2026-01-01"), np.timedelta64(18, "h"))
events = list(plan.open(item).run())
np.testing.assert_equal(len(events), 4)  # initial condition plus three steps
np.testing.assert_equal(events[-1].produced_at, np.datetime64("2026-01-01T18"))
np.testing.assert_array_equal(events[-1].data.values, 3)

# %%
# Because the plan is item-agnostic, source requests are asked for per item.
# This is what predownload tooling reads, without opening a session.

later = WorkItem(np.datetime64("2026-02-01"), np.timedelta64(18, "h"))
(request,) = plan.external_requests(later)
np.testing.assert_equal(request.valid_time, np.datetime64("2026-02-01"))

# %%
# Resume. Supervision would persist these bytes keyed on plan identity and work
# item; it never looks inside the payload. The output cursor records which steps
# were already published, and the resumed session continues from the stored
# state without fetching again.

session = plan.open(item)
stream = session.run()
head = [next(stream), next(stream)]
raw = plan.encode_snapshot(session.snapshot())

tail = list(plan.open(item, plan.decode_snapshot(raw)).run())
np.testing.assert_equal(len(head) + len(tail), len(events))
for resumed, original in zip(head + tail, events):
    xr.testing.assert_identical(resumed.data, original.data)

# %%
# Tier one: post-process, same inputs and outputs
# -----------------------------------------------
# Masking a region -- the pattern DLESyM's eval pipeline uses to blank invalid
# ocean -- changes what is published but not what is declared. Override
# ``outputs`` and nothing else. Its return value never feeds back into the model,
# so the rollout itself is unaffected.


class NorthernHemisphere(SingleModelSession):
    """Publish only the northern hemisphere; zero elsewhere."""

    def outputs(self, x: xr.DataArray) -> Mapping[str, xr.DataArray]:
        return {"forecast": x.where(x["lat"] > 0, 0.0)}


masked = SingleModelPlan(model, source, session=NorthernHemisphere)
last = list(masked.open(item).run())[-1]
np.testing.assert_array_equal(last.data.sel(lat=10).values, 3)
np.testing.assert_array_equal(last.data.sel(lat=0).values, 0)

# %%
# Tier two: publish something new
# -------------------------------
# Adding a derived stream changes what is published, so the declaration has to
# change too -- supervision needs every port's schema before anything runs, to
# lay out output storage. That cost is real and unavoidable. What this design
# controls is *where* it lands: in one classmethod on the same session class,
# not in a plan subclass or a separate declaration file.


class WithWindSpeed(SingleModelSession):
    """Also publish 10 m wind speed on its own port."""

    @classmethod
    def output_ports(
        cls, model: TinyFCN, cadence: FixedCadence
    ) -> tuple[OutputPort, ...]:
        (forecast,) = super().output_ports(model, cadence)
        signature = coord_array_like(forecast.signature, {"variable": ["ws10m"]})
        return forecast, OutputPort("wind_speed", ("ws10m",), signature, cadence)

    def outputs(self, x: xr.DataArray) -> Mapping[str, xr.DataArray]:
        u = x.sel(variable="u10m", drop=True)
        v = x.sel(variable="v10m", drop=True)
        axis = x.get_axis_num("variable")
        speed = np.hypot(u, v).expand_dims(variable=["ws10m"], axis=axis)
        return {"forecast": x, "wind_speed": speed}


derived = SingleModelPlan(model, source, session=WithWindSpeed)
np.testing.assert_equal(
    sorted(ref.port for ref in derived.output_ports), ["forecast", "wind_speed"]
)
speeds = [e for e in derived.open(item).run() if e.port == "wind_speed"]
np.testing.assert_allclose(speeds[-1].data.values, np.hypot(3, 3))

# %%
# A session class changes plan identity, so a snapshot taken under one session
# can never resume a different one.

np.testing.assert_equal(plan.identity == derived.identity, False)
np.testing.assert_equal(
    derived.output_ports[PortRef("model", "wind_speed")].variables, ("ws10m",)
)

# %%
# The measurement. Nothing above imported the component graph package.

np.testing.assert_equal(
    [name for name in sys.modules if name.startswith("earth2studio.coupling")], []
)

# %%
# Below tier two sits the full escape hatch: implement ``RunSession`` and
# ``ExecutionPlan`` directly, for execution the default loop cannot express.
# Supervision handles those identically. ``test/run/test_session.py`` shows one.
