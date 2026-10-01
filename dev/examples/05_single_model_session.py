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

"""Single-model and custom execution through one contract
=====================================================

The direct forecast loop shares model metadata with graph execution, but does
not require a graph. This example shows stateless output transforms and an
ordinary generator wrapped in ``LoopPlan``. No session subclasses are needed.

Uses FCN with an add-one core and a tiny signature, fed by zeros. No model
weights, downloads, or GPU are required. Pipeline itself is not implemented yet;
these plans already share its proposed execution boundary.
"""

import sys
from collections import OrderedDict
from collections.abc import Iterator, Mapping

import numpy as np
import torch
import xarray as xr

from earth2studio.data import Constant
from earth2studio.models.px.fcn import FCN
from earth2studio.run.session import (
    LoopPlan,
    OutputEvent,
    OutputPort,
    PortRef,
    WorkItem,
)
from earth2studio.run.single import OutputTransform, SingleModelPlan
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
# Mask published outputs
# ----------------------
# A transform receives and returns port-keyed arrays. It must not mutate its
# inputs or keep simulation state; transformed values never feed back into FCN.


def northern_hemisphere(
    outputs: Mapping[str, xr.DataArray],
) -> Mapping[str, xr.DataArray]:
    """Zero the southern hemisphere without changing the output schema."""
    x = outputs["forecast"]
    return {"forecast": x.where(x["lat"] > 0, 0.0)}


mask = OutputTransform(northern_hemisphere, identity="north-mask-v1")
masked = SingleModelPlan(model, source, transforms=(mask,))
last = list(masked.open(item).run())[-1]
np.testing.assert_array_equal(last.data.sel(lat=10).values, 3)
np.testing.assert_array_equal(last.data.sel(lat=0).values, 0)

# %%
# Add a derived stream
# --------------------
# When a transform changes outputs, it also supplies a declaration function.
# Both functions compose in the same order. Identity includes configuration and
# must change when behavior changes; it prevents incompatible checkpoint reuse.


def wind_ports(ports: tuple[OutputPort, ...]) -> tuple[OutputPort, ...]:
    """Declare forecast and wind-speed streams without computing values."""
    (forecast,) = ports
    signature = coord_array_like(forecast.signature, {"variable": ["ws10m"]})
    return forecast, OutputPort("wind_speed", ("ws10m",), signature, forecast.schedule)


def wind_speed(outputs: Mapping[str, xr.DataArray]) -> Mapping[str, xr.DataArray]:
    """Add wind speed to the published forecast."""
    x = outputs["forecast"]
    u = x.sel(variable="u10m", drop=True)
    v = x.sel(variable="v10m", drop=True)
    speed = np.hypot(u, v).expand_dims(
        variable=["ws10m"], axis=x.get_axis_num("variable")
    )
    return {**outputs, "wind_speed": speed}


wind = OutputTransform(wind_speed, identity="wind-speed-v1", ports=wind_ports)
derived = SingleModelPlan(model, source, transforms=(mask, wind))
np.testing.assert_equal(
    sorted(ref.port for ref in derived.output_ports), ["forecast", "wind_speed"]
)
speeds = [event for event in derived.open(item).run() if event.port == "wind_speed"]
np.testing.assert_allclose(speeds[-1].data.sel(lat=10), np.hypot(3, 3))
np.testing.assert_array_equal(speeds[-1].data.sel(lat=0), 0)
np.testing.assert_equal(plan.identity == derived.identity, False)

# %%
# Own the run loop
# ----------------
# A bespoke generator needs only its output schema and a stable identity.
# Here a synthetic observation is fetched conditionally and stops the forecast
# early. The request set depends on values, so complete predownload is unknown.
# LoopPlan defaults to no checkpointing and no member batching. Resumable custom
# execution can implement ExecutionPlan and RunSession directly.

observation_fetches: list[np.datetime64] = []


def observation(time: np.datetime64) -> float:
    """Stand in for a conditional observation fetch."""
    observation_fetches.append(time)
    return 1.5


def adaptive(work: WorkItem) -> Iterator[OutputEvent]:
    """Stop when forecast values exceed an observation-derived threshold."""
    events = plan.open(work).run()
    try:
        for event in events:
            yield event
            if float(event.data.max()) >= 2:
                if float(event.data.max()) > observation(event.produced_at):
                    break
    finally:
        events.close()


custom = LoopPlan(adaptive, output_ports=plan.output_ports, identity="adaptive-v1")
np.testing.assert_equal(custom.external_requests(item), None)
np.testing.assert_equal(observation_fetches, [])  # Planning does not run the loop.
np.testing.assert_equal(len(list(custom.open(item).run())), 3)
np.testing.assert_equal(len(observation_fetches), 1)
np.testing.assert_equal(custom.open(item).checkpoint_boundary, False)
np.testing.assert_equal(
    plan.output_ports[PortRef("model", "forecast")].variables, ("u10m", "v10m")
)
np.testing.assert_equal(
    [name for name in sys.modules if name.startswith("earth2studio.coupling")], []
)
