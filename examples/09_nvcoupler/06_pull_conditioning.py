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

# %%
"""
Pull-Pattern Coupling: Feeding a Model That Fetches Its Own Forcing
===================================================================

Coupling a StormCast-style regional model without temp-file staging.

Most models take coupled fields as call arguments — the coupler pushes.
Some do not: StormCast holds a settable ``conditioning_data_source`` and,
inside its own ``__call__``, *pulls* its conditioning by calling the real
``earth2studio.data.utils.fetch_data`` on that attribute. The production
workflow (``serve/server/example_workflows/stormcast_conus_workflow.py``)
copes today by running the full global conditioning forecast first, staging
it to temporary NetCDF files, and handing StormCast an
``InferenceOutputSource`` over those files — a data source masquerading as
GFS. It works, but the entire conditioning forecast must be materialized to
disk before the regional model takes a single step.

nvcoupler's :class:`~earth2studio.nvcoupler.pull.PullAdapter` plays the
same masquerade minus the staging: before each step it installs a
:class:`~earth2studio.nvcoupler.pull.StateDataSource` — an in-memory
DataSource answering fetches from the component's live import State — on
the model's data-source attribute. The model runs its unmodified production
fetch path and receives THIS step's coupled forcing, delivered by a
connector moments earlier in the same run-sequence slot.

In this example you will learn:

- What the pull pattern is and which models need it (StormCast)
- How PullAdapter + StateDataSource replace the temp-file staging
- How an explicit sequential run sequence guarantees fresh conditioning
- How to verify, step by step, that the model pulled the current exchange

One honest caveat up front: the pull path crosses the model's own
xarray/numpy fetch machinery, so pull-coupled components are
**inference-only** — no gradients flow through this exchange (contrast
example 05, where the push-pattern exchange is autograd-clean).
"""

# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# A Pull-Pattern Mock Regional Model
# ----------------------------------
# We stand in for StormCast with a mock that is *protocol-faithful* to its
# coupling surface: a settable ``conditioning_data_source`` attribute, a
# declared list of raw conditioning variables, and — crucially — a fetch
# through the REAL ``earth2studio.data.utils.fetch_data``, the same code
# path the production model runs. If the shim satisfies fetch_data here, it
# satisfies StormCast's mechanics. Its "physics" is a hand-computable
# update of a single radar-reflectivity state:
#
#     refc <- refc + 1 + mean(u10m) + 0.1 * mean(t2m)
#
# and it logs every pull so we can audit exactly what conditioning it saw.

from collections import OrderedDict

import numpy as np
import torch

import earth2studio.nvcoupler as nvc
from earth2studio.data.utils import fetch_data
from earth2studio.nvcoupler.pull import PullAdapter
from earth2studio.nvcoupler.testing import grid_coords

GRID = (8, 16)


class MockPullModel:
    """StormCast stand-in: pulls u10m/t2m via fetch_data inside __call__.

    Mirrors the production coupling surface exactly — the coupler never
    calls anything but ``model(x, coords)``; the conditioning arrives
    through the data source the model itself fetches from.
    """

    conditioning_variables = np.array(["u10m", "t2m"])

    def __init__(self):
        self.conditioning_data_source = None  # PullAdapter sets this
        self.pull_log: list[np.ndarray] = []

    def input_coords(self):
        return OrderedDict(
            {
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["refc"]),
                **grid_coords(*GRID),
            }
        )

    def output_coords(self, input_coords):
        out = OrderedDict({k: v.copy() for k, v in input_coords.items()})
        out["lead_time"] = input_coords["lead_time"] + np.timedelta64(1, "h")
        return out

    def __call__(self, x, coords):
        if self.conditioning_data_source is None:
            raise RuntimeError("conditioning_data_source not set")
        # The REAL fetch path StormCast uses — not a shortcut around it.
        cond, _ = fetch_data(
            self.conditioning_data_source,
            time=np.atleast_1d(coords["time"]),
            variable=self.conditioning_variables,
        )
        self.pull_log.append(cond.numpy().copy())
        u_mean = cond[0, 0, 0].mean()
        t_mean = cond[0, 0, 1].mean()
        return x + 1.0 + u_mean + 0.1 * t_mean, self.output_coords(coords)

    def to(self, device):
        return self


# %%
# The Global Conditioning Component
# ---------------------------------
# A toy "global" model in the Jussi-workflow shape: it exports the two
# fields StormCast conditions on. Its 10 m wind grows by 1 m/s per step
# (so freshness is detectable — each hour's conditioning differs from the
# last) while temperature holds constant at 280 K.


def global_step(x, coords):
    # x stacks [u10m, t2m]; u10m grows 1 m/s per step, t2m constant
    return torch.stack([x[0] + 1.0, x[1]]), coords


glob = nvc.CallableComponent(
    "global",
    global_step,
    timestep="1h",
    exports=["eastward_wind_10m", "air_temperature_2m"],
)

# %%
# Wire the Regional Component with PullAdapter
# --------------------------------------------
# The regional component imports what the global one exports. Its
# ``import_adapter=PullAdapter()`` is the whole pull-pattern story: before
# each step, the adapter installs a StateDataSource over the import State
# on the model's ``conditioning_data_source``, then calls the model
# unchanged. The model pulls "u10m"/"t2m" by its raw names; the shim
# resolves them to the standard names in the State. ``refc`` is not in the
# default field dictionary, so we register it.

dictionary = nvc.FieldDictionary(nvc.DEFAULT_DICTIONARY)
dictionary.register(
    nvc.FieldEntry("radar_reflectivity", "dBZ", aliases=frozenset({"refc"}))
)

model = MockPullModel()
stormcast = nvc.PrognosticComponent(
    "stormcast",
    model,
    imports=["eastward_wind_10m", "air_temperature_2m"],
    exports=["radar_reflectivity"],
    import_adapter=PullAdapter(),
    variable_aliases={"refc": "radar_reflectivity"},
    dictionary=dictionary,
)

# %%
# Explicit Sequential Run Sequence
# --------------------------------
# Freshness is an *ordering* property, so we write the sequence explicitly
# rather than letting the Driver derive the lagged default: in every 1 h
# slot the global model steps, the connector delivers its new exports, and
# only then does the regional model run — so each pull sees conditioning
# valid at the pulled time. This is the sequential-coupling half of
# example 02, applied to the pull pattern.

T0 = np.datetime64("2024-01-01")

driver = nvc.Driver(
    {"global": glob, "stormcast": stormcast},
    sequence="""
    @1h
      global
      global -> stormcast
      stormcast
    @
    """,
    clock=nvc.Clock(T0, "2024-01-01T04:00", "1h"),
    connectors=[nvc.Connector(glob, stormcast)],
)
print(driver.describe())

# %%
# Initialize and Run 4 Hours
# --------------------------
# The global state starts at u10m = 2, t2m = 280; refc starts at 0. Hand
# computation: at hour k the global model has already stepped, so the pull
# sees u = 2 + k, and the refc increment is 1 + (2 + k) + 0.1 * 280 =
# 31 + k — i.e. 32, 33, 34, 35 over four hours, cumulative 134. We iterate
# with ``steps()`` to record refc after every slot.

ic_glob = (
    torch.stack([torch.full(GRID, 2.0), torch.full(GRID, 280.0)]),
    OrderedDict({"variable": np.array(["u10m", "t2m"]), **grid_coords(*GRID)}),
)
ic_sc = model.input_coords()
ic_sc["time"] = np.array([T0])
driver.initialize(
    {"global": ic_glob, "stormcast": (torch.zeros(1, 1, 1, *GRID), ic_sc)}
)

refc_series = []
for time, states in driver.steps():
    refc_series.append(float(stormcast.export_state["radar_reflectivity"].data.mean()))

# %%
# Prove Every Step Pulled Fresh Conditioning
# ------------------------------------------
# The model's pull log is the audit trail: each entry is what its own
# fetch_data returned that step. The pulled u10m must be 3, 4, 5, 6 — the
# global state *after* that hour's step, never a stale or staged value —
# and each refc increment must match the arithmetic above.

print(
    f"{'hour':>6} {'pulled u10m':>12} {'pulled t2m':>11} "
    f"{'refc incr':>10} {'refc':>8}"
)
prev = 0.0
for k, (pull, refc) in enumerate(zip(model.pull_log, refc_series), start=1):
    u, t = float(pull[0, 0, 0].mean()), float(pull[0, 0, 1].mean())
    incr = refc - prev
    prev = refc
    print(f"{k:>5}h {u:>12.1f} {t:>11.1f} {incr:>10.1f} {refc:>8.1f}")

pulled_u = [float(p[0, 0, 0].mean()) for p in model.pull_log]
if pulled_u != [3.0, 4.0, 5.0, 6.0]:
    raise ValueError("a pull saw stale conditioning — sequencing is broken")
if not np.isclose(refc_series[-1], 32.0 + 33.0 + 34.0 + 35.0):
    raise ValueError("refc(4h) does not match the hand-computed value")
print("\nevery regional step pulled that step's fresh conditioning ✓")

# %%
# Probe the Exchange
# ------------------
# The connector's probe shows the last fields it moved — the same fields
# the StateDataSource then served to the model's fetch. Note the u10m value
# matches the final pull (6.0): what crossed the interface is what the
# model consumed, with no files in between.

u10 = driver.probe("global->stormcast")["eastward_wind_10m"]
t2 = driver.probe("global->stormcast")["air_temperature_2m"]
print(f"last global->stormcast transfer: {u10}")
print(f"                                 {t2}")

# %%
# Honest Closing Notes
# --------------------
# Two limits worth stating plainly. First, the pull path goes through the
# model's own fetch_data/xarray machinery, so field data crosses a numpy
# boundary: pull-coupled components are inference-only, and no gradients
# flow through this exchange (push-pattern adapters keep autograd intact —
# see example 05). Second, this example validates the *mechanics* — the
# mock is protocol-faithful, fetching through the real fetch_data — but a
# run against real StormCast weights, replacing the temp-file staging in
# ``stormcast_conus_workflow.py`` end to end, remains future validation.
