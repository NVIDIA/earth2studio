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
Restarting a Deterministic Forecast
===================================

This example shows how to use :py:class:`earth2studio.utils.checkpoint.Checkpoint`
to restart a deterministic forecast after it stops partway through a run.

The example uses :py:class:`earth2studio.data.Random` and
:py:class:`earth2studio.models.px.Persistence` so the restart mechanics are easy
to inspect without downloading a model package. The same checkpoint usage applies
to larger prognostic models.

In this example you will learn:

- Creating a persistent checkpoint
- Running a forecast that stops before the requested final horizon
- Re-opening the IO backend and checkpoint
- Resuming the deterministic workflow from the latest completed lead time
"""
# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Set Up
# ------
# A restartable forecast needs two persistent locations: one for forecast fields
# and one for the checkpoint. The IO backend owns the forecast arrays.
# The checkpoint owns small restart metadata, such as the latest
# completed lead time. Model weights and forecast fields are not copied into the
# checkpoint.

# %%
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import earth2studio.run as run
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Persistence
from earth2studio.utils.checkpoint import Checkpoint, bind_checkpoint_state
from earth2studio.utils.time import to_time_array

os.makedirs("outputs", exist_ok=True)

forecast_store = Path("outputs/04_checkpoint_restart.zarr")
checkpoint_store = Path("outputs/04_checkpoint_restart_checkpoint")

for path in (forecast_store, checkpoint_store):
    if path.exists():
        shutil.rmtree(path)

# %%
# Build a small deterministic forecast problem. In a production run these would
# usually be replaced with a downloaded prognostic model and a real data source.
#
# Full checkpoint state can be staged on the same device used for inference.
# Keeping restart tensors on the active CUDA device can reduce CPU/GPU transfers
# during a run; this example falls back to CPU when CUDA is unavailable.

# %%
compute_device = torch.device(
    f"cuda:{torch.cuda.current_device()}"
    if torch.cuda.is_available()
    else "cpu"
)

domain_coords = OrderedDict(
    {
        "lat": np.linspace(-20, 20, 8),
        "lon": np.linspace(120, 180, 12),
    }
)
variables = ["t2m", "u10m"]
time = ["2024-01-01T00:00:00"]
final_nsteps = 5
first_attempt_nsteps = 2


# %%
# Preallocate the full output store. A real full-length deterministic run does
# this before the first model step. We do it explicitly here because this example
# simulates a mid-run stop by intentionally running only the first two steps.

# %%


def deterministic_output_coords(model, time, nsteps):
    input_coords = model.input_coords()
    output_coords = model.output_coords(input_coords).copy()
    for key, value in model.output_coords(input_coords).items():
        if value.shape == (0,):
            del output_coords[key]

    output_coords["time"] = to_time_array(time)
    output_coords["lead_time"] = np.asarray(
        [model.output_coords(input_coords)["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()
    output_coords.move_to_end("lead_time", last=False)
    output_coords.move_to_end("time", last=False)
    return output_coords


@dataclass
class PersistenceRestartState:
    x: torch.Tensor | None = None
    coord_keys: tuple[str, ...] = ()
    coord_values: tuple[np.ndarray, ...] = ()


class RestartablePersistence(Persistence):
    def __init__(self, variables, domain_coords):
        super().__init__(variables, domain_coords)
        self.restart = bind_checkpoint_state(PersistenceRestartState())

    def create_iterator(self, x, coords):
        restored = False
        if (
            self.restart.checkpoint_state_loaded
            and self.restart.x is not None
            and self.restart.coord_keys
        ):
            x = self.restart.x.to(x.device)
            restored = True
            coords = OrderedDict(
                (key, np.asarray(value).copy())
                for key, value in zip(
                    self.restart.coord_keys, self.restart.coord_values
                )
            )

        iterator = super().create_iterator(x, coords)
        if restored:
            next(iterator)
        for x_out, coords_out in iterator:
            if (
                self.restart.checkpoint_enabled
                and self.restart.checkpoint_state_policy == "full"
            ):
                self.restart.x = x_out.detach().clone().to(self.restart.device)
                self.restart.coord_keys = tuple(coords_out.keys())
                self.restart.coord_values = tuple(
                    np.asarray(value).copy() for value in coords_out.values()
                )
            else:
                self.restart.x = None
                self.restart.coord_keys = ()
                self.restart.coord_values = ()
            yield x_out, coords_out


prealloc_model = RestartablePersistence(variables, domain_coords)

io = ZarrBackend(str(forecast_store), backend_kwargs={"overwrite": True})
coords = deterministic_output_coords(prealloc_model, time, final_nsteps)
var_names = coords.pop("variable")
io.add_array(coords, var_names)

# %%
# First Attempt
# -------------
# Every restartable run should be performed inside a checkpoint context. On an
# empty checkpoint, ``with checkpoint`` opens a new session for future writes.
# Construct restart-aware components inside that context so their dataclass state
# binds to the active checkpoint session. The workflow records a checkpoint row
# after each successful IO write because ``flush_interval=1`` and
# ``mode="append"`` keeps each row in the printed checkpoint table.

# %%
checkpoint = Checkpoint(
    "restart-demo",
    path=checkpoint_store,
    mode="append",
    flush_interval=1,
    state_policy="full",
    device=compute_device,
)

with checkpoint as ckpt:
    data = Random(domain_coords=domain_coords)
    model = RestartablePersistence(variables, domain_coords)
    run.deterministic(
        time=time,
        nsteps=first_attempt_nsteps,
        prognostic=model,
        data=data,
        io=io,
        device=compute_device,
        verbose=False,
        checkpoint=ckpt,
    )

print("Checkpoint after the stopped run:")
print(checkpoint)

# %%
# Resume
# ------
# In a new process, re-open the same IO store and checkpoint. The
# printout above shows the available row ids. Select ``-1`` to resume from the
# latest row.
#
# The selected checkpoint session is used as a context manager so the chosen row is
# the active restart state while components are constructed and while the
# workflow runs. This example model hydrates its restart dataclass during
# construction. Its iterator consumes the selected checkpoint boundary internally
# and yields the next forecast state, while the workflow still fetches the normal
# initial condition and feeds it to the iterator.

# %%
io = ZarrBackend(str(forecast_store))
checkpoint = Checkpoint(
    "restart-demo",
    path=checkpoint_store,
    mode="append",
    state_policy="full",
    device=compute_device,
)

with checkpoint.select(-1) as ckpt:
    data = Random(domain_coords=domain_coords)
    model = RestartablePersistence(variables, domain_coords)
    run.deterministic(
        time=time,
        nsteps=final_nsteps,
        prognostic=model,
        data=data,
        io=io,
        device=compute_device,
        verbose=False,
        checkpoint=ckpt,
    )

print("Checkpoint after resume:")
print(checkpoint)
print(io.root.tree())

# %%
# The latest checkpoint row now points at the final completed lead time. If the
# second process stopped too, selecting ``-1`` again would continue from the new
# latest row.

# %%
latest = checkpoint.select(-1)
print(f"Latest restart lead time: {latest.lead_time}")
