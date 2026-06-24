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
Checkpointing a Deterministic Forecast
======================================

Basic inference workflow checkpointing.

This example shows how to use :py:class:`earth2studio.utils.checkpoint.Checkpoint`
to restart a deterministic forecast after it stops partway through a run. See the
:doc:`checkpointing user guide </userguide/advanced/checkpointing>` for more
information on checkpoint catalogs and restart policies.

The example uses :py:class:`earth2studio.data.Random` and
:py:class:`earth2studio.models.px.FCN`, the FourCastNet AFNO prognostic model.

In this example you will learn:

- Creating a persistent checkpoint
- Running a forecast that stops before the requested final horizon
- Re-opening the IO backend and checkpoint
- Resuming the deterministic workflow from the latest completed lead time
"""
# /// script
# dependencies = [
#   "earth2studio[fcn] @ git+https://github.com/NVIDIA/earth2studio.git",
# ]
# ///

# %%
# Set Up
# ------
# A restartable forecast needs two persistent locations: one for forecast fields
# and one for the checkpoint. The IO backend owns the forecast arrays.
# The checkpoint owns restart metadata plus any model state required to continue
# the rollout. Model weights and forecast fields are not copied into the
# checkpoint.
#
# .. warning::
#
#    Model checkpoint state is opt-in. Before relying on restartable inference,
#    verify that the model you plan to use documents checkpoint support. If a
#    model does not support checkpointing yet, open a feature request on the
#    `Earth2Studio GitHub <https://github.com/NVIDIA/earth2studio/issues>`_.

# %%
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import earth2studio.run as run
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN
from earth2studio.utils.checkpoint import Checkpoint
from earth2studio.utils.time import to_time_array

os.makedirs("outputs", exist_ok=True)

forecast_store = Path("outputs/04_checkpoint_restart.zarr")
checkpoint_store = Path("outputs/04_checkpoint_restart_checkpoint")

# Clean up any left over checkpoints
for path in (forecast_store, checkpoint_store):
    if path.exists():
        shutil.rmtree(path)

# %%
# Load the packaged FCN/AFNO forecast model package. The default package points
# at the FourCastNet model artifacts. Instantiate FCN inside checkpoint contexts
# so its restart state binds to the active checkpoint session.
#
# Rollout checkpoint state can be staged on the same device used for inference.
# Setting ``device`` to the current CUDA device can reduce CPU/GPU transfers for
# restart tensors during a run. Set it to ``torch.device("cpu")`` for CPU-only
# development. Since FCN level 2 checkpoints store the complete autoregressive
# state, keep ``history_size`` small when using ``mode="append"``.

# %%
compute_device = torch.device(
    f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)

model_package = FCN.load_default_package()

output_variables = np.array(["t2m", "u10m"])
time = ["2024-01-01T00:00:00"]
final_nsteps = 3
first_attempt_nsteps = 1


# %%
# Preallocate the full output store. A real full-length deterministic run does
# this before the first model step. We do it explicitly here because this example
# simulates a mid-run stop by intentionally running only the first forecast step.
# The IO store writes only two variables, while the checkpoint keeps FCN's complete
# restart state internally when ``level=2`` is used.

# %%


def deterministic_output_coords(model, time, nsteps, variables):
    """Full output coords for entire roll out"""
    input_coords = model.input_coords()
    output_coords = model.output_coords(input_coords).copy()
    for key, value in model.output_coords(input_coords).items():
        if value.shape == (0,):
            del output_coords[key]

    output_coords["time"] = to_time_array(time)
    output_coords["lead_time"] = np.asarray(
        [model.output_coords(input_coords)["lead_time"] * i for i in range(nsteps + 1)]
    ).flatten()
    output_coords["variable"] = variables
    output_coords.move_to_end("lead_time", last=False)
    output_coords.move_to_end("time", last=False)
    return output_coords


def model_domain_coords(model):
    """Small helper"""
    coords = model.input_coords().copy()
    for key in ("batch", "lead_time", "variable"):
        coords.pop(key)
    return coords


# %%
# First Attempt
# -------------
# Every restartable run should be performed inside a checkpoint context. On an
# empty checkpoint, ``with checkpoint`` opens a new session for future writes.
# Construct restart-aware components inside that context so their state binds to
# the active checkpoint session. The workflow records a checkpoint row
# after each successful IO write because ``flush_interval=1`` and
# ``mode="append"`` keeps each row in the printed checkpoint table.

# %%
checkpoint = Checkpoint(
    "restart-demo",
    path=checkpoint_store,
    mode="append",
    flush_interval=1,
    history_size=4,
    level=2,
    device=compute_device,
)

with checkpoint as ckpt:
    model = FCN.load_model(model_package)
    domain_coords = model_domain_coords(model)
    io = ZarrBackend(str(forecast_store), backend_kwargs={"overwrite": True})
    coords = deterministic_output_coords(model, time, final_nsteps, output_variables)
    var_names = coords.pop("variable")
    io.add_array(coords, var_names)
    data = Random(domain_coords=domain_coords)
    run.deterministic(
        time=time,
        nsteps=first_attempt_nsteps,
        prognostic=model,
        data=data,
        io=io,
        output_coords=OrderedDict({"variable": output_variables}),
        device=compute_device,
        verbose=False,
        checkpoint=ckpt,
    )

print("Checkpoint after the stopped run:")
print(checkpoint)

# %%
# Resume
# ------
# In a new process, re-open the same IO store and checkpoint. The printout above
# shows the available row ids. Select ``-1`` to resume from the latest row.
#
# The selected checkpoint session is used as a context manager so the chosen row
# is the active restart state while components are constructed and while the
# workflow runs. ``FCN`` restores its restart state during construction. Its
# iterator consumes the selected checkpoint boundary internally and yields the
# next forecast state, while the workflow still fetches the normal initial
# condition and feeds it to the iterator.

# %%
io = ZarrBackend(str(forecast_store))
checkpoint = Checkpoint(
    "restart-demo",
    path=checkpoint_store,
    mode="append",
    history_size=4,
    level=2,
    device=compute_device,
)

with checkpoint.select(-1) as ckpt:
    model = FCN.load_model(model_package)
    domain_coords = model_domain_coords(model)
    data = Random(domain_coords=domain_coords)
    run.deterministic(
        time=time,
        nsteps=final_nsteps,
        prognostic=model,
        data=data,
        io=io,
        output_coords=OrderedDict({"variable": output_variables}),
        device=compute_device,
        verbose=False,
        checkpoint=ckpt,
    )

print("Checkpoint after resume:")
print(checkpoint)
print(io.root.tree())
