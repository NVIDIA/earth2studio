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

"""Pipeline: many work items, one store per stream
=================================================

``Pipeline`` drives a runner over a list of work items. It plans every store
from all items, splits items across ranks, filters and regrids each stream,
writes it, and records finished items so an interrupted run resumes where it
stopped. This example runs it single-process; under ``torchrun`` the same
script splits the items across ranks.

Uses FCN's real execution path with an add-one core and a tiny synthetic
signature, fed by a constant zero source, so step ``k`` holds the value ``k``.
Needs no model weights, downloads, or GPU.
"""

import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from earth2studio.data import Constant
from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.io import XarrayBackend, ZarrBackend
from earth2studio.models.dx import Identity
from earth2studio.models.px.fcn import FCN
from earth2studio.run import (
    Pipeline,
    PrognosticRunner,
    ProgressStore,
    WorkItem,
    work_items,
)
from earth2studio.utils import coord_array


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
                "lat": [10.0, 0.0],
                "lon": [0.0, 10.0, 20.0],
            },
            dynamic=("batch",),
        )


model = TinyFCN(AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))
source = Constant(OrderedDict(lat=np.array([10.0, 0.0]), lon=np.arange(0.0, 21, 10)), 0)
times = [np.datetime64("2026-01-01"), np.datetime64("2026-01-02")]
items = work_items(times, horizon=np.timedelta64(18, "h"))
print(items[0])

# %%
# One store per stream
# --------------------
# ``from_model`` wraps a ``PrognosticRunner``. Each stream maps to its own
# backend; a stream left out of ``output`` is not written.

forecast, copy = XarrayBackend(), XarrayBackend()
Pipeline.from_model(
    model,  # type: ignore[arg-type]
    source,
    diagnostics={"copy": Identity()},
    output={"forecast": forecast, "copy": copy},
    device="cpu",
).run(items)
print(forecast.root)
np.testing.assert_array_equal(forecast.root["u10m"][:, :, 0, 0], [np.arange(4)] * 2)

# %%
# Per-stream filtering and regridding
# -----------------------------------
# ``output_variables`` and ``regridder`` are keyed by stream too. Variables are
# filtered first; the regridder's ``source_grid`` must match the stream's grid,
# and the store is planned on its ``target_grid``.

native = LatLonGrid(np.array([10.0, 0.0]), np.arange(0.0, 21, 10))
finer = LatLonGrid(np.array([10.0, 5.0, 0.0]), np.arange(0.0, 21, 5))
regridded = XarrayBackend()
Pipeline.from_model(
    model,  # type: ignore[arg-type]
    source,
    output={"forecast": regridded},
    output_variables={"forecast": ["v10m"]},
    regridder={"forecast": BilinearRegridder(native, finer)},
    device="cpu",
).run(items)
print(dict(regridded.root["v10m"].sizes))

# %%
# Resume
# ------
# A ``ProgressStore`` records each item once its outputs flush. Rerunning the
# same items skips the finished ones. The store also keeps a manifest of the
# stream schemas, seed and tags, and refuses to resume a run that differs.


class FailOnce(PrognosticRunner):
    """Runner that fails the second item once, like a preempted job."""

    failed = False

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Fail the second item the first time it runs."""
        if item.time == times[1] and not self.failed:
            self.failed = True
            raise RuntimeError("preempted")
        yield from super().run_item(item)


with tempfile.TemporaryDirectory() as tmp:
    path, progress = str(Path(tmp) / "forecast.zarr"), Path(tmp) / "progress"
    runner = FailOnce(model, source)  # type: ignore[arg-type]
    try:
        Pipeline(
            runner,
            output={"forecast": ZarrBackend(path)},
            progress=ProgressStore(progress),
            device="cpu",
        ).run(items)
    except RuntimeError as error:
        print(f"First run stopped: {error}")
    print(sorted(p.name for p in progress.iterdir()))

    store = ZarrBackend(path)
    Pipeline(
        runner,
        output={"forecast": store},
        progress=ProgressStore(progress),
        device="cpu",
    ).run(items)
    np.testing.assert_array_equal(store["u10m"][:, :, 0, 0], [np.arange(4)] * 2)

# %%
# Distributed runs
# ----------------
# Launch the same script with ``torchrun`` and call
# ``torch.distributed.init_process_group`` first. Each rank gets a contiguous
# slice of the remaining items; rank 0 creates the stores and lists what is
# left. Build backends on every rank, creating storage on rank 0 first:
#
# .. code-block:: python
#
#     from earth2studio.run.work import rank0_first
#
#     output = rank0_first(lambda: {"forecast": ZarrBackend("out/forecast.zarr")})
#
# Shared stores must use chunks of 1 along ``time`` and ``ensemble``, which is
# ``ZarrBackend``'s default; ``Pipeline`` checks this before running.
