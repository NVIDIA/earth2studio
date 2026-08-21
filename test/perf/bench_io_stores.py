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

"""Head-to-head of AsyncZarrBackend store routes on the same workload.

Routes:
- fsspec/LocalStore (file_name path, the default)
- obstore (zarr.storage.ObjectStore over obstore.store.LocalStore)
- icechunk (session.store)
All non-blocking, 20 steps x 4 vars x 721x1440 fp32 (~332 MB), best of 5.

Not collected by pytest; run directly: `uv run python test/perf/bench_io_stores.py`
Requires the icechunk optional dependency (data extra, python>=3.12).
"""

import os
import shutil
import tempfile
import time
from collections import OrderedDict

import numpy as np
import torch

NSTEPS = 20
VARS = ["t2m", "u10m", "v10m", "msl"]
REPEATS = 5

times = np.array([np.datetime64("2024-01-01")])
leads = np.array([np.timedelta64(6 * i, "h") for i in range(NSTEPS)])
coords = OrderedDict(
    {
        "time": times,
        "lead_time": leads,
        "variable": np.array(VARS),
        "lat": np.linspace(90, -90, 721),
        "lon": np.linspace(0, 360, 1440, endpoint=False),
    }
)
torch.manual_seed(0)
X = torch.randn(1, NSTEPS, len(VARS), 721, 1440, dtype=torch.float32)
PARALLEL = OrderedDict({"time": times, "lead_time": leads})


def step_coords(i: int) -> OrderedDict:
    c = OrderedDict(coords)
    c["lead_time"] = leads[i : i + 1]
    return c


def du(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )


def bench(name, make, finalize):
    rows = []
    for _ in range(REPEATS):
        td = tempfile.mkdtemp()
        try:
            t0 = time.perf_counter()
            io, ctx = make(td)
            io.add_array(coords, "fields")
            t1 = time.perf_counter()
            for i in range(NSTEPS):
                io.write(X[:, i : i + 1], step_coords(i), "fields")
            t2 = time.perf_counter()
            finalize(io, ctx)
            t3 = time.perf_counter()
            rows.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0, du(td)))
        finally:
            shutil.rmtree(td, ignore_errors=True)
    b = min(rows, key=lambda r: r[3])
    med = sorted(r[3] for r in rows)[len(rows) // 2]
    print(
        f"{name:28s} setup {b[0]:5.2f}s  loop {b[1]:5.2f}s  close+commit {b[2]:5.2f}s  "
        f"total(best) {b[3]:5.2f}s  total(med) {med:5.2f}s  {b[4]/1e6:7.1f} MB"
    )


def main() -> None:
    import icechunk
    import obstore.store

    from earth2studio.io import AsyncZarrBackend

    bench(
        "fsspec/LocalStore (default)",
        lambda td: (
            AsyncZarrBackend(
                os.path.join(td, "o.zarr"), parallel_coords=PARALLEL, blocking=False
            ),
            None,
        ),
        lambda io, ctx: io.close(),
    )

    bench(
        "obstore LocalStore",
        lambda td: (
            AsyncZarrBackend(
                "unused",
                parallel_coords=PARALLEL,
                blocking=False,
                store=obstore.store.LocalStore(td),
            ),
            None,
        ),
        lambda io, ctx: io.close(),
    )

    def make_ice(td):
        repo = icechunk.Repository.open_or_create(
            icechunk.local_filesystem_storage(os.path.join(td, "repo"))
        )
        session = repo.writable_session("main")
        io = AsyncZarrBackend(
            "unused", parallel_coords=PARALLEL, blocking=False, store=session.store
        )
        return io, session

    def fin_ice(io, ctx):
        io.close()
        ctx.commit("bench")

    bench("icechunk session store", make_ice, fin_ice)


if __name__ == "__main__":
    main()
