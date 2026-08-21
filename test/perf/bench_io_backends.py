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

"""Benchmark earth2studio IO backends on a forecast-loop workload.

Workload: 1 init time x 20 lead steps x 4 variables x 721x1440 fp32
(~16.6 MB/step, ~332 MB total), written step-by-step like run.deterministic.

Measured per backend:
- setup: backend construction + add_array
- loop: sum of write() calls (what blocks the inference loop)
- finalize: close()/commit() - work to make data durable
- total: setup + loop + finalize
- size: bytes on disk

Not collected by pytest; run directly: `uv run python test/perf/bench_io_backends.py`
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
REPEATS = 3

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


def step_coords(i: int) -> OrderedDict:
    c = OrderedDict(coords)
    c["lead_time"] = leads[i : i + 1]
    return c


def du(path: str | None) -> int:
    if path is None or not os.path.exists(path):
        return 0
    return sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
    )


def run_case(name, make, finalize=None, sizeof=None):
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
            if finalize:
                finalize(io, ctx, td)
            t3 = time.perf_counter()
            size = sizeof(io, ctx, td) if sizeof else du(td)
            rows.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0, size))
            del io
        finally:
            shutil.rmtree(td, ignore_errors=True)
    best = min(rows, key=lambda r: r[3])
    print(
        f"{name:34s} setup {best[0]:6.2f}s  loop {best[1]:6.2f}s  "
        f"final {best[2]:6.2f}s  total {best[3]:6.2f}s  {best[4]/1e6:8.1f} MB"
    )
    return name, best


def main() -> None:
    import icechunk

    from earth2studio.io import (
        AsyncZarrBackend,
        IceChunkBackend,
        KVBackend,
        NetCDF4Backend,
        XarrayBackend,
        ZarrBackend,
    )

    results = []

    results.append(
        run_case(
            "ZarrBackend (local dir)",
            lambda td: (ZarrBackend(os.path.join(td, "o.zarr")), None),
        )
    )

    results.append(
        run_case(
            "NetCDF4Backend",
            lambda td: (
                NetCDF4Backend(os.path.join(td, "o.nc"), backend_kwargs={"mode": "w"}),
                None,
            ),
        )
    )

    def make_async(blocking):
        def _make(td):
            io = AsyncZarrBackend(
                os.path.join(td, "o.zarr"),
                parallel_coords=OrderedDict({"time": times, "lead_time": leads}),
                blocking=blocking,
            )
            return io, None

        return _make

    def async_final(io, ctx, td):
        io.close()

    results.append(
        run_case("AsyncZarrBackend (blocking)", make_async(True), async_final)
    )
    results.append(
        run_case("AsyncZarrBackend (non-blocking)", make_async(False), async_final)
    )

    results.append(
        run_case(
            "IceChunkBackend",
            lambda td: (IceChunkBackend(os.path.join(td, "repo")), None),
            lambda io, ctx, td: io.commit("bench"),
        )
    )

    def make_async_ice(td):
        repo = icechunk.Repository.open_or_create(
            icechunk.local_filesystem_storage(os.path.join(td, "repo"))
        )
        session = repo.writable_session("main")
        io = AsyncZarrBackend(
            "unused",
            parallel_coords=OrderedDict({"time": times, "lead_time": leads}),
            blocking=False,
            store=session.store,
        )
        return io, session

    def async_ice_final(io, ctx, td):
        io.close()
        ctx.commit("bench")

    results.append(
        run_case("AsyncZarr + Icechunk (non-block)", make_async_ice, async_ice_final)
    )

    # In-memory backends (no durability) for reference
    results.append(
        run_case(
            "KVBackend (in-memory)",
            lambda td: (KVBackend(), None),
            sizeof=lambda io, c, td: 0,
        )
    )
    results.append(
        run_case(
            "XarrayBackend (in-memory)",
            lambda td: (XarrayBackend(coords), None),
            sizeof=lambda io, c, td: 0,
        )
    )

    print()
    best = min(results, key=lambda r: r[1][3])
    print(f"fastest total: {best[0]}")


if __name__ == "__main__":
    main()
