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

"""Worker script launched via ``torchrun`` from the multi-GPU pytest tests.

Usage::

    torchrun --nproc_per_node=N --standalone \
        tests/_multigpu_worker.py --test <test_name> --output-dir /tmp/out

Each ``--test`` value maps to a function in this module.  The script exits
with code 0 on success and 1 on failure (assertion or exception).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Ensure the recipe root (parent of tests/) is on sys.path
_RECIPE_ROOT = str(Path(__file__).resolve().parents[1])
if _RECIPE_ROOT not in sys.path:
    sys.path.insert(0, _RECIPE_ROOT)

from physicsnemo.distributed import DistributedManager

from earth2studio.data import Random
from earth2studio.models.px import Persistence

from src.distributed import run_on_rank0_first
from src.inference import run_inference
from src.output import OutputManager
from src.work import WorkItem, build_work_items, distribute_work

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]


# ---------------------------------------------------------------------------
# Test: distribute_work gives each rank a disjoint, covering partition
# ---------------------------------------------------------------------------
def test_distribute_work(output_dir: str) -> None:
    dist = DistributedManager()
    items = list(range(10))
    my_items = distribute_work(items, dist.rank, dist.world_size)

    result_file = Path(output_dir) / f"rank{dist.rank}_items.json"
    result_file.write_text(json.dumps(my_items))

    torch.distributed.barrier()

    if dist.rank == 0:
        all_items: list[int] = []
        for r in range(dist.world_size):
            data = json.loads(
                (Path(output_dir) / f"rank{r}_items.json").read_text()
            )
            all_items.extend(data)

        assert sorted(all_items) == items, (
            f"Items mismatch: {sorted(all_items)} != {items}"
        )
        no_overlap = len(all_items) == len(set(all_items))
        assert no_overlap, "Ranks received overlapping items"


# ---------------------------------------------------------------------------
# Test: run_on_rank0_first executes without deadlock and produces results
# ---------------------------------------------------------------------------
def test_run_on_rank0_first(output_dir: str) -> None:
    dist = DistributedManager()

    def _write_rank_file() -> str:
        path = Path(output_dir) / f"rank{dist.rank}_r0first.txt"
        path.write_text(f"rank={dist.rank}")
        return str(path)

    result = run_on_rank0_first(_write_rank_file)

    torch.distributed.barrier()

    if dist.rank == 0:
        for r in range(dist.world_size):
            p = Path(output_dir) / f"rank{r}_r0first.txt"
            assert p.exists(), f"rank {r} did not produce output"

    assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# Test: end-to-end multi-GPU inference with Persistence model
# ---------------------------------------------------------------------------
def test_end_to_end_inference(output_dir: str) -> None:
    from omegaconf import OmegaConf

    dist = DistributedManager()

    domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
    prognostic = Persistence(variable=VARIABLES, domain_coords=domain)
    data_source = Random(domain_coords=domain)

    nsteps = 2
    ensemble_size = 1

    cfg = OmegaConf.create(
        {
            "project": "test_multigpu",
            "run_id": "e2e",
            "start_times": [
                "2024-01-01 00:00:00",
                "2024-01-02 00:00:00",
                "2024-01-03 00:00:00",
                "2024-01-04 00:00:00",
            ],
            "nsteps": nsteps,
            "ensemble_size": ensemble_size,
            "random_seed": 42,
            "output": {
                "path": output_dir,
                "variables": list(VARIABLES),
                "overwrite": True,
                "thread_writers": 0,
                "chunks": {"time": 1, "lead_time": 1},
            },
        }
    )

    all_items = build_work_items(cfg)
    my_items = distribute_work(all_items, dist.rank, dist.world_size)
    all_times = np.array(sorted({item.time for item in all_items}))

    with OutputManager(
        cfg,
        prognostic=prognostic,
        times=all_times,
        nsteps=nsteps,
        ensemble_size=ensemble_size,
    ) as output_mgr:
        run_inference(
            work_items=my_items,
            prognostic=prognostic,
            data_source=data_source,
            output_mgr=output_mgr,
            nsteps=nsteps,
            device=torch.device(f"cuda:{dist.local_rank}"),
        )

    torch.distributed.barrier()

    if dist.rank == 0:
        store_path = os.path.join(output_dir, "forecast.zarr")
        assert os.path.exists(store_path), f"Zarr store not found at {store_path}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
_TESTS = {
    "distribute_work": test_distribute_work,
    "run_on_rank0_first": test_run_on_rank0_first,
    "end_to_end_inference": test_end_to_end_inference,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, choices=list(_TESTS))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    DistributedManager.initialize()

    try:
        _TESTS[args.test](args.output_dir)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
