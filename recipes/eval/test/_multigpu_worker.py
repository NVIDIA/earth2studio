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
from physicsnemo.distributed import DistributedManager
from src.distributed import run_on_rank0_first
from src.output import OutputManager
from src.pipelines import ForecastPipeline
from src.work import build_work_items, distribute_work

from earth2studio.data import Random
from earth2studio.models.px import Persistence

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
            data = json.loads((Path(output_dir) / f"rank{r}_items.json").read_text())
            all_items.extend(data)

        assert (
            sorted(all_items) == items
        ), f"Items mismatch: {sorted(all_items)} != {items}"
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

    pipeline = ForecastPipeline()
    pipeline.prognostic = prognostic.to(dist.device)
    pipeline.diagnostics = []
    pipeline.perturbation = None
    pipeline.nsteps = nsteps
    pipeline._prognostic_ic = prognostic.input_coords()
    pipeline._spatial_ref = prognostic.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {}

    total_coords = pipeline.build_total_coords(all_times, ensemble_size)

    with OutputManager(cfg) as output_mgr:
        output_mgr.validate_output_store(total_coords, list(VARIABLES))
        pipeline.run(
            work_items=my_items,
            data_source=data_source,
            output_mgr=output_mgr,
            output_variables=list(VARIABLES),
            device=dist.device,
        )

    torch.distributed.barrier()

    if dist.rank == 0:
        store_path = os.path.join(output_dir, "forecast.zarr")
        assert os.path.exists(store_path), f"Zarr store not found at {store_path}"


# ---------------------------------------------------------------------------
# Test: online scoring is invariant to how the ensemble is split across ranks
# ---------------------------------------------------------------------------
def _grouping_invariance(defer: bool) -> None:
    """The strongest available check on the online reductions.

    For a fixed ensemble, the sufficient statistics must not depend on how
    many ranks the ensemble is spread over.  Running the same synthetic
    ensemble through ``G = world_size`` (distributed) and then through
    ``G = 1`` on rank 0 (every member local, no collectives) exercises
    nearly every reduction, normalization and member-indexing path — and,
    with CRPS enabled, the member exchange itself.

    Three lead steps are driven rather than one so that a deferred
    exchange has somewhere to be wrong: a one-step offset in the CRPS
    bookkeeping would leave term 2 on the wrong lead.
    """
    from omegaconf import OmegaConf
    from src.online import (
        GroupComm,
        StepContext,
        build_spatial_weights,
        build_statistics,
        parse_online_settings,
    )
    from src.work import EnsembleGroup, ensemble_group_for_rank, plan_ensemble_groups

    dist = DistributedManager()
    device = dist.device
    world_size = dist.world_size
    ensemble_size = world_size
    n_leads = 3

    settings = parse_online_settings(
        OmegaConf.create(
            {
                "scoring": {
                    "online": {
                        "crps": True,
                        "defer_pairwise_one_step": defer,
                        # float64 on the wire so any mismatch is a real bug,
                        # not bf16 rounding.
                        "pairwise_comm_dtype": "float64",
                        "variable_chunk": 1,
                    }
                }
            }
        )
    )

    spatial = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
    weights = build_spatial_weights(spatial, lat_weights=True).to(device)

    # Same synthetic ensemble on every rank — a fixed seed, not comms.
    generator = torch.Generator(device="cpu").manual_seed(1234)
    shape = (ensemble_size, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
    members = [
        torch.randn(shape, generator=generator).to(device) for _ in range(n_leads)
    ]
    truths = [
        torch.randn(
            len(VARIABLES), len(SMALL_LAT), len(SMALL_LON), generator=generator
        ).to(device)
        for _ in range(n_leads)
    ]

    def accumulate(comm, member_slice: list[int]) -> dict:
        statistics = build_statistics(
            ensemble_size, False, settings, list(VARIABLES)
        )
        for stat in statistics:
            stat.reset(n_leads, len(VARIABLES), ensemble_size, device)

        ctx = None
        for lead in range(n_leads):
            f_local = members[lead][member_slice]
            truth = truths[lead]
            ctx = StepContext(
                lead_index=lead,
                valid_time=np.datetime64("2024-01-01T06", "ns"),
                y=truth,
                clim=None,
                f_local=f_local,
                d_local=f_local - truth,
                weights=weights,
                valid=torch.ones_like(truth, dtype=torch.bool),
                n_spatial=2,
                ensemble_size=ensemble_size,
                member_ids=comm.group.member_ids,
                comm=comm,
            )
            d = ctx.d_local.double()
            s1 = d.sum(dim=0)
            s2 = (d**2).sum(dim=0)
            below = (ctx.f_local < ctx.y).sum(dim=0, dtype=torch.int32)
            comm.reduce(s1)
            comm.reduce(s2)
            comm.reduce(below)
            ctx.s1, ctx.s2, ctx.below = s1, s2, below

            for stat in statistics:
                stat.update(ctx)

        for stat in statistics:
            flush = getattr(stat, "flush", None)
            if flush is not None:
                flush(ctx)

        state: dict = {}
        for stat in statistics:
            state.update(stat.state())
        return state

    # --- Distributed: G = world_size, one member per rank ------------------
    rank_groups = plan_ensemble_groups(world_size, ensemble_size)
    group = ensemble_group_for_rank(dist.rank, rank_groups)
    process_group = None
    for ranks in rank_groups:
        pg = torch.distributed.new_group(ranks=list(ranks))
        if tuple(ranks) == group.ranks:
            process_group = pg
    distributed_state = accumulate(
        GroupComm(group, process_group), list(group.member_ids)
    )

    torch.distributed.barrier()
    if dist.rank != 0:
        return

    # --- Reference: G = 1, every member local, no collectives -------------
    solo_group = EnsembleGroup(
        group_id=0,
        n_groups=1,
        group_rank=0,
        group_size=1,
        members_per_rank=ensemble_size,
        ranks=(0,),
        member_ids=tuple(range(ensemble_size)),
    )
    reference_state = accumulate(
        GroupComm(solo_group, None), list(range(ensemble_size))
    )

    assert "crps_t2" in distributed_state, "CRPS was not accumulated"
    assert set(distributed_state) == set(reference_state), (
        f"field mismatch: {sorted(distributed_state)} vs "
        f"{sorted(reference_state)}"
    )
    for name, got in distributed_state.items():
        expected = reference_state[name]
        assert not torch.isnan(got).any(), f"'{name}' has unfilled lead steps"
        # Exact up to float64 reduction-order noise: the collective sums the
        # same terms as the local path, just in a different association.
        assert torch.allclose(got, expected, rtol=1e-11, atol=1e-12), (
            f"grouping invariance violated for '{name}' (defer={defer}): "
            f"max |diff| = {(got - expected).abs().max().item()}"
        )


def test_online_grouping_invariance(output_dir: str) -> None:
    """Grouping invariance with the deferred member exchange (the default)."""
    _grouping_invariance(defer=True)


def test_online_grouping_invariance_sync(output_dir: str) -> None:
    """Same, completing each exchange in place — must produce identical sums.

    Deferral changes *when* the collective is waited on, not what it
    computes, so the two must agree to the last bit of reduction noise.
    """
    _grouping_invariance(defer=False)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
_TESTS = {
    "distribute_work": test_distribute_work,
    "run_on_rank0_first": test_run_on_rank0_first,
    "end_to_end_inference": test_end_to_end_inference,
    "online_grouping_invariance": test_online_grouping_invariance,
    "online_grouping_invariance_sync": test_online_grouping_invariance_sync,
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
