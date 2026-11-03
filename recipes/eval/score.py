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

"""Scoring entry point: compare stored forecasts against verification data.

Usage
-----
Single process::

    python score.py

With a campaign config::

    python score.py campaign=fcn3_2024_monthly

Distributed (multi-GPU)::

    torchrun --nproc_per_node=$NGPU --standalone score.py \\
        campaign=fcn3_2024_monthly

Prerequisites
-------------
1. ``predownload.py`` with ``predownload.verification.enabled=true`` (or a
   bring-your-own ``verification_source``)
2. ``main.py`` (inference must have completed)

Online runs
-----------
With ``scoring.mode=online`` there is no ``forecast.zarr`` to re-read:
``main.py`` accumulated sufficient statistics as it ran.  This script then
only re-derives ``scores.zarr`` from ``stats.zarr``, which is single-process
and cheap; use it to pick up new derived metrics without re-running
inference.

To score an online run the offline way as a cross-check, possible whenever
the run kept its raw fields (the default ``output`` setting), override the
mode and give the result its own store::

    torchrun --nproc_per_node=$NGPU --standalone score.py \\
        campaign=fcn3_2024_monthly scoring.mode=offline \\
        scoring.output.store_name=scores_offline.zarr
"""

from __future__ import annotations

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from src.campaign import initial_times
from src.logs import configure_logging, finalize_distributed, init_distributed
from src.regions import events_to_attrs, parse_events
from src.scoring import (
    build_input_coords_template,
    build_lead_time_chunks,
    group_score_arrays_by_dims,
    instantiate_metrics,
    online_mode,
    open_prediction_store,
    open_score_store,
    open_verification_source,
    run_scoring,
    scoring_progress_path,
    spatial_coords_from_dataset,
    validate_lead_time_chunking,
)

from earth2studio.run import ProgressStore, work_items
from earth2studio.run.work import (
    barrier,
    broadcast,
    distribute_work,
    rank0_first,
    rank_info,
)


def scoring_device(device: torch.device | str) -> torch.device:
    """Device for metric computations, falling back to CPU without CuPy.

    The library returns CUDA-backed fields through CuPy, so a GPU without
    CuPy installed cannot hold the scored fields; the scores are small and
    CPU scoring stays correct.
    """
    device = torch.device(device)
    if device.type == "cuda":
        try:
            import cupy  # noqa: F401
        except ImportError:
            logger.warning("CuPy is not installed; scoring on CPU instead of CUDA.")
            return torch.device("cpu")
    return device


def run(cfg: DictConfig, device: torch.device | str = "cpu") -> None:
    """Score the campaign described by ``cfg`` on this rank.

    Parameters
    ----------
    cfg : DictConfig
        Full recipe config.
    device : torch.device | str, optional
        Device for metric computations, by default CPU.
    """
    rank, world_size = rank_info()
    device = scoring_device(device)

    # --- Online mode: nothing to re-read -----------------------------------
    # An online run already reduced its forecasts to sufficient statistics
    # during inference, so "scoring" is just the (single-process, cheap)
    # step deriving scores.zarr from stats.zarr.
    if online_mode(cfg):
        from src.online import finalize_online_stats

        if rank == 0:
            finalize_online_stats(cfg)
        barrier()
        logger.success("Scoring finished (online mode: finalized stats.zarr).")
        return

    # --- Open data stores --------------------------------------------------
    prediction_ds = open_prediction_store(cfg)
    verif_source = open_verification_source(cfg)

    times = [np.datetime64(t, "ns") for t in initial_times(cfg)]
    stored = set(np.asarray(prediction_ds.time.values).astype("datetime64[ns]"))
    missing = [str(t) for t in times if t not in stored]
    if missing:
        raise ValueError(
            f"{len(missing)} initial condition(s) of the campaign are not in the "
            f"prediction store (first: {missing[0]}); run main.py for them first."
        )

    lead_times = np.asarray(prediction_ds.lead_time.values)
    if not np.issubdtype(lead_times.dtype, np.timedelta64):
        raise ValueError(
            "score.py pairs each lead_time with a verification time, but this "
            f"store's lead_time is {lead_times.dtype}.  Calendar-window forecasts "
            "hold integer lead windows, which score.py cannot verify yet."
        )
    spatial_coords = spatial_coords_from_dataset(prediction_ds)
    variables = [str(v) for v in cfg.scoring.variables]
    template = build_input_coords_template(prediction_ds, lead_times, variables)

    # --- Metrics and chunking ----------------------------------------------
    metrics = instantiate_metrics(cfg, spatial_coords)
    chunk_size = cfg.scoring.get("lead_time_chunk_size", None)
    validate_lead_time_chunking(metrics, chunk_size, len(lead_times))
    lt_chunks = build_lead_time_chunks(lead_times, chunk_size)

    # --- Score store and progress -------------------------------------------
    array_groups = group_score_arrays_by_dims(metrics, template, np.array(times))
    events = events_to_attrs(parse_events(cfg.scoring.get("events", None)))
    io = rank0_first(lambda: open_score_store(cfg, array_groups, rank, events))

    resume = bool(cfg.scoring.get("resume", False))
    items = work_items(times, np.timedelta64(lead_times[-1], "s"))
    progress = ProgressStore(
        scoring_progress_path(cfg),
        tags={"project": str(cfg.project), "run_id": str(cfg.run_id)},
        overwrite=not resume,
    )
    manifest = {
        "store": str(cfg.scoring.output.get("store_name", "scores.zarr")),
        "metrics": list(metrics),
        "variables": variables,
    }
    attempt = broadcast(progress.setup(manifest) if rank == 0 else None)
    progress.open(manifest, rank, attempt)
    remaining = progress.remaining(items) if resume else list(items)
    if not remaining:
        logger.success("All times already scored; nothing to do.")
    my_items = distribute_work(remaining, rank, world_size)
    logger.info(
        f"Scoring {len(remaining)} initial condition(s) on {world_size} rank(s); "
        f"this rank: {len(my_items)}"
    )

    # --- Run scoring -------------------------------------------------------
    try:
        run_scoring(
            my_items,
            prediction_ds,
            verif_source,
            metrics,
            io,
            variables,
            lead_times,
            lt_chunks,
            spatial_coords,
            device,
            cfg,
            progress=progress,
            rank=rank,
        )
    finally:
        progress.close()
        io.close()
    barrier()
    logger.success("Scoring finished.")


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe scoring entry point: distributed metric computation."""
    configure_logging()
    device = init_distributed()
    try:
        run(cfg, device)
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
