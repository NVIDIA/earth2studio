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

"""Pre-download script for the eval recipe.

Fetches data required by the configured pipeline and writes it into zarr
stores before the GPU inference job runs.  Accepts the same Hydra config
as ``main.py`` so the two scripts stay in sync automatically.

Architecture
------------
Each :class:`~src.pipeline.Pipeline` subclass declares its predownload
requirements by returning a list of :class:`~src.pipeline.PredownloadStore`
entries from :meth:`~src.pipeline.Pipeline.predownload_stores`.  This
script iterates that list and writes each store; no pipeline-specific
logic lives here.

Resume
------
Progress is tracked per-timestamp via marker files.  If interrupted (e.g.
by a SLURM time limit), re-running with the same config skips
already-completed timestamps automatically.  Set
``predownload.overwrite=true`` to recreate stores from scratch.

Typical usage
-------------
Single-process::

    python predownload.py

Multi-process (CPU workers, e.g. on a login or pre-fetch node)::

    torchrun --nproc_per_node=8 --standalone predownload.py

Also pre-fetch verification data (merged into ``data.zarr`` when using the
same source)::

    python predownload.py predownload.verification.enabled=true
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

import hydra
from src.distributed import configure_logging
from src.output import OutputManager, build_predownload_coords, sentinel_path
from src.pipeline import PredownloadStore, build_pipeline
from src.predownload_utils import (
    compute_verification_times,
    infer_step_hours,
    squeeze_lead_time,
    union_variables,
)
from src.work import (
    clear_predownload_progress,
    distribute_work,
    filter_predownload_completed,
    write_predownload_marker,
)

from earth2studio.data import fetch_data

# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# ---------------------------------------------------------------------------
# Existing tests (test/test_predownload.py) import these under their original
# private names.  Keep the aliases until those tests are updated.
_compute_verification_times = compute_verification_times
_infer_step_hours = infer_step_hours
_squeeze_lead_time = squeeze_lead_time
_union_variables = union_variables


# ---------------------------------------------------------------------------
# Generic store download
# ---------------------------------------------------------------------------


def _download_store(
    cfg: DictConfig,
    dist: DistributedManager,
    store: PredownloadStore,
    overwrite: bool,
) -> None:
    """Open a zarr store and download all of *store*'s data into it.

    The per-rank partitioning, resume filtering, and per-timestamp progress
    markers are uniform across all pipelines — only the ``(source, times,
    variables, spatial_ref)`` vary by store.
    """
    store_coords = build_predownload_coords(
        store.spatial_ref, np.array(store.times, dtype="datetime64[ns]")
    )

    with OutputManager(
        cfg,
        store_name=f"{store.name}.zarr",
        overwrite=overwrite,
        resume=not overwrite,
    ) as mgr:
        mgr.validate_output_store(store_coords, store.variables)
        _download_to_store(store=store, output_mgr=mgr, cfg=cfg, dist=dist)


def _download_to_store(
    store: PredownloadStore,
    output_mgr: OutputManager,
    cfg: DictConfig,
    dist: DistributedManager,
) -> None:
    """Fetch each timestamp from *store*'s source and write it to disk.

    Handles resume filtering, work distribution across ranks, and
    per-timestamp progress markers.  Every rank must call this for the
    enclosing :class:`OutputManager` barriers to be satisfied.
    """
    zero_lead = np.array([np.timedelta64(0, "ns")])

    remaining = filter_predownload_completed(list(store.times), cfg, store.name)
    my_times = distribute_work(remaining, dist.rank, dist.world_size)

    logger.info(
        f"Rank {dist.rank}: {store.name} ({store.role}) — "
        f"{len(my_times)}/{len(remaining)} remaining times "
        f"({len(store.times)} total), {len(store.variables)} variables"
    )

    for t in my_times:
        logger.info(f"Rank {dist.rank}: fetching {store.name} {t}")
        x, coords = fetch_data(
            source=store.source,
            time=[t],
            variable=list(store.variables),
            lead_time=zero_lead,
            device=torch.device("cpu"),
        )
        x, coords = squeeze_lead_time(x, coords)
        output_mgr.write(x, coords)
        output_mgr.flush()
        write_predownload_marker(t, cfg, store.name)

    logger.success(f"Rank {dist.rank}: {store.name} download complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="cfg", config_name="predownload")
def main(cfg: DictConfig) -> None:
    """Pre-download data for the eval recipe into zarr stores."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()

    pd_cfg = cfg.predownload
    pd_overwrite = pd_cfg.get("overwrite", False)

    if pd_overwrite and dist.rank == 0:
        clear_predownload_progress(cfg)

    pipeline = build_pipeline(cfg)
    stores = pipeline.predownload_stores(cfg)

    if not stores:
        logger.info("Pipeline declared no predownload stores — nothing to fetch.")
    else:
        logger.info(
            f"Pipeline '{type(pipeline).__name__}' declared "
            f"{len(stores)} predownload store(s): "
            f"{', '.join(f'{s.name} ({s.role})' for s in stores)}"
        )
        for store in stores:
            _download_store(cfg, dist, store, pd_overwrite)

    # --- Sentinel file ------------------------------------------------------
    if dist.distributed:
        torch.distributed.barrier()

    if dist.rank == 0:
        sp = sentinel_path(cfg)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(np.datetime64("now", "s").item().isoformat())
        logger.info(f"Sentinel written: {sp}")

    logger.success("Pre-download finished.")


if __name__ == "__main__":
    main()
