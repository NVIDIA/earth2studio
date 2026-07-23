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
Each :class:`~src.pipelines.Pipeline` subclass declares its predownload
requirements by returning a list of :class:`~src.pipelines.PredownloadStore`
entries from :meth:`~src.pipelines.Pipeline.predownload_stores`.  This
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

from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.data import frame_filename, frame_store_path
from src.distributed import configure_logging
from src.output import OutputManager, build_predownload_coords, sentinel_path
from src.pipelines import PredownloadFrameStore, PredownloadStore, build_pipeline
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

from earth2studio.data import fetch_data, fetch_dataframe

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
# DataFrame (observation) store download
# ---------------------------------------------------------------------------


def _download_frame_store(
    cfg: DictConfig,
    dist: DistributedManager,
    store: PredownloadFrameStore,
    overwrite: bool,
) -> None:
    """Fetch each analysis time's observation DataFrame and write parquet.

    The tabular counterpart of :func:`_download_store` — same per-rank
    partitioning, resume filtering, and per-timestamp progress markers,
    but each timestamp produces one parquet file under
    ``<output.path>/<name>.parquet/`` instead of a zarr slice.  At
    inference time :class:`src.data.PredownloadedFrameSource` serves the
    directory in place of the live observation source.
    """
    store_dir = Path(frame_store_path(cfg.output.path, store.name))

    if dist.rank == 0:
        if overwrite and store_dir.exists():
            import shutil

            shutil.rmtree(store_dir)
            logger.info(f"Overwrite: removed existing frame store {store_dir}")
        store_dir.mkdir(parents=True, exist_ok=True)
    if dist.distributed:
        torch.distributed.barrier()

    remaining = filter_predownload_completed(list(store.times), cfg, store.name)
    my_times = distribute_work(remaining, dist.rank, dist.world_size)

    logger.info(
        f"Rank {dist.rank}: {store.name} ({store.role}) — "
        f"{len(my_times)}/{len(remaining)} remaining times "
        f"({len(store.times)} total), {len(store.variables)} variables"
    )

    empty_count = 0
    for t in my_times:
        logger.info(f"Rank {dist.rank}: fetching {store.name} {t}")
        df = fetch_dataframe(
            source=store.source,
            time=np.array([t], dtype="datetime64[ns]"),
            variable=np.array(store.variables),
            fields=np.array(store.fields),
            device=torch.device("cpu"),
        )
        if len(df) == 0:
            empty_count += 1
            logger.warning(
                f"Rank {dist.rank}: {store.name} {t} returned an empty "
                "frame — storing it anyway (the DA model decides how to "
                "handle missing observations)."
            )
        # fetch_dataframe attaches numpy-array attrs (request_time /
        # request_lead_time) which parquet can't JSON-serialize; drop
        # them before writing.  fetch_dataframe re-attaches request_time
        # when the store is read back at inference time.
        df.attrs = {}
        df.to_parquet(store_dir / frame_filename(t))
        write_predownload_marker(t, cfg, store.name)

    # A few empty frames are normal (archive gaps are tolerated with a
    # warn-and-skip in the obs sources).  Every frame coming back empty is
    # not: it almost always means the requested dates fall outside the
    # observation archive's coverage.  The obs sources no longer raise on
    # missing files, so surface it loudly here instead — otherwise the run
    # would proceed to write all-NaN analyses that only reveal themselves
    # at scoring time.
    if my_times and empty_count == len(my_times):
        logger.error(
            f"Rank {dist.rank}: EVERY fetched frame for {store.name} "
            f"({empty_count}/{len(my_times)}) was empty. This usually means "
            "the requested initial-condition dates are outside the "
            "observation archive's coverage window, or the obs source is "
            "misconfigured. Downstream analyses will be all-NaN. Verify the "
            "campaign's start_times against the source's available range."
        )
    elif empty_count:
        logger.warning(
            f"Rank {dist.rank}: {store.name} — {empty_count}/{len(my_times)} "
            "fetched frames were empty (tolerated archive gaps); the "
            "corresponding analyses will use a reduced observation set."
        )

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
    frame_stores = pipeline.predownload_frame_stores(cfg)

    if not stores and not frame_stores:
        logger.info("Pipeline declared no predownload stores — nothing to fetch.")
    else:
        declared = [f"{s.name} ({s.role})" for s in [*stores, *frame_stores]]
        logger.info(
            f"Pipeline '{type(pipeline).__name__}' declared "
            f"{len(declared)} predownload store(s): {', '.join(declared)}"
        )
        for store in stores:
            _download_store(cfg, dist, store, pd_overwrite)
        for frame_store in frame_stores:
            _download_frame_store(cfg, dist, frame_store, pd_overwrite)

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
