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

Caches the data the configured runner will fetch into local stores before
the GPU inference job runs.  Accepts the same Hydra config as ``main.py`` so
the two scripts stay in sync automatically.

Architecture
------------
The runner describes its fetches through the library protocol:
``runner.data_requests(item)`` returns one
:class:`earth2studio.run.runner.DataRequest` per gridded fetch, and runners
binding observation sources also define ``frame_requests(item)`` returning
:class:`src.data.FrameRequest` entries.  This script groups the requests by
the source they target, names each group after the role the runner gave the
source (``runner.source_roles``) and writes one store per role:

- ``<output.path>/<role>.zarr`` holds every valid time a role needs, one
  array per variable on ``(time, <spatial dims>)``.  ``main.py`` reads it
  through :class:`src.data.PredownloadedSource`.
- ``<output.path>/verification.zarr`` holds the verification fields at every
  forecast valid time, on the scored grid, always as a separate store.
- ``<output.path>/<name>.parquet/`` holds one observation frame per analysis
  time for data assimilation runners.

Resume
------
A :class:`earth2studio.run.ProgressStore` records progress per time
under ``<output.path>/.progress_predownload_<role>``.  Re-running with the
same config skips finished times.  Set ``predownload.overwrite=true`` to
recreate the stores from scratch; changing the campaign's times or variables
also requires it, since the zarr stores have a fixed layout.

Typical usage
-------------
Single process::

    python predownload.py

Multi-process (CPU workers, for example on a pre-fetch node)::

    torchrun --nproc_per_node=8 --standalone predownload.py

Also pre-fetch verification data::

    python predownload.py predownload.verification.enabled=true
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig
from src.campaign import build_items
from src.data import (
    FrameRequest,
    frame_filename,
    frame_store_path,
    instantiate_source,
    store_name,
)
from src.logs import configure_logging, finalize_distributed, init_distributed
from src.output import build_regridders, progress_path, sentinel_path, store_path
from src.runners import build_runner

from earth2studio.data import fetch_data, fetch_dataframe
from earth2studio.grids import GridDefinition, infer_grid
from earth2studio.io import ZarrBackend
from earth2studio.run import ProgressStore, WorkItem, work_items
from earth2studio.run.runner import DataRequest
from earth2studio.run.work import (
    barrier,
    broadcast,
    distribute_work,
    rank0_first,
    rank_info,
)
from earth2studio.utils.coords import coord_array_like

ZERO_LEAD = np.array([np.timedelta64(0, "h")])
VERIFICATION_ROLE = "verification"


@dataclass
class GriddedStore:
    """One zarr store to fill: a source, its valid times and variables.

    Attributes
    ----------
    role : str
        Store name without extension (``data``, ``forcing0``, ``verification``).
    source : Any
        DataSource to fetch from.
    times : np.ndarray
        Valid times to cache, ``datetime64[ns]``, sorted and unique.
    variables : list[str]
        Variables to cache, in first-seen order.
    target_grid : GridDefinition | None
        Grid to regrid fetched fields onto, or None for the source grid.
    """

    role: str
    source: Any
    times: np.ndarray
    variables: list[str]
    target_grid: GridDefinition | None = None


@dataclass
class FrameStore:
    """One parquet directory to fill with observation frames.

    Attributes
    ----------
    name : str
        Slot name; the store is ``<output.path>/<name>.parquet``.
    source : Any
        DataFrameSource to fetch from.
    times : np.ndarray
        Analysis times to cache, ``datetime64[ns]``, sorted and unique.
    variables : np.ndarray
        Variables requested from the source.
    fields : np.ndarray | None
        Columns requested, or None for all.
    """

    name: str
    source: Any
    times: np.ndarray
    variables: np.ndarray
    fields: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Planning: from runner requests to stores
# ---------------------------------------------------------------------------


def _as_times(values: Iterable[Any]) -> np.ndarray:
    """Sorted unique ``datetime64[ns]`` array."""
    return np.unique(np.asarray(list(values), dtype="datetime64[ns]"))


def _valid_times(request: DataRequest) -> np.ndarray:
    """Every ``time + lead_time`` of a request."""
    times = np.asarray(request.time, dtype="datetime64[ns]").reshape(-1)
    leads = np.asarray(request.lead_time, dtype="timedelta64[ns]").reshape(-1)
    return (times[:, None] + leads[None, :]).reshape(-1)


def _collect_requests(
    runner: Any, items: Sequence[WorkItem]
) -> list[DataRequest] | None:
    """Gather every gridded request of the campaign, or None if unknowable."""
    requests: list[DataRequest] = []
    for item in items:
        declared = runner.data_requests(item)
        if declared is None:
            return None
        requests.extend(declared)
    return requests


def plan_gridded_stores(
    cfg: DictConfig, runner: Any, items: Sequence[WorkItem]
) -> list[GriddedStore]:
    """Group the runner's gridded requests into one store per source role.

    Source identity matches each request to ``runner.source_roles``; a source
    the runner did not name gets the role ``source<k>``.  The planner skips the
    ``data`` role when ``ic_source`` is bring-your-own, and every role bound to
    no source.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config.
    runner : Runner
        Campaign runner built with live sources.
    items : Sequence[WorkItem]
        Every work item of the campaign.

    Returns
    -------
    list[GriddedStore]
        Stores in role order of first appearance.
    """
    requests = _collect_requests(runner, items)
    if requests is None:
        logger.warning(
            f"{type(runner).__name__} cannot describe its fetches "
            "(data_requests returned None); skipping gridded caching."
        )
        return []
    roles: dict[int, str] = {}
    for role, source in (getattr(runner, "source_roles", None) or {}).items():
        if source is not None:
            roles.setdefault(id(source), str(role))
    byo_roles: set[str] = set()
    if cfg.get("ic_source") is not None:
        byo_roles.add("data")

    by_role: dict[str, GriddedStore] = {}
    times: dict[str, list[np.ndarray]] = {}
    unnamed = 0
    for request in requests:
        if request.source is None:
            continue
        role = roles.get(id(request.source))
        if role is None:
            role = f"source{unnamed}"
            unnamed += 1
            roles[id(request.source)] = role
        if role in byo_roles:
            continue
        store = by_role.get(role)
        if store is None:
            store = GriddedStore(
                role, request.source, np.array([], "datetime64[ns]"), []
            )
            by_role[role] = store
            times[role] = []
        for name in np.asarray(request.variable).reshape(-1):
            name = str(name)
            if name not in store.variables:
                store.variables.append(name)
        times[role].append(_valid_times(request))
    for role, store in by_role.items():
        store.times = _as_times(np.concatenate(times[role]))
    for role in sorted(byo_roles):
        logger.info(f"Role '{role}' is bring-your-own; not cached.")
    return list(by_role.values())


def _forecast_schema(schemas: Mapping[str, xr.DataArray]) -> xr.DataArray:
    """The ``forecast`` stream schema, or the first stream's."""
    if "forecast" in schemas:
        return schemas["forecast"]
    return schemas[next(iter(schemas))]


def plan_verification_store(
    cfg: DictConfig, runner: Any, items: Sequence[WorkItem]
) -> GriddedStore | None:
    """Describe the verification store, or None when the config does not want one.

    The planner caches verification when ``predownload.verification.enabled``
    holds and ``verification_source`` is not bring-your-own.  The source is
    ``predownload.verification.source`` or, failing that, ``data_source``
    (or the runner's own ``data`` source).  Valid times are every initial
    time plus every lead of the runner's forecast stream; the target grid is
    the scored grid when ``output.regrid`` names one.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config.
    runner : Runner
        Campaign runner built with live sources.
    items : Sequence[WorkItem]
        Every work item of the campaign.

    Returns
    -------
    GriddedStore | None
        The verification store description.
    """
    pd_cfg = cfg.predownload.get("verification", None) or {}
    if not pd_cfg.get("enabled", False):
        return None
    if cfg.get("verification_source") is not None:
        logger.info("verification_source is bring-your-own; not cached.")
        return None
    node = pd_cfg.get("source", None)
    if node is None:
        node = cfg.get("data_source", None)
    source = instantiate_source(node)
    if source is None:
        source = (getattr(runner, "source_roles", None) or {}).get("data")
    if source is None:
        logger.warning(
            "No verification source: set predownload.verification.source or "
            "data_source. Skipping the verification store."
        )
        return None
    scoring = cfg.get("scoring", None) or {}
    variables = [
        str(v) for v in (scoring.get("variables", None) or cfg.output.variables)
    ]
    if not items:
        return None
    final_lead_time = max(item.final_lead_time for item in items)
    schemas = runner.output_coords(final_lead_time)
    schema = _forecast_schema(schemas)
    leads = np.asarray(schema.coords["lead_time"].values, dtype="timedelta64[ns]")
    initial = _as_times(item.time for item in items)
    times = _as_times((initial[:, None] + leads[None, :]).reshape(-1))
    target_grid: GridDefinition | None = None
    if cfg.output.get("regrid"):
        regridders = build_regridders(cfg, schemas)
        stream = "forecast" if "forecast" in schemas else next(iter(schemas))
        target_grid = (
            regridders[stream].target_grid
            if stream in regridders
            else infer_grid(schema)
        )
    return GriddedStore(VERIFICATION_ROLE, source, times, variables, target_grid)


def plan_frame_stores(
    cfg: DictConfig, runner: Any, items: Sequence[WorkItem]
) -> list[FrameStore]:
    """Group the runner's observation-frame requests into one store per slot.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config; ``predownload.observations.enabled`` gates caching.
    runner : Runner
        Campaign runner, optionally defining ``frame_requests(item)``.
    items : Sequence[WorkItem]
        Every work item of the campaign.

    Returns
    -------
    list[FrameStore]
        Stores in slot order of first appearance.
    """
    frame_requests: Callable[[WorkItem], tuple[FrameRequest, ...]] | None = getattr(
        runner, "frame_requests", None
    )
    if frame_requests is None:
        return []
    obs_cfg = cfg.predownload.get("observations", None) or {}
    if not obs_cfg.get("enabled", True):
        logger.info("predownload.observations.enabled=false; not caching frames.")
        return []
    stores: dict[str, FrameStore] = {}
    times: dict[str, list[np.ndarray]] = {}
    for item in items:
        for request in frame_requests(item) or ():
            store = stores.get(request.name)
            if store is None:
                store = FrameStore(
                    request.name,
                    request.source,
                    np.array([], "datetime64[ns]"),
                    np.asarray(request.variable),
                    None if request.fields is None else np.asarray(request.fields),
                )
                stores[request.name] = store
                times[request.name] = []
            times[request.name].append(
                np.asarray(request.time, dtype="datetime64[ns]").reshape(-1)
            )
    for name, store in stores.items():
        store.times = _as_times(np.concatenate(times[name]))
    return list(stores.values())


# ---------------------------------------------------------------------------
# Progress and distribution
# ---------------------------------------------------------------------------


def _progress(
    cfg: DictConfig, name: str, manifest: Mapping[str, Any], overwrite: bool
) -> ProgressStore:
    """Open the per-time progress store of one predownload store, on every rank."""
    rank, _ = rank_info()
    progress = ProgressStore(
        progress_path(cfg, f"progress_predownload_{name}"), overwrite=overwrite
    )
    attempt = broadcast(progress.setup(manifest) if rank == 0 else None)
    try:
        progress.open(manifest, rank, attempt)
    except ValueError as error:
        raise ValueError(
            f"Predownload store '{name}' was written for a different campaign "
            f"({error}); re-run with predownload.overwrite=true."
        ) from error
    return progress


def _my_times(progress: ProgressStore, times: np.ndarray) -> tuple[list[WorkItem], int]:
    """This rank's slice of the unfinished times, and the unfinished count."""
    rank, world_size = rank_info()
    remaining = progress.remaining(work_items(list(times), np.timedelta64(0, "h")))
    return distribute_work(remaining, rank, world_size), len(remaining)


def _remove_dir(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
        logger.info(f"Overwrite: removed {path}")


# ---------------------------------------------------------------------------
# Gridded store download
# ---------------------------------------------------------------------------


def _fetch_field(store: GriddedStore, time: np.datetime64) -> xr.DataArray:
    """Fetch one valid time of a store and drop the lead-time axis."""
    x = fetch_data(
        store.source,
        time=np.array([time], dtype="datetime64[ns]"),
        variable=np.array(store.variables),
        lead_time=ZERO_LEAD,
        device="cpu",
        target_grid=store.target_grid,
    )
    return x.isel(lead_time=0, drop=True)


def download_gridded_store(
    cfg: DictConfig, store: GriddedStore, overwrite: bool
) -> None:
    """Fetch every time of ``store`` and write it to ``<output.path>/<role>.zarr``.

    Every rank must call this: rank 0 creates the store and the others open
    it, then each rank writes its slice of the unfinished times and records
    them in the progress store.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config; the function reads only ``output.path``.
    store : GriddedStore
        Store description.
    overwrite : bool
        Recreate the store and its progress from scratch.
    """
    rank, _ = rank_info()
    path = store_path(cfg, store_name(store.role))
    manifest = {
        "role": store.role,
        "variables": list(store.variables),
        "times": [str(t) for t in store.times.astype("datetime64[s]")],
    }
    if overwrite and rank == 0:
        _remove_dir(path)
    progress = _progress(cfg, store.role, manifest, overwrite)
    my_items, n_remaining = _my_times(progress, store.times)
    logger.info(
        f"{store.role}: {len(my_items)}/{n_remaining} remaining times on this rank "
        f"({len(store.times)} total), variables {store.variables}"
    )
    if n_remaining == 0:
        progress.close()
        return

    # The template comes from a real field so grid metadata and auxiliary
    # coordinates land in the store with the data; rank 0 fetches it and creates the
    # arrays before the other ranks open the store.  add_array leaves existing
    # arrays alone, so resuming into a complete store is a no-op.
    probe_time = my_items[0].time if my_items else store.times[0]
    fetched: dict[np.datetime64, xr.DataArray] = {}

    def open_backend() -> ZarrBackend:
        backend = ZarrBackend(path, backend_kwargs={"overwrite": False})
        if rank == 0:
            first = _fetch_field(store, probe_time)
            fetched[probe_time] = first
            backend.add_array(coord_array_like(first, {"time": store.times}))
        return backend

    backend = rank0_first(open_backend)
    try:
        for item in my_items:
            logger.info(f"{store.role}: fetching {item.time}")
            x = fetched.pop(item.time, None)
            if x is None:
                x = _fetch_field(store, item.time)
            backend.write(x)
            backend.flush()
            progress.mark_done(item)
    finally:
        backend.close()
        progress.close()
    logger.success(f"{store.role}: download complete on this rank.")


# ---------------------------------------------------------------------------
# Observation frame download
# ---------------------------------------------------------------------------


def download_frame_store(cfg: DictConfig, store: FrameStore, overwrite: bool) -> None:
    """Fetch every analysis time of ``store`` and write one parquet file each.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config; the function reads only ``output.path``.
    store : FrameStore
        Store description.
    overwrite : bool
        Recreate the store and its progress from scratch.
    """
    rank, _ = rank_info()
    store_dir = Path(frame_store_path(cfg.output.path, store.name))
    manifest = {
        "name": store.name,
        "variables": [str(v) for v in np.asarray(store.variables).reshape(-1)],
        "fields": (
            None if store.fields is None else [str(f) for f in np.asarray(store.fields)]
        ),
    }
    if rank == 0:
        if overwrite:
            _remove_dir(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    progress = _progress(cfg, store.name, manifest, overwrite)
    my_items, n_remaining = _my_times(progress, store.times)
    logger.info(
        f"{store.name}: {len(my_items)}/{n_remaining} remaining times on this rank "
        f"({len(store.times)} total)"
    )
    empty = 0
    try:
        for item in my_items:
            logger.info(f"{store.name}: fetching {item.time}")
            df = fetch_dataframe(
                store.source,
                time=np.array([item.time], dtype="datetime64[ns]"),
                variable=np.asarray(store.variables),
                fields=store.fields,
                device="cpu",
            )
            if len(df) == 0:
                empty += 1
                logger.warning(
                    f"{store.name} {item.time} returned an empty frame; storing it "
                    "anyway (the DA model decides how to handle missing observations)."
                )
            # fetch_dataframe attaches numpy-array attrs which parquet cannot
            # serialize; it re-attaches them when reading the store back.
            df.attrs = {}
            df.to_parquet(store_dir / frame_filename(item.time))
            progress.mark_done(item)
    finally:
        progress.close()
    _report_empty_frames(store.name, empty, len(my_items))
    logger.success(f"{store.name}: download complete on this rank.")


def _report_empty_frames(name: str, empty: int, total: int) -> None:
    """Warn about empty frames: a few are archive gaps, every one is a misconfiguration."""
    if total and empty == total:
        logger.error(
            f"EVERY fetched frame for {name} ({empty}/{total}) was empty. This "
            "usually means the initial-condition dates fall outside the "
            "observation archive's coverage, or the source is misconfigured. "
            "Downstream analyses will be all-NaN; check start_times against the "
            "source's available range."
        )
    elif empty:
        logger.warning(
            f"{name}: {empty}/{total} fetched frames were empty (tolerated archive "
            "gaps); the corresponding analyses use a reduced observation set."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(cfg: DictConfig, device: torch.device | str = "cpu") -> None:
    """Cache every store the campaign needs on this rank.

    Parameters
    ----------
    cfg : DictConfig
        Full recipe config.
    device : torch.device | str, optional
        Unused; fetching always happens on CPU.  Kept for symmetry with
        ``main.run``.
    """
    rank, world_size = rank_info()
    overwrite = bool(cfg.predownload.get("overwrite", False))

    runner = build_runner(cfg, prefer_predownloaded=False)
    items = build_items(cfg, runner)

    gridded = plan_gridded_stores(cfg, runner, items)
    verification = plan_verification_store(cfg, runner, items)
    if verification is not None:
        gridded.append(verification)
    frames = plan_frame_stores(cfg, runner, items)

    if not gridded and not frames:
        logger.info("Nothing to pre-download for this campaign.")
    else:
        declared = [f"{s.role}.zarr" for s in gridded] + [
            f"{s.name}.parquet" for s in frames
        ]
        logger.info(
            f"Pre-downloading {len(declared)} store(s) on {world_size} rank(s): "
            f"{', '.join(declared)}"
        )
    for store in gridded:
        download_gridded_store(cfg, store, overwrite)
    for frame_store in frames:
        download_frame_store(cfg, frame_store, overwrite)

    barrier()
    if rank == 0:
        sp = sentinel_path(cfg)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(np.datetime64("now", "s").item().isoformat())
        logger.info(f"Sentinel written: {sp}")


@hydra.main(version_base=None, config_path="cfg", config_name="predownload")
def main(cfg: DictConfig) -> None:
    """Pre-download data for the eval recipe into local stores."""
    configure_logging()
    # Fetching is CPU work; hide accelerators so torchrun uses gloo and the
    # models the runner loads stay on the host.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    device = init_distributed()
    try:
        run(cfg, device)
    finally:
        finalize_distributed()
    logger.success("Pre-download finished.")


if __name__ == "__main__":
    main()
