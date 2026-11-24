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
the GPU inference job runs.  Accepts the same Hydra config as ``main.py``, so
both scripts read one campaign definition.

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
  array per variable on ``(time, <spatial dims>)``; for a forecast source
  the store holds every initial time and lead time on ``(time, lead_time,
  <spatial dims>)``.  ``main.py`` reads the store back through
  :func:`src.data.open_predownloaded`.
- ``<output.path>/verification.zarr`` holds the verification fields at every
  forecast valid time, on the scored grid, always as a separate store.
- ``<output.path>/<name>.parquet/`` holds one observation frame per analysis
  time for data assimilation runners.

Resume
------
A :class:`earth2studio.run.ProgressStore` records progress per time
under ``<output.path>/.progress_predownload_<role>``.  Re-running with the
same config skips finished times.  Set ``predownload.overwrite=true`` to
recreate the stores from scratch; changing the campaign's times, variables,
sources, output grid or averaging window also requires it, since the zarr
stores have a fixed layout.  The progress manifest records all five and
refuses a mismatch.

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

import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
import zarr
from loguru import logger
from omegaconf import DictConfig
from src.campaign import build_items
from src.data import (
    WINDOW_MEANS_ATTR,
    FrameRequest,
    frame_filename,
    frame_store_path,
    instantiate_source,
    is_forecast_source,
    store_name,
)
from src.logs import (
    all_agree,
    configure_logging,
    finalize_distributed,
    init_distributed,
    rank0_first_checked,
)
from src.output import (
    OutputRegridder,
    progress_path,
    sentinel_path,
    store_path,
    verification_regridder,
)
from src.runners import (
    DEFAULT_WINDOW,
    build_runner,
    is_window_mean,
    require_window_mean_verification,
    runner_spec,
)

from earth2studio.data import fetch_data, fetch_dataframe
from earth2studio.grids import GridDefinition
from earth2studio.io import ZarrBackend
from earth2studio.run import ProgressStore, WorkItem, work_items
from earth2studio.run.runner import DataRequest
from earth2studio.run.work import (
    MANIFEST_NAME,
    barrier,
    broadcast,
    distribute_work,
    rank_info,
)
from earth2studio.utils.coords import coord_array_like

ZERO_LEAD = np.array([np.timedelta64(0, "h")])
VERIFICATION_ROLE = "verification"


@dataclass
class GriddedStore:
    """One zarr store to fill: a source, its times and variables.

    Attributes
    ----------
    role : str
        Store name without extension (``data``, ``forcing0``, ``verification``).
    source : Any
        DataSource or ForecastSource to fetch from.
    times : np.ndarray
        Times to cache, ``datetime64[ns]``, sorted and unique: valid times of a
        data source, initial times of a forecast source.
    variables : list[str]
        Variables to cache, in first-seen order.
    lead_times : np.ndarray | None
        Lead times of a forecast source, ``timedelta64[ns]``, sorted and
        unique; None for a data source.
    regrid : OutputRegridder | None
        Brings fetched fields onto the ``output.regrid`` grid, or None to keep
        the source grid.
    """

    role: str
    source: Any
    times: np.ndarray
    variables: list[str]
    lead_times: np.ndarray | None = None
    regrid: OutputRegridder | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    """Root attributes written on the store, describing what it holds."""

    @property
    def target_grid(self) -> GridDefinition | None:
        """Grid the stored fields land on, or None for the source grid."""
        return None if self.regrid is None else self.regrid.target_grid


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


def source_signature(source: Any) -> dict[str, str]:
    """Identify a source for a progress manifest.

    The class path, plus the store path of a store-backed source, so a
    campaign re-pointed at another source (or another saved store) no longer
    passes for the one that filled its stores.

    Parameters
    ----------
    source : Any
        A data, forecast or frame source object.

    Returns
    -------
    dict[str, str]
        ``class`` and, when the source has one, ``store_path``.
    """
    signature = {"class": f"{type(source).__module__}.{type(source).__qualname__}"}
    path = getattr(source, "store_path", None)
    if path is not None:
        signature["store_path"] = str(path)
    return signature


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
    no source.  The store of a data source covers every ``time + lead_time``
    of its requests, that of a forecast source every initial time and lead
    time.

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
    leads: dict[str, list[np.ndarray]] = {}
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
            leads[role] = []
        for name in np.asarray(request.variable).reshape(-1):
            name = str(name)
            if name not in store.variables:
                store.variables.append(name)
        if is_forecast_source(request.source):
            times[role].append(
                np.asarray(request.time, dtype="datetime64[ns]").reshape(-1)
            )
            leads[role].append(
                np.asarray(request.lead_time, dtype="timedelta64[ns]").reshape(-1)
            )
        else:
            times[role].append(_valid_times(request))
    for role, store in by_role.items():
        store.times = _as_times(np.concatenate(times[role]))
        if leads[role]:
            store.lead_times = np.unique(np.concatenate(leads[role]))
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
    ``predownload.verification.source`` or, failing that, ``data_source``,
    the runner's own ``data`` source, or a ``verification_source`` the runner
    exposes (StormScope composes its GOES and MRMS observations).  Valid times
    are every initial
    time plus every lead of the runner's forecast stream.  The fields land on
    the ``output.regrid`` grid when the campaign names one, else on the
    forecast grid (see :func:`src.output.verification_regridder`).  The
    planner refuses a ``window_mean`` run falling back to ``data_source``
    (:func:`src.runners.require_window_mean_verification`).

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
    require_window_mean_verification(cfg, node)
    attrs: dict[str, Any] = {}
    if is_window_mean(cfg):
        # The configured source serves window means; mark the store so scoring
        # accepts it for this runner later without the source configured.
        attrs[WINDOW_MEANS_ATTR] = str(runner_spec(cfg).get("window", DEFAULT_WINDOW))
    if node is None:
        node = cfg.get("data_source", None)
    source = instantiate_source(node)
    if source is None:
        source = (getattr(runner, "source_roles", None) or {}).get("data")
    if source is None:
        source = getattr(runner, "verification_source", None)
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
    return GriddedStore(
        VERIFICATION_ROLE,
        source,
        times,
        variables,
        regrid=verification_regridder(cfg, schema),
        attrs=attrs,
    )


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


def _upgrade_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Retire a manifest written before the current keys when the rest matches.

    A store filled before the recipe recorded its source, grid and
    attributes keeps resuming: the progress store writes the full manifest
    again on setup.
    A manifest whose shared keys differ stays, and the resume fails on it.

    Parameters
    ----------
    path : Path
        The stored manifest file.
    manifest : Mapping[str, Any]
        The manifest the current run writes.
    """
    if not path.is_file():
        return
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError:
        return  # the progress store reports the broken file itself
    current = json.loads(json.dumps(dict(manifest), sort_keys=True, default=str))
    added = set(current) - set(stored)
    if not added or set(stored) - set(current):
        return
    if any(stored[key] != current[key] for key in stored):
        return
    logger.info(
        f"Progress manifest {path} predates the keys {sorted(added)}; recording them."
    )
    path.unlink()


def _progress(
    cfg: DictConfig, name: str, manifest: Mapping[str, Any], overwrite: bool
) -> ProgressStore:
    """Open the per-time progress store of one predownload store, on every rank."""
    rank, _ = rank_info()
    path = progress_path(cfg, f"progress_predownload_{name}")
    progress = ProgressStore(path, overwrite=overwrite)

    def setup() -> int:
        if not overwrite:
            _upgrade_manifest(path / MANIFEST_NAME, manifest)
        return progress.setup(manifest)

    attempt = broadcast(all_agree(setup, rank == 0))
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
    """Fetch one time of a store on its target grid.

    A data-source store drops the singleton lead-time axis; a forecast-source
    store keeps every cached lead time.
    """
    x = fetch_data(
        store.source,
        time=np.array([time], dtype="datetime64[ns]"),
        variable=np.array(store.variables),
        lead_time=ZERO_LEAD if store.lead_times is None else store.lead_times,
        device="cpu",
    )
    if store.regrid is not None:
        x = store.regrid(x)
    if store.lead_times is None:
        x = x.isel(lead_time=0, drop=True)
    return x


def _json_value(value: Any) -> Any:
    """``value`` with NumPy scalars and arrays turned into plain JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _store_template(field: xr.DataArray, times: np.ndarray) -> xr.DataArray:
    """The template of a store: ``field`` over every time, with JSON attributes.

    Zarr writes array attributes as JSON, while some sources attach NumPy
    scalars read from their files (NCAR_ERA5 carries ``ecmwf_parameter`` and
    ``quantization_nsd`` as ``int32``), which the JSON encoder rejects.
    """
    template = coord_array_like(field, {"time": times})
    template.attrs = _json_value(dict(template.attrs))
    for name in template.coords:
        variable = template.coords[name].variable
        variable.attrs = _json_value(dict(variable.attrs))
    return template


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
        "lead_times": (
            None
            if store.lead_times is None
            else [str(t) for t in store.lead_times.astype("timedelta64[s]")]
        ),
        "source": source_signature(store.source),
        "grid": None if store.target_grid is None else store.target_grid.fingerprint(),
        # What the fields are, such as the window of window means: a resume
        # must not relabel a store filled for another window.
        "attrs": dict(store.attrs),
    }
    if overwrite and rank == 0:
        _remove_dir(path)
    progress = _progress(cfg, store.role, manifest, overwrite)
    my_items, n_remaining = _my_times(progress, store.times)
    leads = "" if store.lead_times is None else f" x {len(store.lead_times)} leads"
    logger.info(
        f"{store.role}: {len(my_items)}/{n_remaining} remaining times on this rank "
        f"({len(store.times)} total{leads}), variables {store.variables}"
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
            backend.add_array(_store_template(first, store.times))
            if store.attrs:
                zarr.open_group(path, mode="r+").attrs.update(store.attrs)
        return backend

    backend = rank0_first_checked(open_backend)
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
        "source": source_signature(store.source),
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
    # Every rank checks in after each store, so a slice that failed stops the
    # run on every rank before the next store's collectives, and rank 0 never
    # writes the sentinel over a store with missing times.
    for store in gridded:
        all_agree(partial(download_gridded_store, cfg, store, overwrite))
    for frame_store in frames:
        all_agree(partial(download_frame_store, cfg, frame_store, overwrite))

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
    # Ranks fetch different times and meet after every store, so a rank
    # whose archive is slow keeps the others waiting; a backend default
    # timeout (10 min for NCCL) would abort a healthy run.  torchrun still
    # stops every rank when one of them dies.
    device = init_distributed(float(cfg.predownload.get("timeout_s", 21600)))
    try:
        run(cfg, device)
    finally:
        finalize_distributed()
    logger.success("Pre-download finished.")


if __name__ == "__main__":
    main()
