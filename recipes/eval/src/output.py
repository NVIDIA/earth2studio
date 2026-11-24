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

"""Output stores: paths, IO backends and per-stream variable selection.

:class:`earth2studio.run.Pipeline` owns the store lifecycle; this module only
turns the ``output`` config block into backends and decides which of a runner's
streams carry the requested variables.  Every stream writes into the same
``forecast.zarr`` group, so prognostic and diagnostic variables sit side by
side.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.grids import (
    GridDefinition,
    LatLonGrid,
    Regridder,
    infer_grid,
    resolve_grid,
    resolve_regridder,
    same_geometry,
    select_subset,
)
from earth2studio.io import AsyncZarrBackend, IOBackend, ZarrBackend
from earth2studio.run import ProgressStore, WorkItem
from earth2studio.run.work import MANIFEST_NAME, broadcast
from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.type import CoordinateSystem

from .regions import spatial_axes

_ITERATION_DIMS = ("ensemble", "time", "lead_time")


def store_path(cfg: DictConfig, name: str = "forecast.zarr") -> str:
    """Path of a store under ``output.path``."""
    return os.path.join(cfg.output.path, name)


def sentinel_path(cfg: DictConfig) -> Path:
    """Path of the pre-download sentinel ``predownload.py`` writes."""
    return Path(cfg.output.path) / ".predownload.done"


def progress_path(cfg: DictConfig, name: str = "progress") -> Path:
    """Directory of a :class:`~earth2studio.run.ProgressStore`."""
    return Path(cfg.output.path) / f".{name}"


def retain_raw_output(cfg: DictConfig) -> bool:
    """Whether to write the raw forecast store (see the ``output`` config block)."""
    retain = str(cfg.output.get("retain", "all")).lower()
    if retain not in ("all", "none"):
        raise ValueError(f"output.retain must be 'all' or 'none', got {retain!r}")
    return retain == "all"


def parallel_coords(items: Sequence[WorkItem], lead_time: np.ndarray) -> CoordSystem:
    """Full labels of the dimensions written one slice at a time.

    Parameters
    ----------
    items : Sequence[WorkItem]
        Every item of the run.
    lead_time : np.ndarray
        Lead times of the stream schema.

    Returns
    -------
    CoordSystem
        ``ensemble`` (ensemble runs only), ``time`` and ``lead_time`` labels.
    """
    coords: CoordSystem = OrderedDict()
    members = sorted({int(m) for item in items for m in item.member_ids})
    if members:
        coords["ensemble"] = np.array(members, dtype=np.int64)
    coords["time"] = np.unique(
        np.array([item.time for item in items], dtype="datetime64[ns]")
    )
    coords["lead_time"] = np.asarray(lead_time).astype("timedelta64[ns]")
    return coords


def remove_store(path: str) -> None:
    """Delete a store, a directory or a stray file, if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
        logger.info(f"Removed existing store {path}")
    elif os.path.lexists(path):
        os.remove(path)
        logger.info(f"Removed stray file at store path {path}")


def progress_tags(cfg: DictConfig, online: bool) -> dict[str, Any]:
    """Tags every run records in its progress manifest.

    Online and offline runs share the progress store; the ``scoring`` tag
    makes a resume that switches mode fail instead of skipping finished
    items, which would leave the new statistics store without them.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config.
    online : bool
        Whether the run scores online.

    Returns
    -------
    dict[str, Any]
        ``project``, ``run_id`` and ``scoring`` (``online`` or ``offline``).
    """
    return {
        "project": str(cfg.project),
        "run_id": str(cfg.run_id),
        "scoring": "online" if online else "offline",
    }


def recorded_items(
    progress_dir: str | os.PathLike,
    items: Sequence[WorkItem],
    tags: Mapping[str, Any] | None = None,
) -> int:
    """Number of ``items`` a progress store records as done for this run.

    Parameters
    ----------
    progress_dir : str | os.PathLike
        Directory of the :class:`~earth2studio.run.ProgressStore`; a missing
        directory records nothing.
    items : Sequence[WorkItem]
        The run's items.
    tags : Mapping[str, Any] | None, optional
        The run's manifest tags, by default None.  Records under a manifest
        with other tags count as 0: they belong to another run, and the
        pipeline's own manifest check refuses that resume.

    Returns
    -------
    int
        Items whose every unit the store records.
    """
    if not os.path.isdir(progress_dir):
        return 0
    manifest_path = Path(progress_dir) / MANIFEST_NAME
    if tags is not None and manifest_path.is_file():
        try:
            stored = json.loads(manifest_path.read_text()).get("tags") or {}
        except json.JSONDecodeError:
            stored = {}
        normalized = json.loads(json.dumps(dict(tags), sort_keys=True, default=str))
        if stored != normalized:
            return 0
    store = ProgressStore(progress_dir)
    try:
        return len(items) - len(store.remaining(items))
    finally:
        store.close()


def prepare_store(
    path: str, resume: bool, overwrite: bool, rank: int = 0, recorded: int = 0
) -> None:
    """Apply the rerun policy to an existing store before any rank opens it.

    Rank 0 inspects the path and every rank learns the outcome, so a refusal
    raises on all ranks together, each with the same message, before
    :func:`src.logs.rank0_first_checked` opens the store.

    Parameters
    ----------
    path : str
        Store path.
    resume : bool
        Whether the run appends to the existing store.
    overwrite : bool
        Whether rank 0 removes the existing store.  ``resume`` wins.
    rank : int, optional
        This process's rank, by default 0.
    recorded : int, optional
        Items the progress store already records as done
        (:func:`recorded_items`), by default 0.  A resume whose store has
        gone missing while records remain would skip those items and leave
        them out of the new store, so this function refuses it.

    Raises
    ------
    FileExistsError
        If the store exists without ``resume`` or ``overwrite``.
    FileNotFoundError
        With ``resume``, a missing store and a nonzero ``recorded``.
    """
    exists = broadcast(os.path.lexists(path) if rank == 0 else None)
    if not exists:
        if resume and recorded:
            raise FileNotFoundError(
                f"{path} is missing, but the progress store records {recorded} "
                "finished item(s) that it held. A resume would skip them and "
                "leave them out of the new store. Clear the progress store or "
                "run with resume=false to rerun them, or restore the store."
            )
        return
    if resume:
        logger.info(f"Resuming into existing store: {path}")
        return
    if not overwrite:
        raise FileExistsError(
            f"{path} already exists. Set output.overwrite=true to replace it, "
            "resume=true to append, or choose another run_id."
        )
    if rank == 0:
        logger.warning(f"Overwriting existing store: {path}")
        remove_store(path)


def build_backend(
    cfg: DictConfig,
    path: str,
    parallel: CoordSystem,
    overwrite: bool,
    rank: int,
    chunks: Mapping[str, int] | None = None,
) -> IOBackend:
    """Build the configured IO backend for one store.

    Call on every rank inside :func:`src.logs.rank0_first_checked`, so
    rank 0 creates or clears the store before the others open it.  Apply
    :func:`prepare_store` first: it refuses an existing store that the run
    neither resumes nor overwrites.

    Parameters
    ----------
    cfg : DictConfig
        Config with the ``output`` block (``io_backend``, ``chunks``,
        ``thread_writers``, ``shard_coords``, ``max_inflight_shards``).
    path : str
        Store path.
    parallel : CoordSystem
        Labels of the iteration dimensions, see :func:`parallel_coords`.
    overwrite : bool
        Whether rank 0 clears an existing store first.
    rank : int
        This process's rank.
    chunks : Mapping[str, int], optional
        Chunk sizes overriding ``output.chunks``.

    Returns
    -------
    IOBackend
        A :class:`~earth2studio.io.ZarrBackend` or
        :class:`~earth2studio.io.AsyncZarrBackend`.
    """
    output = cfg.output
    backend = str(output.get("io_backend", "async_zarr"))
    if backend not in ("zarr", "async_zarr"):
        raise ValueError(
            f"output.io_backend must be 'zarr' or 'async_zarr', got {backend!r}"
        )
    chunk_sizes = {
        str(k): int(v)
        for k, v in dict(
            chunks if chunks is not None else output.get("chunks", {}) or {}
        ).items()
    }
    # Ranks write different time/ensemble positions, so those dimensions and
    # the per-step lead_time axis get chunk size 1 (see EXECUTION_CONTRACT X14).
    for dim in _ITERATION_DIMS:
        if dim in parallel:
            chunk_sizes.setdefault(dim, 1)
    threads = int(output.get("thread_writers", 0) or 0)
    shards = {
        str(k): int(v) for k, v in dict(output.get("shard_coords", {}) or {}).items()
    }
    if rank == 0 and overwrite:
        remove_store(path)
    if backend == "zarr":
        if threads:
            logger.warning(
                "output.thread_writers is ignored with io_backend=zarr; use "
                "io_backend=async_zarr for threaded writes."
            )
        if shards:
            raise ValueError(
                "output.shard_coords requires output.io_backend=async_zarr"
            )
        return ZarrBackend(
            path, chunks=chunk_sizes, backend_kwargs={"overwrite": False}
        )
    bad = sorted(
        d for d in _ITERATION_DIMS if d in parallel and chunk_sizes.get(d, 1) != 1
    )
    if bad:
        raise ValueError(
            f"output.chunks must use chunk size 1 for {bad} with "
            "io_backend=async_zarr; use output.shard_coords to cut file count."
        )
    chunked = {d: s for d, s in chunk_sizes.items() if d not in parallel}
    return AsyncZarrBackend(
        path,
        parallel_coords=parallel,
        blocking=threads == 0,
        pool_size=max(1, threads),
        chunked_coords=chunked,
        shard_coords=shards,
        max_inflight_shards=int(output.get("max_inflight_shards", 4)),
    )


def _same_axes(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> bool:
    """Whether two spatial axis mappings agree in names, order and values."""
    if list(a) != list(b):
        return False
    for name in a:
        left, right = np.asarray(a[name]), np.asarray(b[name])
        if left.shape != right.shape:
            return False
        try:
            if not np.allclose(left.astype(float), right.astype(float), atol=1e-6):
                return False
        except (TypeError, ValueError):
            if not np.array_equal(left, right):
                return False
    return True


def check_shared_layout(
    schemas: Mapping[str, CoordinateSystem],
    regridders: Mapping[str, Regridder],
    streams: Sequence[str],
) -> None:
    """Require every written stream to share one lead-time axis and one grid.

    The streams of a run write into one store whose ``lead_time`` axis and
    spatial coordinates come from the first stream, so a stream on another
    cadence or grid has no place next to it.  Grids compare by geometry
    where the library infers one, else by their spatial axes; a single
    stream needs no check.

    Parameters
    ----------
    schemas : Mapping[str, CoordinateSystem]
        Stream schemas from ``runner.output_coords``.
    regridders : Mapping[str, Regridder]
        Per-stream output regridders; a regridded stream lands on its
        regridder's ``target_grid``.
    streams : Sequence[str]
        Streams the run writes, first stream first.

    Raises
    ------
    ValueError
        If a stream's lead times or grid differ from the first stream's.
    """
    if len(streams) < 2:
        return
    first = streams[0]
    leads = np.asarray(schemas[first].coords["lead_time"].values)

    def geometry(stream: str) -> GridDefinition | None:
        regridder = regridders.get(stream)
        if regridder is not None:
            return regridder.target_grid
        try:
            return infer_grid(schemas[stream])
        except ValueError:
            return None  # an index grid, or a grid without identity

    grid = geometry(first)
    for stream in streams[1:]:
        other = np.asarray(schemas[stream].coords["lead_time"].values)
        if other.shape != leads.shape or not np.array_equal(other, leads):
            raise ValueError(
                f"Streams {first!r} and {stream!r} publish different lead times "
                f"({len(leads)} and {len(other)} steps), but every stream of a run "
                "writes into one store on one lead_time axis. Drop one of them from "
                "output.variables or give it its own run."
            )
        other_grid = geometry(stream)
        if grid is not None and other_grid is not None:
            same = same_geometry(grid, other_grid)
        elif first in regridders or stream in regridders:
            logger.info(
                f"Streams {first!r} and {stream!r}: one grid has no identity to "
                "compare; the store checks their coordinates when it adds the arrays."
            )
            continue
        else:
            same = _same_axes(
                spatial_axes(schemas[first]), spatial_axes(schemas[stream])
            )
        if not same:
            raise ValueError(
                f"Streams {first!r} and {stream!r} sit on different grids, but "
                "every stream of a run writes into one store on one grid. Set "
                "output.regrid so both land on the same grid, or drop one of them "
                "from output.variables."
            )


def stream_variables(
    schemas: Mapping[str, CoordinateSystem], variables: Sequence[str]
) -> dict[str, list[str]]:
    """Split the requested output variables over the streams carrying them.

    Parameters
    ----------
    schemas : Mapping[str, CoordinateSystem]
        Stream schemas from ``runner.output_coords``.
    variables : Sequence[str]
        Variables to store, in order.

    Returns
    -------
    dict[str, list[str]]
        Variables per stream, omitting streams that carry none.

    Raises
    ------
    ValueError
        If no stream produces a variable.
    """
    wanted = [str(v) for v in variables]
    if not wanted:
        raise ValueError("output.variables must name at least one variable")
    selected: dict[str, list[str]] = {}
    found: set[str] = set()
    for stream, schema in schemas.items():
        if "variable" not in schema.dims:
            continue
        names = set(np.asarray(schema["variable"].values).astype(str))
        mine = [v for v in wanted if v in names and v not in found]
        if mine:
            selected[stream] = mine
            found.update(mine)
    missing = [v for v in wanted if v not in found]
    if missing:
        available = sorted(
            {
                str(v)
                for schema in schemas.values()
                if "variable" in schema.dims
                for v in np.asarray(schema["variable"].values)
            }
        )
        raise ValueError(
            f"output.variables {missing} are produced by no stream; available: "
            f"{available}"
        )
    return selected


def _target_grid(spec: Any) -> str | LatLonGrid:
    """Resolve an ``output.regrid.grid`` entry: a registered name or lat/lon axes."""
    if isinstance(spec, str):
        return spec
    spec = dict(spec)
    axes = []
    for key in ("lat", "lon"):
        axis = spec.get(key)
        if axis is None:
            raise ValueError("output.regrid.grid needs 'lat' and 'lon' or a grid name")
        if isinstance(axis, Mapping):
            axes.append(
                np.linspace(float(axis["start"]), float(axis["stop"]), int(axis["num"]))
            )
        else:
            axes.append(np.asarray(axis, dtype=float))
    return LatLonGrid(axes[0], axes[1])


@dataclass(frozen=True)
class RegridSpec:
    """The ``output.regrid`` block: the grid every stream and the verification land on.

    Attributes
    ----------
    target : str | LatLonGrid
        Registered grid name, or explicit latitude and longitude axes.
    method : str
        Registered regridder method, ``nearest`` by default.
    options : dict[str, Any]
        Keyword arguments of the regridder, such as ``fill_value``.
    """

    target: str | LatLonGrid
    method: str = "nearest"
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def target_grid(self) -> GridDefinition:
        """The resolved target grid."""
        if isinstance(self.target, str):
            return resolve_grid(self.target)
        return self.target

    def regridder(self, source_grid: GridDefinition) -> Regridder | None:
        """Regridder from ``source_grid`` onto the target, or None when already there."""
        if same_geometry(source_grid, self.target_grid):
            return None
        return resolve_regridder(source_grid, self.target, self.method, **self.options)


def regrid_spec(cfg: DictConfig) -> RegridSpec | None:
    """Parse ``output.regrid``; ``null`` keeps every stream on its native grid.

    The block names a ``grid`` (a registered name such as ``latlon-0.25deg``, or
    ``{lat: {start, stop, num}, lon: {...}}``), an optional ``method`` (default
    ``nearest``) and the regridder's keyword arguments, under ``options`` or as
    further keys of the block.

    Parameters
    ----------
    cfg : DictConfig
        Config with the ``output`` block.

    Returns
    -------
    RegridSpec | None
        The parsed block, or None when regridding is off.
    """
    spec = cfg.output.get("regrid")
    if not spec:
        return None
    block = dict(OmegaConf.to_container(spec, resolve=True))  # type: ignore[arg-type]
    if "grid" not in block:
        raise ValueError("output.regrid needs a 'grid' entry")
    target = _target_grid(block.pop("grid"))
    method = str(block.pop("method", None) or "nearest")
    options = dict(block.pop("options", None) or {})
    options.update(block)
    return RegridSpec(target, method, options)


class OutputRegridder:
    """Bring fetched fields onto the ``output.regrid`` grid with its method.

    A fetched field reveals its grid only once it arrives, so the first call
    builds the regridder and later calls reuse it while the source grid stays
    the same.  Fields already on the target pass through, and exact crops
    select the subset, as :func:`earth2studio.data.fetch_data` does for a
    ``target_grid``.

    Parameters
    ----------
    spec : RegridSpec
        The parsed ``output.regrid`` block.
    """

    def __init__(self, spec: RegridSpec) -> None:
        self.spec = spec
        self.target_grid: GridDefinition = spec.target_grid
        self._source_grid: GridDefinition | None = None
        self._regridder: Regridder | None = None

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Return ``x`` on the target grid."""
        source_grid = infer_grid(x)
        if same_geometry(source_grid, self.target_grid):
            return x
        subset = select_subset(x, source_grid, self.target_grid)
        if subset is not None:
            return subset
        if (
            self._regridder is None
            or self._source_grid is None
            or not same_geometry(self._source_grid, source_grid)
        ):
            self._regridder = resolve_regridder(
                source_grid, self.spec.target, self.spec.method, **self.spec.options
            )
            self._source_grid = source_grid
            logger.info(
                f"Regridding fetched {source_grid.topology} fields "
                f"{dict(zip(source_grid.dims, source_grid.shape))} -> "
                f"{dict(zip(self.target_grid.dims, self.target_grid.shape))} "
                f"({self.spec.method})"
            )
        return self._regridder(x)


def output_regridder(cfg: DictConfig) -> OutputRegridder | None:
    """An :class:`OutputRegridder` for ``output.regrid``, or None without the block."""
    spec = regrid_spec(cfg)
    return None if spec is None else OutputRegridder(spec)


def verification_regridder(
    cfg: DictConfig, spatial: xr.DataArray | Mapping[str, Any]
) -> OutputRegridder | None:
    """The regridder bringing verification onto the scored grid.

    ``output.regrid`` when the campaign names one.  Otherwise the forecast
    grid itself: verification fetched on a finer latitude/longitude grid
    lands on the forecast grid by subset selection (a 1.5 degree model under
    0.25 degree reanalysis) or by nearest neighbour, as the legacy runner
    mapped source fields onto the model grid.

    Parameters
    ----------
    cfg : DictConfig
        Config with the ``output`` block.
    spatial : xr.DataArray | Mapping[str, Any]
        The forecast stream's signature, or its spatial axes by name.

    Returns
    -------
    OutputRegridder | None
        The regridder, or None for a forecast grid without latitude/longitude
        axes (projected nowcasts), whose sources already deliver the model grid.
    """
    regridder = output_regridder(cfg)
    if regridder is not None:
        return regridder
    try:
        grid = (
            infer_grid(spatial)
            if isinstance(spatial, xr.DataArray)
            else LatLonGrid(np.asarray(spatial["lat"]), np.asarray(spatial["lon"]))
        )
    except Exception:  # noqa: BLE001 - no latitude/longitude grid to align with
        return None
    if not isinstance(grid, LatLonGrid):
        return None
    return OutputRegridder(RegridSpec(grid, "nearest"))


def build_regridders(
    cfg: DictConfig, schemas: Mapping[str, CoordinateSystem]
) -> dict[str, Regridder]:
    """Build the per-stream output regridders from ``output.regrid``.

    Streams already on the target grid get no regridder; see
    :func:`regrid_spec` for the block's layout.

    Parameters
    ----------
    cfg : DictConfig
        Config with the ``output`` block.
    schemas : Mapping[str, CoordinateSystem]
        Stream schemas from ``runner.output_coords``.

    Returns
    -------
    dict[str, Regridder]
        Regridder per stream needing one.
    """
    spec = regrid_spec(cfg)
    if spec is None:
        return {}
    target_grid = spec.target_grid
    regridders: dict[str, Regridder] = {}
    for stream, schema in schemas.items():
        source_grid = infer_grid(schema)
        regridder = spec.regridder(source_grid)
        if regridder is None:
            continue
        regridders[stream] = regridder
        logger.info(
            f"Stream '{stream}': regridding {source_grid.topology} "
            f"{dict(zip(source_grid.dims, source_grid.shape))} -> "
            f"{dict(zip(target_grid.dims, target_grid.shape))} ({spec.method})"
        )
    return regridders
