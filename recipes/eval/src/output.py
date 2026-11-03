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
``forecast.zarr`` group, as the recipe has always stored prognostic and
diagnostic variables side by side.
"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.grids import (
    LatLonGrid,
    Regridder,
    infer_grid,
    resolve_grid,
    resolve_regridder,
    same_geometry,
)
from earth2studio.io import AsyncZarrBackend, IOBackend, ZarrBackend
from earth2studio.run import WorkItem
from earth2studio.utils.coords import CoordSystem
from earth2studio.utils.type import CoordinateSystem

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
    """Delete a store directory if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
        logger.info(f"Removed existing store {path}")


def build_backend(
    cfg: DictConfig,
    path: str,
    parallel: CoordSystem,
    overwrite: bool,
    rank: int,
    chunks: Mapping[str, int] | None = None,
) -> IOBackend:
    """Build the configured IO backend for one store.

    Call on every rank inside :func:`earth2studio.run.work.rank0_first`, so
    rank 0 creates or clears the store before the others open it.

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


def build_regridders(
    cfg: DictConfig, schemas: Mapping[str, CoordinateSystem]
) -> dict[str, Regridder]:
    """Build the per-stream output regridders from ``output.regrid``.

    ``output.regrid`` is ``null`` (store every stream on its native grid) or a
    mapping with ``grid`` (a registered grid name such as ``latlon-0.25deg``,
    or ``{lat: {start, stop, num}, lon: {...}}``), an optional ``method``
    (default ``nearest``) and further regridder options such as
    ``fill_value`` or ``max_dist_km``.  Streams already on the target grid
    get no regridder.

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
    spec = cfg.output.get("regrid")
    if not spec:
        return {}
    options = dict(OmegaConf.to_container(spec, resolve=True))  # type: ignore[arg-type]
    target = _target_grid(options.pop("grid"))
    method = str(options.pop("method", "nearest"))
    target_grid = resolve_grid(target) if isinstance(target, str) else target
    regridders: dict[str, Regridder] = {}
    for stream, schema in schemas.items():
        source_grid = infer_grid(schema)
        if same_geometry(source_grid, target_grid):
            continue
        regridders[stream] = resolve_regridder(source_grid, target, method, **options)
        logger.info(
            f"Stream '{stream}': regridding {source_grid.topology} "
            f"{dict(zip(source_grid.dims, source_grid.shape))} -> "
            f"{dict(zip(target_grid.dims, target_grid.shape))} ({method})"
        )
    return regridders
