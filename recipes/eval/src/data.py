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

"""Data source wrappers for predownloaded zarr stores."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import DictConfig

from earth2studio.data import DataSource
from earth2studio.utils.type import TimeArray, VariableArray


def resolve_ic_source(
    cfg: DictConfig,
    *,
    store_name: str = "data.zarr",
    byo: Any = None,
    live_source: Any = None,
) -> DataSource:
    """Resolve the initial-condition data source for a pipeline.

    Resolution order:

    1. *byo* — explicit user-provided (BYO) override config node.  When
       provided, Hydra-instantiated and returned directly.
    2. ``<cfg.output.path>/<store_name>`` — predownloaded zarr cache.
       When present, wrapped in :class:`PredownloadedSource`.
    3. *live_source* — live source config node.  Hydra-instantiated
       fresh.  Required if neither *byo* nor a cache is available.

    Used by :mod:`main` (single-source: ``byo=cfg.get("ic_source")``,
    ``live_source=cfg.data_source``) and by
    :class:`~src.pipelines.stormscope.StormScopePipeline` (multi-source:
    one call per model with ``store_name="data_<side>.zarr"`` and
    ``live_source=cfg.model.<side>.ic_source``).

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config — only ``cfg.output.path`` is consulted.
    store_name : str
        Filename of the cached zarr under ``cfg.output.path``.  Defaults
        to ``"data.zarr"``.
    byo : Any, optional
        Explicit BYO override config (DictConfig node).  ``None`` means
        no BYO override.
    live_source : Any, optional
        Live source config (DictConfig node) to fall back to.  Required
        when there is no BYO override and no cache.

    Returns
    -------
    DataSource

    Raises
    ------
    ValueError
        If no cache is present, no BYO override is provided, and
        *live_source* is ``None``.
    """
    if byo is not None:
        logger.info("Using user-provided ic source (BYO).")
        return hydra.utils.instantiate(byo)

    cache_path = os.path.join(cfg.output.path, store_name)
    if os.path.exists(cache_path):
        logger.info(f"Using predownloaded data store: {cache_path}")
        return PredownloadedSource(cache_path)

    if live_source is None:
        raise ValueError(
            f"resolve_ic_source: no cache at '{cache_path}', no BYO override, "
            "and no live_source was provided."
        )
    logger.info(f"No cache at '{cache_path}' — instantiating live source directly.")
    return hydra.utils.instantiate(live_source)


class PredownloadedSource:
    """DataSource backed by a predownloaded zarr store.

    The zarr store is expected to have dimensions
    ``(time, variable, <spatial...>)`` with no ``lead_time`` dimension.
    Because ``lead_time`` is absent from :meth:`__call__`, ``fetch_data``
    automatically handles lead-time expansion by calling
    ``source(ic_time + lead, variable)`` for each requested lead time.

    Parameters
    ----------
    store_path : str
        Path to the zarr store on disk.
    """

    def __init__(self, store_path: str) -> None:
        # to_array puts "variable" first; transpose so time leads per DataSource protocol.
        self._da = (
            xr.open_zarr(store_path)
            .to_array("variable")
            .transpose("time", "variable", ...)
        )

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return data for the requested times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Variable names to return.

        Returns
        -------
        xr.DataArray
            Data with dimensions ``[time, variable, <spatial...>]``.
        """
        if not isinstance(time, (list, np.ndarray)):
            time = [time]
        if not isinstance(variable, (list, np.ndarray)):
            variable = [variable]
        return self._da.sel(time=time, variable=variable)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async version of :meth:`__call__`."""
        return self(time, variable)


class CompositeSource:
    """DataSource composed of multiple sources dispatched by variable.

    Each request for one or more variables is split into per-source
    sub-requests according to which component source advertises that
    variable, then the sub-results are concatenated along the
    ``variable`` dim.  Designed for multi-source pipelines such as
    StormScope where different variables live in different predownloaded
    zarrs (``data_goes.zarr`` holds ABI channels; ``data_mrms.zarr``
    holds ``refc``).

    All component sources must produce DataArrays with the same
    non-variable dimensions (time, lead_time if any, spatial dims) and
    the same spatial coordinate values — otherwise concatenation will
    raise.  The component sources' variables must be mutually disjoint.

    Parameters
    ----------
    sources : dict[str, DataSource]
        Mapping from a descriptive name (for error messages) to a
        ``DataSource`` instance.  Iteration order is preserved; when the
        same variable is advertised by more than one source the first
        one wins.
    variable_index : dict[str, str]
        Mapping from variable name to the component source name that
        should handle it.  Typically built by inspecting each zarr's
        array list — see :meth:`from_predownloaded_stores`.
    """

    def __init__(
        self,
        sources: Mapping[str, DataSource],
        variable_index: dict[str, str],
    ) -> None:
        unknown = sorted(set(variable_index.values()) - set(sources))
        if unknown:
            raise ValueError(f"variable_index references unknown sources: {unknown}")
        self._sources = dict(sources)
        self._var_index = dict(variable_index)

    @classmethod
    def from_predownloaded_stores(cls, stores: dict[str, str]) -> CompositeSource:
        """Build a :class:`CompositeSource` by inspecting each zarr's vars.

        Parameters
        ----------
        stores : dict[str, str]
            Mapping from source name → zarr store path.  Each store is
            opened once; its top-level array names become the variable
            index entries that dispatch to that source.

        Returns
        -------
        CompositeSource
        """
        sources: dict[str, PredownloadedSource] = {}
        variable_index: dict[str, str] = {}
        for name, path in stores.items():
            src = PredownloadedSource(path)
            sources[name] = src
            # PredownloadedSource._da is the result of to_array("variable");
            # its "variable" coord enumerates the store's data vars.
            for v in src._da.coords["variable"].values:
                v_str = str(v)
                if v_str not in variable_index:
                    variable_index[v_str] = name
        return cls(sources, variable_index)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        if not isinstance(variable, (list, np.ndarray)):
            variables: list[str] = [str(variable)]
        else:
            variables = [str(v) for v in variable]

        missing = [v for v in variables if v not in self._var_index]
        if missing:
            known = sorted(self._var_index)
            raise KeyError(
                f"CompositeSource: variables {missing} not found in any "
                f"component source (known: {known})"
            )

        # Preserve caller-requested variable order; group by source while
        # remembering each variable's target index in the output.
        per_source: dict[str, list[str]] = {}
        for v in variables:
            per_source.setdefault(self._var_index[v], []).append(v)

        partial_das: list[xr.DataArray] = []
        for src_name, vars_for_src in per_source.items():
            partial_das.append(self._sources[src_name](time, vars_for_src))

        # Concatenate along the variable dim, then reorder to match the
        # caller's requested variable order.
        combined = xr.concat(partial_das, dim="variable")
        return combined.sel(variable=variables)

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)


class CadenceRoundedSource:
    """DataSource wrapper that rounds valid-time requests to a fixed cadence.

    Used when a caller (e.g. a 10-min StormScope model) queries the
    underlying store at a finer cadence than its native resolution
    (e.g. hourly GFS or ERA5).  Each valid time in the request is
    rounded to the nearest cadence boundary before being forwarded;
    the returned ``xr.DataArray`` is *relabeled* with the caller's
    original (pre-round) times so downstream consumers see the
    requested valid times rather than the coarser underlying ones.

    Deduplication is automatic — if the request produces duplicate
    rounded times, the underlying source is only called with unique
    values, and the result is re-indexed on the way out.

    Parameters
    ----------
    source : DataSource
        Wrapped source.  Must speak the ``(time, variable)`` protocol.
    cadence : str | timedelta | np.timedelta64
        Native resolution of the underlying data.  Parsed via
        ``pd.Timedelta`` so strings like ``"1h"``, ``"30min"`` work.
    """

    def __init__(
        self,
        source: Any,
        cadence: Any,
    ) -> None:
        self._source = source
        self._cadence_ns: np.timedelta64 = (
            pd.Timedelta(cadence).to_timedelta64().astype("timedelta64[ns]")
        )
        if int(self._cadence_ns.astype("int64")) <= 0:
            raise ValueError(f"cadence must be positive, got {cadence}")

    def _round_nearest(self, times: np.ndarray) -> np.ndarray:
        """Round each datetime64 to the nearest multiple of the cadence."""
        t_int = np.asarray(times, dtype="datetime64[ns]").astype("int64")
        c = int(self._cadence_ns.astype("int64"))
        # (x + c/2) // c * c — nearest-multiple rounding for non-negative offsets;
        # also correct for the epoch-relative timestamps we deal with here.
        rounded = ((t_int + c // 2) // c) * c
        return rounded.astype("datetime64[ns]")

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        if not isinstance(time, (list, np.ndarray)):
            times = np.array([time], dtype="datetime64[ns]")
        else:
            times = np.asarray(time, dtype="datetime64[ns]")

        rounded = self._round_nearest(times)

        # Dedupe to unique rounded values, preserving insertion order.
        unique_seen: dict[int, np.datetime64] = {}
        for r in rounded:
            unique_seen.setdefault(int(r.astype("int64")), r)
        unique_rounded = np.array(list(unique_seen.values()), dtype="datetime64[ns]")

        da = self._source(unique_rounded, variable)

        # Expand back to the caller's requested-time count/order.
        idx_map = {int(t.astype("int64")): i for i, t in enumerate(unique_rounded)}
        indices = np.array(
            [idx_map[int(r.astype("int64"))] for r in rounded],
            dtype=np.int64,
        )
        out = da.isel(time=indices)
        # Relabel the time axis with the original requested values so
        # downstream selection / dim handling sees what was asked for.
        out = out.assign_coords(time=times)
        return out

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)


class ValidTimeForecastAdapter:
    """Adapt a :class:`ForecastSource` to the :class:`DataSource` protocol.

    Wraps a forecast source (``__call__(time, lead_time, variable)``) so
    that callers can request data by *valid time* alone, matching the
    ``DataSource`` contract.  Each valid time is looked up in
    ``lookup`` to resolve which ``(init_time, lead_time)`` pair to
    query — the forecast source is called once per valid time and the
    returned DataArray's ``(time, lead_time)`` axes are squeezed and
    relabeled with the requested valid time.

    Used at predownload to flatten forecast fetches into a
    valid-time-keyed zarr that :class:`PredownloadedSource` can serve
    at inference time in place of a live forecast source (e.g.
    substituting a local zarr for ``GFS_FX`` in StormScope's
    ``conditioning_data_source``).

    Parameters
    ----------
    source : ForecastSource
        The wrapped forecast source.  Only its ``__call__`` is used.
    lookup : dict[np.datetime64, tuple[np.datetime64, np.timedelta64]]
        Mapping ``valid_time → (init_time, lead_time)``.  Every valid
        time the adapter will be asked for must appear here; missing
        keys raise ``KeyError``.

    Notes
    -----
    This adapter is stateless beyond ``lookup`` and not threadsafe for
    concurrent mutation.  It does not attempt to dedupe or batch fetches
    across valid times — the predownload loop already iterates one
    valid time at a time, so per-call fan-out is fine.
    """

    def __init__(
        self,
        source: Any,
        lookup: dict[Any, tuple[Any, Any]],
    ) -> None:
        self._source = source
        # Normalize all lookup entries to ns-unit numpy scalars so the
        # downstream membership check is robust to mixed-unit inputs
        # (pd.Timestamp, np.datetime64[s], datetime, ...).
        self._lookup: dict[np.datetime64, tuple[np.datetime64, np.timedelta64]] = {
            np.datetime64(pd.Timestamp(k), "ns"): (
                np.datetime64(pd.Timestamp(v[0]), "ns"),
                np.timedelta64(pd.Timedelta(v[1]), "ns"),
            )
            for k, v in lookup.items()
        }

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        if not isinstance(time, (list, np.ndarray)):
            times_iter: list = [time]
        else:
            times_iter = list(time)
        if not isinstance(variable, (list, np.ndarray)):
            variables = [variable]
        else:
            variables = list(variable)

        pieces: list[xr.DataArray] = []
        for t in times_iter:
            key = np.datetime64(pd.Timestamp(t), "ns")
            if key not in self._lookup:
                known = sorted(self._lookup)
                first = known[0] if known else "<empty>"
                last = known[-1] if known else "<empty>"
                raise KeyError(
                    f"ValidTimeForecastAdapter: valid_time {t} not in lookup "
                    f"(known range: {first} → {last}, N={len(known)})."
                )
            init, lead = self._lookup[key]
            init_dt = pd.Timestamp(init).to_pydatetime()
            da = self._source([init_dt], np.array([lead]), variables)
            # Collapse the (now-constant) (time, lead_time) pair and
            # relabel with the caller-supplied valid time.
            da = da.isel(time=0, lead_time=0, drop=True)
            da = da.expand_dims(time=[np.datetime64(pd.Timestamp(t), "ns")])
            pieces.append(da)

        if len(pieces) == 1:
            return pieces[0]
        return xr.concat(pieces, dim="time")

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        return self(time, variable)
