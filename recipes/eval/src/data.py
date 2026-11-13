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

"""Local data stores and source resolution.

The library handles regridding, composition and temporal statistics
(:class:`~earth2studio.data.RegriddedSource`,
:class:`~earth2studio.data.CompositeSource`, qualified variable labels such as
``t2m:mean:-1h:23h``); this module holds what is specific to the recipe:
the zarr and parquet stores ``predownload.py`` writes, and the rule deciding
whether a runner slot reads a bring-your-own source, a predownloaded store or
the live source.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from inspect import signature
from types import SimpleNamespace
from typing import Any

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from omegaconf import DictConfig

from earth2studio.utils.coords import E2S_STATISTICS
from earth2studio.utils.time_statistics import time_statistic_metadata
from earth2studio.utils.type import LeadTimeArray, TimeArray, VariableArray


@dataclass(frozen=True)
class FrameRequest:
    """One observation-frame fetch a runner makes, for predownload.

    The tabular counterpart of :class:`earth2studio.run.runner.DataRequest`.
    Runners binding DataFrame sources expose ``frame_requests(item)`` returning
    a tuple of these; ``predownload.py`` caches them under
    ``<output.path>/<name>.parquet``.

    Attributes
    ----------
    name : str
        Slot name, also the frame store name.
    source : Any
        DataFrameSource to fetch from.
    time : np.ndarray
        Analysis times to fetch.
    variable : np.ndarray
        Variables requested from the source.
    fields : np.ndarray | None
        Columns requested, or None for all.
    """

    name: str
    source: Any
    time: np.ndarray
    variable: np.ndarray
    fields: np.ndarray | None = None


def store_name(role: str) -> str:
    """Zarr store file name of a predownload role (``data`` -> ``data.zarr``)."""
    return f"{role}.zarr"


def instantiate_source(node: Any) -> Any:
    """Instantiate a Hydra source node, passing through live source objects."""
    if node is None:
        return None
    if isinstance(node, (DictConfig, dict)) and "_target_" in node:
        return hydra.utils.instantiate(node)
    return node


def resolve_source(
    cfg: DictConfig,
    role: str,
    byo: Any = None,
    live: Any = None,
    prefer_predownloaded: bool = True,
) -> Any:
    """Pick the source feeding one runner slot.

    Resolution order: a bring-your-own node, then the predownloaded store
    ``<output.path>/<role>.zarr`` when present and allowed, then the live node.
    ``None`` leaves the slot to the model's own ``default_sources()``.

    Parameters
    ----------
    cfg : DictConfig
        Config; this reads only ``output.path``.
    role : str
        Slot role naming the predownloaded store (``data``, ``forcing0``, ...).
    byo : Any, optional
        User-provided source node or object, by default None
    live : Any, optional
        Live source node or object, by default None
    prefer_predownloaded : bool, optional
        Whether an existing predownloaded store takes precedence over ``live``,
        by default True.  ``predownload.py`` passes False.

    Returns
    -------
    Any
        A data source, or ``None`` for the model default.
    """
    if byo is not None:
        logger.info(f"Slot '{role}': using the bring-your-own source")
        return instantiate_source(byo)
    path = os.path.join(cfg.output.path, store_name(role))
    if prefer_predownloaded and os.path.isdir(path):
        logger.info(f"Slot '{role}': reading predownloaded store {path}")
        return open_predownloaded(path)
    if live is None:
        logger.info(f"Slot '{role}': using the model's recommended source")
        return None
    logger.info(f"Slot '{role}': using the live source")
    return instantiate_source(live)


def is_forecast_source(source: Any) -> bool:
    """Whether ``source`` takes ``(time, lead_time, variable)`` requests.

    Mirrors the dispatch of :func:`earth2studio.data.fetch_data`: a boolean
    ``is_forecast`` attribute decides, else the call signature.
    """
    declared = getattr(source, "is_forecast", None)
    if isinstance(declared, bool):
        return declared
    return "lead_time" in signature(source.__call__).parameters


def open_predownloaded(
    store_path: str,
) -> PredownloadedSource | PredownloadedForecastSource:
    """Open a predownloaded zarr store as the kind of source it caches.

    A store with a ``lead_time`` dimension caches a
    :class:`~earth2studio.data.ForecastSource`; any other caches a
    :class:`~earth2studio.data.DataSource`.

    Parameters
    ----------
    store_path : str
        Path to the zarr store.

    Returns
    -------
    PredownloadedSource | PredownloadedForecastSource
        The source serving the store.
    """
    if not os.path.isdir(store_path):
        raise FileNotFoundError(f"No predownloaded store at '{store_path}'")
    with xr.open_zarr(store_path, consolidated=False) as ds:
        forecast = "lead_time" in ds.dims
    if forecast:
        return PredownloadedForecastSource(store_path)
    return PredownloadedSource(store_path)


class _PredownloadedStore:
    """Fields of one predownloaded zarr store, served by exact label lookups.

    The store holds one array per variable label on ``(time, <spatial...>)``,
    or on ``(time, lead_time, <spatial...>)`` for a forecast source, with its
    grid coordinates, as ``predownload.py`` writes it through the library's
    :class:`~earth2studio.io.ZarrBackend`.  A time, lead time or variable that
    was not predownloaded raises ``KeyError``.

    The store holds qualified labels such as ``t2m:mean:-1h:23h`` already
    reduced and serves them natively: its ``lexicon.TIME_STATISTICS`` declares the
    methods of the stored labels, :func:`earth2studio.data.fetch_data` passes
    those labels through unchanged, and the returned field carries their
    statistics metadata.

    Parameters
    ----------
    store_path : str
        Path to the zarr store.
    """

    is_forecast: bool = False

    def __init__(self, store_path: str) -> None:
        if not os.path.isdir(store_path):
            raise FileNotFoundError(f"No predownloaded store at '{store_path}'")
        self.store_path = store_path
        ds = xr.open_zarr(store_path, consolidated=False)
        if ("lead_time" in ds.dims) != self.is_forecast:
            kind = "forecast-source" if "lead_time" in ds.dims else "data-source"
            raise ValueError(
                f"'{store_path}' is a {kind} store; open it with open_predownloaded()"
            )
        leading = ("time", "lead_time") if self.is_forecast else ("time",)
        self._da = ds.to_array("variable").transpose(*leading, "variable", ...)
        self._da.attrs.update(ds.attrs)
        self._labels = [str(v) for v in self._da.coords["variable"].values]
        self._statistics: dict[str, Any] = {}
        for label in self._labels:
            _, separator, modifier = label.partition(":")
            if separator:
                self._statistics[label] = time_statistic_metadata(modifier)
        methods = sorted({str(meta["method"]) for meta in self._statistics.values()})
        self.lexicon = SimpleNamespace(TIME_STATISTICS=tuple(methods))
        axis = self._da.coords["lead_time" if self.is_forecast else "time"].values
        steps = np.unique(np.diff(axis)) if axis.size > 1 else np.array([])
        self.time_step: np.timedelta64 | None = (
            np.timedelta64(steps[0]) if steps.size == 1 else None
        )

    @property
    def variables(self) -> list[str]:
        """Stored variable labels."""
        return list(self._labels)

    @property
    def times(self) -> np.ndarray:
        """Stored times: valid times, or initial times for a forecast store."""
        return self._da.coords["time"].values

    def _select(
        self, names: Sequence[str], indexers: dict[str, np.ndarray]
    ) -> xr.DataArray:
        """Stored fields for ``names`` at the requested ``indexers``."""
        missing = sorted(set(names) - set(self._labels))
        if missing:
            raise KeyError(
                f"Predownloaded store '{self.store_path}' lacks variables "
                f"{missing}; re-run predownload.py with them in the campaign."
            )
        try:
            da = self._da.sel(variable=list(names), **indexers)
        except KeyError as error:
            wanted = {k: [str(v) for v in vals] for k, vals in indexers.items()}
            axes = "times or lead times" if self.is_forecast else "times"
            raise KeyError(
                f"Predownloaded store '{self.store_path}' lacks some of the "
                f"{axes} {wanted}; re-run predownload.py with the campaign's "
                "initial conditions and lead times."
            ) from error
        da.attrs = dict(self._da.attrs)
        da.attrs.pop(E2S_STATISTICS, None)
        statistics = {n: self._statistics[n] for n in names if n in self._statistics}
        if statistics:
            da.attrs[E2S_STATISTICS] = statistics
        return da.load()


WINDOW_MEANS_ATTR = "window_means"
"""Root attribute ``predownload.py`` writes on a verification store of window means."""


def store_attrs(store_path: str) -> dict[str, Any]:
    """Root attributes of a predownloaded zarr store, or an empty dict without one.

    Parameters
    ----------
    store_path : str
        Path to the zarr store.

    Returns
    -------
    dict[str, Any]
        The store's root attributes.
    """
    if not os.path.isdir(store_path):
        return {}
    with xr.open_zarr(store_path, consolidated=False) as ds:
        return dict(ds.attrs)


class PredownloadedSource(_PredownloadedStore):
    """Data source backed by a predownloaded store on ``(time, <spatial...>)``.

    See :class:`_PredownloadedStore` for the store layout and the native
    time statistics.

    Parameters
    ----------
    store_path : str
        Path to the zarr store.
    """

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return stored fields for the requested times and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Valid times to return.
        variable : str | list[str] | VariableArray
            Variables to return.

        Returns
        -------
        xr.DataArray
            Field with dimensions ``[time, variable, <spatial...>]``.
        """
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        return self._select(variables, {"time": times})

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async version of :meth:`__call__`."""
        return self(time, variable)


class PredownloadedForecastSource(_PredownloadedStore):
    """Forecast source backed by a store on ``(time, lead_time, <spatial...>)``.

    ``predownload.py`` writes it for runner slots bound to a
    :class:`~earth2studio.data.ForecastSource`, caching every requested lead
    time of every initial time.

    Parameters
    ----------
    store_path : str
        Path to the zarr store.
    """

    is_forecast = True

    @property
    def lead_times(self) -> np.ndarray:
        """Stored forecast lead times."""
        return self._da.coords["lead_time"].values

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return stored forecasts for the requested times, leads and variables.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Initial times to return.
        lead_time : timedelta | list[timedelta] | LeadTimeArray
            Forecast lead times to return.
        variable : str | list[str] | VariableArray
            Variables to return.

        Returns
        -------
        xr.DataArray
            Field with dimensions ``[time, lead_time, variable, <spatial...>]``.
        """
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        leads = np.atleast_1d(np.asarray(lead_time, dtype="timedelta64[ns]"))
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        return self._select(variables, {"time": times, "lead_time": leads})

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async version of :meth:`__call__`."""
        return self(time, lead_time, variable)


def frame_store_path(output_path: str, name: str) -> str:
    """Directory of a predownloaded DataFrame store: ``<output>/<name>.parquet``."""
    return os.path.join(output_path, f"{name}.parquet")


def frame_filename(time: np.datetime64) -> str:
    """Per-time parquet file name inside a frame store (``20240101T000000.parquet``)."""
    ts = str(np.datetime64(time, "s")).replace("-", "").replace(":", "")
    return f"{ts}.parquet"


class PredownloadedFrameSource:
    """DataFrame source backed by a predownloaded parquet directory.

    Serves the observation frames ``predownload.py`` cached, one parquet file
    per analysis time, through the ``DataFrameSource`` protocol so
    :func:`earth2studio.data.fetch_dataframe` works on it unchanged.  Each
    stored frame already holds the full observation window of its analysis
    time, so lookups are exact file matches.

    Parameters
    ----------
    store_path : str
        Path to the ``<name>.parquet`` directory.
    """

    def __init__(self, store_path: str) -> None:
        if not os.path.isdir(store_path):
            raise FileNotFoundError(f"No predownloaded frame store at '{store_path}'")
        self.store_path = store_path

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: Any = None,
    ) -> pd.DataFrame:
        """Return the stored observation frames for the requested times.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Analysis times; each must exist in the predownloaded store.
        variable : str | list[str] | VariableArray
            Variables to keep when the frame carries a ``variable`` column.
        fields : array-like, optional
            Columns to return, by default every stored column.

        Returns
        -------
        pd.DataFrame
            Concatenation of the per-time frames in request order.
        """
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        pieces: list[pd.DataFrame] = []
        for t in times:
            path = os.path.join(self.store_path, frame_filename(t))
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Predownloaded observation frame for {t} not found at "
                    f"'{path}'; re-run predownload.py with this time included."
                )
            pieces.append(pd.read_parquet(path))
        df = pieces[0] if len(pieces) == 1 else pd.concat(pieces, ignore_index=True)
        if variable is not None and "variable" in df.columns:
            requested = [str(v) for v in np.atleast_1d(np.asarray(variable))]
            df = df[df["variable"].isin(requested)].reset_index(drop=True)
        if fields is not None:
            wanted = [str(f) for f in np.atleast_1d(np.asarray(fields))]
            missing = [f for f in wanted if f not in df.columns]
            if missing:
                raise KeyError(
                    f"Frame store '{self.store_path}' lacks fields {missing} "
                    f"(stored: {list(df.columns)}); re-run predownload.py with "
                    "predownload.overwrite=true."
                )
            df = df[wanted]
        return df
