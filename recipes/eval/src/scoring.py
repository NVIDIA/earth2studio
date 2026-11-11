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

"""Offline scoring: compare stored forecasts against verification data.

Opens the ``forecast.zarr`` store ``main.py`` wrote, fetches verification at
the valid times of every forecast through :func:`earth2studio.data.fetch_data`,
applies the configured ``earth2studio.statistics`` metrics on
:class:`xarray.DataArray` fields and writes the results to a shared
``scores.zarr`` store through :class:`earth2studio.io.ZarrBackend`.

Designed for multi-GPU use: each rank scores an independent subset of
initial-condition times, and a :class:`earth2studio.run.ProgressStore` records
finished times so a run can resume.  Lead-time chunking keeps memory bounded.

Score store schema
------------------
One array per metric and variable, named ``<metric>__<variable>`` (or just
``<metric>`` when the metric reduces the ``variable`` dimension), over
``(time, <metric output dimensions>)`` -- typically ``(time, lead_time)``,
``(time, region, lead_time)`` with ``scoring.regions`` or
``(time, ensemble, lead_time)`` for per-member metrics.
"""

from __future__ import annotations

import inspect
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from earth2studio.data import fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.run import ProgressStore, WorkItem
from earth2studio.statistics.weights import lat_weight
from earth2studio.utils.coords import CoordSystem, coord_array
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

from .campaign import online_scoring
from .data import WINDOW_MEANS_ATTR, resolve_source, store_attrs, store_name
from .output import (
    OutputRegridder,
    progress_path,
    store_path,
    verification_regridder,
)
from .regions import (
    NON_SPATIAL,
    SpatialCoords,
    region_masks,
    scoring_regions,
    spatial_axes,
)
from .runners import is_window_mean, require_window_mean_verification

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def online_mode(cfg: DictConfig) -> bool:
    """Whether ``scoring.mode`` selects the online pathway.

    The one check every entry point shares,
    :func:`src.campaign.online_scoring`.

    Raises
    ------
    ValueError
        On an unknown mode, rather than silently falling back to offline.
    """
    return online_scoring(cfg)


def score_store_path(cfg: DictConfig) -> str:
    """Path of the score store: ``<output.path>/<scoring.output.store_name>``."""
    return store_path(cfg, str(cfg.scoring.output.get("store_name", "scores.zarr")))


def scoring_progress_path(cfg: DictConfig) -> os.PathLike:
    """Directory of the scoring :class:`~earth2studio.run.ProgressStore`."""
    return progress_path(cfg, "progress_scoring")


# ---------------------------------------------------------------------------
# Statistic / Metric protocol detection
# ---------------------------------------------------------------------------


def _is_statistic(obj: Any) -> bool:
    """Return True if *obj* follows the Statistic (one-array) protocol.

    ``earth2studio.statistics.Statistic.__call__`` takes ``x`` only, while
    ``Metric.__call__`` takes ``x`` and ``y``.  Wrappers may declare which
    protocol they forward to through an ``is_statistic`` attribute.
    """
    flag = getattr(obj, "is_statistic", None)
    if flag is not None:
        return bool(flag)
    sig = inspect.signature(type(obj).__call__)
    required = [
        p
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.default is inspect.Parameter.empty
    ]
    return len(required) == 1


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------


def _with_region(inner: xr.DataArray, names: Sequence[str]) -> CoordinateSystem:
    """Signature of *inner* behind a leading, string-labelled ``region`` axis."""
    coords: dict[Any, Any] = {
        key: value.variable for key, value in inner.coords.items()
    }
    coords["region"] = np.array(list(names))
    sizes = {dim: inner.sizes[dim] for dim in inner.dims}
    return coord_array(("region", *inner.dims), coords, sizes=sizes, dtype=inner.dtype)


class RegionalMetric:
    """One metric applied per named region, stacked on a ``region`` axis.

    Wraps per-region instances of the same underlying metric (each built
    with region-masked weights) behind the Metric protocol, so the store
    schema, chunking validation and write path treat it as any other
    metric; its output carries a leading string-labelled ``region``
    dimension.

    Parameters
    ----------
    per_region : OrderedDict[str, Any]
        Metric instances keyed by region name, in config order.
    """

    def __init__(self, per_region: OrderedDict[str, Any]) -> None:
        self._per_region = per_region
        first = next(iter(per_region.values()))
        self.reduction_dimensions = first.reduction_dimensions
        self.ensemble_dimension = getattr(first, "ensemble_dimension", None)
        self.is_statistic = _is_statistic(first)

    @property
    def regions(self) -> list[str]:
        """Region labels in output order."""
        return list(self._per_region)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Inner metric's output signature behind a leading ``region`` axis."""
        first = next(iter(self._per_region.values()))
        return _with_region(first.output_coords(input_coords), self.regions)

    def __call__(self, x: xr.DataArray, y: xr.DataArray | None = None) -> xr.DataArray:
        """Apply every region's instance and stack the results."""
        values: list[torch.Tensor] = []
        last: xr.DataArray | None = None
        for metric in self._per_region.values():
            last = metric(x) if self.is_statistic else metric(x, y)
            values.append(last.e2s.to_torch()[0])
        if last is None:
            raise ValueError("RegionalMetric needs at least one region")
        return from_torch(torch.stack(values, dim=0), _with_region(last, self.regions))


def _embed_mask(
    mask: torch.Tensor,
    red_dims: list[str],
    grid_dims: list[str],
) -> torch.Tensor:
    """Reshape a spatial-shaped region mask onto a metric's reduction dims.

    The mask carries one axis per spatial dimension (grid order, and not
    separable for union regions); the metric wants weights with exactly one
    axis per reduction dimension.  Permute the mask's axes into reduction
    order and insert size-1 axes for non-spatial reduction dims.
    """
    perm = [grid_dims.index(d) for d in red_dims if d in grid_dims]
    out = mask.permute(perm)
    for i, d in enumerate(red_dims):
        if d not in grid_dims:
            out = out.unsqueeze(i)
    return out


def instantiate_metrics(
    cfg: DictConfig,
    spatial_coords: SpatialCoords,
) -> OrderedDict:
    """Create metric instances from the ``scoring.metrics`` config block.

    Each entry is resolved via ``hydra.utils.get_class`` on its ``_target_``
    key.  If ``scoring.lat_weights`` is true and ``lat`` appears among a
    metric's ``reduction_dimensions``, cosine latitude weights are injected
    automatically.

    With ``scoring.regions`` (plus any boxes declared by ``scoring.events``),
    every metric that reduces over all spatial dimensions and accepts a
    ``weights`` argument becomes a :class:`RegionalMetric`, adding a
    labelled ``region`` axis to its scores.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (reads ``cfg.scoring.metrics``,
        ``cfg.scoring.lat_weights`` and ``cfg.scoring.regions``).
    spatial_coords : Mapping[str, Any] | xr.DataArray
        Spatial coordinate arrays of the scored grid, or a coordinate
        signature carrying them (used for latitude weights and masks).

    Returns
    -------
    OrderedDict[str, Metric]
        Metric instances keyed by their config name.
    """
    grid = spatial_axes(spatial_coords)
    grid_dims = list(grid)
    use_lat_weights = cfg.scoring.get("lat_weights", False)
    regions = scoring_regions(cfg.scoring)
    masks = region_masks(grid, regions) if regions is not None else None
    metrics: OrderedDict = OrderedDict()

    for name, metric_cfg in cfg.scoring.metrics.items():
        cfg_dict: dict = OmegaConf.to_container(metric_cfg, resolve=True)
        target = cfg_dict.pop("_target_")
        cls = hydra.utils.get_class(target)
        red_dims = list(cfg_dict.get("reduction_dimensions", []))
        shape = [len(grid[d]) if d in grid else 1 for d in red_dims]

        # Inject cosine latitude weights when appropriate.  Weights must
        # have exactly len(reduction_dimensions) dims with sizes matching
        # each reduction dim's coordinate length.
        if (
            use_lat_weights
            and "lat" in grid
            and "lat" in red_dims
            and "weights" not in cfg_dict
        ):
            weights_1d = lat_weight(torch.tensor(grid["lat"], dtype=torch.float32))
            view_shape = [1] * len(red_dims)
            view_shape[red_dims.index("lat")] = len(grid["lat"])
            cfg_dict["weights"] = torch.ones(shape) * weights_1d.reshape(view_shape)

        # Region-mask the metric when it can carry the masking: it must
        # reduce over every spatial dimension (a partial spatial reduction
        # would leave masked-out gridpoints in the output) and take a
        # weights argument to fold the mask into.
        maskable = (
            set(grid_dims) <= set(red_dims)
            and "weights" in inspect.signature(cls.__init__).parameters
        )
        if masks is not None and not maskable:
            logger.warning(
                f"Metric '{name}' cannot be region-masked (needs a 'weights' "
                "argument and a full spatial reduction); scored on the whole "
                "grid only."
            )

        if masks is not None and maskable:
            base = cfg_dict.pop("weights", None)
            base = torch.ones(shape) if base is None else torch.as_tensor(base)
            per_region: OrderedDict = OrderedDict()
            for region_name, mask in masks.items():
                embedded = _embed_mask(mask, red_dims, grid_dims).to(base.dtype)
                per_region[region_name] = cls(**cfg_dict, weights=base * embedded)
            metrics[name] = RegionalMetric(per_region)
            logger.info(
                f"Instantiated metric '{name}': {target} ({len(per_region)} regions)"
            )
        else:
            metrics[name] = cls(**cfg_dict)
            logger.info(f"Instantiated metric '{name}': {target}")

    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def open_prediction_store(cfg: DictConfig) -> xr.Dataset:
    """Open the inference output store ``forecast.zarr`` as a Dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``output.path``.

    Returns
    -------
    xr.Dataset

    Raises
    ------
    FileNotFoundError
        If the prediction store does not exist.
    """
    path = store_path(cfg)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Prediction store not found at '{path}'.\n"
            "Run inference (main.py) before scoring."
        )
    logger.info(f"Opened prediction store: {path}")
    return xr.open_zarr(path, consolidated=False)


def open_verification_source(cfg: DictConfig) -> Any:
    """Resolve the verification data source.

    A bring-your-own ``verification_source`` wins; otherwise the
    predownloaded ``<output.path>/verification.zarr``; otherwise the live
    ``predownload.verification.source`` or ``data_source``.  A ``window_mean``
    run accepts only sources of window means: the two configured ones, or a
    saved store that ``predownload.py`` marked as holding them
    (:data:`src.data.WINDOW_MEANS_ATTR`); it never falls back to
    ``data_source``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config.

    Returns
    -------
    DataSource

    Raises
    ------
    ValueError
        If no config entry names a verification source, or a ``window_mean``
        run has no source of window means
        (:func:`src.runners.require_window_mean_verification`).
    """
    byo = cfg.get("verification_source")
    predownload = cfg.get("predownload", None) or {}
    configured = (predownload.get("verification", None) or {}).get("source", None)
    if byo is None and configured is None and is_window_mean(cfg):
        saved = store_path(cfg, store_name("verification"))
        if not store_attrs(saved).get(WINDOW_MEANS_ATTR):
            require_window_mean_verification(cfg, None)
    live = configured if configured is not None else cfg.get("data_source")
    source = resolve_source(cfg, "verification", byo=byo, live=live)
    if source is None:
        raise ValueError(
            "No verification source: set verification_source, run predownload.py "
            "with predownload.verification.enabled=true, or configure data_source."
        )
    return source


def spatial_coords_from_dataset(ds: xr.Dataset) -> CoordSystem:
    """Extract spatial coordinate arrays from a prediction Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Opened prediction zarr store.

    Returns
    -------
    CoordSystem
        Spatial coordinate arrays (e.g. ``{lat: [...], lon: [...]}``, or
        ``{x: [...], y: [...]}`` for non-lat/lon grids).  A dimension
        without a coordinate gets ``arange(size)``.
    """
    # The dimension order comes from a stored field, not from ``ds.dims``:
    # that mapping follows the listing order of the arrays, which a zarr 3
    # store returns in the order its concurrent reads completed.
    reference = next(iter(ds.data_vars.values()), None)
    dims = tuple(ds.dims) if reference is None else tuple(reference.dims)
    coords: CoordSystem = OrderedDict()
    for dim in dims:
        if dim in NON_SPATIAL:
            continue
        if dim in ds.coords:
            coords[str(dim)] = np.asarray(ds.coords[dim].values)
        else:
            coords[str(dim)] = np.arange(ds.sizes[dim])
    return coords


def build_input_coords_template(
    prediction_ds: xr.Dataset,
    lead_times: np.ndarray,
    variables: Sequence[str],
) -> CoordinateSystem:
    """Build the coordinate signature of one forecast for metric introspection.

    Used to call ``metric.output_coords()`` to determine the shape of the
    score arrays before loading any data.

    Parameters
    ----------
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    lead_times : np.ndarray
        All lead time values from the prediction store.
    variables : Sequence[str]
        Variable names to be scored.

    Returns
    -------
    CoordinateSystem
        Signature with ``(ensemble?, lead_time, variable, <spatial...>)``.
    """
    coords: dict[str, Any] = {}
    dims: list[str] = []
    if "ensemble" in prediction_ds.dims:
        dims.append("ensemble")
        coords["ensemble"] = np.asarray(prediction_ds.coords["ensemble"].values)
    dims += ["lead_time", "variable"]
    coords["lead_time"] = np.asarray(lead_times)
    coords["variable"] = np.array([str(v) for v in variables])
    for dim, values in spatial_coords_from_dataset(prediction_ds).items():
        dims.append(dim)
        coords[dim] = values
    return coord_array(dims, coords)


def _dimension_coords_only(da: xr.DataArray) -> xr.DataArray:
    """Drop auxiliary coordinates, keeping one coordinate per dimension."""
    extra = [c for c in da.coords if c not in da.dims]
    return da.drop_vars(extra) if extra else da


def load_prediction_chunk(
    prediction_ds: xr.Dataset,
    time: np.datetime64,
    lead_times: np.ndarray,
    variables: Sequence[str],
    device: torch.device | str,
) -> xr.DataArray:
    """Load one initial time and lead-time range of the prediction store.

    Parameters
    ----------
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    time : np.datetime64
        Initial-condition time to select.
    lead_times : np.ndarray
        Lead-time values for this chunk.
    variables : Sequence[str]
        Variable names to load.
    device : torch.device | str
        Device of the returned field (NumPy-backed on CPU, CuPy on CUDA).

    Returns
    -------
    xr.DataArray
        Float32 field with dimensions
        ``(ensemble?, lead_time, variable, <spatial...>)``.
    """
    subset = prediction_ds[list(variables)].sel(time=time, lead_time=lead_times)
    da = subset.to_array(dim="variable")
    spatial = [d for d in da.dims if d not in NON_SPATIAL and d != "variable"]
    order = (["ensemble"] if "ensemble" in da.dims else []) + ["lead_time", "variable"]
    da = _dimension_coords_only(da.transpose(*order, *spatial))
    da = da.assign_coords(variable=np.array([str(v) for v in da["variable"].values]))
    tensor = torch.as_tensor(np.ascontiguousarray(da.values), dtype=torch.float32)
    return from_torch(tensor.to(device), da)


def load_verification_chunk(
    source: Any,
    time: np.datetime64,
    lead_times: np.ndarray,
    variables: Sequence[str],
    spatial_coords: SpatialCoords,
    device: torch.device | str,
    regrid: OutputRegridder | None = None,
) -> xr.DataArray:
    """Load verification data aligned to a prediction chunk.

    Fetches the valid times ``time + lead_time`` through
    :func:`earth2studio.data.fetch_data`, brings them onto the ``output.regrid``
    grid when the campaign names one, and relabels the result with the
    prediction grid's coordinates so it aligns with the prediction chunk.

    Parameters
    ----------
    source : DataSource
        Verification data source.
    time : np.datetime64
        Initial-condition time.
    lead_times : np.ndarray
        Lead-time values for this chunk.
    variables : Sequence[str]
        Variable names to load.
    spatial_coords : Mapping[str, Any] | xr.DataArray
        Spatial coordinates of the prediction grid.
    device : torch.device | str
        Device of the returned field.
    regrid : OutputRegridder, optional
        Brings the fetched fields onto the prediction grid (see
        :func:`src.output.verification_regridder`), by default None (the
        source must already sit on the prediction grid).

    Returns
    -------
    xr.DataArray
        Float32 field with dimensions ``(lead_time, variable, <spatial...>)``.

    Raises
    ------
    ValueError
        If the verification grid does not match the prediction grid.
    """
    grid = spatial_axes(spatial_coords)
    try:
        da = fetch_data(
            source,
            time=np.array([time], dtype="datetime64[ns]"),
            variable=np.array([str(v) for v in variables]),
            lead_time=np.asarray(lead_times).astype("timedelta64[ns]"),
            device=device,
        )
    except KeyError as exc:
        raise KeyError(
            f"Scoring requested variables {list(variables)} at the valid times of "
            f"{time} but the verification source could not supply them ({exc}). "
            "Ensure predownload completed for every lead time, or drop the "
            "missing variable(s) from cfg.scoring.variables."
        ) from exc
    if regrid is not None:
        da = regrid(da)
    da = da.isel(time=0, drop=True)
    spatial = [d for d in da.dims if d not in ("lead_time", "variable")]
    if set(spatial) != set(grid) or any(da.sizes[d] != len(grid[d]) for d in grid):
        raise ValueError(
            "Verification grid "
            f"{ {d: da.sizes[d] for d in spatial} } does not match the prediction "
            f"grid { {d: len(v) for d, v in grid.items()} }."
        )
    # Transposed by name: the source's own dimension order is not the store's.
    da = _dimension_coords_only(da.transpose("lead_time", "variable", *grid))
    da = da.assign_coords(
        lead_time=np.asarray(lead_times), **{d: grid[d] for d in grid}
    )
    tensor, _ = da.e2s.to_torch()
    return from_torch(tensor.to(torch.float32), da)


# ---------------------------------------------------------------------------
# Lead-time chunking
# ---------------------------------------------------------------------------


def build_lead_time_chunks(
    lead_times: np.ndarray,
    chunk_size: int | None,
) -> list[np.ndarray]:
    """Partition lead times into chunks for memory-bounded processing.

    Parameters
    ----------
    lead_times : np.ndarray
        Full array of lead-time values.
    chunk_size : int | None
        Maximum number of lead times per chunk.  ``None`` or ``<= 0``
        means no chunking (all lead times in one chunk).

    Returns
    -------
    list[np.ndarray]
        List of lead-time sub-arrays.
    """
    if chunk_size is None or chunk_size <= 0 or chunk_size >= len(lead_times):
        return [lead_times]
    return [
        lead_times[i : i + chunk_size] for i in range(0, len(lead_times), chunk_size)
    ]


def validate_lead_time_chunking(
    metrics: Mapping[str, Any],
    chunk_size: int | None,
    n_lead_times: int,
) -> None:
    """Check that chunking is safe for the configured metrics.

    Chunking over ``lead_time`` produces incorrect results for metrics
    that reduce over it, because partial reductions do not combine.

    Parameters
    ----------
    metrics : Mapping[str, Metric]
        Metric instances.
    chunk_size : int | None
        Configured chunk size (``None`` means no chunking).
    n_lead_times : int
        Total number of lead-time values.

    Raises
    ------
    ValueError
        If chunking is active and a metric reduces over ``lead_time``.
    """
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n_lead_times:
        return
    for name, metric in metrics.items():
        if "lead_time" in metric.reduction_dimensions:
            raise ValueError(
                f"Metric '{name}' reduces over 'lead_time', but "
                f"lead_time_chunk_size={chunk_size} is active.  "
                "Either remove 'lead_time' from reduction_dimensions or "
                "set lead_time_chunk_size to null to disable chunking."
            )


# ---------------------------------------------------------------------------
# Score store schema
# ---------------------------------------------------------------------------


def score_variable_names(
    metric_name: str,
    metric_output_coords: CoordinateSystem,
) -> list[str]:
    """Determine the zarr array names of a metric's output.

    If the metric preserves the ``variable`` dimension, names are
    ``{metric_name}__{variable_name}``.  If ``variable`` was reduced away,
    the single array name is just ``metric_name``.

    Parameters
    ----------
    metric_name : str
        Config name of the metric (e.g. ``"rmse"``).
    metric_output_coords : CoordinateSystem
        Output signature from ``metric.output_coords()``.

    Returns
    -------
    list[str]
        Zarr array names for this metric.
    """
    if "variable" in metric_output_coords.dims:
        return [f"{metric_name}__{v}" for v in metric_output_coords["variable"].values]
    return [metric_name]


def all_score_variable_names(
    metrics: Mapping[str, Any],
    input_coords_template: CoordinateSystem,
) -> list[str]:
    """Collect all zarr array names across all configured metrics.

    Parameters
    ----------
    metrics : Mapping[str, Metric]
        Metric instances keyed by name.
    input_coords_template : CoordinateSystem
        Representative input signature for ``output_coords()`` calls.

    Returns
    -------
    list[str]
        Flat list of all score array names.
    """
    names: list[str] = []
    for metric_name, metric in metrics.items():
        names.extend(
            score_variable_names(
                metric_name, metric.output_coords(input_coords_template)
            )
        )
    return names


def _store_dims(output: CoordinateSystem) -> list[str]:
    """Dimensions of a metric's score arrays: ``time`` then its output dims."""
    return ["time"] + [str(d) for d in output.dims if d not in ("variable", "time")]


def group_score_arrays_by_dims(
    metrics: Mapping[str, Any],
    input_coords_template: CoordinateSystem,
    times: np.ndarray,
) -> list[tuple[CoordSystem, list[str]]]:
    """Group score arrays by their output dimension structure.

    Metrics that reduce different dimensions produce arrays with different
    shapes.  This function groups them so each group's arrays share the
    same coordinate axes and can be created in a single
    ``ZarrBackend.add_array`` call.

    Parameters
    ----------
    metrics : Mapping[str, Metric]
        Metric instances keyed by name.
    input_coords_template : CoordinateSystem
        Representative input signature for ``output_coords()`` calls.
    times : np.ndarray
        All initial-condition times for the ``time`` coordinate.

    Returns
    -------
    list[tuple[CoordSystem, list[str]]]
        Each entry is ``(group_coords, array_names)`` where all arrays in
        the group share the same dimensions, ``time`` first.
    """
    groups: dict[tuple[str, ...], tuple[CoordSystem, list[str]]] = {}
    for metric_name, metric in metrics.items():
        out = metric.output_coords(input_coords_template)
        metric_coords: CoordSystem = OrderedDict()
        metric_coords["time"] = np.asarray(times).astype("datetime64[ns]")
        for dim in _store_dims(out)[1:]:
            metric_coords[dim] = np.asarray(out[dim].values)
        key = tuple(metric_coords)
        names = score_variable_names(metric_name, out)
        if key in groups:
            groups[key][1].extend(names)
        else:
            groups[key] = (metric_coords, list(names))
    return list(groups.values())


def add_score_arrays(
    io: ZarrBackend,
    array_groups: list[tuple[CoordSystem, list[str]]],
) -> None:
    """Create the score arrays of the zarr store and check the ones present.

    Safe to call on every rank inside :func:`src.logs.rank0_first_checked`:
    rank 0 creates the arrays, the other ranks find them present and no-op.
    The backend leaves an existing array alone when its layout matches and
    raises when it does not, so a resumed scoring into a store laid out for
    other times or lead times fails here, before any work.

    Each group becomes one IO template whose ``variable`` labels name the
    score arrays and whose other dimensions size them.

    Parameters
    ----------
    io : ZarrBackend
        The score store backend.
    array_groups : list[tuple[CoordSystem, list[str]]]
        Groups from :func:`group_score_arrays_by_dims`.

    Raises
    ------
    ValueError
        If an existing array or coordinate differs from the template.
    """
    for group_coords, group_vars in array_groups:
        coords = {dim: np.asarray(vals) for dim, vals in group_coords.items()}
        coords["variable"] = np.array(list(group_vars))
        io.add_array(coord_array((*group_coords, "variable"), coords))


def open_score_store(
    cfg: DictConfig,
    array_groups: list[tuple[CoordSystem, list[str]]],
    rank: int,
    events: Mapping[str, Any] | None = None,
) -> ZarrBackend:
    """Create or open the score store and add the score arrays.

    Call on every rank inside :func:`src.logs.rank0_first_checked`, after
    :func:`src.output.prepare_store` applied the rerun policy
    (``scoring.resume``, ``scoring.output.overwrite``) to an existing store.
    The chunk sizes come from ``scoring.output.chunks`` (default
    ``{time: 1}``) and the parsed events go onto the store attributes so the
    scorecard exporter can window the per-IC scores.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    array_groups : list[tuple[CoordSystem, list[str]]]
        Groups from :func:`group_score_arrays_by_dims`.
    rank : int
        This process's rank.
    events : Mapping[str, Any] | None
        JSON-friendly events (see :func:`src.regions.events_to_attrs`).

    Returns
    -------
    ZarrBackend
    """
    path = score_store_path(cfg)
    chunks = {
        str(k): int(v)
        for k, v in dict(cfg.scoring.output.get("chunks", None) or {"time": 1}).items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    io = ZarrBackend(path, chunks=chunks, backend_kwargs={"overwrite": False})
    add_score_arrays(io, array_groups)
    if events is not None and rank == 0:
        io.root.attrs["events"] = dict(events)
    logger.info(f"Score store ready: {path}")
    return io


# ---------------------------------------------------------------------------
# Core scoring loop
# ---------------------------------------------------------------------------


def _parse_nan_policy(cfg: DictConfig) -> str:
    policy = str(cfg.scoring.get("nan_policy", "propagate")).lower()
    if policy == "mask":
        raise ValueError(
            "scoring.nan_policy=mask drops non-finite gridpoints from each "
            "statistic, which only online scoring supports; offline metrics "
            "reduce whole fields. Use propagate or zero_fill with scoring.mode=offline."
        )
    if policy not in ("propagate", "zero_fill"):
        raise ValueError(
            f"Invalid scoring.nan_policy '{policy}'; "
            "expected 'propagate', 'zero_fill' or 'mask' (online only)."
        )
    if policy == "zero_fill":
        logger.info(
            "Scoring nan_policy=zero_fill: NaNs in prediction and verification "
            "chunks are replaced with 0 before metric evaluation.  RMSE-style "
            "metrics normalize over all grid points, so invalid points contribute "
            "zero to the sum of squares and bias the score low by "
            "sqrt(N_valid/N_total)."
        )
    return policy


def _parse_valid_ranges(cfg: DictConfig) -> dict:
    valid_ranges_cfg = cfg.scoring.get("valid_ranges", None)
    if valid_ranges_cfg is None:
        return {}
    valid_ranges = OmegaConf.to_container(valid_ranges_cfg, resolve=True) or {}
    if valid_ranges:
        logger.info(
            f"Scoring valid_ranges: clamping {sorted(valid_ranges)} in both "
            "prediction and verification before metric evaluation."
        )
    return dict(valid_ranges)


def _zero_fill(da: xr.DataArray) -> xr.DataArray:
    tensor, _ = da.e2s.to_torch()
    return from_torch(torch.nan_to_num(tensor, nan=0.0), da)


def run_scoring(
    my_items: Sequence[WorkItem],
    prediction_ds: xr.Dataset,
    verif_source: Any,
    metrics: Mapping[str, Any],
    io: ZarrBackend,
    variables: Sequence[str],
    lead_times: np.ndarray,
    lead_time_chunks: Sequence[np.ndarray],
    spatial_coords: SpatialCoords,
    device: torch.device | str,
    cfg: DictConfig,
    progress: ProgressStore | None = None,
    rank: int = 0,
) -> None:
    """Score predictions against verification for the assigned work items.

    For each item (one initial-condition time), loads prediction and
    verification data in lead-time chunks, applies every configured metric
    and writes the results to the score store, then records the item in
    the progress store.

    Parameters
    ----------
    my_items : Sequence[WorkItem]
        Items assigned to this rank, one per initial time.
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    verif_source : DataSource
        Verification data source.
    metrics : Mapping[str, Metric]
        Metric instances keyed by name.
    io : ZarrBackend
        Score store with its arrays added (see :func:`open_score_store`).
    variables : Sequence[str]
        Variable names to score.
    lead_times : np.ndarray
        Full array of lead-time values (for concatenating chunks).
    lead_time_chunks : Sequence[np.ndarray]
        Partitioned lead-time arrays.
    spatial_coords : Mapping[str, Any] | xr.DataArray
        Spatial coordinates of the prediction grid.
    device : torch.device | str
        Device for metric computations.
    cfg : DictConfig
        Full Hydra config (``scoring.nan_policy``, ``scoring.valid_ranges``).
    progress : ProgressStore, optional
        Opened progress store recording finished items, by default None
    rank : int, optional
        This process's rank, which gates the progress bar, by default 0
    """
    if not my_items:
        logger.info(f"Rank {rank}: no times assigned for scoring.")
        return

    nan_policy = _parse_nan_policy(cfg)
    valid_ranges = _parse_valid_ranges(cfg)
    variables = [str(v) for v in variables]
    regrid = verification_regridder(cfg, spatial_coords)

    for item in tqdm(my_items, desc="Scoring", disable=rank != 0):
        time = np.datetime64(item.time, "ns")
        chunk_results: dict[str, list[xr.DataArray]] = {name: [] for name in metrics}

        for lt_chunk in lead_time_chunks:
            x = load_prediction_chunk(prediction_ds, time, lt_chunk, variables, device)
            y = load_verification_chunk(
                verif_source, time, lt_chunk, variables, spatial_coords, device, regrid
            )
            if nan_policy == "zero_fill":
                x, y = _zero_fill(x), _zero_fill(y)
            if valid_ranges:
                x = apply_valid_ranges(x, valid_ranges)
                y = apply_valid_ranges(y, valid_ranges)

            for metric_name, metric in metrics.items():
                if _is_statistic(metric):
                    score = metric(x)
                else:
                    score = metric(x, _observation_for(metric, x, y))
                chunk_results[metric_name].append(score.e2s.as_numpy())

        for metric_name, chunks in chunk_results.items():
            score = _concat_chunks(chunks, lead_times)
            io.write(_score_field(score, metric_name, time))

        io.flush()
        if progress is not None:
            progress.mark_done(item)

    logger.success("Scoring complete.")


def _observation_for(metric: Any, x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
    """Shape the observation to the forecast for one metric.

    Metrics compare ``x`` and ``y`` on identical dimensions.  A per-member
    metric on an ensemble forecast (one that neither reduces the
    ``ensemble`` dimension nor averages over it through
    ``ensemble_dimension``) sees the observation broadcast along
    ``ensemble``; every other metric gets ``y`` unchanged.
    """
    if "ensemble" not in x.dims or "ensemble" in y.dims:
        return y
    if "ensemble" in metric.reduction_dimensions:
        return y
    if getattr(metric, "ensemble_dimension", None) == "ensemble":
        return y
    return y.expand_dims(ensemble=x["ensemble"].values).transpose(*x.dims)


def _score_field(
    score: xr.DataArray, metric_name: str, time: np.datetime64
) -> xr.DataArray:
    """Turn one metric result into the field written to the score store.

    Variable labels become ``<metric>__<variable>`` (a reduced ``variable``
    dimension comes back with the metric name as its only label) and a
    leading ``time`` axis locates the initial condition.  Auxiliary
    coordinates go away so only dimension labels take part in matching.
    """
    score = _dimension_coords_only(score)
    if "variable" in score.dims:
        labels = [f"{metric_name}__{v}" for v in score["variable"].values]
        score = score.assign_coords(variable=np.array(labels))
    else:
        score = score.expand_dims(variable=np.array([metric_name]))
    score = score.expand_dims(time=np.array([time], dtype="datetime64[ns]"))
    dims = ["time"] + [d for d in score.dims if d not in ("time", "variable")]
    return score.transpose(*dims, "variable")


def apply_valid_ranges(
    da: xr.DataArray, valid_ranges: Mapping[str, Any]
) -> xr.DataArray:
    """Clamp per-variable values of a field to the configured physical range.

    ``valid_ranges`` maps variable names to ``{min: ..., max: ...}`` entries;
    either bound may be ``None`` / absent for a one-sided clamp.  Variables
    not listed stay untouched.

    Use case: verification sources (e.g. MRMS ``refc``) encode no-data
    pixels as large fill values (``-9999``) that otherwise
    dominate the sum of squares in RMSE-style metrics.  Clamping both
    prediction and verification to the physical range turns those pixels
    into a shared in-range value (typically 0 for non-negative fields),
    contributing about zero to the error.

    Parameters
    ----------
    da : xr.DataArray
        Field with a ``variable`` dimension.
    valid_ranges : Mapping[str, Mapping[str, float | None]]
        Mapping from variable name to ``{min, max}``.

    Returns
    -------
    xr.DataArray
        Clamped copy of *da*.
    """
    if "variable" not in da.dims:
        return da
    tensor, _ = da.e2s.to_torch()
    coords: CoordSystem = OrderedDict(
        (
            str(d),
            (
                np.asarray(da.coords[d].values)
                if d in da.coords
                else np.arange(da.sizes[d])
            ),
        )
        for d in da.dims
    )
    return from_torch(_apply_valid_ranges(tensor.clone(), coords, valid_ranges), da)


def _apply_valid_ranges(
    x: torch.Tensor,
    coords: CoordSystem,
    valid_ranges: Mapping[str, Any],
) -> torch.Tensor:
    """Clamp per-variable values in *x* in place along its ``variable`` axis.

    Tensor-level core of :func:`apply_valid_ranges`, shared with the online
    scorer.

    Parameters
    ----------
    x : torch.Tensor
        Tensor with a ``variable`` axis.
    coords : CoordSystem
        Matching coordinate system (locates the variable axis and maps
        variable names to axis indices).
    valid_ranges : Mapping[str, Mapping[str, float | None]]
        Mapping from variable name to ``{min, max}``.

    Returns
    -------
    torch.Tensor
        *x*, clamped in place.
    """
    if "variable" not in coords:
        return x
    var_axis = list(coords.keys()).index("variable")
    var_names = [str(v) for v in coords["variable"]]

    for var_name, spec in valid_ranges.items():
        if var_name not in var_names:
            continue
        idx = var_names.index(var_name)
        vmin = spec.get("min") if isinstance(spec, Mapping) else None
        vmax = spec.get("max") if isinstance(spec, Mapping) else None
        if vmin is None and vmax is None:
            continue
        slicer: list[Any] = [slice(None)] * x.ndim
        slicer[var_axis] = idx
        key = tuple(slicer)
        x[key] = torch.clamp(
            x[key],
            min=float(vmin) if vmin is not None else None,
            max=float(vmax) if vmax is not None else None,
        )
    return x


def _concat_chunks(
    chunks: Sequence[xr.DataArray],
    full_lead_times: np.ndarray,
) -> xr.DataArray:
    """Concatenate lead-time chunk results into a single score array.

    Parameters
    ----------
    chunks : Sequence[xr.DataArray]
        Per-chunk results of a single metric, NumPy-backed.
    full_lead_times : np.ndarray
        Complete lead-time array of the forecast.

    Returns
    -------
    xr.DataArray
        Concatenated score.
    """
    if len(chunks) == 1:
        return chunks[0]
    if "lead_time" in chunks[0].dims:
        out = xr.concat(list(chunks), dim="lead_time")
        return out.assign_coords(lead_time=np.asarray(full_lead_times))
    # The metric reduced lead_time, so concatenation cannot combine the
    # chunks; validate_lead_time_chunking() blocks this path.
    return chunks[-1]
