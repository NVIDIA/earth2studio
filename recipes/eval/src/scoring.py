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

"""Scoring module: compare predictions against verification data.

Loads inference outputs from a zarr store, aligns them with predownloaded
verification data, applies configured ``earth2studio.statistics`` metrics,
and writes the resulting scores to a shared ``scores.zarr`` store.

Designed for multi-GPU use: each rank scores an independent subset of
initial-condition times.  Lead-time chunking keeps memory bounded.
"""

from __future__ import annotations

import inspect
import os
from collections import OrderedDict
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.statistics.weights import lat_weight
from earth2studio.utils.coords import CoordSystem

from .data import PredownloadedSource
from .output import OutputManager
from .work import write_scoring_marker

# Dimensions that are never spatial.
_NON_SPATIAL = frozenset({"ensemble", "time", "lead_time", "batch"})


# ---------------------------------------------------------------------------
# Statistic / Metric protocol detection
# ---------------------------------------------------------------------------


def _is_statistic(obj: Any) -> bool:
    """Return True if *obj* follows the Statistic (2-arg) protocol.

    The ``earth2studio.statistics.base.Statistic`` protocol defines
    ``__call__(self, x, coords)`` (2 required positional args after self),
    while ``Metric`` defines ``__call__(self, x, x_coords, y, y_coords)``
    (4 required positional args).  We distinguish the two by inspecting
    the call signature.
    """
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
    return len(required) == 2


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------


def instantiate_metrics(
    cfg: DictConfig,
    spatial_coords: CoordSystem,
) -> OrderedDict:
    """Create metric instances from the ``scoring.metrics`` config block.

    Each entry is resolved via ``hydra.utils.get_class`` on its ``_target_``
    key.  If ``scoring.lat_weights`` is true and ``lat`` appears among a
    metric's ``reduction_dimensions``, cosine latitude weights are injected
    automatically.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config (reads ``cfg.scoring.metrics`` and
        ``cfg.scoring.lat_weights``).
    spatial_coords : CoordSystem
        Spatial coordinate arrays from the prediction store (used to
        compute latitude weights).

    Returns
    -------
    OrderedDict[str, Metric]
        Metric instances keyed by their config name.
    """
    use_lat_weights = cfg.scoring.get("lat_weights", False)
    metrics: OrderedDict = OrderedDict()

    for name, metric_cfg in cfg.scoring.metrics.items():
        cfg_dict: dict = OmegaConf.to_container(metric_cfg, resolve=True)
        target = cfg_dict.pop("_target_")
        cls = hydra.utils.get_class(target)

        # Inject cosine latitude weights when appropriate.
        # Weights must have exactly len(reduction_dimensions) dims with
        # sizes matching each reduction dim's coordinate length.
        if (
            use_lat_weights
            and "lat" in spatial_coords
            and "lat" in cfg_dict.get("reduction_dimensions", [])
            and "weights" not in cfg_dict
        ):
            lat_vals = spatial_coords["lat"]
            weights_1d = lat_weight(torch.tensor(lat_vals, dtype=torch.float32))
            red_dims = cfg_dict["reduction_dimensions"]
            shape = [
                len(spatial_coords[d]) if d in spatial_coords else 1 for d in red_dims
            ]
            lat_idx = red_dims.index("lat")
            # Start with ones at the full shape, then broadcast lat in.
            weights = torch.ones(shape)
            view_shape = [1] * len(red_dims)
            view_shape[lat_idx] = len(lat_vals)
            weights = weights * weights_1d.reshape(view_shape)
            cfg_dict["weights"] = weights

        metrics[name] = cls(**cfg_dict)
        logger.info(f"Instantiated metric '{name}': {target}")

    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def open_prediction_store(cfg: DictConfig) -> xr.Dataset:
    """Open the inference output zarr store as an xarray Dataset.

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
    store_path = os.path.join(cfg.output.path, "forecast.zarr")
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Prediction store not found at '{store_path}'.\n"
            "Run inference (main.py) before scoring."
        )
    logger.info(f"Opened prediction store: {store_path}")
    return xr.open_zarr(store_path)


def open_verification_source(cfg: DictConfig) -> PredownloadedSource:
    """Open the verification data source.

    Checks for ``verification.zarr`` first (separate verification store),
    then falls back to ``data.zarr`` (merged IC + verification store).

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``output.path``.

    Returns
    -------
    PredownloadedSource

    Raises
    ------
    FileNotFoundError
        If neither verification nor data store exists.
    """
    verif_path = os.path.join(cfg.output.path, "verification.zarr")
    if os.path.exists(verif_path):
        logger.info(f"Using verification store: {verif_path}")
        return PredownloadedSource(verif_path)

    data_path = os.path.join(cfg.output.path, "data.zarr")
    if os.path.exists(data_path):
        logger.info(f"Using merged data store for verification: {data_path}")
        return PredownloadedSource(data_path)

    raise FileNotFoundError(
        f"No verification data found in '{cfg.output.path}'.\n"
        "Run predownload.py with predownload.verification.enabled=true."
    )


def spatial_coords_from_dataset(ds: xr.Dataset) -> CoordSystem:
    """Extract spatial coordinate arrays from a prediction Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Opened prediction zarr store.

    Returns
    -------
    CoordSystem
        Spatial coordinate arrays (e.g. ``{lat: [...], lon: [...]}``,
        or ``{x: [...], y: [...]}`` for non-lat/lon grids).
    """
    coords: CoordSystem = OrderedDict()
    for dim in ds.dims:
        if dim not in _NON_SPATIAL:
            coords[dim] = ds.coords[dim].values
    return coords


def build_input_coords_template(
    prediction_ds: xr.Dataset,
    lead_times: np.ndarray,
    variables: list[str],
) -> CoordSystem:
    """Build a representative input coordinate system for metric introspection.

    Used to call ``metric.output_coords()`` in order to determine the
    shape of the score arrays before any data is loaded.

    Parameters
    ----------
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    lead_times : np.ndarray
        All lead time values from the prediction store.
    variables : list[str]
        Variable names to be scored.

    Returns
    -------
    CoordSystem
        Template with ``(ensemble?, lead_time, variable, <spatial...>)``.
    """
    template: CoordSystem = OrderedDict()
    if "ensemble" in prediction_ds.dims:
        template["ensemble"] = prediction_ds.coords["ensemble"].values
    template["lead_time"] = lead_times
    template["variable"] = np.array(variables)
    for dim in prediction_ds.dims:
        if dim not in _NON_SPATIAL:
            template[dim] = prediction_ds.coords[dim].values
    return template


def load_prediction_chunk(
    prediction_ds: xr.Dataset,
    time: np.datetime64,
    lead_times: np.ndarray,
    variables: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Load a chunk of prediction data for one IC time and lead-time range.

    Parameters
    ----------
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    time : np.datetime64
        Initial-condition time to select.
    lead_times : np.ndarray
        Lead-time values for this chunk.
    variables : list[str]
        Variable names to load.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tensor and coords with dimensions
        ``(ensemble?, lead_time, variable, <spatial...>)``.
    """
    subset = prediction_ds[variables].sel(time=time, lead_time=lead_times)
    da = subset.to_array(dim="variable")

    has_ensemble = "ensemble" in da.dims
    spatial_dims = [
        d for d in da.dims if d not in {"variable", "ensemble", "lead_time"}
    ]

    if has_ensemble:
        dim_order = ["ensemble", "lead_time", "variable"] + spatial_dims
    else:
        dim_order = ["lead_time", "variable"] + spatial_dims

    da = da.transpose(*dim_order)
    tensor = torch.from_numpy(da.values.copy()).to(device=device, dtype=torch.float32)

    coords: CoordSystem = OrderedDict()
    for dim in dim_order:
        coords[dim] = np.array(da.coords[dim].values)
    return tensor, coords


def load_verification_chunk(
    source: PredownloadedSource,
    time: np.datetime64,
    lead_times: np.ndarray,
    variables: list[str],
    spatial_coords: CoordSystem,
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Load verification data aligned to a prediction chunk.

    For each lead time, the valid time ``time + lead_time`` is computed and
    fetched from the verification source.  The returned tensor uses
    ``lead_time`` (not valid time) as its first dimension so that it aligns
    with the prediction chunk.

    Parameters
    ----------
    source : PredownloadedSource
        Verification data source.
    time : np.datetime64
        Initial-condition time.
    lead_times : np.ndarray
        Lead-time values for this chunk.
    variables : list[str]
        Variable names to load.
    spatial_coords : CoordSystem
        Spatial coordinate arrays (for building the output CoordSystem).
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tensor and coords with dimensions
        ``(lead_time, variable, <spatial...>)``.
    """
    valid_times = time + lead_times
    da = source(list(valid_times), list(variables))

    # da has dims (time, variable, <spatial...>); reorder consistently.
    spatial_dims = [d for d in da.dims if d not in {"time", "variable"}]
    dim_order = ["time", "variable"] + spatial_dims
    da = da.transpose(*dim_order)

    tensor = torch.from_numpy(da.values.copy()).to(device=device, dtype=torch.float32)

    # Relabel the time axis as lead_time for alignment with predictions.
    coords: CoordSystem = OrderedDict()
    coords["lead_time"] = lead_times
    coords["variable"] = np.array(variables)
    for dim in spatial_dims:
        coords[dim] = spatial_coords[dim]
    return tensor, coords


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


# ---------------------------------------------------------------------------
# Score variable naming
# ---------------------------------------------------------------------------


def score_variable_names(
    metric_name: str,
    metric_output_coords: CoordSystem,
) -> list[str]:
    """Determine zarr array names for a metric's output.

    If the metric preserves the ``variable`` dimension, names are
    ``{metric_name}__{variable_name}``.  If ``variable`` was reduced away,
    the single array name is just ``metric_name``.

    Parameters
    ----------
    metric_name : str
        Config name of the metric (e.g. ``"rmse"``).
    metric_output_coords : CoordSystem
        Output coords from ``metric.output_coords()``.

    Returns
    -------
    list[str]
        Zarr array names for this metric.
    """
    if "variable" in metric_output_coords:
        return [f"{metric_name}__{v}" for v in metric_output_coords["variable"]]
    return [metric_name]


def all_score_variable_names(
    metrics: OrderedDict,
    input_coords_template: CoordSystem,
) -> list[str]:
    """Collect all zarr array names across all configured metrics.

    Parameters
    ----------
    metrics : OrderedDict[str, Metric]
        Metric instances keyed by name.
    input_coords_template : CoordSystem
        Representative input coords for ``output_coords()`` calls.

    Returns
    -------
    list[str]
        Flat list of all score array names.
    """
    names: list[str] = []
    for metric_name, metric in metrics.items():
        out_coords = metric.output_coords(input_coords_template)
        names.extend(score_variable_names(metric_name, out_coords))
    return names


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def build_superset_score_coords(
    metrics: OrderedDict,
    input_coords_template: CoordSystem,
    times: np.ndarray,
) -> CoordSystem:
    """Build the superset coordinate system for the score zarr store.

    Collects the *union* of all non-variable output dimensions across
    every metric and prepends a ``time`` axis.  This defines the store's
    coordinate axes; individual metric arrays may use any subset.

    Parameters
    ----------
    metrics : OrderedDict[str, Metric]
        Metric instances keyed by name.
    input_coords_template : CoordSystem
        Representative input coords for ``output_coords()`` calls.
    times : np.ndarray
        All initial-condition times that will appear in the store.

    Returns
    -------
    CoordSystem
        ``(time, <union of surviving dims across all metrics>)``
    """
    all_dims: dict[str, np.ndarray] = {}
    for metric in metrics.values():
        out = metric.output_coords(input_coords_template)
        for dim, vals in out.items():
            if dim != "variable" and dim not in all_dims:
                all_dims[dim] = vals

    total: CoordSystem = OrderedDict()
    total["time"] = times
    for dim, vals in all_dims.items():
        if dim != "time":
            total[dim] = vals
    return total


def group_score_arrays_by_dims(
    metrics: OrderedDict,
    input_coords_template: CoordSystem,
    times: np.ndarray,
) -> list[tuple[CoordSystem, list[str]]]:
    """Group score arrays by their output dimension structure.

    Metrics that reduce different dimensions produce arrays with different
    shapes.  This function groups them so each group's arrays share the
    same coordinate axes and can be created in a single
    ``ZarrBackend.add_array`` call.

    Parameters
    ----------
    metrics : OrderedDict[str, Metric]
        Metric instances keyed by name.
    input_coords_template : CoordSystem
        Representative input coords for ``output_coords()`` calls.
    times : np.ndarray
        All initial-condition times for the ``time`` coordinate.

    Returns
    -------
    list[tuple[CoordSystem, list[str]]]
        Each entry is ``(group_coords, variable_names)`` where all
        variables in the group share the same dimension structure.
    """
    groups: dict[tuple[str, ...], tuple[CoordSystem, list[str]]] = {}

    for metric_name, metric in metrics.items():
        out = metric.output_coords(input_coords_template)

        # This metric's store coords: time + non-variable output dims.
        metric_coords: CoordSystem = OrderedDict()
        metric_coords["time"] = times
        for dim, vals in out.items():
            if dim not in ("variable", "time"):
                metric_coords[dim] = vals

        dim_key = tuple(metric_coords.keys())
        var_names = score_variable_names(metric_name, out)

        if dim_key in groups:
            groups[dim_key][1].extend(var_names)
        else:
            groups[dim_key] = (metric_coords, list(var_names))

    return list(groups.values())


def add_score_arrays(
    io: Any,
    array_groups: list[tuple[CoordSystem, list[str]]],
) -> None:
    """Add score arrays to the zarr store, skipping any that already exist.

    Safe to call on every rank via ``run_on_rank0_first`` — rank 0 creates
    the arrays, subsequent ranks find them already present and no-op.

    Parameters
    ----------
    io : ZarrBackend
        The underlying I/O backend (from ``OutputManager.io``).
    array_groups : list[tuple[CoordSystem, list[str]]]
        Groups from :func:`group_score_arrays_by_dims`.
    """
    for group_coords, group_vars in array_groups:
        new_vars = [v for v in group_vars if v not in io]
        if new_vars:
            io.add_array(group_coords, np.array(new_vars))


def validate_lead_time_chunking(
    metrics: OrderedDict,
    chunk_size: int | None,
    n_lead_times: int,
) -> None:
    """Validate that chunking is safe for the configured metrics.

    Chunking over ``lead_time`` produces incorrect results for metrics
    that reduce over it, because partial reductions cannot be combined.

    Parameters
    ----------
    metrics : OrderedDict[str, Metric]
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
        return  # no chunking, nothing to validate

    for name, metric in metrics.items():
        if "lead_time" in metric.reduction_dimensions:
            raise ValueError(
                f"Metric '{name}' reduces over 'lead_time', but "
                f"lead_time_chunk_size={chunk_size} is active.  "
                "Either remove 'lead_time' from reduction_dimensions or "
                "set lead_time_chunk_size to null to disable chunking."
            )


# ---------------------------------------------------------------------------
# Core scoring loop
# ---------------------------------------------------------------------------


def run_scoring(
    my_times: list[np.datetime64],
    prediction_ds: xr.Dataset,
    verif_source: PredownloadedSource,
    metrics: OrderedDict,
    output_mgr: OutputManager,
    variables: list[str],
    lead_times: np.ndarray,
    lead_time_chunks: list[np.ndarray],
    spatial_coords: CoordSystem,
    device: torch.device,
    cfg: DictConfig,
) -> None:
    """Score predictions against verification for assigned IC times.

    For each initial-condition time, loads prediction and verification data
    in lead-time chunks, applies every configured metric, and writes the
    results to the output store.

    Parameters
    ----------
    my_times : list[np.datetime64]
        IC times assigned to this rank.
    prediction_ds : xr.Dataset
        Opened prediction zarr store.
    verif_source : PredownloadedSource
        Verification data source.
    metrics : OrderedDict[str, Metric]
        Metric instances keyed by name.
    output_mgr : OutputManager
        Validated output manager for the score store.
    variables : list[str]
        Variable names to score.
    lead_times : np.ndarray
        Full array of lead-time values (for concatenating chunks).
    lead_time_chunks : list[np.ndarray]
        Partitioned lead-time arrays.
    spatial_coords : CoordSystem
        Spatial coordinate arrays from the prediction store.
    device : torch.device
        Device for tensor computations.
    cfg : DictConfig
        Full Hydra config.
    """
    if not my_times:
        logger.warning("No times assigned for scoring — skipping.")
        return

    resume = cfg.scoring.get("resume", False)
    rank = DistributedManager().rank

    for time in tqdm(my_times, desc="Scoring", disable=rank != 0):
        # Accumulate chunk results per metric.
        chunk_results: dict[str, list[tuple[torch.Tensor, CoordSystem]]] = {
            name: [] for name in metrics
        }

        for lt_chunk in lead_time_chunks:
            x, x_coords = load_prediction_chunk(
                prediction_ds, time, lt_chunk, variables, device
            )
            y, y_coords = load_verification_chunk(
                verif_source,
                time,
                lt_chunk,
                variables,
                spatial_coords,
                device,
            )

            for metric_name, metric in metrics.items():
                if _is_statistic(metric):
                    score_val, score_coords = metric(x, x_coords)
                else:
                    score_val, score_coords = metric(x, x_coords, y, y_coords)
                chunk_results[metric_name].append((score_val.cpu(), score_coords))

        # Concatenate chunks along lead_time and write scores for this time.
        for metric_name, chunks in chunk_results.items():
            score, score_coords = _concat_chunks(chunks, lead_times)

            # Prefix variable names for the zarr store.
            write_coords: CoordSystem = OrderedDict()
            write_coords["time"] = np.array([time])

            for dim, vals in score_coords.items():
                if dim == "variable":
                    write_coords["variable"] = np.array(
                        [f"{metric_name}__{v}" for v in vals]
                    )
                else:
                    write_coords[dim] = vals

            # If metric reduced the variable dim, inject it.
            if "variable" not in write_coords:
                score = score.unsqueeze(-1)
                write_coords["variable"] = np.array([metric_name])

            # Prepend time dimension.
            score = score.unsqueeze(0)
            output_mgr.write(score, write_coords)

        if resume:
            output_mgr.flush()
            write_scoring_marker(time, cfg)

    logger.success("Scoring complete.")


def _concat_chunks(
    chunks: list[tuple[torch.Tensor, CoordSystem]],
    full_lead_times: np.ndarray,
) -> tuple[torch.Tensor, CoordSystem]:
    """Concatenate lead-time chunk results into a single score tensor.

    Parameters
    ----------
    chunks : list[tuple[torch.Tensor, CoordSystem]]
        Per-chunk ``(score, coords)`` pairs from a single metric.
    full_lead_times : np.ndarray
        Complete lead-time array for the full trajectory.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Concatenated score and coords.
    """
    if len(chunks) == 1:
        return chunks[0]

    score_coords = chunks[0][1].copy()

    if "lead_time" in score_coords:
        lt_dim_idx = list(score_coords.keys()).index("lead_time")
        score = torch.cat([c[0] for c in chunks], dim=lt_dim_idx)
        score_coords["lead_time"] = full_lead_times
    else:
        # lead_time was reduced — chunks cannot be combined by concatenation.
        # This path should be blocked by validate_lead_time_chunking().
        score = chunks[-1][0]

    return score, score_coords
