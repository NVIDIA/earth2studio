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

"""Score aggregation and data-loading helpers for report generation."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default lead times for summary tables (as timedelta strings)
_DEFAULT_SUMMARY_LEAD_TIMES = [
    "1 days",
    "3 days",
    "5 days",
    "7 days",
    "10 days",
    "14 days",
]

# Metric names whose proper aggregation is sqrt(mean(x²)) not mean(x).
# This is the legacy path for any RMSE values stored directly.
_QUADRATIC_METRICS = frozenset({"rmse"})

# Metrics that store pre-sqrt values (MSE/variance).  Correct time
# aggregation is sqrt(mean(x)) — take the mean first, then sqrt.
_SQRT_AFTER_MEAN = frozenset({"mse", "ensemble_mean_mse"})

# Human-readable display names for metric keys.
_DISPLAY_NAMES: dict[str, str] = {
    "mse": "RMSE",
    "ensemble_mean_mse": "RMSE (ens. mean)",
    "ensemble_variance": "Ens. Variance",
    "rmse": "RMSE",
    "crps": "CRPS",
    "mae": "MAE",
}


def display_name(metric_name: str) -> str:
    """Return a human-readable label for *metric_name*."""
    return _DISPLAY_NAMES.get(metric_name, metric_name.upper())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SectionOutput:
    """Result from a section renderer."""

    markdown: str
    figures: dict[str, Figure] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scores(cfg: DictConfig) -> xr.Dataset:
    """Open the scores zarr store as an xarray Dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``output.path`` and ``scoring.output.store_name``.

    Returns
    -------
    xr.Dataset

    Raises
    ------
    FileNotFoundError
        If the score store does not exist.
    """
    store_name = cfg.scoring.output.get("store_name", "scores.zarr")
    store_path = os.path.join(cfg.output.path, store_name)
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Score store not found at '{store_path}'.\n"
            "Run scoring (score.py) before generating a report."
        )
    logger.info(f"Opened score store: {store_path}")
    return xr.open_zarr(store_path)


def parse_score_arrays(
    ds: xr.Dataset,
) -> dict[str, list[str]]:
    """Parse score array names into {metric: [variable, ...]} groups.

    Array names follow the ``{metric}__{variable}`` convention.  Arrays
    without ``__`` are treated as single-variable metrics (the metric name
    is the array name, variable is ``None``).

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.

    Returns
    -------
    dict[str, list[str]]
        Mapping from metric name to list of variable names.
    """
    groups: dict[str, list[str]] = {}
    for name in ds.data_vars:
        if "__" in name:
            metric, variable = name.split("__", 1)
        else:
            metric = name
            variable = name
        groups.setdefault(metric, []).append(variable)
    return groups


def _open_data_stores(
    cfg: DictConfig,
) -> tuple[xr.Dataset | None, xr.Dataset | None]:
    """Open prediction and verification zarr stores for visualization.

    Returns ``(None, None)`` if either store is missing — visualization
    sections degrade gracefully.

    Verification store discovery delegates to
    :meth:`src.pipelines.base.Pipeline.verification_zarr_paths` so the
    report inherits pipeline-specific overrides.  When multiple zarrs
    are returned, they are merged by variable with :func:`xarray.merge`;
    all component stores must share ``time`` and spatial coords.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``output.path``.

    Returns
    -------
    tuple[xr.Dataset | None, xr.Dataset | None]
        ``(prediction_ds, verification_ds)``
    """
    # Local imports avoid a circular dependency through src.pipelines → src.report.
    from ..pipelines import build_pipeline
    from ..pipelines.base import default_verification_zarr_paths

    pred_path = os.path.join(cfg.output.path, "forecast.zarr")
    pred_ds = None
    if os.path.exists(pred_path):
        pred_ds = xr.open_zarr(pred_path)
    else:
        logger.info(
            f"Prediction store not found at '{pred_path}' — "
            "visualization sections will be skipped."
        )

    # Honor pipeline-specific overrides of verification_zarr_paths when a
    # pipeline is configured; otherwise use the default discovery inline.
    if cfg.get("pipeline") is not None:
        paths = build_pipeline(cfg).verification_zarr_paths(cfg)
    else:
        paths = default_verification_zarr_paths(cfg)

    verif_ds: xr.Dataset | None
    if not paths:
        verif_ds = None
        logger.info(
            "Verification store not found — visualization sections will be skipped."
        )
    elif len(paths) == 1:
        verif_ds = xr.open_zarr(paths[0])
    else:
        logger.info(
            "Verification: merging per-model stores "
            f"({', '.join(os.path.basename(p) for p in paths)})"
        )
        verif_ds = xr.merge([xr.open_zarr(p) for p in paths])

    return pred_ds, verif_ds


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_over_time(
    da: xr.DataArray,
    metric_name: str,
) -> xr.DataArray:
    """Aggregate a score array over the time dimension.

    Three modes depending on what the metric stores:

    * **Quadratic** (``rmse``): stores RMSE per time → ``sqrt(mean(x²))``
      to recover global RMSE.
    * **Sqrt-after-mean** (``mse``, ``ensemble_mean_mse``): stores MSE
      per time → ``sqrt(mean(x))`` to produce RMSE.
    * **Default**: simple ``mean(x)`` (e.g. CRPS, MAE, variance).

    Parameters
    ----------
    da : xr.DataArray
        Score array with a ``time`` dimension.
    metric_name : str
        Metric name (used to determine aggregation method).

    Returns
    -------
    xr.DataArray
        Time-aggregated score (time dimension removed).
    """
    if metric_name in _SQRT_AFTER_MEAN:
        return np.sqrt(da.mean(dim="time"))
    if metric_name in _QUADRATIC_METRICS:
        return np.sqrt((da**2).mean(dim="time"))
    return da.mean(dim="time")


def aggregate_over_ensemble(
    da: xr.DataArray,
) -> xr.DataArray:
    """Mean over the ensemble dimension if present.

    Parameters
    ----------
    da : xr.DataArray
        Score array, optionally with an ``ensemble`` dimension.

    Returns
    -------
    xr.DataArray
        Ensemble-averaged score.
    """
    if "ensemble" in da.dims:
        return da.mean(dim="ensemble")
    return da


def resolve_time_groups(
    time_groups_cfg: Any,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Match IC times to named groups based on date ranges.

    Parameters
    ----------
    time_groups_cfg : dict
        Mapping of group names to lists of ``{start, end}`` date ranges.
    times : np.ndarray
        Array of datetime64 values from the score store.

    Returns
    -------
    dict[str, np.ndarray]
        Group name to matching time values.
    """
    groups: dict[str, np.ndarray] = {}
    for name, ranges in time_groups_cfg.items():
        mask = np.zeros(len(times), dtype=bool)
        for r in ranges:
            start = np.datetime64(r["start"])
            end = np.datetime64(r["end"])
            mask |= (times >= start) & (times <= end)
        matched = times[mask]
        if len(matched) > 0:
            groups[name] = matched
        else:
            logger.warning(f"Time group '{name}' matched no IC times — skipping.")
    return groups


def snapshot_at_lead_times(
    ds: xr.Dataset,
    metric_groups: dict[str, list[str]],
    lead_times: list[str],
) -> pd.DataFrame:
    """Extract scores at specific lead times into a summary DataFrame.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` from :func:`parse_score_arrays`.
    lead_times : list[str]
        Lead time strings to snapshot (e.g. ``["1 days", "5 days"]``).

    Returns
    -------
    pd.DataFrame
        Columns: ``metric, variable, lead_time, value``.
    """
    rows: list[dict[str, Any]] = []
    # Convert via ``to_timedelta64`` to land on a concrete ns-unit numpy scalar;
    # ``np.timedelta64(pd.Timedelta(...))`` can return the pandas subclass and
    # the downstream ``in`` / ``sel`` comparisons then silently miss for short
    # units like ``"60 min"`` that parse to an object numpy doesn't equate to
    # the zarr's ``timedelta64[ns]`` values.
    target_lts = [pd.Timedelta(lt).to_timedelta64() for lt in lead_times]

    for metric_name, variables in metric_groups.items():
        for var in variables:
            array_name = f"{metric_name}__{var}"
            if array_name not in ds:
                continue
            da = aggregate_over_ensemble(ds[array_name])
            da = aggregate_over_time(da, metric_name)

            if "lead_time" not in da.dims:
                continue
            da_lts = np.asarray(da.lead_time.values).astype("timedelta64[ns]")

            for lt, lt_str in zip(target_lts, lead_times):
                lt_ns = lt.astype("timedelta64[ns]")
                if not (da_lts == lt_ns).any():
                    continue
                val = float(da.sel(lead_time=lt_ns).values)
                rows.append(
                    {
                        "metric": metric_name,
                        "variable": var,
                        "lead_time": lt_str,
                        "value": val,
                    }
                )

    return pd.DataFrame(rows)


def build_summary_csv(
    ds: xr.Dataset,
    metric_groups: dict[str, list[str]],
    run_id: str,
) -> pd.DataFrame:
    """Build the full aggregation CSV with all lead times.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset.
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` from :func:`parse_score_arrays`.
    run_id : str
        Run/model identifier for the ``model`` column.

    Returns
    -------
    pd.DataFrame
        Columns: ``model, metric, variable, lead_time, value``.
    """
    rows: list[dict[str, Any]] = []

    for metric_name, variables in metric_groups.items():
        for var in variables:
            array_name = f"{metric_name}__{var}"
            if array_name not in ds:
                continue
            da = aggregate_over_ensemble(ds[array_name])
            da = aggregate_over_time(da, metric_name)

            if "lead_time" not in da.dims:
                rows.append(
                    {
                        "model": run_id,
                        "metric": metric_name,
                        "variable": var,
                        "lead_time": "aggregate",
                        "value": float(da.values),
                    }
                )
                continue

            for lt in da.lead_time.values:
                val = float(da.sel(lead_time=lt).values)
                rows.append(
                    {
                        "model": run_id,
                        "metric": metric_name,
                        "variable": var,
                        "lead_time": str(pd.Timedelta(lt)),
                        "value": val,
                    }
                )

    return pd.DataFrame(rows)


def compute_spread_skill(
    ds: xr.Dataset,
    mse_metric: str,
    variance_metric: str,
    variables: list[str],
    time_sel: np.ndarray | None = None,
) -> dict[str, xr.Dataset]:
    """Compute WB2-correct RMSE, Spread, and Spread-Skill Ratio.

    Parameters
    ----------
    ds : xr.Dataset
        Score dataset with MSE and variance arrays.
    mse_metric : str
        Config key used for the ensemble-mean MSE metric
        (e.g. ``"ensemble_mean_mse"``).
    variance_metric : str
        Config key used for the ensemble variance metric
        (e.g. ``"ensemble_variance"``).
    variables : list[str]
        Variable names to compute over.
    time_sel : np.ndarray | None
        Optional time subset for seasonal / grouped analysis.

    Returns
    -------
    dict[str, xr.Dataset]
        Keys: ``"rmse"``, ``"spread"``, ``"ratio"`` — each an
        ``xr.Dataset`` with one ``DataArray`` per variable.
    """
    rmse_arrays: dict[str, xr.DataArray] = {}
    spread_arrays: dict[str, xr.DataArray] = {}
    ratio_arrays: dict[str, xr.DataArray] = {}

    for var in variables:
        mse_name = f"{mse_metric}__{var}"
        var_name = f"{variance_metric}__{var}"
        if mse_name not in ds or var_name not in ds:
            continue

        mse_da = ds[mse_name]
        var_da = ds[var_name]
        if time_sel is not None:
            mse_da = mse_da.sel(time=time_sel)
            var_da = var_da.sel(time=time_sel)

        rmse_vals = np.sqrt(mse_da.mean(dim="time"))
        spread_vals = np.sqrt(var_da.mean(dim="time"))

        rmse_arrays[var] = rmse_vals
        spread_arrays[var] = spread_vals
        ratio_arrays[var] = spread_vals / rmse_vals

    return {
        "rmse": xr.Dataset(rmse_arrays),
        "spread": xr.Dataset(spread_arrays),
        "ratio": xr.Dataset(ratio_arrays),
    }


def _resolve_variable_groups(
    report_cfg: DictConfig,
    metric_groups: dict[str, list[str]],
) -> OrderedDict[str, list[str]]:
    """Resolve variable groups from config, or fall back to one plot per variable.

    When ``variable_groups`` is configured, variables that appear in a group
    share a plot.  Variables present in the score store but not listed in any
    group each get their own single-variable plot.

    When ``variable_groups`` is not configured, every variable gets its own
    plot (avoids mixing incompatible physical scales).

    Parameters
    ----------
    report_cfg : DictConfig
        Report config (may contain ``variable_groups``).
    metric_groups : dict[str, list[str]]
        ``{metric: [variable, ...]}`` from score arrays.

    Returns
    -------
    OrderedDict[str, list[str]]
        Named variable groups.
    """
    all_vars = list(dict.fromkeys(v for vs in metric_groups.values() for v in vs))

    if "variable_groups" in report_cfg:
        groups: OrderedDict[str, list[str]] = OrderedDict(
            OmegaConf.to_container(report_cfg.variable_groups, resolve=True)
        )
        # Variables not in any explicit group get their own plot.
        grouped = {v for vs in groups.values() for v in vs}
        for v in all_vars:
            if v not in grouped:
                groups[v] = [v]
        return groups

    # Fallback: one plot per variable
    return OrderedDict((v, [v]) for v in all_vars)
