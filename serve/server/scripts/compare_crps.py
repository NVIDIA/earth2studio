#!/usr/bin/env python3
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

"""Compare two ensemble forecast zarr outputs using CRPS against GFS verification.

Scores both forecasts against the same GFS ground truth and asserts their CRPS
values are within a configurable threshold of each other. Designed to verify that
two different inference systems (e.g. ferroflux 8-GPU vs python-serve 1-GPU)
produce statistically equivalent ensemble forecasts.

Usage
-----
    python compare_crps.py --forecast-a /path/to/ferroflux/forecast.zarr \\
                           --forecast-b /path/to/python_serve/forecast.zarr \\
                           --variables t2m,z500,tcwv \\
                           --threshold 0.01

Expected zarr layout
--------------------
Both zarrs must have dimensions: (time, ensemble, lead_time, <variables as data vars>, lat, lon).
The ``time`` dimension holds IC (initial condition) timestamps. The ``lead_time``
dimension holds forecast horizons as timedelta values. Each variable (e.g. t2m)
is a separate data variable in the dataset.

Testing
-------
Requires physicsnemo and earth2studio installed with the statistics extra.
Run from the repository root::

    python -m pytest test/serve/server/test_compare_crps.py -v
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data import GFS
from earth2studio.statistics import crps
from earth2studio.statistics.weights import lat_weight

warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")

_NON_SPATIAL = frozenset({"ensemble", "time", "lead_time", "batch"})

CoordSystem = OrderedDict


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
    source: Any,
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
    source
        Verification data source (callable accepting times and variables).
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

    # Align verification grid to the prediction's spatial coordinates.
    spatial_dims = [d for d in da.dims if d not in {"time", "variable"}]
    sel_kwargs = {
        dim: xr.DataArray(spatial_coords[dim], dims=[dim])
        for dim in spatial_dims
        if dim in spatial_coords
    }
    if sel_kwargs:
        da = da.sel(**sel_kwargs, method="nearest")

    dim_order = ["time", "variable"] + spatial_dims
    da = da.transpose(*dim_order)

    tensor = torch.from_numpy(da.values.copy()).to(device=device, dtype=torch.float32)

    coords: CoordSystem = OrderedDict()
    coords["lead_time"] = lead_times
    coords["variable"] = np.array(variables)
    for dim in spatial_dims:
        coords[dim] = spatial_coords[dim]
    return tensor, coords


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CRPS comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare two ensemble forecast zarrs via CRPS against GFS.",
    )
    parser.add_argument(
        "--forecast-a",
        required=True,
        help="Path to first forecast zarr store (e.g. ferroflux output)",
    )
    parser.add_argument(
        "--forecast-b",
        required=True,
        help="Path to second forecast zarr store (e.g. python-serve output)",
    )
    parser.add_argument(
        "--variables",
        default=None,
        help="Comma-separated list of variables to score (e.g. t2m,z500,tcwv). "
        "If omitted, uses the intersection of variables present in both zarrs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Max allowable relative CRPS difference as a fraction "
        "(default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--ensemble-dim",
        default="ensemble",
        help="Name of ensemble dimension in zarrs (default: ensemble)",
    )
    parser.add_argument(
        "--lead-time-chunk-size",
        type=int,
        default=1,
        help="Number of lead times per CRPS batch (default: 1)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache",
        default=True,
        help="Disable GFS download caching (caching is enabled by default)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for computation (default: cuda)",
    )
    return parser.parse_args()


def assert_compatible(ds_a: xr.Dataset, ds_b: xr.Dataset) -> None:
    """Validate both zarrs cover the same time and lead_time ranges."""
    times_a = set(ds_a.time.values)
    times_b = set(ds_b.time.values)
    common_times = times_a & times_b
    if not common_times:
        raise ValueError(
            "Forecast zarrs have no overlapping IC times.\n"
            f"  A: {sorted(times_a)[:3]}...\n"
            f"  B: {sorted(times_b)[:3]}..."
        )
    if times_a != times_b:
        print(
            f"WARNING: zarrs have different time coverage. "
            f"Scoring {len(common_times)} common times "
            f"(A has {len(times_a)}, B has {len(times_b)}).",
            file=sys.stderr,
        )

    lt_a = ds_a.lead_time.values
    lt_b = ds_b.lead_time.values
    if not np.array_equal(lt_a, lt_b):
        raise ValueError(
            "Forecast zarrs have different lead_time values.\n"
            f"  A: {lt_a[:5]}...\n"
            f"  B: {lt_b[:5]}..."
        )


def build_metric(
    ensemble_dim: str,
    spatial_coords: OrderedDict,
) -> crps:
    """Instantiate a CRPS metric with cosine latitude weighting."""
    reduction_dims = ["lat", "lon"]
    lat_vals = spatial_coords["lat"]
    weights_lat = lat_weight(torch.tensor(lat_vals, dtype=torch.float32))
    lon_size = len(spatial_coords["lon"])
    weights = weights_lat.unsqueeze(-1).expand(-1, lon_size)

    return crps(
        ensemble_dimension=ensemble_dim,
        reduction_dimensions=reduction_dims,
        weights=weights,
    )


def score_forecast(
    prediction_ds: xr.Dataset,
    verif_source: GFS,
    variables: list[str],
    lead_times: np.ndarray,
    lead_time_chunks: list[np.ndarray],
    spatial_coords: OrderedDict,
    device: torch.device,
    metric: crps,
) -> np.ndarray:
    """Score a forecast zarr against verification. Returns (n_times, n_lead_times, n_vars) CRPS."""
    times = sorted(prediction_ds.time.values)
    n_times = len(times)
    n_lead_times = len(lead_times)
    n_vars = len(variables)

    scores = np.full((n_times, n_lead_times, n_vars), np.nan, dtype=np.float32)

    for t_idx, time in enumerate(times):
        lt_offset = 0
        for lt_chunk in lead_time_chunks:
            x, x_coords = load_prediction_chunk(
                prediction_ds, time, lt_chunk, variables, device
            )
            y, y_coords = load_verification_chunk(
                verif_source, time, lt_chunk, variables, spatial_coords, device
            )
            score_val, score_coords = metric(x, x_coords, y, y_coords)

            chunk_scores = score_val.cpu().numpy()
            chunk_len = len(lt_chunk)
            scores[t_idx, lt_offset : lt_offset + chunk_len, :] = chunk_scores
            lt_offset += chunk_len

        print(f"  Scored IC time {t_idx + 1}/{n_times}: {time}")

    return scores


def report_results(
    crps_a: np.ndarray,
    crps_b: np.ndarray,
    variables: list[str],
    lead_times: np.ndarray,
    threshold: float,
) -> bool:
    """Print comparison table and return True if all relative diffs are within threshold."""
    mean_a = np.nanmean(crps_a, axis=0)  # (n_lead_times, n_vars)
    mean_b = np.nanmean(crps_b, axis=0)

    all_pass = True
    max_rel_diff = 0.0
    _EPS = 1e-12

    print("\n" + "=" * 80)
    print("CRPS COMPARISON REPORT")
    print("=" * 80)
    print(
        f"{'Variable':<12} {'Lead Time':<14} {'CRPS A':>10} {'CRPS B':>10} "
        f"{'Rel Diff%':>10} {'Status':>8}"
    )
    print("-" * 80)

    for v_idx, var in enumerate(variables):
        for lt_idx, lt in enumerate(lead_times):
            a_val = float(mean_a[lt_idx, v_idx])
            b_val = float(mean_b[lt_idx, v_idx])
            lt_hours = lt / np.timedelta64(1, "h")

            if np.isnan(a_val) or np.isnan(b_val):
                all_pass = False
                print(
                    f"{var:<12} {lt_hours:>6.0f}h       "
                    f"{a_val:>10.4f} {b_val:>10.4f}       NaN     FAIL"
                )
                continue

            denom = max(a_val, b_val)
            if denom < _EPS:
                rel_diff = 0.0
            else:
                rel_diff = abs(a_val - b_val) / denom

            max_rel_diff = max(max_rel_diff, rel_diff)
            status = "PASS" if rel_diff <= threshold else "FAIL"
            if rel_diff > threshold:
                all_pass = False

            print(
                f"{var:<12} {lt_hours:>6.0f}h       "
                f"{a_val:>10.4f} {b_val:>10.4f} {rel_diff * 100:>9.4f}% {status:>8}"
            )

    print("-" * 80)
    print(f"Max relative CRPS difference: {max_rel_diff * 100:.4f}%")
    print(f"Threshold:                    {threshold * 100:.4f}%")
    print(f"Result:                       {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)

    return all_pass


def main() -> None:
    """Entry point: score two forecast zarrs against GFS and compare CRPS."""
    args = parse_args()
    device = torch.device(args.device)

    print(f"Opening forecast A: {args.forecast_a}")
    ds_a = xr.open_zarr(args.forecast_a)
    print(f"Opening forecast B: {args.forecast_b}")
    ds_b = xr.open_zarr(args.forecast_b)

    assert_compatible(ds_a, ds_b)

    if args.variables:
        variables = [v.strip() for v in args.variables.split(",")]
    else:
        coord_dims = set(ds_a.dims) | set(ds_b.dims)
        vars_a = set(ds_a.data_vars) - coord_dims
        vars_b = set(ds_b.data_vars) - coord_dims
        variables = sorted(vars_a & vars_b)
        if not variables:
            print(
                "ERROR: No common variables found between zarrs.\n"
                f"  A has: {sorted(vars_a)}\n"
                f"  B has: {sorted(vars_b)}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected {len(variables)} common variables: {variables}")

    common_times = sorted(set(ds_a.time.values) & set(ds_b.time.values))
    ds_a = ds_a.sel(time=common_times)
    ds_b = ds_b.sel(time=common_times)

    lead_times = ds_a.lead_time.values
    lead_time_chunks = build_lead_time_chunks(lead_times, args.lead_time_chunk_size)
    spatial_coords = spatial_coords_from_dataset(ds_a)

    print(f"Variables: {variables} ({len(variables)} total, scored one at a time)")
    print(f"IC times: {len(common_times)}")
    print(f"Lead times: {len(lead_times)} steps")
    print(f"Ensemble dim: {args.ensemble_dim}")
    print(f"Threshold: {args.threshold * 100:.2f}%")

    metric = build_metric(args.ensemble_dim, spatial_coords)
    gfs = GFS(cache=args.cache)

    n_times = len(common_times)
    n_lead_times = len(lead_times)
    n_vars = len(variables)
    crps_a = np.full((n_times, n_lead_times, n_vars), np.nan, dtype=np.float32)
    crps_b = np.full((n_times, n_lead_times, n_vars), np.nan, dtype=np.float32)

    for v_idx, var in enumerate(variables):
        print(f"\n[{v_idx + 1}/{n_vars}] Scoring variable: {var}")

        print("  Forecast A...")
        var_scores_a = score_forecast(
            ds_a,
            gfs,
            [var],
            lead_times,
            lead_time_chunks,
            spatial_coords,
            device,
            metric,
        )
        crps_a[:, :, v_idx] = var_scores_a[:, :, 0]

        print("  Forecast B...")
        var_scores_b = score_forecast(
            ds_b,
            gfs,
            [var],
            lead_times,
            lead_time_chunks,
            spatial_coords,
            device,
            metric,
        )
        crps_b[:, :, v_idx] = var_scores_b[:, :, 0]

    all_pass = report_results(crps_a, crps_b, variables, lead_times, args.threshold)

    if not all_pass:
        print("\nFAILED: CRPS difference exceeds threshold.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nPASSED: Both systems produce equivalent CRPS scores.")


if __name__ == "__main__":
    main()
