# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any

import numpy as np
import pandas as pd
from data_handling import (
    compute_averages_of_errors_over_lead_time,
    extract_tracks,
    extract_tracks_from_file,
    get_ensemble_averages,
    match_tracks,
)
from plotting_helpers import (
    plot_errors_over_lead_time,
    plot_extreme_extremes_histograms,
    plot_ib_era5,
    plot_over_time,
    plot_relative_over_time,
    plot_spaghetti,
)
from tqdm import tqdm

_DEFAULT_TIME_STEP = np.timedelta64(6, "h")


def load_tracks(
    case: str,
    pred_track_dir: str,
    tru_track_dir: str,
    out_dir: str | None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any], int, str | None]:
    """Load predicted and reference tracks for a named storm.

    Parameters
    ----------
    case : str
        Storm identifier in the format ``{name}_{year}_{basin}``.
    pred_track_dir : str
        Directory containing predicted track CSV files.
    tru_track_dir : str
        Directory containing the reference track CSV file.
    out_dir : str | None
        Base output directory for plots.  A case-specific sub-directory
        is created automatically.
    time_step : np.timedelta64, optional
        Model time step, by default 6 h.

    Returns
    -------
    tuple
        ``(tru_track, pred_tracks, ens_mean, n_members, out_dir)``
    """
    tru_track = extract_tracks_from_file(
        os.path.join(tru_track_dir, f"reference_track_{case}.csv")
    )
    tru_track["dist"] = np.zeros(len(tru_track))

    pred_tracks = extract_tracks(in_dir=os.path.join(pred_track_dir))
    n_members = len(pred_tracks)

    pred_tracks = match_tracks(pred_tracks, tru_track)

    if out_dir:
        out_dir = os.path.join(out_dir, case)
        os.makedirs(out_dir, exist_ok=True)

    ens_mean = get_ensemble_averages(
        pred_tracks=pred_tracks, tru_track=tru_track, time_step=time_step
    )

    return tru_track, pred_tracks, ens_mean, n_members, out_dir


def analyse_individual_storms(
    cases: str | list[str],
    pred_track_dir: str,
    tru_track_dir: str,
    out_path: str | None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> None:
    """Generate a full set of analysis plots for each storm individually.

    Parameters
    ----------
    cases : str | list[str]
        Storm identifier(s).
    pred_track_dir : str
        Directory containing predicted track CSV files.
    tru_track_dir : str
        Directory containing reference track CSV files.
    out_path : str | None
        Base output directory for plots.
    time_step : np.timedelta64, optional
        Model time step, by default 6 h.
    """
    if isinstance(cases, str):
        cases = [cases]

    for case in tqdm(cases):
        tru_track, pred_tracks, ens_mean, n_members, out_dir = load_tracks(
            case=case,
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_dir=out_path,
            time_step=time_step,
        )

        plot_spaghetti(
            true_track=tru_track,
            pred_tracks=pred_tracks,
            ensemble_mean=ens_mean["mean"],
            case=case,
            n_members=n_members,
            out_dir=out_dir,
        )

        plot_over_time(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            ensemble_mean=ens_mean,
            case=case,
            n_members=n_members,
            out_dir=out_dir,
            time_step=time_step,
        )

        plot_relative_over_time(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            ensemble_mean=ens_mean,
            case=case,
            n_members=n_members,
            out_dir=out_dir,
            time_step=time_step,
        )

        plot_ib_era5(
            tru_track=tru_track,
            case=case,
            variables=["msl", "wind_speed"],
            out_dir=out_dir,
        )

        plot_extreme_extremes_histograms(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            ensemble_mean=ens_mean,
            case=case,
            out_dir=out_dir,
        )

        err_dict, _ = compute_averages_of_errors_over_lead_time(
            pred_tracks=pred_tracks,
            tru_track=tru_track,
            variables=["wind_speed", "msl", "dist"],
        )

        plot_errors_over_lead_time(
            err_dict=err_dict,
            case=case,
            ic=pred_tracks[0]["ic"],
            n_members=n_members,
            n_tracks=len(pred_tracks),
            out_dir=out_dir,
            time_step=time_step,
        )


def stack_metrics(err_dict: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    """Stack per-variable, per-metric arrays into a single 3-D array.

    Parameters
    ----------
    err_dict : dict[str, dict[str, np.ndarray]]
        Per-variable error metrics.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_vars, n_metrics, lead_time)``.
    """
    var_errs = []
    for var in err_dict.keys():
        metrics = np.stack([err_dict[var][metric] for metric in err_dict[var]], axis=0)
        var_errs.append(metrics)

    return np.stack(var_errs, axis=0)


def stack_cases(storm_metrics: dict[str, Any], max_len: int) -> dict[str, Any]:
    """Pad per-storm metric and weight arrays to *max_len* and stack them.

    Parameters
    ----------
    storm_metrics : dict[str, Any]
        Accumulator with ``"data"`` (list of 3-D arrays) and ``"weights"``
        (list of 1-D arrays).
    max_len : int
        Target lead-time dimension (shorter storms are NaN-padded for
        ``"data"`` and zero-padded for ``"weights"``).

    Returns
    -------
    dict[str, Any]
        Updated *storm_metrics* with ``"data"`` as a 4-D ``np.ndarray``
        of shape ``(n_cases, n_vars, n_metrics, max_len)`` and
        ``"weights"`` as a 2-D ``np.ndarray`` of shape
        ``(n_cases, max_len)``.
    """
    for ii in range(len(storm_metrics["case"])):
        storm_metrics["data"][ii] = np.pad(
            storm_metrics["data"][ii],
            pad_width=(
                (0, 0),
                (0, 0),
                (0, max_len - storm_metrics["data"][ii].shape[-1]),
            ),
            mode="constant",
            constant_values=np.nan,
        )
        storm_metrics["weights"][ii] = np.pad(
            storm_metrics["weights"][ii],
            pad_width=(0, max_len - storm_metrics["weights"][ii].shape[-1]),
            mode="constant",
            constant_values=0,
        )

    storm_metrics["data"] = np.stack(storm_metrics["data"], axis=0)
    storm_metrics["weights"] = np.stack(storm_metrics["weights"], axis=0).astype(int)

    should_shape = (
        len(storm_metrics["case"]),
        len(storm_metrics["var"]),
        len(storm_metrics["metric"]),
        max_len,
    )
    if storm_metrics["data"].shape != should_shape:
        raise ValueError(
            f"shapes not matching when stacking cases: "
            f'{storm_metrics["data"].shape=} {should_shape=}'
        )

    return storm_metrics


def get_individual_storm_metrics(
    cases: list[str],
    pred_track_dir: str,
    tru_track_dir: str,
    out_path: str | None,
    variables: list[str] | None = None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    """Compute per-storm error metrics and collect ensemble averages and extremes.

    Parameters
    ----------
    cases : list[str]
        Storm identifiers.
    pred_track_dir : str
        Directory containing predicted track CSV files.
    tru_track_dir : str
        Directory containing reference track CSV files.
    out_path : str | None
        Base output directory for plots.
    variables : list[str] | None, optional
        Variable names to evaluate, by default
        ``["wind_speed", "msl", "dist"]``
    time_step : np.timedelta64, optional
        Model time step used for the lead-time axis, by default 6 h.

    Returns
    -------
    tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]
        ``(storm_metrics, max_len, ensemble_averages, extremes)``
    """
    if variables is None:
        variables = ["wind_speed", "msl", "dist"]

    storm_metrics: dict[str, Any] = {
        "case": [],
        "var": None,
        "metric": None,
        "lead time": None,
        "data": [],
        "weights": [],
    }
    max_len: int = 0
    ensemble_averages: dict[str, Any] = {}
    extremes: dict[str, Any] = {}
    for case in tqdm(cases, desc="loading storm data"):
        tru_track, pred_tracks, ens_mean, _n_members, _out_dir = load_tracks(
            case=case,
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_dir=out_path,
            time_step=time_step,
        )
        ensemble_averages[case] = ens_mean

        err_dict, _max_len = compute_averages_of_errors_over_lead_time(
            pred_tracks=pred_tracks, tru_track=tru_track, variables=variables
        )

        # Strip per-storm extremes and ensemble-member counts from the metric
        # axis so only the real error metrics remain.  ``n_members`` is the
        # same array for every variable, so we read it once per storm.
        extremes[case] = {}
        storm_weights: np.ndarray | None = None
        for var in variables:
            extremes[case][var] = {}
            for ext, npfun in zip(["min", "max"], [np.nanmin, np.nanmax]):
                extremes[case][var][ext + "_pred"] = err_dict[var].pop(ext)
                extremes[case][var][ext + "_tru"] = npfun(tru_track[var])
            counts = err_dict[var].pop("n_members")
            if storm_weights is None:
                storm_weights = np.nan_to_num(counts, nan=0).astype(int)

        max_len = max(max_len, _max_len)
        storm_metrics["case"].append(case)
        storm_metrics["data"].append(stack_metrics(err_dict))
        storm_metrics["weights"].append(storm_weights)

    storm_metrics["var"] = list(err_dict.keys())
    storm_metrics["metric"] = list(err_dict[list(err_dict.keys())[0]].keys())
    storm_metrics["lead time"] = np.arange(max_len) * time_step

    return storm_metrics, max_len, ensemble_averages, extremes


def reduce_over_all_storms(
    storm_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Average error metrics across all storms.

    Parameters
    ----------
    storm_metrics : dict[str, Any]
        Stacked storm metrics with ``"weights"`` key.

    Returns
    -------
    dict[str, Any]
        Per-variable metrics averaged over storms, plus aggregate
        ``"n_members"`` counts.
    """
    ensemble_metrics: dict[str, Any] = {}
    for var in storm_metrics["var"]:
        ensemble_metrics[var] = {}
        var_idx = storm_metrics["var"].index(var)
        for metric in storm_metrics["metric"]:
            met_idx = storm_metrics["metric"].index(metric)
            ensemble_metrics[var][metric] = np.nanmean(
                storm_metrics["data"][:, var_idx, met_idx, :], axis=0
            )

    ensemble_metrics["n_members"] = np.sum(storm_metrics["weights"], axis=0)

    return ensemble_metrics


def analyse_ensemble_of_storms(
    cases: list[str],
    pred_track_dir: str,
    tru_track_dir: str,
    out_path: str | None,
    time_step: np.timedelta64 = _DEFAULT_TIME_STEP,
) -> dict[str, Any]:
    """Compute and aggregate error metrics across multiple storms.

    Parameters
    ----------
    cases : list[str]
        Storm identifiers.
    pred_track_dir : str
        Directory containing predicted track CSV files.
    tru_track_dir : str
        Directory containing reference track CSV files.
    out_path : str | None
        Base output directory for plots.
    time_step : np.timedelta64, optional
        Model time step used for the lead-time axis, by default 6 h.

    Returns
    -------
    dict[str, Any]
        Ensemble-aggregated error metrics.
    """
    storm_metrics, max_len, _ens_means, _extremes = get_individual_storm_metrics(
        cases, pred_track_dir, tru_track_dir, out_path, time_step=time_step
    )

    storm_metrics = stack_cases(storm_metrics, max_len)

    ensemble_metrics = reduce_over_all_storms(storm_metrics)

    return ensemble_metrics


def analyse_n_plot_tracks() -> None:
    """Entry point for batch analysis of multiple storm cases."""
    cases = [
        "amphan_2020_north_indian",  # 00
        "beryl_2024_north_atlantic",  # 01
        "debbie_2017_southern_pacific",  # 02
        "dorian_2019_north_atlantic",  # 03
        "harvey_2017_north_atlantic",  # 04
        "hato_2017_west_pacific",  # 05
        "helene_2024_north_atlantic",  # 06
        "ian_2022_north_atlantic",  # 07
        "iota_2020_north_atlantic",  # 08
        "irma_2017_north_atlantic",  # 09
        "lan_2017_west_pacific",  # 10
        "lee_2023_north_atlantic",  # 11
        "lorenzo_2019_north_atlantic",  # 12
        "maria_2017_north_atlantic",  # 13
        "mawar_2023_west_pacific",  # 14
        "michael_2018_north_atlantic",  # 15
        "milton_2024_north_atlantic",  # 16
        "ophelia_2017_north_atlantic",  # 17
        "yagi_2024_west_pacific",  # 18
    ]

    # case_selection = list(range(len(cases)))
    case_selection = [6, 13]
    individual_storms = False
    ensemble_of_storms = True
    time_step = _DEFAULT_TIME_STEP

    pred_track_dir = "/path/to/predictions/cyclone_tracks_te"
    tru_track_dir = "/path/to/reference_tracks"
    out_dir = "./plots"

    if individual_storms:
        analyse_individual_storms(
            cases=[cases[ii] for ii in case_selection],
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_path=out_dir,
            time_step=time_step,
        )

    if ensemble_of_storms:
        analyse_ensemble_of_storms(
            cases=[cases[ii] for ii in case_selection],
            pred_track_dir=pred_track_dir,
            tru_track_dir=tru_track_dir,
            out_path=out_dir,
            time_step=time_step,
        )


if __name__ == "__main__":
    analyse_n_plot_tracks()
