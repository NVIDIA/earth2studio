# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os

import hydra
import numpy as np
import torch
import xarray as xr
from AI_WQ_package.forecast_evaluation import conditional_obs_probs, work_out_RPSS
from AI_WQ_package.retrieve_evaluation_data import retrieve_land_sea_mask
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.data import DataSource
from earth2studio.data.utils import fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.utils.coords import CoordSystem
from src.aiwq_utilities import (
    convert_to_quintile_probs,
    get_quintile_clim,
    get_verif_data,
)
from src.s2s_utilities import build_io_dict, run_with_rank_ordered_execution


def initialize(cfg: DictConfig) -> tuple[dict, dict, dict, DataSource]:
    """Initialize the scoring system

    Parameters
    ----------
    cfg : DictConfig
        The configuration object

    Returns
    -------
    io_dict : dict
        The IO backend for loading forecast data
    metric_dict : dict
        The dictionary of metric objects to use for scoring
    score_io_dict : dict
        The IO backend for writing scores
    """

    dist = DistributedManager()

    # Initialize the IO backend to load forecast data
    if "file_output" in cfg and "cropboxes" in cfg["file_output"]:
        region_keys = cfg["file_output"]["cropboxes"].keys()
    else:
        region_keys = ["global"]

    io_dict = build_io_dict(cfg, region_keys, create_store=False)
    metric_dict = {
        m: hydra.utils.instantiate(cfg.scoring.metrics[m]) for m in cfg.scoring.metrics
    }

    # Initialize the IO backend to write scores
    score_io_dict = run_with_rank_ordered_execution(
        prepare_score_io_dict, cfg, io_dict, create_store=(dist.rank == 0)
    )

    # get data source
    data_source = hydra.utils.instantiate(cfg.data_source)

    return io_dict, metric_dict, score_io_dict, data_source


def prepare_score_io_dict(
    cfg: DictConfig, io_dict: dict, create_store: bool = False
) -> dict:
    """Prepare the IO backend to write scores

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    io_dict : dict
        The IO backend for loading forecast data
    create_store : bool, optional
        Whether to create a new store for the IO backend, or use an existing one. Defaults to False.

    Returns
    -------
    score_io_dict : dict
        The IO backend for writing scores
    """
    # Initialize the IO backend to write scores
    score_io_dict = build_io_dict(
        cfg, io_dict.keys(), create_store=create_store, file_name="score"
    )

    # Populate the IO backend with the metrics and variables being scored
    metrics = [m for m in cfg.scoring.metrics]
    variables = cfg.scoring.variables
    array_list = [m + "_" + v for m in metrics for v in variables]

    for k in io_dict.keys():

        # Prepare output score coordinates
        score_coords = io_dict[k].coords.copy()
        score_coords.pop("ensemble")
        for dim in ["time", "lead_time", "lat", "lon"]:
            # Manually reset dimension order as io_dict[k].coords does not preserve order
            score_coords.move_to_end(dim, last=True)

        lead_times = score_coords["lead_time"]
        score_coords["lead_time"] = lead_times[lead_times >= 0]
        if "temporal_aggregation" in cfg.scoring:
            if cfg.scoring.temporal_aggregation == "weekly":
                score_coords["lead_time"] = np.array(
                    sorted(
                        list(set(score_coords["lead_time"].astype("timedelta64[W]")))
                    )
                ).astype("timedelta64[ns]")
            else:
                raise ValueError(
                    f"Temporal aggregation {cfg.scoring.temporal_aggregation} not supported"
                )

        if create_store:
            for m in metrics:
                for v in variables:
                    write_coords = score_coords.copy()
                    if "reduction_dimensions" in cfg.scoring.metrics[m]:
                        for dim in cfg.scoring.metrics[m].reduction_dimensions:
                            write_coords.pop(dim)
                    score_io_dict[k].add_array(write_coords, m + "_" + v)
            if "aiwq" in cfg.scoring:
                for v in cfg.scoring.aiwq.variables:
                    write_coords = score_coords.copy()
                    for d in ["lead_time", "lat", "lon"]:
                        write_coords.pop(d)
                    score_io_dict[k].add_array(write_coords, "rpss_wk3_" + v)
                    score_io_dict[k].add_array(write_coords, "rpss_wk4_" + v)
        else:
            for a in array_list:
                if a not in score_io_dict[k]:
                    raise ValueError(
                        f"Array {a} not found in initialized {k} IO backend"
                    )

    return score_io_dict


def distribute_ics(all_ics: np.ndarray) -> np.ndarray:
    """Distribute the initial conditions to score across ranks

    Parameters
    ----------
    all_ics : np.ndarray
        The full set of initial conditions to score

    Returns
    -------
    ics_to_score : np.ndarray
        The initial conditions to score for the current rank
    """
    dist = DistributedManager()

    ics_per_rank = len(all_ics) // dist.world_size
    ics_to_score = all_ics[dist.rank * ics_per_rank : (dist.rank + 1) * ics_per_rank]
    return ics_to_score


@torch.inference_mode()
def run_scoring(
    cfg: DictConfig,
    io_dict: dict,
    metric_dict: dict,
    score_io_dict: dict,
    data_source: DataSource,
) -> None:
    """Run the scoring system on the supplied configuration

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    io_dict : dict
        The IO backend for loading forecast data
    metric_dict : dict
        The dictionary of metric objects to use for scoring
    score_io_dict : dict
        The IO backend for writing scores
    data_source : DataSource
        The data source for loading verification data

    Returns
    -------
    None
    """

    dist = DistributedManager()
    device = dist.device

    for k in io_dict.keys():
        all_ics = io_dict[k].coords["time"]
        ics_to_score = distribute_ics(all_ics)

        score_coords = score_io_dict[k].coords.copy()
        io_backend = io_dict[k]

        for ic in ics_to_score:

            for var in cfg.scoring.variables:

                # Prepare evaluation data
                fcst_data, fcst_coords = load_forecast_data(
                    cfg,
                    io_backend,
                    ic,
                    array=var,
                    score_coords=score_coords,
                    device=device,
                )
                verif_data, verif_coords = load_verification_data(
                    cfg,
                    data_source,
                    ic,
                    fcst_coords=io_backend.coords,
                    variable=var,
                    score_coords=score_coords,
                    device=device,
                )

                # Score the forecast
                scores = score_forecast(
                    fcst_data, fcst_coords, verif_data, verif_coords, metric_dict, var
                )

                # Write results
                write_scores(score_io_dict, scores)

            # If initial condition is on a Thursday, compute the AIWQ RPSS if specified
            ic_dayofweek = (ic.astype("datetime64[D]").view("int64") - 4) % 7
            if "aiwq" in cfg.scoring and ic_dayofweek == 3:
                for var in cfg.scoring.aiwq.variables:
                    fcst_data, fcst_coords = load_forecast_for_aiwq(
                        io_backend, ic, array=var, device=device
                    )
                    rpss_wk3, rpss_wk4 = compute_aiwq_rpss(fcst_data, fcst_coords, var)
                    write_aiwq_scores(score_io_dict, rpss_wk3, rpss_wk4, var)


def load_forecast_data(
    cfg: DictConfig,
    io: ZarrBackend,
    ic: np.datetime64,
    array: str,
    score_coords: CoordSystem,
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Load the forecast data for scoring

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    io : ZarrBackend
        The IO backend for loading forecast data
    ic : np.datetime64
        The initial condition to score
    array : str
        The array to load
    score_coords : CoordSystem
        The coordinates of the scored data

    Returns
    -------
    forecast_data : torch.Tensor
        The forecast data
    forecast_coords : CoordSystem
        The coordinates of the forecast data
    """

    # Populate coordiantes of the read data in proper dimension order
    # (accessing io.coords directly does not preserve dimension order)
    read_coords = CoordSystem({})
    for dim in io.root[array].metadata.dimension_names:
        read_coords[dim] = io.coords[dim]
    read_coords["time"] = np.array([ic])
    read_data, read_coords = io.read(
        coords=read_coords, array_name=array, device=device
    )

    # Apply temporal aggregation if specified
    if "temporal_aggregation" in cfg.scoring:
        fcst_data, fcst_coords = apply_temporal_aggregation(
            cfg.scoring.temporal_aggregation, read_data, read_coords, score_coords
        )
    else:
        fcst_data, fcst_coords = read_data, read_coords

    return fcst_data, fcst_coords


def load_verification_data(
    cfg: DictConfig,
    data_source: DataSource,
    ic: np.datetime64,
    fcst_coords: CoordSystem,
    variable: str,
    score_coords: CoordSystem,
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Load the verification data for scoring

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    data_source : DataSource
        The data source for loading verification data
    ic : np.datetime64
        The initial condition to score
    fcst_coords : CoordSystem
        The coordinates of the forecast data
    variable : str
        The variable to load
    score_coords : CoordSystem
        The coordinates of the scored data
    device : torch.device
        The device to load the data onto

    Returns
    -------
    verif_data : torch.Tensor
        The verification data
    verif_coords : CoordSystem
        The coordinates of the verification data
    """

    interp_coords = {
        "_lat": fcst_coords["lat"],
        "_lon": fcst_coords["lon"],
    }
    read_data, read_coords = run_with_rank_ordered_execution(
        fetch_data,
        data_source,
        time=np.array([ic]),
        variable=np.array([variable]),
        lead_time=fcst_coords["lead_time"],
        interp_to=interp_coords,
        device=device,
    )

    # Pop out the singleton variable dimension, reset lat/lon
    read_data = read_data[:, :, 0, ...]
    read_coords.pop("variable")
    lats, lons = read_coords.pop("_lat"), read_coords.pop("_lon")
    read_coords["lat"] = lats
    read_coords["lon"] = lons

    # Apply temporal aggregation if specified
    if "temporal_aggregation" in cfg.scoring:
        verif_data, verif_coords = apply_temporal_aggregation(
            cfg.scoring.temporal_aggregation, read_data, read_coords, score_coords
        )
    else:
        verif_data, verif_coords = read_data, read_coords

    return verif_data, verif_coords


def apply_temporal_aggregation(
    agg_window: str, data: torch.Tensor, coords: CoordSystem, score_coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Apply temporal aggregation to the data

    Parameters
    ----------
    agg_window : str
        The temporal aggregation window to apply
    data : torch.Tensor
        The data to apply temporal aggregation to
    coords : CoordSystem
        The coordinates of the data
    score_coords : CoordSystem
        The target coordinates of the scored data

    Returns
    -------
    fcst_data : torch.Tensor
        The aggregated forecast data
    fcst_coords : CoordSystem
        The coordinates of the aggregated forecast data
    """

    if "ensemble" not in coords:
        # Spoof an ensemble dimension so this works for single-member forecasts/verification data as well
        data = data.unsqueeze(0)
        coords["ensemble"] = np.array([0])
        coords.move_to_end("ensemble", last=False)
        spoofed_ensemble = True
    else:
        spoofed_ensemble = False

    if agg_window == "weekly":
        # Apply weekly averaging starting from lead time 0 onwards
        leads = coords["lead_time"]
        data = data[:, :, leads >= 0, ...]
        coords["lead_time"] = leads[leads >= 0]
        leads = coords["lead_time"]

        tgt_shape = list(data.size())
        tgt_shape[2] = len(score_coords["lead_time"])
        fcst_data = torch.empty(*tgt_shape, device=data.device)
        for i, wk in enumerate(score_coords["lead_time"]):
            avg_slice = np.where((leads >= wk) & (leads < wk + np.timedelta64(7, "D")))[
                0
            ]
            fcst_data[:, :, i, ...] = data[:, :, avg_slice, ...].mean(dim=2)

        fcst_coords = coords.copy()
        fcst_coords["lead_time"] = score_coords["lead_time"]

    else:
        raise ValueError(f"Temporal aggregation {agg_window} not supported")

    if spoofed_ensemble:
        fcst_data = fcst_data.squeeze(0)
        fcst_coords.pop("ensemble")

    return fcst_data, fcst_coords


def score_forecast(
    fcst_data: torch.Tensor,
    fcst_coords: CoordSystem,
    verif_data: torch.Tensor,
    verif_coords: CoordSystem,
    metric_dict: dict,
    var: str,
) -> dict:
    """Score the forecast

    Parameters
    ----------
    fcst_data : torch.Tensor
        The forecast data
    fcst_coords : CoordSystem
        The coordinates of the forecast data
    verif_data : torch.Tensor
        The verification data
    verif_coords : CoordSystem
        The coordinates of the verification data
    metric_dict : dict
        The dictionary of metric objects to use for scoring
    var : str
        The variable to score

    Returns
    -------
    scores : dict
        The scores for the forecast
    """

    scores = {}
    for metric_name, metric in metric_dict.items():
        m, m_coords = metric(
            x=fcst_data, x_coords=fcst_coords, y=verif_data, y_coords=verif_coords
        )
        scores[metric_name + "_" + var] = (m, m_coords)

    return scores


def write_scores(score_io_dict: dict, scores: dict) -> None:
    """Write the scores to the score IO backend

    Parameters
    ----------
    score_io_dict : dict
        The IO backend for writing scores
    scores : dict
        The scores to write

    Returns
    -------
    None
    """

    for k in score_io_dict.keys():
        for m, (data, coords) in scores.items():
            score_io_dict[k].write(data, coords=coords, array_name=m)


def load_forecast_for_aiwq(
    io: ZarrBackend, ic: np.datetime64, array: str, device: torch.device
) -> tuple[torch.Tensor, CoordSystem]:
    """Load the forecast data for AIWQ scoring

    Parameters
    ----------
    io : ZarrBackend
        The IO backend for loading forecast data
    ic : np.datetime64
        The initial condition to score
    array : str
        The array to load
    device : torch.device
        The device to load the data onto
    """

    # Populate coordiantes of the read data in proper dimension order
    read_coords = CoordSystem({})
    for dim in io.root[array].metadata.dimension_names:
        read_coords[dim] = io.coords[dim]
    read_coords["time"] = np.array([ic])
    read_data, read_coords = io.read(
        coords=read_coords, array_name=array, device=device
    )

    # Apply temporal aggregation over days 19-25, 26-32
    wk3_start_day, wk3_end_day = 19, 25
    wk4_start_day, wk4_end_day = 26, 32
    weekly_averaged = []
    for wk_start, wk_end in [
        (wk3_start_day, wk3_end_day),
        (wk4_start_day, wk4_end_day),
    ]:
        lead_time_slice = np.where(
            (read_coords["lead_time"] >= np.timedelta64(wk_start, "D"))
            & (read_coords["lead_time"] <= np.timedelta64(wk_end, "D"))
        )[0]
        weekly_averaged.append(read_data[:, :, lead_time_slice, ...].mean(dim=2))

    return weekly_averaged, read_coords


def compute_aiwq_rpss(
    fcst_data: list[torch.Tensor], fcst_coords: CoordSystem, var: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the AIWQ RPSS

    Parameters
    ----------
    fcst_data : list[torch.Tensor]
        The forecast data
    fcst_coords : CoordSystem
        The coordinates of the forecast data
    var : str
        The variable being scored

    Returns
    -------
    rpss_wk3 : torch.Tensor
        The AIWQ RPSS for the first week
    rpss_wk4 : torch.Tensor
        The AIWQ RPSS for the second week
    """

    ecmwf_names = {
        "t2m": "tas",
        "mslp": "mslp",
        "tp": "pr",
    }

    forecast_date = np.datetime_as_string(fcst_coords["time"][0], unit="D")

    ecmwf_varname = ecmwf_names[var]
    verif_wk3, verif_wk4 = run_with_rank_ordered_execution(
        get_verif_data, forecast_date.replace("-", ""), ecmwf_varname
    )
    q_clim_wk3, q_clim_wk4 = run_with_rank_ordered_execution(
        get_quintile_clim, forecast_date.replace("-", ""), ecmwf_varname
    )
    land_sea_mask = run_with_rank_ordered_execution(
        retrieve_land_sea_mask, os.getenv("AIWQ_SUBMIT_PWD")
    )
    eval_lat, eval_lon = verif_wk3.latitude.values, verif_wk3.longitude.values

    # Load the forecast data into xarray
    da_coords = {
        "ensemble": fcst_coords["ensemble"],
        "time": fcst_coords["time"],
        "lat": fcst_coords["lat"],
        "lon": fcst_coords["lon"],
    }
    data_wk3 = xr.DataArray(fcst_data[0].cpu().numpy(), coords=da_coords).isel(time=0)
    data_wk4 = xr.DataArray(fcst_data[1].cpu().numpy(), coords=da_coords).isel(time=0)

    # Interpolate to the evaluation grid and convert to quintile probabilites
    data_wk3 = data_wk3.interp(lat=eval_lat, lon=eval_lon)
    data_wk4 = data_wk4.interp(lat=eval_lat, lon=eval_lon)

    data_wk3 = data_wk3.rename({"lat": "latitude", "lon": "longitude"})
    data_wk4 = data_wk4.rename({"lat": "latitude", "lon": "longitude"})

    # Compute the quintile probabilities
    verif_q_pbs_wk3 = conditional_obs_probs(verif_wk3, q_clim_wk3)
    verif_q_pbs_wk4 = conditional_obs_probs(verif_wk4, q_clim_wk4)

    data_q_pbs_wk3 = convert_to_quintile_probs(data_wk3, q_clim_wk3)
    data_q_pbs_wk4 = convert_to_quintile_probs(data_wk4, q_clim_wk4)

    rpss_wk3 = work_out_RPSS(
        data_q_pbs_wk3,
        verif_q_pbs_wk3,
        ecmwf_varname,
        land_sea_mask,
        quantile_dim="quintile",
    )
    rpss_wk4 = work_out_RPSS(
        data_q_pbs_wk4,
        verif_q_pbs_wk4,
        ecmwf_varname,
        land_sea_mask,
        quantile_dim="quintile",
    )

    return rpss_wk3, rpss_wk4


def write_aiwq_scores(
    score_io_dict: dict, rpss_wk3: torch.Tensor, rpss_wk4: torch.Tensor, var: str
) -> None:
    """Write the AIWQ scores to the score IO backend

    Parameters
    ----------
    score_io_dict : dict
        The IO backend for writing scores
    rpss_wk3 : torch.Tensor
        The AIWQ RPSS for the first week
    rpss_wk4 : torch.Tensor
        The AIWQ RPSS for the second week
    var : str
        The variable being scored

    Returns
    -------
    None
    """

    for k in score_io_dict.keys():
        for wk, rpss in zip([3, 4], [rpss_wk3, rpss_wk4]):
            write_coords = {"time": rpss.time.values.astype("datetime64[ns]")}
            write_data = torch.from_numpy(rpss.values)
            score_io_dict[k].write(
                write_data, coords=write_coords, array_name=f"rpss_wk{wk}_{var}"
            )
