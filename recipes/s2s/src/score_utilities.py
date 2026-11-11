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

import os

import hydra
import numpy as np
import torch
import xarray as xr
from AI_WQ_package.forecast_evaluation import conditional_obs_probs, work_out_RPSS
from AI_WQ_package.retrieve_evaluation_data import retrieve_land_sea_mask
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.data import DataSource, fetch_data
from earth2studio.grids import LatLonGrid
from earth2studio.io import ZarrBackend
from earth2studio.run.work import distribute_work
from earth2studio.utils.coords import coord_array
from src.aiwq_utilities import (
    convert_to_quintile_probs,
    get_quintile_clim,
    get_verif_data,
)
from src.s2s_utilities import build_io_dict, run_with_rank_ordered_execution

WEEK = np.timedelta64(7, "D")


def initialize(cfg: DictConfig) -> tuple[dict, dict, dict, DataSource]:
    """Initialize the scoring system

    Parameters
    ----------
    cfg : DictConfig
        The configuration object

    Returns
    -------
    forecasts : dict[str, xr.Dataset]
        The forecast store of each output region
    metric_dict : dict
        The dictionary of metric objects to use for scoring
    score_io_dict : dict[str, ZarrBackend]
        The IO backend for writing scores, per output region
    data_source : DataSource
        The data source for verification data
    """
    dist = DistributedManager()

    if "file_output" in cfg and "cropboxes" in cfg["file_output"]:
        region_keys = list(cfg["file_output"]["cropboxes"].keys())
    else:
        region_keys = ["global"]

    forecasts = {
        k: xr.open_zarr(os.path.join(cfg.file_output.path, k, "forecast.zarr"))
        for k in region_keys
    }
    metric_dict = {
        m: hydra.utils.instantiate(cfg.scoring.metrics[m]) for m in cfg.scoring.metrics
    }

    # Initialize the IO backend to write scores, rank 0 first
    score_io_dict = run_with_rank_ordered_execution(
        prepare_score_io_dict, cfg, forecasts, create_store=(dist.rank == 0)
    )

    # Download the shared AIWQ land-sea mask once, before ranks score in parallel
    if "aiwq" in cfg.scoring:
        run_with_rank_ordered_execution(
            retrieve_land_sea_mask, os.getenv("AIWQ_SUBMIT_PWD")
        )

    data_source = hydra.utils.instantiate(cfg.data_source)

    return forecasts, metric_dict, score_io_dict, data_source


def score_lead_times(cfg: DictConfig, lead_times: np.ndarray) -> np.ndarray:
    """Lead times of the scores: nonnegative forecast leads, or their weeks."""
    lead_times = lead_times[lead_times >= np.timedelta64(0, "s")]
    if "temporal_aggregation" not in cfg.scoring:
        return lead_times
    if cfg.scoring.temporal_aggregation != "weekly":
        raise ValueError(
            f"Temporal aggregation {cfg.scoring.temporal_aggregation} not supported"
        )
    return np.unique((lead_times // WEEK) * WEEK).astype("timedelta64[ns]")


def prepare_score_io_dict(
    cfg: DictConfig, forecasts: dict[str, xr.Dataset], create_store: bool = False
) -> dict[str, ZarrBackend]:
    """Prepare the IO backend to write scores

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    forecasts : dict[str, xr.Dataset]
        The forecast store of each output region
    create_store : bool, optional
        Whether to create a new store for the IO backend, or use an existing one. Defaults to False.

    Returns
    -------
    score_io_dict : dict[str, ZarrBackend]
        The IO backend for writing scores
    """
    score_io_dict = build_io_dict(
        cfg, list(forecasts), create_store=create_store, file_name="score"
    )
    if not create_store:
        return score_io_dict

    for k, ds in forecasts.items():
        coords = {
            "time": ds.time.values,
            "lead_time": score_lead_times(cfg, ds.lead_time.values),
            "lat": ds.lat.values,
            "lon": ds.lon.values,
        }
        for m in cfg.scoring.metrics:
            reduced = cfg.scoring.metrics[m].get("reduction_dimensions", [])
            dims = [d for d in coords if d not in reduced]
            for v in cfg.scoring.variables:
                score_io_dict[k].add_array(
                    coord_array(dims, {d: coords[d] for d in dims}, name=f"{m}_{v}")
                )
        if "aiwq" in cfg.scoring:
            for v in cfg.scoring.aiwq.variables:
                for wk in (3, 4):
                    score_io_dict[k].add_array(
                        coord_array(
                            ("time",), {"time": coords["time"]}, name=f"rpss_wk{wk}_{v}"
                        )
                    )

    return score_io_dict


@torch.inference_mode()
def run_scoring(
    cfg: DictConfig,
    forecasts: dict[str, xr.Dataset],
    metric_dict: dict,
    score_io_dict: dict[str, ZarrBackend],
    data_source: DataSource,
) -> None:
    """Run the scoring system on the supplied configuration

    Initial conditions are split over ranks. Scores of each initial condition
    are written to the region's ``score.zarr``.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object
    forecasts : dict[str, xr.Dataset]
        The forecast store of each output region
    metric_dict : dict
        The dictionary of metric objects to use for scoring
    score_io_dict : dict[str, ZarrBackend]
        The IO backend for writing scores
    data_source : DataSource
        The data source for loading verification data
    """
    dist = DistributedManager()
    device = dist.device

    for k, ds in forecasts.items():
        ics = distribute_work(list(ds.time.values), dist.rank, dist.world_size)
        for ic in ics:
            for var in cfg.scoring.variables:
                fcst = load_forecast_data(ds, ic, var)
                verif = load_verification_data(data_source, fcst, var)
                if "temporal_aggregation" in cfg.scoring:
                    fcst, verif = weekly_mean(fcst), weekly_mean(verif)
                for name, metric in metric_dict.items():
                    score = metric(_to_device(fcst, device), _to_device(verif, device))
                    score = score.e2s.to_backend("numpy", device="cpu")
                    score_io_dict[k].write(
                        score.astype(np.float32).rename(f"{name}_{var}")
                    )

            # If initial condition is on a Thursday, compute the AIWQ RPSS if specified
            ic_dayofweek = (ic.astype("datetime64[D]").view("int64") - 4) % 7
            if "aiwq" in cfg.scoring and ic_dayofweek == 3:
                for var in cfg.scoring.aiwq.variables:
                    weekly = load_forecast_for_aiwq(ds, ic, var)
                    for wk, rpss in zip((3, 4), compute_aiwq_rpss(weekly, ic, var)):
                        score_io_dict[k].write(
                            xr.DataArray(
                                np.atleast_1d(rpss.values).astype(np.float32),
                                dims=["time"],
                                coords={"time": [ic]},
                                name=f"rpss_wk{wk}_{var}",
                            )
                        )
            logger.info(f"Rank {dist.rank}: scored {k} initial condition {ic}")

    for io in score_io_dict.values():
        io.close()


def _to_device(x: xr.DataArray, device: torch.device) -> xr.DataArray:
    backend = "cupy" if device.type == "cuda" else "numpy"
    return x.e2s.to_backend(backend, device=device)


def weekly_mean(x: xr.DataArray) -> xr.DataArray:
    """Average over weeks of lead time, labeled by each week's first lead time."""
    weeks = xr.DataArray(
        ((x.lead_time.values // WEEK) * WEEK).astype("timedelta64[ns]"),
        dims="lead_time",
        name="week",
    )
    return x.groupby(weeks).mean("lead_time").rename(week="lead_time")


def load_forecast_data(ds: xr.Dataset, ic: np.datetime64, var: str) -> xr.DataArray:
    """Load one initial condition's forecast of ``var`` at nonnegative lead times.

    Returns
    -------
    xr.DataArray
        Forecast with dimensions (ensemble, time, lead_time, lat, lon)
    """
    fcst = ds[var].sel(time=[ic])
    fcst = fcst.sel(lead_time=fcst.lead_time >= np.timedelta64(0, "s")).load()
    return fcst.transpose("ensemble", "time", "lead_time", "lat", "lon")


def load_verification_data(
    data_source: DataSource, fcst: xr.DataArray, var: str
) -> xr.DataArray:
    """Load verification data on the forecast's grid, times and lead times.

    Parameters
    ----------
    data_source : DataSource
        The data source for loading verification data
    fcst : xr.DataArray
        Forecast being scored
    var : str
        The variable to load

    Returns
    -------
    xr.DataArray
        Verification data with dimensions (time, lead_time, lat, lon)
    """
    verif = fetch_data(
        source=data_source,
        time=fcst.time.values,
        variable=np.array([var]),
        lead_time=fcst.lead_time.values,
        target_grid=LatLonGrid(fcst.lat.values, fcst.lon.values),
        device="cpu",
    ).isel(variable=0, drop=True)
    return verif.transpose("time", "lead_time", "lat", "lon")


def load_forecast_for_aiwq(
    ds: xr.Dataset, ic: np.datetime64, var: str
) -> list[xr.DataArray]:
    """Average one initial condition's forecast over days 19-25 and 26-32."""
    fcst = ds[var].sel(time=ic)
    weekly_averaged = []
    for wk_start, wk_end in [(19, 25), (26, 32)]:
        leads = fcst.lead_time
        window = (leads >= np.timedelta64(wk_start, "D")) & (
            leads <= np.timedelta64(wk_end, "D")
        )
        weekly_averaged.append(fcst.sel(lead_time=window).mean("lead_time").load())
    return weekly_averaged


def compute_aiwq_rpss(
    fcst_data: list[xr.DataArray], ic: np.datetime64, var: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute the AIWQ RPSS

    Parameters
    ----------
    fcst_data : list[xr.DataArray]
        Week 3 and week 4 mean forecasts, with dimensions (ensemble, lat, lon)
    ic : np.datetime64
        Initial condition of the forecast
    var : str
        The variable being scored

    Returns
    -------
    rpss_wk3 : xr.DataArray
        The AIWQ RPSS for week 3
    rpss_wk4 : xr.DataArray
        The AIWQ RPSS for week 4
    """
    ecmwf_names = {
        "t2m": "tas",
        "mslp": "mslp",
        "tp": "pr",
    }
    forecast_date = np.datetime_as_string(ic, unit="D").replace("-", "")
    ecmwf_varname = ecmwf_names[var]

    verif_wk3, verif_wk4 = get_verif_data(forecast_date, ecmwf_varname)
    q_clim_wk3, q_clim_wk4 = get_quintile_clim(forecast_date, ecmwf_varname)
    land_sea_mask = retrieve_land_sea_mask(os.getenv("AIWQ_SUBMIT_PWD"))
    eval_lat, eval_lon = verif_wk3.latitude.values, verif_wk3.longitude.values

    rpss = []
    for data, verif, q_clim in zip(
        fcst_data, (verif_wk3, verif_wk4), (q_clim_wk3, q_clim_wk4)
    ):
        # Interpolate to the evaluation grid and convert to quintile probabilites
        data = data.interp(lat=eval_lat, lon=eval_lon).rename(
            {"lat": "latitude", "lon": "longitude"}
        )
        rpss.append(
            work_out_RPSS(
                convert_to_quintile_probs(data, q_clim),
                conditional_obs_probs(verif, q_clim),
                ecmwf_varname,
                land_sea_mask,
                quantile_dim="quintile",
            )
        )
    return rpss[0], rpss[1]
