import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from physicsnemo.distributed import DistributedManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.localh5 import LocalArchiveHDF5

from earth2studio.data import fetch_data, prep_data_array
from earth2studio.statistics import crps, lat_weight, mean, variance
from earth2studio.utils.type import TimeArray, VariableArray


def open_ds(fl):
    if fl.endswith(".zarr") or (
        os.path.isdir(fl) and os.path.exists(os.path.join(fl, ".zgroup"))
    ):
        ds = xr.open_dataset(fl, engine="zarr")
    else:
        ds = xr.open_dataset(fl)

    return ds


def pick_vars_for_rank(vars):
    DistributedManager.initialize()

    dist = DistributedManager()
    if dist.world_size == 1:
        return vars

    if dist.world_size != len(vars):
        raise ValueError(f"{dist.world_size=} != {len(vars)=}")

    vars = [vars[dist.rank]]

    return vars


def compute_mse(ds):

    vars = list(ds.data_vars.keys())
    vars.sort()
    vars = pick_vars_for_rank(vars)
    mse = torch.zeros(len(vars), len(ds.lead_time.values))

    for ivar, var in enumerate(vars):
        print(f"processing variable {var} ({ivar+1}/{len(vars)})")
        var_ds = ds[var]

        se = torch.zeros(
            len(var_ds.time.values),
            len(var_ds.lead_time.values),
            len(var_ds.lat.values),
            len(var_ds.lon.values),
        )

        for ii, ic in enumerate(var_ds.time.values):
            print(
                f"processing time {ic} ({ii+1}/{len(var_ds.time.values)})",
                end="\r",
                flush=True,
            )

            # fetch ground truth data
            data_source = LocalArchiveHDF5(
                dirs=[
                    "/lustre/fsw/coreai_climate_earth2/tkurth/73varQ-hourly/out_of_sample"
                ],
                metadata_file="/lustre/fsw/coreai_climate_earth2/tkurth/73varQ-hourly/metadata/data.json",
            )

            x0, _ = fetch_data(
                data_source,
                time=[ic],
                lead_time=var_ds.lead_time.values,
                variable=var_ds.name,
            )
            x0 = x0.squeeze()

            mean_ds = var_ds.sel(time=ic).mean(dim="ensemble")
            se[ii, ...] = (x0 - mean_ds.values) ** 2
        print("\n")

        weights = (
            torch.as_tensor(lat_weight(np.linspace(ds.lat[-1], ds.lat[0], len(ds.lat))))
            .unsqueeze(1)
            .repeat(1, 1440)
        )
        mse[ivar, ...] = (se * weights).mean(dim=(0, 2, 3))

    return mse


def compute_rmse(var_ds):
    return torch.sqrt(compute_mse(var_ds))


def write_to_csv(rmse, ds, identifier):
    # Get variable names and lead times for DataFrame
    vars = list(ds.data_vars.keys())
    vars.sort()
    vars = pick_vars_for_rank(vars)

    lead_times = ds.lead_time.values

    # Convert tensor to numpy and create DataFrame
    rmse_np = rmse.numpy()

    # Create DataFrame with variables as rows and lead times as columns
    rmse_df = pd.DataFrame(
        rmse_np.T,
        index=[f"{lt}" for lt in lead_times],
        columns=vars,
    )
    rmse_df.index.name = "lead_time"

    # Generate output filename based on input filename
    out_file = identifier
    for var in vars:
        out_file += f"_{var}"
    out_file += ".csv"

    rmse_df.to_csv(out_file)

    return


def compute_crps(ds):

    vars = list(ds.data_vars.keys())
    vars.sort()
    vars = pick_vars_for_rank(vars)

    _crps = crps(
        ensemble_dimension="ensemble",
        reduction_dimensions=["lat", "lon"],
        weights=torch.as_tensor(
            lat_weight(np.linspace(ds.lat[-1], ds.lat[0], len(ds.lat)))
        )
        .unsqueeze(1)
        .repeat(1, 1440),
        fair=True,
    )

    # fetch ground truth data
    _truth = LocalArchiveHDF5(
        dirs=["/lustre/fsw/coreai_climate_earth2/tkurth/73varQ-hourly/out_of_sample"],
        metadata_file="/lustre/fsw/coreai_climate_earth2/tkurth/73varQ-hourly/metadata/data.json",
    )

    ccrr = torch.zeros(len(vars), len(ds.lead_time.values))
    for ivar, var in enumerate(vars):
        print(f"processing variable {var} ({ivar+1}/{len(vars)})")
        ccc = torch.zeros(len(ds.time.values), len(ds.lead_time.values), device="cpu")
        for ii, ic in enumerate(ds.time.values):
            print(
                f"processing time {ic} ({ii+1}/{len(ds.time.values)})",
                end="\r",
                flush=True,
            )

            tru, tru_coords = fetch_data(
                _truth,
                time=[ic],
                lead_time=ds.lead_time.values,
                variable=var,
            )
            tru = tru.squeeze()
            tru_coords.pop("time")
            tru_coords.pop("variable")

            pred, pred_coords = prep_data_array(ds[var].sel(time=ic))

            _ccc, _ = _crps(
                x=pred.to(DistributedManager().device),
                x_coords=pred_coords,
                y=tru.to(DistributedManager().device),
                y_coords=tru_coords,
            )
            ccc[ii, :] = _ccc.to("cpu")

        print("\n")

        ccrr[ivar, :] = ccc.mean(dim=0)

    return ccrr


def score(fl):
    ds = open_ds(fl)

    rmse = False
    crps = True

    if False:
        ds = (
            ds.sel(time=ds.time[:5])
            .sel(ensemble=ds.ensemble[:50])
            .sel(lead_time=ds.lead_time[:10])
        )

    if rmse:
        then = time.time()
        rmse = compute_rmse(ds)
        print(f"time taken: {time.time() - then} seconds")
        write_to_csv(rmse, ds, "rmse")

    if crps:
        then = time.time()
        crps = compute_crps(ds)
        print(f"time taken: {time.time() - then} seconds")
        write_to_csv(crps, ds, "crps")
    return


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score_results.py <netcdf_file>")
        print(
            "Example: python score_results.py score_2020_2020-01-01T00.00.00_mems0000-0049.nc"
        )
        exit()

    score(sys.argv[1])
