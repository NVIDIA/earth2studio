#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from earth2studio.data.arco import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.models.px.atlas import Atlas
from earth2studio.utils.coords import map_coords



def main() -> None:
    torch.backends.cudnn.enabled = False

    t = np.datetime64("2023-01-01T00:00")
    n_steps = 4
    device = torch.cuda.current_device()

    # Load model
    package = Atlas.load_default_package()
    model = Atlas.load_model(package).to(device)

    # Model-required inputs
    prognostic_ic = model.input_coords()
    lead_time = prognostic_ic["lead_time"]
    variable = prognostic_ic["variable"]

    # Single initialization time
    time = np.array([t])

    # Fetch initial conditions from ARCO
    data = ARCO(cache=True, verbose=True)
    x, coords = fetch_data(
        source=data,
        time=time,
        variable=variable,
        lead_time=lead_time,
        device=device,
    )

    # Ensure coords match model expectations
    # x, coords = map_coords(x, coords, prognostic_ic)
    # Add a batch dim
    bs = 1
    x = x.unsqueeze(0).repeat(bs, 1, 1, 1, 1, 1)
    coords["batch"] = np.arange(bs)
    coords.move_to_end("batch", last=False)

    print(f"X shape: {x.shape}")
    print(f"X coords: {[(k, v.shape) for k, v in coords.items()]}")

    # Test forward call
    y, ycoords = model(x, coords)
    print(f"Forward call returned shape {y.shape} and coords {[(k, v.shape) for k, v in ycoords.items()]}, lead time {ycoords['lead_time'].astype('timedelta64[h]')}")

    # Plot the prediction
    labeldict = {
        "t2m": "2m Temperature (K)",
        "u10m": "10m U-Wind (m/s)",
        "tcwv": "Total Column Water Vapor (kg/m²)",
    }
    lats = ycoords["lat"]
    lons = ycoords["lon"]

    for plotvar in labeldict.keys():
        chidx = list(ycoords["variable"]).index(plotvar)
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.coastlines()
        im = ax.pcolormesh(lons, lats, y[0, 0, 0, chidx, :, :].detach().cpu().numpy(), cmap="turbo", transform=ccrs.PlateCarree())
        plt.colorbar(im, orientation="horizontal", label=labeldict[plotvar])
        plt.title(f"{ycoords['time'][0].astype('datetime64[h]')} - Lead time {ycoords['lead_time'][0].astype('timedelta64[h]')}")
        plt.savefig(f"outputs/20_atlas_prediction_{plotvar}.jpg", dpi=300)
        plt.close()

    # Create iterator and roll out n_steps (iterator yields initial state as step 0)
    it = model.create_iterator(x, coords)

    print(f"Starting rollout from time {time[0]} on device {device}...")
    with torch.no_grad():
        for step, (y, ycoords) in enumerate(it):
            print(
                f"Step {step}: lead_time={ycoords['lead_time'].astype('timedelta64[h]')}, "
                f"vars={len(ycoords['variable'])}, grid=({len(ycoords['lat'])}x{len(ycoords['lon'])}), "
                f"tensor_shape={tuple(y.shape)}"
            )
            if step == n_steps:
                break

    for plotvar in labeldict.keys():
        chidx = list(ycoords["variable"]).index(plotvar)
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.Robinson())
        ax.coastlines()
        im = ax.pcolormesh(lons, lats, y[0, 0, 0, chidx, :, :].detach().cpu().numpy(), cmap="turbo", transform=ccrs.PlateCarree())
        plt.colorbar(im, orientation="horizontal", label=labeldict[plotvar])
        plt.title(f"{ycoords['time'][0].astype('datetime64[h]')} - Lead time {ycoords['lead_time'][0].astype('timedelta64[h]')}")
        plt.savefig(f"outputs/20_atlas_prediction_{plotvar}.jpg", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()