
from datetime import datetime
import numpy as np
import torch
from loguru import logger

from earth2studio.utils.time import to_time_array

from earth2studio.data import DataSource, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import OrbitGlobalPrecip
from collections import OrderedDict
from earth2studio.utils.coords import map_coords, split_coords

import xarray as xr
from matplotlib import pyplot as plt
import os
import earth2grid

def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    orbit: OrbitGlobalPrecip,
    data: DataSource,
    io: IOBackend,
    file_name: str,
    data_check: bool,
    inference_check: bool,
    inference_check_file: str,
    plot_inference: bool,
) -> IOBackend:

    logger.info("Running ORBIT-2 inference!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    orbit = orbit.to(device)

    time = to_time_array(time)
    x, coords = prep_data_array(
        data(time, orbit.input_coords()["variable"]), device=device
    )

    sst, coords = prep_data_array(
        data(time, "sst"), device=device
    )

    t2_sst_combined = torch.where(torch.isnan(sst[:,0]), x[:,0], sst[:,0])
    x[:,0] = t2_sst_combined

    for i in range(24):
        time_ = (np.array(time) + np.timedelta64(1, 'h') * (-1 * i)).astype(str)
        time_ = to_time_array(time_)
        p, coords = prep_data_array(
            data(time_, ["cp", "lsp", "t2m", "sst"]), device=device
        )

        #tuple of t (1, 4, 721, 1440) tensors
        batch_split = torch.split(p, 1)
        if i == 0:
            batch_p = torch.zeros((len(time), 24, p.shape[-3], p.shape[-2], p.shape[-1]), device=device)
        for j in range(len(batch_split)):
            batch_p[j,i] = batch_split[j][0]
                
        
    total_p_24hr = batch_p[:,:,0] + batch_p[:,:,1]
    total_p_24hr = torch.sum(total_p_24hr, dim=1)
    t2_sst_combined = torch.where(torch.isnan(batch_p[:,:,3]), batch_p[:,:,2], batch_p[:,:,3])
    t2_max = torch.max(t2_sst_combined, dim=1).values
    t2_min = torch.min(t2_sst_combined, dim=1).values
    total_p_24hr = total_p_24hr.unsqueeze(1)
    t2_max = t2_max.unsqueeze(1)
    t2_min = t2_min.unsqueeze(1)
    x = torch.cat((x,total_p_24hr,t2_max,t2_min), dim=1)
    logger.success(f"Fetched data from {data.__class__.__name__}")

    if data_check:
        check_data(x[0].unsqueeze(0), time[0], orbit.in_variables)

    input_coords = OrderedDict({
        k: v for k, v in
        orbit.input_coords().items()
        if k != "batch" # remove placeholder batch dim
    })
    input_coords["time"] = time
    input_coords.move_to_end("time", last=False)

    total_coords = OrderedDict({
        k: v for k, v in
        orbit.output_coords(orbit.input_coords()).items()
        if k != "batch" # remove placeholder batch dim
    })
    total_coords["time"] = time
    total_coords.move_to_end("time", last=False)

    io.add_array(total_coords, total_coords["variable"], overwrite=True)
    logger.info("Inference Starting")
    x, output_coords = orbit(x, input_coords)
    io.write(x, output_coords, total_coords["variable"][0])
    logger.success("Inference complete")

    data = xr.open_zarr(file_name)

    #CHECK_DIFFERENCE
    if inference_check:
        gt = np.load(inference_check_file)
        ae = np.absolute(data[total_coords["variable"][0]].values[0]-gt)
        mae = ae.mean()
        print ("MAE: ", mae)

    #PLOT_OUTPUT
    if plot_inference:
        logger.info("Plotting Output")
        for i in range(len(time)):
            img_min = np.min(data[total_coords["variable"][0]].values[i])
            img_max = np.max(data[total_coords["variable"][0]].values[i])
            plt.figure(
                figsize=(
                    data[total_coords["variable"][0]].values[i].shape[2] / 100,
                    data[total_coords["variable"][0]].values[i].shape[1] / 100,
                )
            )
            plt.imshow(data[total_coords["variable"][0]].values[i,0], cmap="coolwarm", vmin=img_min, vmax=img_max)
            plt.savefig("pred_"+str(i)+".png")
            plt.close()
        logger.success("Plot Saved")

    return io

def check_data(x, time, in_variables):
    ORBIT_VARIABLE_MAPPING = [
        "2m_temperature",
        "temperature_200",
        "temperature_500",
        "temperature_850",
        "10m_u_component_of_wind",
        "u_component_of_wind_200",
        "u_component_of_wind_500",
        "u_component_of_wind_850",
        "10m_v_component_of_wind",
        "v_component_of_wind_200",
        "v_component_of_wind_500",
        "v_component_of_wind_850",
        "specific_humidity_200",
        "specific_humidity_500",
        "specific_humidity_850",
        "volumetric_soil_water_layer_1",
        "total_precipitation_24hr",
        "2m_temperature_max",
        "2m_temperature_min",
    ]

    src_grid = earth2grid.latlon.LatLonGrid(
        lat=np.linspace(90, -90, 721),
        lon=np.linspace(0, 360, 1440, endpoint=False)
    )

    # Define target grid (lat ascending: -90 -> 90)
    dst_grid = earth2grid.latlon.LatLonGrid(
        lat=np.linspace(-90, 90, 721),
        lon=np.arange(0, 360, 0.25)
    )

    # Build regridder (stays on GPU)
    regridder = earth2grid.get_regridder(src_grid, dst_grid)
    regridder = regridder.to(x.device).float()

    # Regrid — expects (..., lat, lon), handles time+variable dims automatically
    # x shape: (time, variables, lat, lon)
    x = regridder(x)

    x = x[:,:,1:,:]

    year = int(time.astype(str)[0:4])
    if year < 2018:
        data_folder = '/lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/train'
    elif year >= 2018 and year < 2020:
        data_folder = '/lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/val'
    elif year == 2020:
        data_folder = '/lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/test'
    month = int(time.astype(str)[5:7])
    day = int(time.astype(str)[8:10])
    day = day-1
    hour = int(time.astype(str)[11:13])

    #Calulate absolute day
    absolute_day = 0
    for i in range(month):
        if i+1 == month:
            break
        if i == 0 or i == 2 or i == 4 or i == 6 or i == 7 or i == 9 or i == 11:
            absolute_day = absolute_day + 31
        elif i == 3 or i == 5 or i==8 or i==10:
            absolute_day = absolute_day + 30
        else:
            absolute_day = absolute_day + 28
    absolute_day = absolute_day + day

    #ERA5
    file_number = ((absolute_day)*24 + hour) // 438
    index = ((absolute_day)*24 + hour) % 438

    #ERA5-IMERG
    #file_number = absolute_day // 5
    #index = absolute_day % 5
    ##Make index 0 based
    #index = index-1

    data_path = os.path.join(data_folder, str(year)+"_"+str(file_number)+".npz")

    os.makedirs("error_plots",exist_ok=True)
    data = np.load(data_path)
    indexer = 0
    for i in range(len(in_variables)-4):
        var = data[in_variables[i]][index]
        ae = np.absolute(var-x.cpu().numpy()[0,indexer])
        mae = ae.mean()
        print("MAE: ", in_variables[i], mae)

        fig, ax = plt.subplots()
        im = ax.imshow(np.squeeze(var))
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(os.path.join("error_plots","orbit_"+in_variables[i]+".png"), bbox_inches='tight', dpi=200)

        fig, ax = plt.subplots()
        im = ax.imshow(np.squeeze(x.cpu().numpy()[0,indexer]))
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(os.path.join("error_plots","e2s_"+str(indexer)+".png"), bbox_inches='tight', dpi=200)

        plt.clf()
        plt.imshow(np.squeeze(ae))
        plt.colorbar()
        plt.savefig(os.path.join("error_plots","ae_"+str(indexer)+".png"), bbox_inches='tight', dpi=200)
        
        indexer = indexer+1




