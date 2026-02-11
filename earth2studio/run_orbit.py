
from datetime import datetime
import numpy as np
import torch
from loguru import logger

from earth2studio.utils.time import to_time_array

from earth2studio.data import DataSource, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import OrbitGlobalPrecip9_5M
from collections import OrderedDict
from earth2studio.utils.coords import map_coords, split_coords

import xarray as xr
from matplotlib import pyplot as plt
import os
import xesmf as xe

def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    orbit: OrbitGlobalPrecip9_5M,
    data: DataSource,
    io: IOBackend,
) -> IOBackend:

    logger.info("Running ORBIT-2 inference!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    orbit = orbit.to(device)

    time_in = time

    #time_list = []
    #for i in range(24):
    #    time_list.append((time_in[0] + np.timedelta64(1, 'h') * (-1 * i)).astype(str))
    #print(time_list)
    #time = to_time_array(time_list)
    time = to_time_array(time)
    x, coords = prep_data_array(
        data(time, orbit.input_coords()["variable"]), device=device
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")
    #x = torch.mean(x, dim=0)
    #x = x.unsqueeze(0)
    #print("X_MEAN_SHAPE :", x.shape)

    #initial_hour = int(time_in[0].astype(str)[11:13]) 

    #time_list = []
    #for i in range(24):
    #    time_list.append((time_in[0] + np.timedelta64(1, 'h') * (-1 * i)).astype(str))
    #print(time_list)
    #time = to_time_array(time_list)
    #print(time)
    #p, coords = prep_data_array(
    #    data(time, ["cp"]), device=device
    #    #data(time, ["cp", "lsp"]), device=device
    #)
    #print("P", p.shape)
    
    for i in range(24):
        time_i = time_in + np.timedelta64(1, 'h') * (-1 * i)
        time = to_time_array(time_i)
        p, coords = prep_data_array(
            data(time, ["cp", "lsp"]), device=device
        )
        t, coords = prep_data_array(
            data(time, ["t2"]), device=device
        )
        if i == 0:
            p_total = p
            t_max = t
            t_min = t
        else:
            p_total = p_total + p
            t_max = torch.max(t_max, t)
            t_min = torch.min(t_min, t)
    p1 = p_total[:,0]
    p2 = p_total[:,1]
    p = p1+p2
    p = p.unsqueeze(1)
    t_min.squeeze(0)
    t_max.squeeze(0)
    x = torch.cat((x,p,t_max,t_min), dim=1)

    check_data(x, time_in[0], orbit.in_variables)
        

    output_coords = orbit.output_coords(orbit.input_coords())
    total_coords = OrderedDict(
        {
            "time": time,
            #"variable": output_coords["variable"],
            "lat": output_coords["lat"],
            "lon": output_coords["lon"],
        }
    )
    #io.add_array(output_coords, output_coords["variable"], overwrite=True)
    io.add_array(total_coords, output_coords["variable"], overwrite=True)

    logger.info("Inference Starting")
    x, coords = orbit(x, coords)
    print("INF_SHAPE", x.shape)
    #io.write(x, coords, "total_precipitation_24hr")
    #io.write(x, coords, output_coords["variable"])
    io.write(*split_coords(x, coords))
    logger.success("Inference complete")

    #PLOT_OUTPUT
    data = xr.open_zarr("outputs/aifs_forecast3.zarr")
    #data = xr.open_zarr(io.file_name)
    plt.imshow(data['total_precipitation_24hr'][0])
    #plt.imshow(data[output_coords["variable"]][0])
    plt.colorbar()
    plt.savefig("OUTPUT.png")

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

    #time, variables, lat, lon
    ddeg_out = 0.25
    grid_out = xr.Dataset(
        {
            #"lat": (["lat"], np.arange(-90 + ddeg_out / 2, 90, ddeg_out)),
            "lat": np.arange(-90 + ddeg_out, 90+ddeg_out, ddeg_out),
            "lon": np.arange(0, 360, ddeg_out),
        }
    )
    #lat = np.arange(-90 + ddeg_out, 90+ddeg_out, ddeg_out)

    x = x.detach().cpu().numpy()
    x = xr.DataArray(
        x,
        dims=("time", "variables", "lat", "lon"),
        coords={
            #"time": [10, 20],
            "time": [1],
            "variables": ORBIT_VARIABLE_MAPPING,
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False)
        },
        name="dummy_data"
    )
    regridder = xe.Regridder(
        x, grid_out, "bilinear", periodic=True, reuse_weights=False
    )
    x = regridder(x, keep_attrs=True).astype("float32")
    print("XARRAY shape", x.values.shape)

    ##Remove 90 degree latitude from data
    #x = x[:,:,1:,:].to(torch.float32)
    ##Flip latitude (89.75, -90) -> (-90, 89.75)
    #x = torch.flip(x, dims=(2,))

    #data_folder = '/lustre/orion/lrn036/world-shared/data/superres/ERA5-IMERG-FUSED/0.25_deg/train'
    data_folder = '/lustre/orion/lrn036/world-shared/data/superres/era5/0.25_deg/train'
    year = int(time.astype(str)[0:4])
    assert year < 2020, "ORBIT doesn't exist for 2020 and beyond"
    month = int(time.astype(str)[5:7])
    day = int(time.astype(str)[8:10])
    #Actual day we want from ORBIT Data file is day-1
    #TODO: Add capability to go back a month if day is 1
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
    print("ABS_DAY: ", absolute_day)

    #ERA5
    file_number = ((absolute_day)*24 + hour) // 438
    index = ((absolute_day)*24 + hour) % 438

    #ERA5-IMERG
    #file_number = absolute_day // 5
    #index = absolute_day % 5
    ##Make index 0 based
    #index = index-1

    data_path = os.path.join(data_folder, str(year)+"_"+str(file_number)+".npz")
    print("DATA_PATH", data_path)
    print("INDEX", index)

    data = np.load(data_path)
    indexer = 0
    for i in range(len(in_variables)-4):
        print("VARIABLE: ", in_variables[i])
        print("DAY_OF_YEAR: ", data['days_of_year'][index][0,0,0])
        print("TIME_OF_DAY: ", data['time_of_day'][index][0,0,0])
        var = data[in_variables[i]][index]
        print("VAR_SHAPE: ", var.shape)
        #mae = np.absolute(var-x[:,indexer].cpu().numpy()).mean()
        mae = np.absolute(var-x.values[:,indexer]).mean()
        print("MAE: ", mae)

        fig, ax = plt.subplots()
        #im = ax.imshow(var)
        im = ax.imshow(np.squeeze(var))
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(os.path.join("./PLOT_01012017","orbit_"+in_variables[i]+".png"), bbox_inches='tight', dpi=200)

        fig, ax = plt.subplots()
        #im = ax.imshow(x.values[:,indexer])
        im = ax.imshow(np.squeeze(x.values[:,indexer]))
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(os.path.join("./PLOT_01012017","e2s_"+str(indexer)+".png"), bbox_inches='tight', dpi=200)
        
        indexer = indexer+1




