# Simple example: initialize ObsCastGOES, fetch GOES input, run one step, and plot one channel

from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
import xarray as xr

from earth2studio.data import GOES, GFS_FX, fetch_data, HRRR, CurvilinearNNInterp
from earth2studio.models.px.obscast import ObsCastGOES, ObsCastBase

def main() -> None:
    # Choose a valid GOES timestamp (scan_mode 'C' has 5-minute cadence)
    t = datetime(2023, 6, 1, 18, 0, 0)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model from a local package (update path via load_default_package if needed)
    package = ObsCastBase.load_default_package()
    model = ObsCastGOES.load_model(package=package, conditioning_data_source=GFS_FX())
    model = model.to(device)
    model.eval()

    # GOES data source: we use a custom subclass to handle custom interpolation scheme
   
    

    # Pull GOES data and interpolate to model grid
    # Model stores lat/lon as 2D tensors; convert to numpy for interpolation target
    satellite, scan_mode = "goes16", "C"
    variables = model.input_coords()["variable"]
    lat_out = model.latitudes.detach().cpu().numpy()
    lon_out = model.longitudes.detach().cpu().numpy()
    y, x = model.y, model.x
    base_goes = GOES(satellite=satellite, scan_mode=scan_mode)
    goes_lat, goes_lon = GOES.grid(satellite=satellite, scan_mode=scan_mode)
    goes = CurvilinearNNInterp(
        base=base_goes,
        lat_in=goes_lat,
        lon_in=goes_lon,
        lat_out=lat_out,
        lon_out=lon_out,
        y=y,
        x=x,
        max_distance_km=6.0,
    )

    x, x_coords = fetch_data(
        goes,
        time=np.array([np.datetime64(t)]),
        variable=np.array(variables),
        lead_time=np.array([np.timedelta64(0, "h")]),
        device=device,
    )

    # np.save("invalid_gridpoints.npy", np.where(np.isnan(x[0, 0, 0, :, :].cpu().numpy()), 1, 0))
    # print("SAVED INVALID GRIDPOINTS: ", np.load("invalid_gridpoints.npy").shape)

    # Add batch dimension: [B, T, L, C, H, W]
    if x.dim() == 5:
        x = x.unsqueeze(0)
        x_coords["batch"] = np.empty(0)
        x_coords.move_to_end("batch", last=False)

    print("GOES INPUT COORDS: ", x.shape, [(k, v.shape) for k, v in x_coords.items()])
    print("GOES RANGE: ", np.nanmin(x[0, 0, 0, :, :, :].cpu().numpy()), np.nanmax(x[0, 0, 0, :, :, :].cpu().numpy()))

    for ch in range(x.shape[3]):
        plt.figure()
        plt.imshow(x[0, 0, 0, ch, :, :].cpu().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title(f"GOES DATA CH {ch}")
        plt.savefig(f"xxx_goes_data_ch{ch}.png", dpi=300)
        plt.close()

    plt.figure()
    plt.imshow(np.load("invalid_gridpoints.npy"), cmap="viridis")
    plt.colorbar()
    plt.title("INVALID GRIDPOINTS")
    plt.savefig("xxx_invalid_gridpoints.png", dpi=300)
    plt.close()

    real_nans = np.isnan(x[0, 0, 0, 0, :, :].cpu().numpy())
    discrepant_mask = real_nans != model.invalid_mask.cpu().numpy()
    plt.figure()
    plt.imshow(discrepant_mask, cmap="viridis")
    plt.colorbar()
    plt.title("DISCREPANCY MASK")
    plt.savefig("xxx_discrepant_mask.png", dpi=300)
    plt.close()

    # Run one prognostic step
    with torch.inference_mode():
        y, y_coords = model(x, x_coords)

    # Select a single channel to plot (e.g., abi09c)
    channel = "abi03c"
    ch_idx = list(variables).index(channel)
    field = y[0, 0, 0, ch_idx].detach().cpu().numpy()
    print("FIELD", field.shape, np.sum(np.isnan(field)))

    # Prepare HRRR Lambert Conformal projection and plot
    proj_hrrr = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=proj_hrrr)

    # Use geodetic transform for pcolormesh on lat/lon grid
    lat_plot = lat_out
    lon_plot = lon_out
    ax.set_extent(
        (np.nanmin(lon_plot), np.nanmax(lon_plot), np.nanmin(lat_plot), np.nanmax(lat_plot)),
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="50m", linewidth=1.0)
    ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor="black")

    im = ax.pcolormesh(
        lon_plot,
        lat_plot,
        field,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        shading="auto",
    )

    plt.title(f"Predicted GOES output ({channel}) at {t} UTC (lead +1h)")
    plt.colorbar(im, label=channel, orientation="horizontal", pad=0.05)
    plt.tight_layout()
    plt.savefig("outputs/20_obscast_goes_example.png", dpi=300)


if __name__ == "__main__":
    main()


