# Simple example: initialize ObsCastGOES, fetch GOES input, run one step, and plot one channel
# TODO make this a proper example with sphinx formatting and explanations
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import GFS_FX, GOES, MRMS, fetch_data
from earth2studio.models.px.obscast import ObsCastBase, ObsCastGOES, ObsCastMRMS
from earth2studio.utils.type import CoordSystem


# Khronos PBR Neutral Tone Mapping
# https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/README.md#pbr-neutral-specification
def tonemap(rgb):
    start_compression = 0.8 - 0.04
    desaturation = 0.15
    d = 1.0 - start_compression

    height, width, _ = rgb.shape
    result = rgb.reshape(-1, 3)

    def to_3d(col):
        return np.repeat(col[:, np.newaxis], repeats=3, axis=1)

    # make array 1d
    x = result.min(axis=1)

    # apply offset
    mask = x < 0.08
    offset = np.full_like(x, 0.04)
    offset[mask] = x[mask] - 6.25 * x[mask] ** 2
    result -= to_3d(offset)
    # return result.reshape(height, width, 3)

    # calculate peak value (after applying offset)
    peak = result.max(axis=1)
    # anything not in the peak mask is left untouched from now on
    peak_mask = peak >= start_compression

    if not peak_mask.any():
        return result.reshape(height, width, 3)

    new_peak = 1.0 - d * d / (peak[peak_mask] + d - start_compression)
    result[peak_mask] *= to_3d(new_peak / peak[peak_mask])

    g = 1.0 - 1.0 / (desaturation * (peak[peak_mask] - new_peak) + 1.0)
    result[peak_mask] = to_3d(1 - g) * result[peak_mask] + to_3d(g * new_peak)
    return result.reshape(height, width, 3)


def rgb_composite(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Create a RGB composite from a 3-channel input array."""
    rgb = np.stack([r, g, b], axis=-1)

    # Process RGB data
    rgb = np.nan_to_num(rgb, nan=1.0)
    rgb = tonemap(2 * rgb)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.concatenate(
        [rgb, np.ones_like(r)[..., np.newaxis]], axis=-1
    )  # Add alpha channel

    # Set invalid values to transparent
    full_nanmask = np.stack(
        [np.isnan(r), np.isnan(g), np.isnan(b), np.isnan(r)], axis=-1
    )
    transparent_rgba = np.ones_like(rgb)
    transparent_rgba[:, :, -1] = 0.0
    return np.where(full_nanmask, transparent_rgba, rgb)


def main() -> None:

    # Initialization time(s)
    t = datetime(2024, 4, 2, 12, 0, 0)
    goes_satellite = "goes19" if t >= datetime(2025, 4, 7, 0, 0, 0) else "goes16"
    inits = [np.datetime64(t)]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models from a local package
    # Model names:
    #  - "6km_60min_natten_cos_zenith_input_eoe_v2" for 1hr timestep GOES model
    #  - "6km_10min_natten_pure_obs_zenith_6steps" for 10min timestep GOES model
    #  - "6km_60min_natten_cos_zenith_input_mrms_eoe" for 1hr timestep MRMS model
    #  - "6km_10min_natten_pure_obs_mrms_obs_6steps" for 10min timestep MRMS model
    goes_model_name = "6km_60min_natten_cos_zenith_input_eoe_v2"
    mrms_model_name = "6km_60min_natten_cos_zenith_input_mrms_eoe"
    package = ObsCastBase.load_default_package()
    model = ObsCastGOES.load_model(
        package=package,
        conditioning_data_source=GFS_FX(),  # can be set to None if using 10min GOES model
        model_name=goes_model_name,
    )
    model = model.to(device)
    model.eval()

    model_mrms = ObsCastMRMS.load_model(
        package=package,
        conditioning_data_source=GOES(),  # can be set to None if using 10min MRMS model
        model_name=mrms_model_name,
    )
    model_mrms = model_mrms.to(device)
    model_mrms.eval()

    # Setup GOES data source and prep the interpolation using lat/lon coords
    scan_mode = "C"
    variables = model.input_coords()["variable"]
    lat_out = model.latitudes.detach().cpu().numpy()
    lon_out = model.longitudes.detach().cpu().numpy()
    y, x = model.y, model.x
    goes = GOES(satellite=goes_satellite, scan_mode=scan_mode)
    goes_lat, goes_lon = GOES.grid(satellite=goes_satellite, scan_mode=scan_mode)
    model.build_input_interpolator(goes_lat, goes_lon, max_dist_km=12.0)
    model.build_conditioning_interpolator(
        GFS_FX.GFS_LAT, GFS_FX.GFS_LON, max_dist_km=26.0
    )  # interpolating from 25km global grid, we don't want NaNs
    in_coords = model.input_coords()

    x, x_coords = fetch_data(
        goes,
        time=inits,
        variable=np.array(variables),
        lead_time=in_coords["lead_time"],
        device=device,
    )

    # Setup MRMS data source and prep MRMS model interpolators
    mrms = MRMS()
    mrms_in_coords = model_mrms.input_coords()
    x_mrms, x_coords_mrms = fetch_data(
        mrms,
        time=inits,
        variable=np.array(["refc"]),
        lead_time=mrms_in_coords["lead_time"],
        device=device,
    )
    model_mrms.build_input_interpolator(
        x_coords_mrms["lat"], x_coords_mrms["lon"], max_dist_km=12.0
    )
    model_mrms.build_conditioning_interpolator(goes_lat, goes_lon, max_dist_km=12.0)

    # Add batch dimension: [B, T, L, C, H, W]
    batch_size = 1
    if x.dim() == 5:
        x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
        x_coords["batch"] = np.arange(batch_size)
        x_coords.move_to_end("batch", last=False)
    if x_mrms.dim() == 5:
        x_mrms = x_mrms.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
        x_coords_mrms["batch"] = np.arange(batch_size)
        x_coords_mrms.move_to_end("batch", last=False)

    x = x.to(dtype=torch.float32)
    x_mrms = x_mrms.to(dtype=torch.float32)

    # Iterative prognostic steps
    y, y_coords = x, x_coords
    y_mrms, y_coords_mrms = x_mrms, x_coords_mrms
    for step_idx in range(12):

        # Run one prognostic step with GOES
        y_pred, y_pred_coords = model(y, y_coords)

        # Run one prognostic step with MRMS
        y_mrms_pred, y_coords_mrms_pred = model_mrms.call_with_conditioning(
            y_mrms, y_coords_mrms, conditioning=y, conditioning_coords=y_coords
        )

        plot_step(
            y_pred,
            y_pred_coords,
            y_mrms_pred,
            y_coords_mrms_pred,
            composite=True,
            channel="refc",
            cmap="inferno",
            valid_mask=model.valid_mask,
            lat_plot=lat_out,
            lon_plot=lon_out,
            step_idx=step_idx,
            ch_idx=list(model_mrms.variables).index("refc"),
        )

        # Update sliding window with new prediction (no-op if model doesn't use sliding window)
        y_pred, y_pred_coords = model.next_input(y_pred, y_pred_coords, y, y_coords)
        y_mrms_pred, y_coords_mrms_pred = model_mrms.next_input(
            y_mrms_pred, y_coords_mrms_pred, y_mrms, y_coords_mrms
        )

        y = y_pred
        y_coords = y_pred_coords
        y_mrms = y_mrms_pred
        y_coords_mrms = y_coords_mrms_pred
        print(
            "STEP",
            step_idx,
            y_pred_coords["lead_time"][-1].astype("timedelta64[m]").item(),
        )

    # Test iterator mode
    # iterator = model.create_iterator(x, x_coords)
    # step_idx = 0
    # for y_pred, y_pred_coords in iterator:
    #     print("ITERATOR", step_idx, y_pred_coords["lead_time"][-1].astype('timedelta64[m]').item())
    #     step_idx += 1

    #     # Note plots will use outdated MRMS data; MRMS model doesn't work w/ iterator mode without a GOES_FX data source
    #     plot_step(
    #         y_pred,
    #         y_pred_coords,
    #         y_mrms_pred,
    #         y_coords_mrms_pred,
    #         composite=True,
    #         channel="refc",
    #         cmap="inferno",
    #         valid_mask=model.valid_mask,
    #         lat_plot=lat_out,
    #         lon_plot=lon_out,
    #         step_idx=step_idx,
    #         ch_idx=list(model_mrms.variables).index("refc")
    #     )
    #     if step_idx >= 11:
    #         break


def plot_step(
    y: torch.Tensor,
    y_coords: CoordSystem,
    y_mrms: torch.Tensor,
    y_coords_mrms: CoordSystem,
    composite: bool,
    channel: str,
    cmap: str,
    valid_mask: torch.Tensor,
    lat_plot: np.ndarray,
    lon_plot: np.ndarray,
    step_idx: int,
    ch_idx: int,
) -> None:

    # Select a single channel to plot (e.g., refc), or make a composite (see below)
    y = torch.where(valid_mask, y, torch.nan)  # Nan-fill invalid gridpoints

    # Prepare HRRR Lambert Conformal projection and plot
    proj_hrrr = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )
    plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=proj_hrrr)

    # Dual layer coast/state lines for better day/night visibility

    # Black halo (thicker)
    ax.coastlines(color="black", linewidth=1.2)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=1.0)

    # White inner line (thinner)
    ax.coastlines(color="white", linewidth=0.4)
    ax.add_feature(cfeature.STATES, edgecolor="white", linewidth=0.3)

    if composite:
        b = y[0, 0, 0, 0].detach().cpu().numpy()  # abi01c == ~blue
        r = y[0, 0, 0, 1].detach().cpu().numpy()  # abi02c == ~red
        g = y[0, 0, 0, 2].detach().cpu().numpy()  # abi03c == ~green
        field = rgb_composite(r, 0.45 * r + 0.1 * g + 0.45 * b, b)
        pcolor_kwargs = {
            "shading": "gouraud",
        }
    else:
        field = y_mrms[0, 0, 0, ch_idx].detach().cpu().numpy()
        pcolor_kwargs = {
            "cmap": cmap,
            "shading": "auto",
        }

    im = ax.pcolormesh(
        lon_plot,
        lat_plot,
        field,
        transform=ccrs.PlateCarree(),
        **pcolor_kwargs,
    )

    # Overlay MRMS on top of GOES
    if composite:

        # Set low refc and invlaid points to nan
        field_mrms = y_mrms[0, 0, 0, ch_idx]
        field_mrms = (
            torch.where(~valid_mask, torch.nan, field_mrms).detach().cpu().numpy()
        )
        field_mrms = np.where(field_mrms <= 0, np.nan, field_mrms)
        im_mrms = ax.pcolormesh(
            lon_plot,
            lat_plot,
            field_mrms,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading="auto",
            vmin=0.0,
            vmax=55.0,
        )
        plt.colorbar(im_mrms, label=channel, orientation="horizontal", pad=0.05)
    else:
        plt.colorbar(im, label=channel, orientation="horizontal", pad=0.05)

    label = "composite" if composite else channel
    time = y_coords["time"][0].item()
    lead_time = y_coords["lead_time"][0]
    plt.title(
        f"Predicted GOES/MRMS output ({label}) from {time} UTC initialization (lead {lead_time.astype('timedelta64[m]').item()})"
    )

    plt.tight_layout()
    plt.savefig(f"outputs/20_obscast_goes_example_step{step_idx:02d}.png", dpi=300)


if __name__ == "__main__":
    main()
