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

"""
StormScope Satellite and Radar Nowcasting
========================================

Run coupled GOES and MRMS/GLM forecasts with StormScope or StormScope Flash.

In this example you will learn:

- Select baseline or Flash weights using the same public model classes.
- Download and cache the selected checkpoints automatically from Hugging Face.
- Run full-domain forecasts or request regional context with Flash.
- Save every lead to Zarr, a final-lead JPG, and a separate GIF.

Use ``--model 3km_10min_flash`` to select Flash; the default is the baseline
``3km_10min`` model. Pass ``--lat SOUTH NORTH --lon WEST EAST`` for Flash
regional inference, or omit both bounds for the full native domain.
"""

# /// script
# dependencies = [
#   "earth2studio[data,stormscope-flash] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy", "matplotlib", "pillow", "xarray",
# ]
# ///

# %%
# Model and region selection
# --------------------------
# Latitude/longitude select the output box. The models independently add real
# context, expand each input axis to at least 200 pixels, and align to patches.
import argparse
from datetime import datetime
from pathlib import Path

# %%
# Forecast configuration — edit these values in Python, as in the baseline.
# The selected checkpoint revision downloads automatically from Hugging Face.
start_date = datetime(2024, 3, 13, 23, 30)
n_steps = 18  # Ten-minute leads; 18 steps = three hours.
padding = 25  # Available real context per side, applied once.
seed = 1234
output_dir = Path("outputs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Select the model variant and optional Flash geographic bounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("3km_10min", "3km_10min_flash"),
        default="3km_10min",
        help="StormScope variant; both models download their own checkpoints.",
    )
    parser.add_argument("--lat", type=float, nargs=2, metavar=("SOUTH", "NORTH"))
    parser.add_argument("--lon", type=float, nargs=2, metavar=("WEST", "EAST"))
    args = parser.parse_args(argv)
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be supplied together")
    if args.lat is not None and args.model != "3km_10min_flash":
        parser.error("Regional inputs require --model 3km_10min_flash")
    return args


def create_output_path(
    region: dict[str, tuple[float, float]] | None, model_name: str
) -> Path:
    """Reserve a separate run directory without replacing previous products."""
    domain = "full"
    if region is not None:
        south, north = region["lat"]
        west, east = region["lon"]
        domain = f"lat_{south:g}_{north:g}_lon_{west:g}_{east:g}"
    root = output_dir / f"stormscope_{model_name}_{start_date:%Y%m%dT%H%M}_{domain}"
    root.mkdir(parents=True, exist_ok=True)
    run = 1
    while True:
        directory = root / f"run_{run:03d}"
        try:
            directory.mkdir()
        except FileExistsError:
            run += 1
        else:
            return directory / "stormscope.zarr"


def goes_east_satellite(time: datetime) -> str:
    """Select the east satellite from the baseline data source's operational dates."""
    from earth2studio.data import GOES

    for satellite in ("goes19", "goes16"):
        start, end = GOES.GOES_HISTORY_RANGE[satellite]
        if start <= time and (end is None or time < end):
            return satellite
    raise ValueError(f"No operational GOES-East satellite for {time} UTC")


# %%
# Coupled inference
# -----------------
# Importing this example does not download observations or initialize CUDA.
def main(argv: list[str] | None = None) -> None:
    """Run a full-domain or regional forecast and save all leads, a final JPG, and a GIF."""
    from collections import OrderedDict

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import xarray as xr
    from dotenv import load_dotenv
    from loguru import logger
    from matplotlib.animation import PillowWriter

    from earth2studio.data import GOES, MRMS, GOESGLMGrid, fetch_data
    from earth2studio.models.px import StormScopeGOES, StormScopeMRMS

    load_dotenv()
    args = parse_args(argv)
    region = (
        None if args.lat is None else {"lat": tuple(args.lat), "lon": tuple(args.lon)}
    )
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    model_name = args.model
    is_flash = model_name == "3km_10min_flash"
    model_label = "StormScope Flash" if is_flash else "StormScope"
    package = StormScopeGOES.load_default_package(model_name=model_name)
    if is_flash:
        from earth2studio.models.px._stormscope_flash.region import resolve_region

        # Check bounds on the native grid before loading checkpoints or CUDA.
        _, entry = StormScopeGOES._resolve_model_entry(package, model_name)
        latitude, longitude, *_ = StormScopeGOES._build_grid_and_times(package, entry)
        try:
            resolve_region(latitude.numpy(), longitude.numpy(), region, padding)
        except ValueError as exc:
            raise ValueError(f"Invalid geographic request {region}: {exc}") from exc
    output_path = create_output_path(region, model_name)
    logger.info("Saving this run to {}", output_path.parent.resolve())
    device = torch.device("cuda")
    time = np.array([np.datetime64(start_date)])

    # Baseline keeps its existing AMP dtype and Heun sampler. Only Flash needs
    # the regional and FP16 configuration; both variants enable compilation.
    options = {"model_name": model_name, "amp": True, "compile": True}
    if is_flash:
        options.update(region=region, padding=padding, amp_dtype=torch.float16)
    goes = StormScopeGOES.load_model(package, **options).to(device).eval()
    mrms = (
        StormScopeMRMS.load_model(
            package, glm_data_source=GOESGLMGrid(satellite="east"), **options
        )
        .to(device)
        .eval()
    )
    if is_flash:
        logger.info("Region: {}", goes.region_info)

    # Load observation histories
    # -------------------------
    # GLM is interpolated as raw counts, then normalized by the model. We use the
    # coupled API, so predicted GLM is retained after initialization.
    # Select by observation time so a history crossing the satellite handover
    # can use both sources. Both use the same GOES-East CONUS navigation grid.
    satellites = [
        goes_east_satellite(t.astype("datetime64[us]").astype(datetime))
        for t in time[0] + goes.input_times
    ]
    pieces, order = [], []
    goes_lat, goes_lon = GOES.grid(satellite=satellites[-1], scan_mode="C")
    goes.build_input_interpolator(goes_lat, goes_lon)
    for satellite in dict.fromkeys(satellites):
        lat, lon = GOES.grid(satellite=satellite, scan_mode="C")
        if not (
            np.array_equal(lat, goes_lat, equal_nan=True)
            and np.array_equal(lon, goes_lon, equal_nan=True)
        ):
            raise ValueError("GOES history sources must share the same navigation grid")
        indices = [i for i, value in enumerate(satellites) if value == satellite]
        field, gc = fetch_data(
            GOES(satellite=satellite, scan_mode="C"),
            time=time,
            variable=goes.variables,
            lead_time=goes.input_times[indices],
            device=device,
        )
        pieces.append(field)
        order.extend(indices)
    gx = torch.cat(pieces, dim=1)[:, np.argsort(order)]
    gc["lead_time"] = goes.input_times
    gx, gc = goes.prep_input(gx.unsqueeze(0), OrderedDict(batch=np.arange(1), **gc))
    radar, rc = fetch_data(
        MRMS(),
        time=time,
        variable=np.array(["refc", "refc_base"]),
        lead_time=mrms.input_times,
        device=device,
    )
    mrms.build_input_interpolator(rc["lat"], rc["lon"])
    radar = mrms.input_interp(radar)
    mrms.conditioning_valid_mask = goes.valid_mask.clone()
    glm_coords = mrms.input_coords()
    glm_coords["time"] = time
    glm, _ = mrms.fetch_glm(glm_coords, device=device)
    mx = torch.cat((radar, glm), dim=2).unsqueeze(0).float()
    mc = OrderedDict(
        batch=np.arange(1),
        time=time,
        lead_time=mrms.input_times,
        variable=mrms.variables,
        y=mrms.y,
        x=mrms.x,
    )

    # Forecast and save
    # -----------------
    # Both models see the histories from the same forecast time. Only after both
    # predictions finish do we advance either history. Cropped outputs are separate
    # tensors, leaving the full input context available for later leads.
    torch.manual_seed(seed)
    steps = n_steps
    satellite_fields, radar_fields = [], []
    with torch.inference_mode():
        for _ in range(steps):
            gp, gpc = goes(gx, gc)
            mp, mpc = mrms.call_with_conditioning(mx, mc, gx, gc)
            cropped_goes, _ = goes.crop_output(gp, gpc)
            cropped_mrms, _ = mrms.crop_output(mp, mpc)
            satellite_fields.append(cropped_goes[0, 0, 0].cpu().numpy())
            radar_fields.append(cropped_mrms[0, 0, 0].cpu().numpy())
            gx, gc = goes.next_input(gp, gpc, gx, gc)
            mx, mc = mrms.next_input(mp, mpc, mx, mc)
    ys, xs = goes.region_info.output_slices if is_flash else (slice(None), slice(None))
    metadata = {
        "model": model_label,
        "model_name": model_name,
        "amp_dtype": "float16" if is_flash else str(torch.get_autocast_dtype("cuda")),
        "torch_compile": True,
        "input_shape": list(goes.latitudes.shape),
        "seed": seed,
        "goes_history_satellites": satellites,
    }
    if is_flash:
        metadata.update(
            goes_nfe=5,
            mrms_nfe=7,
            actual_padding=list(goes.region_info.padding),
            requested_bounds=goes.region_info.requested_bounds,
            input_bounds=list(goes.region_info.input_bounds),
            output_bounds=list(goes.region_info.output_bounds),
        )
    else:
        metadata.update(
            goes_nfe=2 * int(goes.sampler_args.get("num_steps", 100)) - 1,
            mrms_nfe=2 * int(mrms.sampler_args.get("num_steps", 100)) - 1,
        )
    lat = goes.latitudes[ys, xs].cpu().numpy()
    lon = goes.longitudes[ys, xs].cpu().numpy()
    dataset = xr.Dataset(
        {
            "goes": (
                ("lead_time", "goes_variable", "y", "x"),
                np.stack(satellite_fields),
            ),
            "mrms": (("lead_time", "mrms_variable", "y", "x"), np.stack(radar_fields)),
        },
        coords={
            "lead_time": np.arange(1, steps + 1) * np.timedelta64(10, "m"),
            "goes_variable": goes.variables,
            "mrms_variable": mrms.variables,
            "y": goes.y[ys],
            "x": goes.x[xs],
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
            "initialization": time[0],
        },
        attrs=metadata,
    )
    dataset.to_zarr(output_path, mode="w-")

    # Visualize every lead using fixed scales; save the final frame separately.
    # -----------------------------------------------------------------------
    plt.close("all")
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    panels = (
        (
            dataset.goes.sel(goes_variable="abi13c"),
            "GOES ABI13",
            "gray_r",
            190,
            310,
            "K",
        ),
        (
            dataset.mrms.sel(mrms_variable="refc"),
            "MRMS reflectivity",
            "turbo",
            0,
            75,
            "dBZ",
        ),
    )
    meshes = []
    for ax, (fields, title, cmap, vmin, vmax, unit) in zip(axes, panels):
        mesh = ax.pcolormesh(
            lon,
            lat,
            fields.isel(lead_time=0).values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        meshes.append(mesh)
        ax.coastlines()
        fig.colorbar(mesh, ax=ax, shrink=0.75, label=unit)
    fig.suptitle(f"{model_label} | initialized {start_date:%Y-%m-%d %H:%M} UTC")
    writer = PillowWriter(fps=2)
    with writer.saving(fig, str(output_path.with_suffix(".gif")), dpi=100):
        for i in range(steps):
            for ax, mesh, (fields, title, *_) in zip(axes, meshes, panels):
                mesh.set_array(fields.isel(lead_time=i).values)
                ax.set_title(f"{title} | +{(i + 1) * 10} min")
            writer.grab_frame()
    fig.savefig(output_path.with_suffix(".jpg"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "Saved forecast products in {} (.zarr, .jpg, .gif)",
        output_path.parent.resolve(),
    )


if __name__ == "__main__":
    main()
