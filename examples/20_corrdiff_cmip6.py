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

# %%
"""
Running CorrDiffCMIP6 on CMIP6 Data
===================================

This example demonstrates how to run the `CorrDiffCMIP6` downscaling model on CMIP6
data and write outputs into an in-memory IO backend.

In this example you will learn:

- How to load a `CorrDiffCMIP6` model from a local `Package`
- How to construct a CMIP6 datasource and wrap it using the model's `TimeWindow` metadata
- How to run downscaling and compare against the coarse CMIP6 input
- How to run multiple samples to visualize downscaling uncertainty (members, mean, std)
"""
# /// script
# dependencies = [
#   "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git",
#   "xarray",
#   "zarr",
#   "matplotlib",
#   "tqdm",    # optional: progress bars
#   "cartopy", # optional: coastlines
# ]
# ///

# Imports used throughout the example.
# %% Set Up (imports)

from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from earth2studio.data import CMIP6, CMIP6MultiRealm, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import CorrDiffCMIP6
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.time import timearray_to_datetime, to_time_array

# Configuration knobs: device selection, model package path, CMIP6 dataset IDs, and evaluation times.
# %% Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PACKAGE = CorrDiffCMIP6.load_default_package()
CMIP6_KWARGS = dict(
    experiment_id="ssp585",
    source_id="CanESM5",
    variant_label="r1i1p2f1",
    exact_time_match=True,
)
TIMES = [datetime(2037, 9, 3, 12, 0, 0)]


# Helper functions used by the workflow below:
# - datasource construction + TimeWindow wrapping
# - a minimal diagnostic runner that writes to an in-memory Zarr backend
# - plotting utilities for deterministic and ensemble outputs
# %% Helper functions (datasource + diagnostic + plotting)
def normalize_to_noon(dt: datetime) -> datetime:
    """Normalize times to 12:00 to match CMIP6 daily tables used in this example."""
    return dt.replace(hour=12, minute=0, second=0, microsecond=0)


def create_cmip6_timewindow_source(model: CorrDiffCMIP6, **cmip6_kwargs):
    """Create a CMIP6MultiRealm datasource wrapped by the model's TimeWindow config."""
    data = CMIP6MultiRealm(
        [CMIP6(table_id=t, **cmip6_kwargs) for t in ("day", "Eday", "SIday")]
    )
    return model.create_time_window_wrapper(data, time_fn=normalize_to_noon)


def diagnostic_from_data_cmip6_corrdiff(
    *,
    time: list,
    diagnostic: CorrDiffCMIP6,
    data,
    io,
    output_variables: list[str],
    device: torch.device | None = None,
) -> ZarrBackend:
    """A small, example-local variant of `run.diagnostic_from_data`.

    CorrDiffCMIP6 uses `coords["time"]` as validity timestamps for time-dependent
    features. In this demo we treat `time` as a real tensor dimension (singleton axis),
    which keeps `@batch_func` happy without modifying `run.py`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostic = diagnostic.to(device)

    # Fetch model-required (suffix-expanded) inputs for all times
    time_arr = to_time_array(time)
    x_all, coords_all = fetch_data(
        source=data,
        time=time_arr,
        variable=np.asarray(diagnostic.input_variables),
        device=device,
    )

    # Remove lead_time dimension (fetch_data adds it for non-forecast sources)
    if "lead_time" in coords_all:
        x_all = x_all[:, 0]
        del coords_all["lead_time"]

    output_coords = OrderedDict({"variable": np.asarray(output_variables)})

    def coords_for_idx(i: int):
        # IMPORTANT: coords keys must match x dims. Here x has dims:
        #   [batch, time, variable, lat, lon]
        return OrderedDict(
            {
                "batch": np.arange(1),
                "time": coords_all["time"][i : i + 1],
                "variable": coords_all["variable"],
                "lat": coords_all["lat"],
                "lon": coords_all["lon"],
            }
        )

    # Infer output layout from one forward pass, then allocate full IO time axis
    # Make batch explicit: [batch=1, time=1, variable, lat, lon]
    y0, ycoords0 = diagnostic(x_all[0:1].unsqueeze(0), coords_for_idx(0))
    y0, ycoords0 = map_coords(y0, ycoords0, output_coords)

    if "batch" in ycoords0:
        bdim = list(ycoords0).index("batch")
        y0 = y0.squeeze(bdim)
        ycoords0 = ycoords0.copy()
        del ycoords0["batch"]

    # Add singleton time axis for slice, and full time axis for IO layout
    ycoords0 = ycoords0.copy()
    ycoords0["time"] = coords_all["time"][0:1]
    ycoords0.move_to_end("time", last=False)
    y0 = y0.unsqueeze(0)

    total_coords = ycoords0.copy()
    total_coords["time"] = coords_all["time"]
    total_coords.move_to_end("time", last=False)
    var_names = total_coords.pop("variable")
    io.add_array(total_coords, var_names)

    # Write timestep 0 using the already-computed y0 (avoids a duplicate forward / progress bar)
    io.write(*split_coords(y0, ycoords0))

    # Write remaining timesteps
    for i in range(1, len(coords_all["time"])):
        y, ycoords = diagnostic(x_all[i : i + 1].unsqueeze(0), coords_for_idx(i))
        y, ycoords = map_coords(y, ycoords, output_coords)

        if "batch" in ycoords:
            bdim = list(ycoords).index("batch")
            y = y.squeeze(bdim)
            ycoords = ycoords.copy()
            del ycoords["batch"]

        ycoords = ycoords.copy()
        ycoords["time"] = coords_all["time"][i : i + 1]
        ycoords.move_to_end("time", last=False)
        y = y.unsqueeze(0)

        io.write(*split_coords(y, ycoords))

    return io


# Plotting helpers
def plot_compare_2x2(
    io: ZarrBackend,
    *,
    output_variables: list[str],
    cmip6_kwargs: dict,
    input_time_fn,
    unit_map: dict[str, str] | None = None,
) -> None:
    """Plot a fixed 2x2 comparison.

    Layout:
      - left column: coarse CMIP6 input
      - right column: CorrDiff downscaled output
      - rows: variables (exactly 2)
    """
    # This helper plots exactly two rows:
    #  - temperature (CMIP6 tas vs CorrDiff t2m)
    #  - wind speed (CMIP6 sqrt(uas^2+vas^2) vs CorrDiff sqrt(u10m^2+v10m^2))
    required_out = {"t2m", "u10m", "v10m"}
    missing_out = sorted(required_out - set(output_variables))
    if missing_out:
        raise ValueError(
            f"plot_compare_2x2 requires output_variables to include {sorted(required_out)}, "
            f"but is missing {missing_out}."
        )

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
        import xarray as xr  # noqa: WPS433
        from matplotlib.gridspec import GridSpec  # noqa: WPS433

        try:
            import cartopy.crs as ccrs  # noqa: WPS433
        except Exception:  # pragma: no cover
            ccrs = None
    except Exception as e:  # pragma: no cover
        print(f"Plotting skipped (missing deps): {e}")
        return

    if unit_map is None:
        unit_map = {
            "tas": "K",
            "t2m": "K",
            "uas": "m/s",
            "vas": "m/s",
            "u10m": "m/s",
            "v10m": "m/s",
            "ws10m": "m/s",
        }

    ds = xr.open_zarr(io.store, consolidated=False)
    t0 = ds["time"].values[0]

    def _select_first_non_spatial(da):
        for d in list(da.dims):
            if d in ("lat", "lon"):
                continue
            if d == "time":
                continue
            da = da.isel({d: 0})
        return da

    # Fetch coarse CMIP6 inputs for the same "normalized" time
    if isinstance(t0, datetime):
        t_fetch = input_time_fn(t0)
    else:
        t_fetch = input_time_fn(timearray_to_datetime(np.asarray([t0]))[0])
    # Only fetch what we need for the comparison plot.
    input_variables = ["tas", "uas", "vas"]
    base = CMIP6MultiRealm(
        [CMIP6(table_id=t, **cmip6_kwargs) for t in ("day", "Eday", "SIday")]
    )
    x_in, coords_in = fetch_data(
        source=base,
        time=to_time_array([t_fetch]),
        variable=np.asarray(input_variables),
        device=torch.device("cpu"),
    )
    if "lead_time" in coords_in:
        x_in = x_in[:, 0]
        del coords_in["lead_time"]

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.08, hspace=0.18)

    if ccrs is not None:

        proj = ccrs.PlateCarree(central_longitude=180)
        ax_in = [fig.add_subplot(gs[r, 0], projection=proj) for r in range(2)]
        ax_out = [fig.add_subplot(gs[r, 1], projection=proj) for r in range(2)]
    else:
        ax_in = [fig.add_subplot(gs[r, 0]) for r in range(2)]
        ax_out = [fig.add_subplot(gs[r, 1]) for r in range(2)]
    ax_cbar = [fig.add_subplot(gs[r, 2]) for r in range(2)]

    def _pcolormesh(ax, lon, lat, z, *, vmin, vmax, cmap: str):
        if ccrs is not None:
            im = ax.pcolormesh(
                lon,
                lat,
                z,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                # Input data lon/lat are in geographic coordinates.
                transform=ccrs.PlateCarree(),
            )
            ax.coastlines(linewidth=0.8)
            ax.set_global()
            return im
        return ax.pcolormesh(
            lon, lat, z, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap
        )

    # Row 0: temperature
    if "t2m" not in ds:
        raise KeyError(
            f"Output store does not contain 't2m'. Available: {list(ds.data_vars)}"
        )
    tas_idx = list(coords_in["variable"]).index("tas")
    z_in_t = x_in[0, tas_idx].cpu().numpy()
    z_out_t = np.asarray(_select_first_non_spatial(ds["t2m"].sel(time=t0)).values)
    vmin = float(np.nanmin([np.nanmin(z_in_t), np.nanmin(z_out_t)]))
    vmax = float(np.nanmax([np.nanmax(z_in_t), np.nanmax(z_out_t)]))
    _pcolormesh(
        ax_in[0],
        coords_in["lon"],
        coords_in["lat"],
        z_in_t,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    im1 = _pcolormesh(
        ax_out[0],
        ds["lon"],
        ds["lat"],
        z_out_t,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax_in[0].set_title("tas input")
    ax_out[0].set_title("t2m downscaled")
    ax_in[0].set_xlabel("lon")
    ax_in[0].set_ylabel("lat")
    ax_out[0].set_xlabel("lon")
    ax_out[0].set_ylabel("lat")
    cbar = fig.colorbar(im1, cax=ax_cbar[0])
    cbar.set_label(unit_map.get("tas", ""))
    cbar.ax.tick_params(labelsize=9)

    # Row 1: 10m wind speed
    for vreq in ("u10m", "v10m"):
        if vreq not in ds:
            raise KeyError(
                f"Output store does not contain '{vreq}'. Available: {list(ds.data_vars)}"
            )
    uas_idx = list(coords_in["variable"]).index("uas")
    vas_idx = list(coords_in["variable"]).index("vas")
    z_in_ws = np.sqrt(
        np.square(x_in[0, uas_idx].cpu().numpy())
        + np.square(x_in[0, vas_idx].cpu().numpy())
    )
    z_out_u = np.asarray(_select_first_non_spatial(ds["u10m"].sel(time=t0)).values)
    z_out_v = np.asarray(_select_first_non_spatial(ds["v10m"].sel(time=t0)).values)
    z_out_ws = np.sqrt(np.square(z_out_u) + np.square(z_out_v))
    vmin = float(np.nanmin([np.nanmin(z_in_ws), np.nanmin(z_out_ws)]))
    vmax = float(np.nanmax([np.nanmax(z_in_ws), np.nanmax(z_out_ws)]))
    _pcolormesh(
        ax_in[1],
        coords_in["lon"],
        coords_in["lat"],
        z_in_ws,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    im1 = _pcolormesh(
        ax_out[1],
        ds["lon"],
        ds["lat"],
        z_out_ws,
        vmin=vmin,
        vmax=vmax,
        cmap="magma",
    )
    ax_in[1].set_title("ws10m input (from uas/vas)")
    ax_out[1].set_title("ws10m downscaled (from u10m/v10m)")
    ax_in[1].set_xlabel("lon")
    ax_in[1].set_ylabel("lat")
    ax_out[1].set_xlabel("lon")
    ax_out[1].set_ylabel("lat")
    cbar = fig.colorbar(im1, cax=ax_cbar[1])
    cbar.set_label(unit_map.get("ws10m", "m/s"))
    cbar.ax.tick_params(labelsize=9)

    out_path = Path("outputs/20_corrdiff_cmip6_quicklook.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def plot_ws10m_ensemble_3x3(
    ws_members,
    *,
    lon,
    lat,
    cmap_members: str = "magma",
    cmap_std: str = "viridis",
) -> None:
    """Plot 7 ws10m members + mean + std in a 3x3 grid.

    Layout (3x3):
      - 7 ensemble members
      - mean
      - std
    """
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
        from matplotlib.colors import Normalize  # noqa: WPS433
        from matplotlib.gridspec import GridSpec  # noqa: WPS433

        try:
            import cartopy.crs as ccrs  # noqa: WPS433
        except Exception:  # pragma: no cover
            ccrs = None
    except Exception as e:  # pragma: no cover
        print(f"Plotting skipped (missing deps): {e}")
        return

    # Normalize inputs to numpy
    if hasattr(ws_members, "values"):  # xarray
        ws_np = ws_members.values
    elif hasattr(ws_members, "detach"):  # torch
        ws_np = ws_members.detach().cpu().numpy()
    else:
        ws_np = np.asarray(ws_members)

    if ws_np.shape[0] != 7:
        raise ValueError(f"Expected ws_members shape [7,H,W], got {ws_np.shape}")

    ws_mean = ws_np.mean(axis=0)
    ws_std = ws_np.std(axis=0)

    # Shared scaling for members + mean
    vmin = float(np.nanmin(ws_np))
    vmax = float(np.nanmax(ws_np))
    norm_members = Normalize(vmin=vmin, vmax=vmax)

    # Separate scaling for std (usually much smaller)
    std_vmin = 0.0
    std_vmax = float(np.nanmax(ws_std))
    norm_std = Normalize(vmin=std_vmin, vmax=std_vmax)

    # Use an explicit GridSpec with a dedicated right-side colorbar column.
    # This avoids label overlap and reduces vertical whitespace (important with cartopy GeoAxes).
    # Slightly shorter figure + negative hspace to reduce the perceived vertical gaps
    # between cartopy GeoAxes rows.
    fig = plt.figure(figsize=(12.8, 7.6))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        # Add an explicit spacer column between the two colorbars.
        # Make the spacer noticeably wider so the bars don't feel cramped.
        width_ratios=[1, 1, 1, 0.06, 0.12, 0.06],
        wspace=0.05,
        hspace=-0.08,
    )

    if ccrs is not None:
        proj = ccrs.PlateCarree(central_longitude=180)
        axes = [
            fig.add_subplot(gs[r, c], projection=proj)
            for r in range(3)
            for c in range(3)
        ]
    else:
        axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    # Two vertical colorbar axes on the right, with an explicit spacer column between them.
    cax_members = fig.add_subplot(gs[:, 3])
    cax_std = fig.add_subplot(gs[:, 5])

    data_items = [ws_np[i] for i in range(7)] + [ws_mean, ws_std]
    titles = [f"member {i+1}" for i in range(7)] + ["mean", "std"]

    mappables = []
    for ax, da, title in zip(axes, data_items, titles):
        z = da
        if title == "std":
            cmap = cmap_std
            norm = norm_std
        else:
            cmap = cmap_members
            norm = norm_members

        if ccrs is not None:
            m = ax.pcolormesh(
                lon,
                lat,
                z,
                shading="auto",
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
            )
            ax.coastlines(linewidth=0.6)
            ax.set_global()
        else:
            m = ax.pcolormesh(lon, lat, z, shading="auto", cmap=cmap, norm=norm)

        ax.set_title(title, fontsize=9, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        mappables.append(m)

    # Two vertical colorbars (no overlap, readable labels).
    cbar1 = fig.colorbar(mappables[0], cax=cax_members, orientation="vertical")
    cbar1.set_label("ws10m [m/s]", rotation=270, labelpad=18, fontsize=10)
    cbar1.ax.set_title("members+mean", fontsize=9, pad=6)
    cbar1.ax.tick_params(labelsize=9)

    cbar2 = fig.colorbar(mappables[-1], cax=cax_std, orientation="vertical")
    cbar2.set_label("ws10m std [m/s]", rotation=270, labelpad=18, fontsize=10)
    cbar2.ax.tick_params(labelsize=9)

    # Make both colorbars ~20% shorter and vertically centered (reduces visual dominance).
    for cax in (cax_members, cax_std):
        pos = cax.get_position()
        new_h = pos.height * 0.8
        new_y = pos.y0 + (pos.height - new_h) / 2
        cax.set_position([pos.x0, new_y, pos.width, new_h])

    # Tighten outer margins (leave room for two colorbar columns + spacer)
    fig.subplots_adjust(left=0.03, right=0.975, top=0.94, bottom=0.03)

    out_path = Path("outputs/20_corrdiff_cmip6_ws10m_ensemble_3x3.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def plot_ws10m_ensemble_3x3_from_io(
    io: ZarrBackend,
    *,
    time_index: int = 0,
    cmap_members: str = "magma",
    cmap_std: str = "viridis",
) -> None:
    """Load u10m/v10m samples from a ZarrBackend and plot a ws10m 3x3 ensemble grid."""
    import xarray as xr

    ds = xr.open_zarr(io.store, consolidated=False)
    for v in ("u10m", "v10m"):
        if v not in ds:
            raise KeyError(
                f"Missing '{v}' in output store. Available: {list(ds.data_vars)}"
            )

    u = ds["u10m"].isel(time=time_index)
    v = ds["v10m"].isel(time=time_index)
    if "sample" not in u.dims:
        raise ValueError(f"Expected a 'sample' dimension, got dims={u.dims}")
    if u.sizes["sample"] < 7:
        raise ValueError(f"Expected >=7 samples, got {u.sizes['sample']}")

    ws_members = ((u**2 + v**2) ** 0.5).isel(sample=slice(0, 7)).values  # [7,H,W]
    plot_ws10m_ensemble_3x3(
        ws_members,
        lon=ds["lon"].values,
        lat=ds["lat"].values,
        cmap_members=cmap_members,
        cmap_std=cmap_std,
    )


# Load the CorrDiffCMIP6 model package and move it to the configured device.
# %% Run 1: load model
model = CorrDiffCMIP6.load_model(MODEL_PACKAGE).to(DEVICE)
# Fix the RNG seed for reproducible sampling
model.seed = 0

# Construct the CMIP6 datasource and wrap it using the model's TimeWindow metadata.
# %% Run 1: create datasource
data = create_cmip6_timewindow_source(model, **CMIP6_KWARGS)

# Run a single-sample downscaling pass (one member) and write outputs to an in-memory Zarr backend.
# %% Run 1: single-sample downscaling
#
# Compare coarse CMIP6 inputs to the downscaled outputs:
# - tas vs t2m
# - ws10m = 10 m wind speed (magnitude): sqrt(u10m^2 + v10m^2)
#   - input: computed from CMIP6 (uas, vas)
#   - output: computed from downscaled (u10m, v10m)
model.number_of_samples = 1


# NOTE: CorrDiffCMIP6 produces a full output state internally. Here we choose to store only a
# subset to the IO backend to reduce write time and storage (and to keep plotting focused).
OUTPUT_VARIABLES = ["t2m", "u10m", "v10m"]
io_det = ZarrBackend()
io_det = diagnostic_from_data_cmip6_corrdiff(
    time=TIMES,
    diagnostic=model,
    data=data,
    io=io_det,
    output_variables=OUTPUT_VARIABLES,
    device=DEVICE,
)

# Plot a side-by-side comparison between coarse CMIP6 input and downscaled output.
# Keep in mind: the CMIP6 source used here is daily, while the downscaled
# fields are hourly values at the chosen timestamp Treat this comparison as
# qualitative rather than a strict time-mean match.
# %% Run 1: plot single-sample comparison
plot_compare_2x2(
    io_det,
    output_variables=OUTPUT_VARIABLES,
    cmip6_kwargs=CMIP6_KWARGS,
    input_time_fn=normalize_to_noon,
)

# Configure the model for stochastic sampling; enable CPU streaming/progress bars to keep GPU
# memory usage manageable for multi-member ensembles.
# %% Run 2: stochastic downscaling (uncertainty visualization)
#
# Generate multiple stochastic members and visualize their variability via member panels,
# mean, and standard deviation (ws10m).
model.number_of_samples = 7
# Reduce peak GPU memory during multi-sample inference by generating members one-by-one and
# immediately moving each sample to CPU (tradeoff: a bit more host RAM + transfer time).
model.stream_samples_to_cpu = True
model.show_sample_progress = True

# Generate and write an ensemble of downscaled members (u10m, v10m) to in-memory Zarr.
# %% Run 2: generate 7 members (u10m, v10m)
io_ens = ZarrBackend()
io_ens = diagnostic_from_data_cmip6_corrdiff(
    time=TIMES,
    diagnostic=model,
    data=data,
    io=io_ens,
    output_variables=["u10m", "v10m"],
    device=DEVICE,
)

# Load the ensemble from IO, derive ws10m (10 m wind speed magnitude), and plot members + mean + std
# in a 3x3 grid.
# %% Run 2: plot ws10m ensemble (members + mean + std)
plot_ws10m_ensemble_3x3_from_io(io_ens, time_index=0)
