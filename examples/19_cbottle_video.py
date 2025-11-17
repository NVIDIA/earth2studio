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
CBottle Video Inference
=======================

Climate in a Bottle (cBottle) video model inference workflows.

This example will demonstrate the cBottle video diffusion model for generating
temporal sequences of global weather states. The CBottleVideo model predicts 12
frames at a time (in 6-hour increments) and can operate in both unconditional and
conditional modes.

For more information on cBottle see:

- https://arxiv.org/abs/2505.06474v1

In this example you will learn:

- Running unconditional video generation with CBottleVideo
- Running conditional video generation using ERA5 data with CBottleInfill
- Post-processing and visualizing the temporal forecasts
- Creating animated videos of the forecasts

This example can be run interactively or from command line:
  python 19_cbottle_video.py --mode unconditional --create-video --field msl
  python 19_cbottle_video.py --mode conditional --create-video --field t2m
"""
# /// script
# dependencies = [
#   "earth2studio[cbottle] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
# ]
# ///

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import Annotated

from earth2studio.models.px import CBottleVideo
from earth2studio.data import WB2ERA5
from earth2studio.models.dx import CBottleInfill
from earth2studio.data.utils import fetch_data

try:
    from cbottle.dataclass_parser import Help, parse_args

    HAS_PARSER = True
except ImportError:
    HAS_PARSER = False


@dataclass
class InferenceOptions:
    mode: Annotated[
        str, Help("Inference mode: 'unconditional' or 'conditional'")
    ] = "unconditional"
    year: Annotated[int, Help("Year for forecast initialization")] = 2022
    month: Annotated[int, Help("Month for forecast initialization")] = 6
    day: Annotated[int, Help("Day for forecast initialization")] = 1
    field: Annotated[str, Help("Variable to visualize")] = "msl"
    n_frames: Annotated[int, Help("Number of frames to generate")] = 12
    seed: Annotated[int, Help("Random seed for reproducibility")] = 42
    create_video: Annotated[
        bool, Help("Create animated video instead of static plot")
    ] = False
    video_fps: Annotated[int, Help("Video frames per second")] = 2
    video_dpi: Annotated[int, Help("Video DPI quality")] = 100
    style: Annotated[str, Help("Matplotlib style")] = "dark_background"


def get_field_ranges() -> dict:
    """Get visualization ranges for different fields"""
    return {
        "clivi": (-1e-12, 1.5),
        "cllvi": (0, 11),
        "pr": (0, 0.07),
        "pres_msl": (95000, 104000),
        "msl": (95000, 104000),
        "pres_sfc": (39000, 107000),
        "prw": (0, 70),
        "tcwv": (0, 70),
        "tclw": (0, 11),
        "tciw": (-1e-12, 1.5),
        "rlds": (70, 500),
        "rlus": (110, 800),
        "rlut": (70, 360),
        "rsds": (0, 1130),
        "rsdt": (0, 1380),
        "rsus": (0, 620),
        "rsut": (0, 1010),
        "sfcwind": (0, 45),
        "sic": (0, 1),
        "sit": (0, 12),
        "uas": (-40, 40),
        "u10m": (-40, 40),
        "vas": (-40, 40),
        "v10m": (-40, 40),
        "speed": (0, 30),
        "tas": (200, 320),
        "t2m": (200, 320),
        "sst": (270, 305),
        "tpf": (0, 0.07),
    }


def get_field_cmap(field: str) -> str:
    """Get colormap for field"""
    if field in ["pr", "tcwv", "prw", "tclw", "tciw", "tpf"]:
        return "gist_ncar"
    elif field in ["tas", "t2m", "uas", "vas", "u10m", "v10m"]:
        return "RdBu_r"
    elif field in ["sst"]:
        return "turbo"
    elif field in ["msl", "pres_msl", "pres_sfc"]:
        return "viridis"
    else:
        return "bone"


# %%
# Set Up
# ------
# For this example we will use the CBottleVideo prognostic model. Unlike typical
# prognostic models, CBottleVideo is a diffusion-based video model that generates
# 12 frames (0-66 hours in 6-hour steps) at once.


def run_unconditional_inference(args: InferenceOptions):
    """Run unconditional video generation"""
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CBottleVideo model
    print("Loading CBottleVideo model...")
    package = CBottleVideo.load_default_package()
    cbottle_video = CBottleVideo.load_model(package, seed=args.seed)
    cbottle_video = cbottle_video.to(device)

    # %%
    # Unconditional Video Generation
    # -------------------------------
    # First, let's generate a video sequence unconditionally. This means the model will
    # generate plausible weather states based only on the timestamp and SST data, without
    # conditioning on any specific initial atmospheric state. We do this by providing a
    # tensor of NaN values as input.

    # Prepare input coordinates
    time = np.array([datetime(args.year, args.month, args.day)], dtype="datetime64[ns]")
    coords = cbottle_video.input_coords()
    coords["time"] = time
    coords["batch"] = np.array([0])

    # Create NaN tensor for unconditional sampling
    x_uncond = torch.full(
        (1, 1, 1, len(cbottle_video.VARIABLES), 721, 1440),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )

    # Run inference - the model generates 12 frames at once
    print(f"Running unconditional generation for {args.n_frames} frames...")
    iterator = cbottle_video.create_iterator(x_uncond, coords)
    outputs = []
    coords_list = []
    for step, (output, output_coords) in enumerate(iterator):
        lead_time = output_coords["lead_time"][0]
        hours = int(lead_time / np.timedelta64(1, "h"))
        print(f"Step {step}: lead_time = +{hours}h")
        outputs.append(output.cpu())
        coords_list.append(output_coords)
        if step >= args.n_frames - 1:
            break

    return outputs, coords_list, "unconditional"


def run_conditional_inference(args: InferenceOptions):
    """Run conditional video generation with ERA5"""
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CBottleVideo model
    print("Loading CBottleVideo model...")
    package = CBottleVideo.load_default_package()
    cbottle_video = CBottleVideo.load_model(package, seed=args.seed)
    cbottle_video = cbottle_video.to(device)

    # %%
    # Conditional Video Generation with ERA5
    # ---------------------------------------
    # Next, let's demonstrate conditional generation using real ERA5 data. Since ERA5
    # doesn't contain all 45 variables required by CBottleVideo, we first use the
    # CBottleInfill model to generate the missing variables, then use that complete
    # state to condition the video model.

    # Load ERA5 data source
    print("Loading ERA5 data source...")
    era5_ds = WB2ERA5()

    # Load CBottleInfill model to generate all required variables
    input_variables = [
        "u10m",
        "v10m",
        "t2m",
        "msl",
        "z50",
        "u50",
        "v50",
        "z500",
        "u500",
        "v500",
        "z1000",
        "u1000",
        "v1000",
    ]

    print("Loading CBottleInfill model...")
    package_infill = CBottleInfill.load_default_package()
    cbottle_infill = CBottleInfill.load_model(
        package_infill, input_variables=input_variables, sampler_steps=18
    )
    cbottle_infill = cbottle_infill.to(device)
    cbottle_infill.set_seed(args.seed)

    # Fetch ERA5 data
    print("Fetching ERA5 data and running infill...")
    times = np.array([datetime(args.year, args.month, args.day)], dtype="datetime64[ns]")
    era5_x, era5_coords = fetch_data(era5_ds, times, input_variables, device=device)

    # Infill to get all 45 CBottleVideo variables
    infilled_x, infilled_coords = cbottle_infill(era5_x, era5_coords)

    # Reshape for CBottleVideo input: [batch, time, lead_time, variable, lat, lon]
    x_cond = infilled_x.unsqueeze(2)  # Add lead_time dimension
    print(f"Conditioned input shape: {x_cond.shape}")

    # Update coordinates for CBottleVideo
    coords_cond = cbottle_video.input_coords()
    coords_cond["time"] = times
    coords_cond["batch"] = np.array([0])
    coords_cond["variable"] = infilled_coords["variable"]

    # Run conditional inference
    print(f"Running conditional generation for {args.n_frames} frames...")
    cbottle_video.set_seed(args.seed)  # Set seed for reproducibility
    iterator_cond = cbottle_video.create_iterator(x_cond, coords_cond)
    outputs = []
    coords_list = []
    for step, (output, output_coords) in enumerate(iterator_cond):
        lead_time = output_coords["lead_time"][0]
        hours = int(lead_time / np.timedelta64(1, "h"))
        print(f"Step {step}: lead_time = +{hours}h")
        outputs.append(output.cpu())
        coords_list.append(output_coords)
        if step >= args.n_frames - 1:
            break

    return outputs, coords_list, "conditional"


def create_static_visualization(
    outputs: list, coords_list: list, field: str, mode: str
):
    """Create static visualization comparing time steps"""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    # %%
    # Post Processing and Visualization
    # ----------------------------------
    # Let's visualize the results by plotting the selected variable at different
    # time steps to see how the weather patterns evolve.

    var_idx = np.where(coords_list[0]["variable"] == field)[0]
    if len(var_idx) == 0:
        raise ValueError(
            f"Variable '{field}' not found. Available: {list(coords_list[0]['variable'])}"
        )
    var_idx = var_idx[0]

    # Select time steps to visualize (0, 24h, 48h, 66h)
    time_steps = [0, min(4, len(outputs) - 1), min(8, len(outputs) - 1), len(outputs) - 1]

    plt.close("all")
    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=45.0)

    # Create a figure with subplots
    fig, axes = plt.subplots(
        1, 4, subplot_kw={"projection": projection}, figsize=(16, 4)
    )

    # Get field range and colormap
    field_ranges = get_field_ranges()
    vmin, vmax = field_ranges.get(
        field, (outputs[0][0, 0, 0, var_idx].min(), outputs[0][0, 0, 0, var_idx].max())
    )
    cmap = get_field_cmap(field)

    def plot_field(ax, data, coords, title):
        """Helper function to plot a field"""
        im = ax.pcolormesh(
            coords["lon"],
            coords["lat"],
            data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.coastlines()
        ax.gridlines()
        ax.set_title(title)
        return im

    # Plot selected time steps
    for i, step in enumerate(time_steps):
        lead_time = coords_list[step]["lead_time"][0]
        hours = int(lead_time / np.timedelta64(1, "h"))
        im = plot_field(
            axes[i],
            outputs[step][0, 0, 0, var_idx].numpy(),
            coords_list[step],
            f"{mode.capitalize()}: +{hours}h",
        )

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation="horizontal", pad=0.05, label=field)

    plt.tight_layout()
    output_file = f"outputs/19_cbottle_video_{mode}_{field}.jpg"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_file}")


def create_video_animation(
    outputs: list, coords_list: list, field: str, mode: str, args: InferenceOptions
):
    """Create animated video from model outputs"""
    import matplotlib.animation as animation
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import pandas as pd

    var_idx = np.where(coords_list[0]["variable"] == field)[0]
    if len(var_idx) == 0:
        raise ValueError(
            f"Variable '{field}' not found. Available: {list(coords_list[0]['variable'])}"
        )
    var_idx = var_idx[0]

    print(f"Creating animation for variable: {field}")

    # Get field range and colormap
    field_ranges = get_field_ranges()
    vmin, vmax = field_ranges.get(
        field, (outputs[0][0, 0, 0, var_idx].min(), outputs[0][0, 0, 0, var_idx].max())
    )
    cmap_name = get_field_cmap(field)

    # Set up plot
    plt.style.use(args.style)
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(12, 8))
    projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=45.0)
    ax = fig.add_subplot(111, projection=projection)

    # Process first frame
    data = outputs[0][0, 0, 0, var_idx].numpy()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    img = ax.pcolormesh(
        coords_list[0]["lon"],
        coords_list[0]["lat"],
        data,
        transform=ccrs.PlateCarree(),
        cmap=plt.get_cmap(cmap_name),
        norm=norm,
    )
    ax.coastlines()
    ax.gridlines()

    plt.colorbar(img, ax=ax, orientation="horizontal", shrink=0.5, pad=0.05, label=field)

    lead_time = coords_list[0]["lead_time"][0]
    hours = int(lead_time / np.timedelta64(1, "h"))
    time_str = pd.Timestamp(
        coords_list[0]["time"][0] + coords_list[0]["lead_time"][0]
    ).strftime("%Y-%m-%d %H:%M")
    title = ax.set_title(f"{field} {mode.capitalize()} +{hours:03d}h ({time_str})")

    fig.tight_layout()

    def update(frame):
        """Update animation frame"""
        data = outputs[frame][0, 0, 0, var_idx].numpy()
        img.set_array(data.ravel())

        lead_time = coords_list[frame]["lead_time"][0]
        hours = int(lead_time / np.timedelta64(1, "h"))
        time_str = pd.Timestamp(
            coords_list[frame]["time"][0] + coords_list[frame]["lead_time"][0]
        ).strftime("%Y-%m-%d %H:%M")
        title.set_text(f"{field} {mode.capitalize()} +{hours:03d}h ({time_str})")
        return [img, title]

    # Create animation
    print(f"Rendering {len(outputs)} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(outputs), interval=1000 / args.video_fps, blit=True
    )

    # Save video
    output_file = f"outputs/19_cbottle_video_{mode}_{field}.mp4"
    print(f"Saving video to {output_file}...")
    writer = animation.FFMpegWriter(fps=args.video_fps)
    anim.save(output_file, writer=writer, dpi=args.video_dpi)
    plt.close()
    print(f"Video saved successfully!")


def main():
    """Main execution function"""
    # Parse arguments if available
    if HAS_PARSER:
        args = parse_args(InferenceOptions)
    else:
        # Use defaults if parser not available
        args = InferenceOptions()

    # Run inference based on mode
    if args.mode == "unconditional":
        outputs, coords_list, mode = run_unconditional_inference(args)
    elif args.mode == "conditional":
        outputs, coords_list, mode = run_conditional_inference(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose 'unconditional' or 'conditional'")

    # Create visualization or video
    if args.create_video:
        create_video_animation(outputs, coords_list, args.field, mode, args)
    else:
        create_static_visualization(outputs, coords_list, args.field, mode)


if __name__ == "__main__":
    main()
