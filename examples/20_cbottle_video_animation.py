# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
CBottle Video Animation
=======================

Create animated visualizations from CBottleVideo model forecasts.

This example demonstrates how to run the CBottleVideo diffusion model and create
animated videos of the generated weather forecasts on the HEALPix grid. The model
generates 12 frames at a time (0-66 hours in 6-hour steps).

For more information on cBottle see:

- https://arxiv.org/abs/2505.06474v1

In this example you will learn:

- Running CBottleVideo inference on HEALPix grid
- Processing and visualizing HEALPix data
- Creating animated videos of weather forecasts
"""
# /// script
# dependencies = [
#   "earth2studio[cbottle] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "matplotlib",
# ]
# ///

#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Annotated

import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from earth2studio.models.px import CBottleVideo

try:
    from cbottle.dataclass_parser import Help, parse_args
    from earth2grid import healpix
except ImportError:
    print("Warning: cbottle and earth2grid required for this example")
    print("Install with: pip install earth2studio[cbottle]")
    raise


@dataclass
class VideoOptions:
    output: Annotated[str, Help("Output video file (.mp4)")] = "outputs/cbottle_video.mp4"
    year: Annotated[int, Help("Year for forecast initialization")] = 2022
    month: Annotated[int, Help("Month for forecast initialization")] = 6
    day: Annotated[int, Help("Day for forecast initialization")] = 1
    field: Annotated[str, Help("Variable to visualize")] = "msl"
    style: Annotated[str, Help("Matplotlib style")] = "dark_background"
    fps: Annotated[int, Help("Frames per second")] = 2
    dpi: Annotated[int, Help("DPI for video quality")] = 100
    seed: Annotated[int, Help("Random seed for reproducibility")] = 42
    n_frames: Annotated[int, Help("Number of frames to generate")] = 12


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
    }


def get_field_cmap(field: str) -> str:
    """Get colormap for field"""
    if field in ["pr", "tcwv", "prw"]:
        return "gist_ncar"
    elif field in ["tas", "t2m", "uas", "vas", "u10m", "v10m"]:
        return "RdBu_r"
    elif field in ["sst"]:
        return "turbo"
    else:
        return "bone"


def process_field(data: np.ndarray, field: str) -> tuple:
    """Process field data for visualization"""
    vmin, vmax = get_field_ranges().get(field, (data.min(), data.max()))

    if field == "pr":
        data = 10 * np.log10((data * 3600).clip(0.1))
        vmin, vmax = -5, 20
    elif field in ["tas", "t2m"]:
        data = data - 273.15
        vmin, vmax = -40, 40

    cmap = get_field_cmap(field)
    return data, vmin, vmax, cmap


def run_cbottle_video_inference(args: VideoOptions) -> tuple:
    """Run CBottleVideo inference and return outputs"""
    from datetime import datetime

    print("Loading CBottleVideo model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model on HEALPix grid (lat_lon=False)
    package = CBottleVideo.load_default_package()
    model = CBottleVideo.load_model(package, lat_lon=False, seed=args.seed)
    model = model.to(device)

    # Prepare input coordinates
    time = np.array([datetime(args.year, args.month, args.day)], dtype="datetime64[ns]")
    coords = model.input_coords()
    coords["time"] = time
    coords["batch"] = np.array([0])

    print(f"Running unconditional generation for {args.n_frames} frames...")
    print(f"Initialization time: {time[0]}")

    # Create NaN tensor for unconditional sampling
    # HEALPix format: [batch, time, lead_time, variable, hpx]
    x = torch.full(
        (1, 1, 1, len(model.VARIABLES), 4**6 * 12),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )

    # Run inference
    iterator = model.create_iterator(x, coords)
    outputs = []
    times = []
    lead_times = []

    for step, (output, output_coords) in enumerate(iterator):
        lead_time = output_coords["lead_time"][0]
        hours = int(lead_time / np.timedelta64(1, "h"))
        print(f"Generated frame {step}: +{hours}h")

        outputs.append(output.cpu())
        times.append(time[0] + lead_time)
        lead_times.append(hours)

        if step >= args.n_frames - 1:
            break

    return outputs, times, lead_times, model.VARIABLES


def create_animation(args: VideoOptions, outputs: list, times: list, lead_times: list, variables: np.ndarray):
    """Create animated visualization from model outputs"""
    import os

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Find variable index
    var_idx = np.where(variables == args.field)[0]
    if len(var_idx) == 0:
        raise ValueError(
            f"Variable '{args.field}' not found. Available: {list(variables)}"
        )
    var_idx = var_idx[0]

    print(f"Creating animation for variable: {args.field}")

    # Set up plot
    plt.style.use(args.style)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    # Process first frame
    data = outputs[0][0, 0, 0, var_idx].numpy()
    data, vmin, vmax, cmap = process_field(data, args.field)
    x = healpix.to_double_pixelization(data, fill_value=float("nan"))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    img = ax.imshow(
        np.ma.masked_invalid(x),
        extent=[0, 2, 0, 1],
        cmap=plt.get_cmap(cmap),
        norm=norm,
    )
    plt.colorbar(img, ax=ax, orientation="horizontal", shrink=0.5, pad=0.02)

    time_str = pd.Timestamp(times[0]).strftime("%Y-%m-%d %H:%M")
    title = ax.set_title(f"{args.field} +{lead_times[0]:03d}h ({time_str})")
    ax.axis("off")
    fig.tight_layout()

    def update(frame):
        """Update animation frame"""
        data = outputs[frame][0, 0, 0, var_idx].numpy()
        data, vmin, vmax, cmap = process_field(data, args.field)
        x = healpix.to_double_pixelization(data, fill_value=float("nan"))
        x = np.ma.masked_invalid(x)
        img.set_data(x)

        time_str = pd.Timestamp(times[frame]).strftime("%Y-%m-%d %H:%M")
        title.set_text(f"{args.field} +{lead_times[frame]:03d}h ({time_str})")
        return [img, title]

    # Create animation
    print(f"Rendering {len(outputs)} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(outputs), interval=1000 / args.fps, blit=True
    )

    # Save video
    print(f"Saving video to {args.output}...")
    writer = animation.FFMpegWriter(fps=args.fps)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    plt.close()
    print(f"Video saved successfully!")


def main():
    args = parse_args(VideoOptions)

    # Run inference
    outputs, times, lead_times, variables = run_cbottle_video_inference(args)

    # Create animation
    create_animation(args, outputs, times, lead_times, variables)


if __name__ == "__main__":
    main()
