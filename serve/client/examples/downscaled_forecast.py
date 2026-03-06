#!/usr/bin/env python3
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
Forecast example with local diagnostic using Earth2Studio Client SDK.

This example shows how to:
1. Access a remote workflow
2. Submit a simple deterministic forecast request and retrieve results as xarray dataset
3. Run precipitation diagnostic model locally to get precipitation
4. Create and save a matplotlib line plot of the precipitation forecast
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from earth2studio.data import HRRR  # type: ignore[import-untyped]
from earth2studio.io import XarrayBackend  # type: ignore[import-untyped]
from earth2studio.models.px import StormCast  # type: ignore[import-untyped]
from earth2studio.run import deterministic  # type: ignore[import-untyped]
from earth2studio.serve.client.e2client import RemoteEarth2Workflow

# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///


def main(
    plot_file: str = "downscaled_t2m_plot.png",
    y: float = 200.0,
    x: float = 200.0,
    start_time: datetime = datetime(2025, 8, 21, 6),
    num_steps: int = 10,
) -> None:
    """Run a remote FourCastNet 3 forecast and run a StormCast downscaling locally.

    Args:
        plot_file: Path to save the temperature plot (default: 'downscaled_t2m_plot.png')
        x: HRRR x coordinate for temperature extraction (default: 200.0)
        y: HRRR y coordinate for temperature extraction (default: 200.0)
    """

    # Create client (configurable via environment variable)
    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="stormcast_fcn3_workflow",
        device="cuda" if torch.cuda.is_available() else "cpu",
        token=api_token,
    )

    # Check if API is healthy
    try:
        health = workflow.client.health_check()
        print(f"✓ API Status: {health.status}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        return

    # Call workflow and get result as a data source
    try:
        conditioning_source = workflow(
            start_time=start_time, num_hours=num_steps, run_stormcast=False
        ).as_data_source()
    except Exception as e:
        print(f"\n❌ Forecast failed: {e}")
        return

    stormcast = StormCast.from_pretrained()
    stormcast.conditioning_data_source = conditioning_source
    io = XarrayBackend()
    hrrr_ic = HRRR()

    deterministic(
        [start_time], num_steps, stormcast, hrrr_ic, io, device=workflow.device
    )

    # Extract t2m data for the specified location
    ds = io.root
    t2m = ds["t2m"].sel(hrrr_x=x, hrrr_y=y, method="nearest").values.ravel()

    # Extract time coordinate
    time_coord = ds["lead_time"].values.astype("timedelta64[h]")

    # Create line plot of temperature
    print(f"   Creating plot: {plot_file}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_coord, t2m, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Lead Time (h)", fontsize=12)
    ax.set_ylabel("2-meter Temperature (K)", fontsize=12)
    ax.set_title(
        f"Temperature Forecast at x={x}, y={y}", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Save the plot
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n🎉 Forecast complete!")
    print(f"   Plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
