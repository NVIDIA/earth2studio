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
Basic deterministic forecast example using Earth2Studio Client SDK.

This example shows how to:
1. Access a remote workflow
2. Submit a simple deterministic forecast request and retrieve results as xarray dataset
3. Extract 2-meter temperature data at a specific location
4. Create and save a matplotlib line plot of the forecast
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from earth2studio.serve.client.e2client import RemoteEarth2Workflow

# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///


def main(
    plot_file: str = "t2m_plot.png",
    lat: float = 37.4,
    lon: float = -122.0,
    start_time: datetime = datetime(2025, 8, 21, 6),
    num_steps: int = 10,
) -> None:
    """Run a basic deterministic forecast and save a t2m plot.

    Args:
        plot_file: Path to save the temperature plot (default: 't2m_plot.png')
        lat: Latitude for temperature extraction (default: 37.4)
        lon: Longitude for temperature extraction (default: -122.0)
    """

    # Create client (configurable via environment variable)
    api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
    api_token = os.getenv("EARTH2STUDIO_API_TOKEN", "")
    workflow = RemoteEarth2Workflow(
        api_url,
        workflow_name="deterministic_earth2_workflow",
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

    # Call workflow and get result
    try:
        ds = workflow(start_time=[start_time], num_steps=num_steps).as_dataset()
    except Exception as e:
        print(f"\n❌ Forecast failed: {e}")
        return

    # Extract t2m data for the specified location
    t2m = ds["t2m"].sel(lat=lat, lon=lon, method="nearest").values.ravel()

    # Extract time coordinate
    time_coord = ds["lead_time"].values.astype("timedelta64[h]")

    # Create line plot of temperature
    print(f"   Creating plot: {plot_file}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_coord, t2m, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Lead Time (h)", fontsize=12)
    ax.set_ylabel("2-meter Temperature (K)", fontsize=12)
    ax.set_title(
        f"Temperature Forecast starting {start_time} at {lat}°N, {lon}°E",
        fontsize=14,
        fontweight="bold",
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
