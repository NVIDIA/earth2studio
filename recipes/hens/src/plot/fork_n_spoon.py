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

from datetime import timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import animation

helene_ibtracs = pd.DataFrame(
    {
        "time": pd.date_range("2024-09-21 18:00:00", "2024-09-28 12:00:00", freq="3h"),
        "lat": [
            13.60,
            13.80,
            14.00,
            14.20,
            14.40,
            14.60,
            14.80,
            15.00,
            15.20,
            15.40,
            15.60,
            15.70,
            16.00,
            16.60,
            17.20,
            17.60,
            17.90,
            18.10,
            18.20,
            18.40,
            18.60,
            19.00,
            19.30,
            19.40,
            19.40,
            19.50,
            19.80,
            20.00,
            20.30,
            20.70,
            21.10,
            21.50,
            22.00,
            22.40,
            22.80,
            23.20,
            23.60,
            24.10,
            24.70,
            25.60,
            26.70,
            27.70,
            28.70,
            29.90,
            31.30,
            32.90,
            34.40,
            35.70,
            36.70,
            37.60,
            38.10,
            37.90,
            37.40,
            37.00,
            36.60,
        ],
        "lon": [
            277.3,
            277.4,
            277.4,
            277.4,
            277.4,
            277.4,
            277.4,
            277.4,
            277.4,
            277.4,
            277.5,
            277.7,
            278.0,
            278.1,
            278.2,
            278.2,
            278.1,
            278.0,
            277.9,
            277.7,
            277.3,
            276.8,
            276.3,
            275.8,
            275.4,
            275.0,
            274.7,
            274.4,
            274.1,
            273.9,
            273.8,
            273.7,
            273.5,
            273.4,
            273.3,
            273.3,
            273.5,
            273.7,
            274.1,
            274.6,
            275.1,
            275.4,
            275.7,
            276.2,
            276.7,
            276.9,
            276.8,
            276.1,
            275.1,
            274.2,
            273.4,
            272.5,
            272.0,
            272.0,
            272.4,
        ],
    }
)


def create_track_animation_florida(
    track_files: list[str], out_dir: str, fps: int = 10
) -> None:
    """Create an animation of cyclone tracks evolving over time over florida

    Parameters
    ----------
    track_files : list
        List of paths to track NetCDF files with the hurricane tracks from Earth2Studio
        TC trackers.
    out_dir : str
        Directory to save the animation
    fps : int, optional
        Frames per second for the animation, by default 10
    """
    # Set up the figure and projection
    plt.close("all")
    fig = plt.figure(figsize=(10, 8))
    projection = ccrs.LambertConformal(
        central_longitude=280.0, central_latitude=28.0, standard_parallels=(18.0, 38.0)
    )
    ax = plt.axes(projection=projection)
    # Reduce white borders
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.02, top=0.98)

    # Load all tracks
    all_tracks = [xr.load_dataarray(track_file) for track_file in track_files]

    # Combine all tracks
    if len(all_tracks) > 1:
        tracks = xr.concat(all_tracks, dim="ensemble")
    else:
        tracks = all_tracks[0]

    # Get the maximum number of time steps
    max_steps = tracks.sizes["step"]

    def _make_frame(step: int) -> plt.Figure:
        """Create a single frame of the animation."""
        ax.clear()

        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.RIVERS, lw=0.5)
        ax.add_feature(cfeature.LAND, alpha=0.1)
        ax.gridlines(draw_labels=True, alpha=0.6)
        ax.set_extent([260, 300, 10, 40], crs=ccrs.PlateCarree())

        # Plot tracks up to current step
        for ensemble in tracks.coords["ensemble"].values:
            for path in tracks.coords["path_id"].values:
                tracks_path = tracks.sel(ensemble=ensemble).isel(time=0)

                # Get lat/lon coordinates up to current step
                lats = tracks_path.isel(
                    path_id=path, variable=0, step=slice(0, step + 1)
                ).values
                lons = tracks_path.isel(
                    path_id=path, variable=1, step=slice(0, step + 1)
                ).values

                mask = ~np.isnan(lats) & ~np.isnan(lons)
                if mask.any() and len(lons[mask]) > 2:
                    # Plot track
                    ax.plot(
                        lons[mask],
                        lats[mask],
                        color="b",
                        linestyle="-.",
                        transform=ccrs.PlateCarree(),
                    )
                    # Plot current position
                    if step >= 0:
                        current_lon = lons[step] if step < len(lons) else lons[-1]
                        current_lat = lats[step] if step < len(lats) else lats[-1]
                        if not np.isnan(current_lon) and not np.isnan(current_lat):
                            ax.plot(
                                current_lon,
                                current_lat,
                                "bx",
                                transform=ccrs.PlateCarree(),
                                markersize=4,
                            )

        current_time = pd.Timestamp(tracks.time.values[0]) + timedelta(hours=6 * step)
        mask = helene_ibtracs["time"] <= current_time
        if mask.any():
            # Plot ground truth track in red
            ax.plot(
                helene_ibtracs.loc[mask, "lon"],
                helene_ibtracs.loc[mask, "lat"],
                color="red",
                linewidth=2,
                transform=ccrs.PlateCarree(),
            )
            # Plot current ground truth position
            ax.plot(
                helene_ibtracs.loc[mask, "lon"].iloc[-1],
                helene_ibtracs.loc[mask, "lat"].iloc[-1],
                "ro",  # Red circle for ground truth
                markersize=8,
                transform=ccrs.PlateCarree(),
            )

        # Add manual legend and timestamp
        ax.legend(
            handles=[
                plt.Line2D(
                    [0], [0], color="red", linewidth=2, label="IBTrACS (Ground Truth)"
                ),
                plt.Line2D([0], [0], color="blue", linestyle="-.", label="HENS"),
            ],
            loc="upper left",
        )
        ax.set_title(f"Time: {current_time.strftime('%Y-%m-%d %H:%M')}")

        return fig

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        lambda frame: _make_frame(frame),
        frames=max_steps,
        interval=1000 / fps,  # milliseconds between frames
        blit=False,
        repeat=False,
    )

    # Save animation
    ani.save(
        f"{out_dir}/cyclone_tracks.gif",
        writer="pillow",
        fps=fps,
        dpi=200,
        metadata=dict(artist="Earth2Studio"),
    )
    plt.close()
