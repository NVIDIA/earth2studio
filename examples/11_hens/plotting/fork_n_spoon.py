import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


def ibtracs_helene():
    """Get the IBTrACS coordinates for Hurricane Helene.

    Returns
    -------
    pd.DataFrame
        data frame containing latitude and longitude coordinates of TC centres

    """
    centre_coords = pd.DataFrame(
        {
            "time": pd.date_range(
                "2024-09-21 18:00:00", "2024-09-28 12:00:00", freq="3h"
            ),
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

    return centre_coords


def get_file_list(dir):
    """Get a sorted list of CSV files from the specified directory.

    Parameters
    ----------
    dir : str
        Directory path to search for CSV files

    Returns
    -------
    list
        Sorted list of full paths to CSV files
    """
    return sorted(
        [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(".csv")]
    )


def extract_tracks_from_csv(
    csv_file: str, ic: str, tc_centres, max_dist, min_len, max_stp
):
    """Extract and filter tracks from a CSV file based on initial conditions and TC centers.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing track data
    ic : str
        Initial condition time to filter tracks
    tc_centres : pd.DataFrame
        DataFrame containing TC center positions
    max_dist : float
        Maximum distance from TC center to consider
    min_len : int
        Minimum track length to include
    max_stp : int
        Maximum number of steps to consider

    Returns
    -------
    tuple
        List of filtered tracks and updated max_stp
    """
    # read csv
    tracks = pd.read_csv(csv_file, sep=",")

    # exract runs starting at ic and actual position of TC
    # tracks = tracks.loc[pd.to_datetime(tracks['ic']) == pd.to_datetime(ic)]
    tracks = tracks.loc[tracks["ic"] == ic]
    tc_pos = tc_centres.loc[tc_centres["time"] == ic]
    tc_lat, tc_lon = tc_pos["lat"].item(), tc_pos["lon"].item()

    # get track_ids of hurricane
    new_borns = tracks.loc[tracks["point_number"] == 0]
    new_borns = new_borns[
        new_borns["tc_lat"].between(tc_lat - max_dist, tc_lat + max_dist)
    ]
    new_borns = new_borns[
        new_borns["tc_lon"].between(tc_lon - max_dist, tc_lon + max_dist)
    ]
    track_ids = list(new_borns["track_id"])

    # extract TC tracks
    tracks = tracks.loc[tracks["track_id"].isin(track_ids)]

    # get all tracks above ceratin length
    track_list = []
    for id in sorted(track_ids):
        track = tracks.loc[tracks["track_id"] == id]
        track_len = track.tail(1)["point_number"].item()
        max_stp = max(track_len, max_stp)
        if track_len >= min_len:
            track_list.append(
                {
                    "lat": np.array(list(track["tc_lat"])),
                    "lon": np.array(list(track["tc_lon"])),
                    "ens_member": list(track["ens_member"])[0],
                }
            )
    return track_list, max_stp


def interpolate_track_buildup(track_list, fac):
    """Interpolate track points to increase temporal resolution.

    Parameters
    ----------
    track_list : list
        List of track dictionaries containing lat/lon coordinates
    fac : int
        Interpolation factor (1 means no interpolation)

    Returns
    -------
    list
        List of interpolated track dictionaries
    """
    if fac == 1:
        return track_list

    for track in track_list:
        hilf = np.arange((len(track["lat"]) - 1) * fac + 1)
        for dir in ("lat", "lon"):
            track[dir] = np.interp(hilf, hilf[::fac], track[dir])

    return track_list


def plot_tracks(
    track_list,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    dots_per_deg,
    alpha: float = 0.7,
    line_width: float = 2,
    max_len: int = 9999,
):
    """Plot hurricane tracks on a map.

    Parameters
    ----------
    track_list : list
        List of track dictionaries containing lat/lon coordinates
    lat_min : float
        Minimum latitude for plot bounds
    lat_max : float
        Maximum latitude for plot bounds
    lon_min : float
        Minimum longitude for plot bounds
    lon_max : float
        Maximum longitude for plot bounds
    dots_per_deg : int
        Resolution in dots per degree
    alpha : float, optional
        Line transparency, by default 0.7
    line_width : float, optional
        Line width, by default 2
    max_len : int, optional
        Maximum number of points to plot per track, by default 9999

    Returns
    -------
    tuple
        matplotlib Figure and Axes objects
    """
    plt.close("all")

    fig_size = (lon_max - lon_min, lat_max - lat_min)

    # Set the style to dark background
    plt.style.use("dark_background")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size, dpi=dots_per_deg)

    # Plot the line in white
    for track in track_list:
        ax.plot(
            track["lon"][:max_len],
            track["lat"][:max_len],
            color="white",
            linewidth=line_width,
            alpha=alpha,
        )

    # Remove extra space around the plot
    plt.tight_layout(pad=0)

    # Show only the plot area by removing spines and ticks
    for pos in ["top", "right", "bottom", "left"]:
        ax.spines[pos].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    return fig, ax


# define plots
def make_figure(projection: ccrs.Projection = ccrs.PlateCarree()):
    """Create a figure with a map projection and basic geographic features.

    Parameters
    ----------
    projection : ccrs.Projection, optional
        Cartopy projection to use for the map, by default ccrs.PlateCarree()

    Returns
    -------
    tuple
        matplotlib Figure and Axes objects with configured map projection
    """
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.COASTLINE, lw=0.5)
    ax.add_feature(cfeature.RIVERS, lw=0.5)

    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    return fig, ax


class make_frame:
    """Class for creating animation frames with variable data and track overlays.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to draw on
    ax : matplotlib.axes.Axes
        Axes object with map projection
    var_ds : xarray.DataArray
        dataset to plot
    ensemble_member : int
        Index of ensemble member to plot
    track_list : list
        List of track dictionaries containing lat/lon coordinates
    max_frames : int
        Maximum number of frames to process
    min_val : float
        Minimum value for colormap
    max_val : float
        Maximum value for colormap
    projection : ccrs.Projection
        Cartopy projection used for the map
    reg_ds : xarray.Dataset
        Dataset containing latitude and longitude coordinates
    time_str : str
        Time string for the frame title
    """

    def __init__(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        var_ds: xr.DataArray,
        ensemble_member: int,
        track_list: list,
        max_frames: int,
        min_val: float,
        max_val: float,
        projection: ccrs.Projection,
        reg_ds: xr.Dataset,
        time_str: str,
    ):
        self.fig = fig
        self.ax = ax
        self.var_ds = var_ds
        self.ensemble_member = ensemble_member
        self.track_list = track_list
        self.max_frames = max_frames
        self.min_val = min_val
        self.max_val = max_val
        self.projection = projection
        self.reg_ds = reg_ds
        self.time_str = time_str

    def __call__(self, frame: int) -> plt.pcolormesh:
        """Generate a single frame for the animation.

        Parameters
        ----------
        frame : int
            Frame number to generate (-1 for initialization frame)

        Returns
        -------
        matplotlib.pyplot.pcolormesh
            The pcolormesh object for the current frame
        """
        print(
            f"\rprocessing frame {frame+1} of {min(self.max_frames, self.var_ds.shape[2])}",
            end="",
        )
        plot_ds = self.var_ds[self.ensemble_member, 0, max(frame, 0), :, :]
        pc = self.ax.pcolormesh(
            self.reg_ds.lon,
            self.reg_ds.lat,
            plot_ds,
            transform=self.projection,
            cmap="plasma",
            vmin=self.min_val,
            vmax=self.max_val,
        )

        if frame == -1:
            # create colorbar
            _ = self.fig.colorbar(pc, extend="both", shrink=0.8, ax=self.ax)
        else:
            # plot track
            track = self.track_list[self.ensemble_member]
            max_len = min(frame, len(track["lon"]))
            self.ax.plot(
                track["lon"][:max_len] - 360,
                track["lat"][:max_len],
                color="white",
                linewidth=2,
                alpha=1,
            )

        header = self.time_str + " " + f"{frame*6}:00:00"
        self.ax.set_title(header, fontsize=14)

        return pc
