# import io
# import json
import os

# from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from PIL import Image

# E2CC_CONFIG_TEMPLATE = {
#     "features": [
#         {
#             "name": None,
#             "type": "Image",
#             "projection": "latlong",
#             "sources": {},
#             "alpha_sources": {},
#             "latlon_min": None,
#             "latlon_max": None,
#             "remapping": {
#                 "input_min": 0.0,
#                 "input_max": 1.0,
#                 "output_min": 0.0,
#                 "output_max": 1.0,
#                 "output_gamma": 1.0,
#             },
#             "colormap": None,
#         }
#     ]
# }


# def initialise_e2cc_config(cfg):
#     e2cc_config = deepcopy(E2CC_CONFIG_TEMPLATE)
#     feature = e2cc_config["features"][0]
#     feature["name"] = cfg.name

#     feature["latlon_min"] = [float(cfg.lat_min), float(cfg.lon_min)]
#     feature["latlon_max"] = [float(cfg.lat_max), float(cfg.lon_max)]
#     feature["colormap"] = cfg.colour_map

#     # create out dir if not exist
#     os.makedirs(cfg.out_dir, exist_ok=True)

#     return e2cc_config


# def save_e2cc_config(e2cc_config, out_dir):
#     e2cc_conf_path = os.path.join(out_dir, "0000-config.json")
#     with open(e2cc_conf_path, "w") as conf:
#         json.dump(e2cc_config, conf)
#     print(f"done :)  ---> all data to be found under {out_dir}")

#     return


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


# def plot_to_image(fig, ax):
#     # Extract the plot as an image
#     with io.BytesIO() as buff:
#         fig.savefig(buff, format="raw", transparent=True)
#         buff.seek(0)
#         data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
#     ww, hh = fig.get_size_inches() * fig.dpi
#     arr = data.reshape((int(hh), int(ww), -1))

#     arr = arr.copy()
#     for channel in range(3):
#         arr[:, :, channel] = arr[:, :, -1]
#     arr = arr[..., -1]
#     cont_img = Image.fromarray(arr).convert("L")

#     arr[arr != 0] = 255
#     binary_img = Image.fromarray(arr).convert("L")

#     return cont_img, binary_img


# def image_to_file(cont_img, binary_img, time, out_dir, e2cc_config, value_to):
#     if value_to == "colour":
#         imgs = {"source": cont_img, "alpha_source": binary_img}
#     elif value_to == "alpha":
#         imgs = {"source": binary_img, "alpha_source": cont_img}
#     elif value_to == "both":
#         imgs = {"source": cont_img, "alpha_source": cont_img}
#     elif value_to == "none":
#         imgs = {"source": binary_img, "alpha_source": binary_img}
#     else:
#         raise ValueError(
#             f"value_to is set as {value_to}, but has to "
#             + "be 'colour', 'alpha', 'both' or 'none'."
#         )

#     for component in ["source", "alpha_source"]:
#         file_name = f"{time.replace(':', '-').replace(' ', 'T')}_{component}.jpg"
#         out_path = os.path.join(out_dir, file_name)
#         imgs[component].save(out_path, quality=100, subsampling=0)

#         # add info to e2cc_config
#         e2cc_config["features"][0][component + "s"][time] = "./" + file_name
