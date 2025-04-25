# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, open_dict
from physicsnemo.distributed import DistributedManager
from reproduce_utilities import (
    ensure_all_torch_seeds_are_unique,
    get_reproducibility_settings,
)

from earth2studio.data import DataSource
from earth2studio.io import IOBackend, KVBackend, XarrayBackend
from earth2studio.models.auto import Package
from earth2studio.models.dx import TCTrackerWuDuan
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.time import to_time_array


def initialise_perturbation(
    model: PrognosticModel,
    data: DataSource,
    start_time: np.ndarray[np.datetime64],
    cfg: DictConfig,
) -> Perturbation:
    """
    Initialise perturbation method. Some methods need to be initialized with model, data, start_time, etc.
    which can not always be defined in the config requireing partial instantiation.

    Parameters
    ----------
    model : PrognosticModel
        forecast model.
    data : DataSource
        Data source from which to obtain ICs
    start_time : np.ndarray[np.datetime64]
        IC times
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    Perturbation
        Perturbation method.
    """
    perturbation = hydra.utils.instantiate(cfg.perturbation)

    if isinstance(perturbation, partial):  # inform about model, IC etc
        if perturbation.func.__name__ == "HENSPerturbation":
            perturbation = perturbation(model=model, start_time=start_time, data=data)
        elif perturbation.func.__name__ == "BredVector":
            perturbation = perturbation(model=model)
        else:
            raise ValueError(
                f"perturbation method {perturbation.func.__name__} not implemented for partial instantiation"
            )

    return perturbation


def build_package_list(cfg: DictConfig) -> list[str]:
    """
    Find all available model packages.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    list[str]
        Available model packages.
    """
    if "package" in cfg.forecast_model:  # pointing to single package
        if cfg.forecast_model.package == "default":
            return ["default"]

        elif os.path.isfile(os.path.join(cfg.forecast_model.package, "config.json")):
            return [cfg.forecast_model.package]

        else:  # pointing to directory of packages
            max_num_ckpts = 29
            if "max_num_checkpoints" in cfg.forecast_model:
                max_num_ckpts = cfg.forecast_model.max_num_checkpoints
            packages = []
            for pkg in os.listdir(cfg.forecast_model.package):
                pth = os.path.abspath(os.path.join(cfg.forecast_model.package, pkg))
                if os.path.isdir(pth) and os.path.isfile(
                    os.path.join(pth, "config.json")
                ):
                    packages.append(pth)
            if len(packages) == 0:
                ValueError(
                    f"Found no valid model packages under {cfg.forecast_model.package}."
                )
            return (sorted(packages))[:max_num_ckpts]

    else:
        return ["default"]


def build_model_dict(cfg: DictConfig) -> dict:
    """
    Build a dictionary of loaded model, model class and package name.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    dict
        Dictionary containing model, model class and model package.
    """
    return {
        "model": None,
        "class": hydra.utils.get_class(cfg.forecast_model.architecture),
        "package": None,
    }


def get_model(cfg: DictConfig) -> tuple[dict, list[str]]:
    """
    get a model dictionary and a list of available model packages.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    tuple[dict, list[str]]
        Dictionary containing model, model class and model package.
    """
    return build_model_dict(cfg), build_package_list(cfg)


def set_initial_times(cfg: DictConfig) -> list[np.datetime64]:
    """
    build list of IC times.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    list[np.datetime64]
        Dictionary containing model, model class and model package.
    """
    # list of ICs
    if "start_times" in cfg:
        if "ic_block_start" in cfg:
            raise ValueError(
                "either provide a list of start times or define a block, not both"
            )
        ics = to_time_array(sorted(cfg.start_times))

    # block of ICs
    else:
        ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
        ics = np.arange(
            ics[0],
            ics[1] + np.timedelta64(cfg.ic_block_step, "h"),
            np.timedelta64(cfg.ic_block_step, "h"),
        )

    return ics


def initialise_output(
    cfg: DictConfig, time: np.datetime64, model_dict: dict, output_coords_dict
) -> dict[str, IOBackend]:
    """
    Initialise data output.

    This function sets up the data output based on the provided configuration. It creates an IO handler for storing the
    forecast data either in memory or on disk. If file output is enabled in the configuration, it constructs the file path
    and name, creates the necessary directories, and initializes the IO backend according to the specified format.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object containing settings for the file output.
    time : np.datetime64
        Initial Condition (IC) time for which the output is being initialized.
    model_dict : dict
        Dictionary containing the prognostic model, its class, and the name of its package.
    output_coords_dict : dict
        Dictionary containing the output coordinates for different cropbox areas.

    Returns
    -------
    dict[str, IOBackend]
        A dictionary where the keys are the names of different cropbox areas (e.g., 'Global', 'North', 'South'),
        and the values are the corresponding IOBackend objects (e.g., XarrayBackend, KVBackend) for storing the data.

    Notes
    -----
    - If the configuration does not include file output settings (`file_output`), the function returns an empty dictionary.
    - The function constructs a file name based on the project name, initial condition time, and model package name.
    - It supports different file formats such as NetCDF and Zarr.
    - The function ensures that the output directory exists before attempting to write data.
    - In a distributed setting, only the process with rank 0 creates the necessary directories to avoid race conditions.
    """
    if "file_output" not in cfg:
        return None

    # Create the IO handler, store in memory
    if "path" not in cfg.file_output:
        with open_dict(cfg):
            cfg["file_output"]["path"] = "outputs/"

    pkg = model_dict["package"]
    if pkg != "vanilla":
        pkg = "_pkg_" + pkg.split("_")[-1]

    file_name = cfg.project + "_" + str(time)[:13] + pkg

    io_dict = {}
    for k in output_coords_dict.keys():
        out_path_base = os.path.join(cfg.file_output.path, k)
        out_path = os.path.join(out_path_base, file_name)
        if DistributedManager().rank == 0:
            os.makedirs(out_path_base, exist_ok=True)

        io = hydra.utils.instantiate(cfg.file_output.format)
        if isinstance(io, partial):  # add out file names
            if cfg.file_output.format._target_.split(".")[-1].startswith("NetCDF"):
                file_name = out_path + ".nc"
            elif cfg.file_output.format._target_.split(".")[-1].startswith("Zarr"):
                file_name = out_path + ".zarr"
            else:
                raise ValueError(
                    f"no file name extension implemented for {io}. It's a one-liner tho, do it quickly ;)"
                )
            io = io(file_name=file_name)
        io_dict[k] = io

    return io_dict


def pair_packages_ics(
    ics: list,
    model_packages: list,
    ensemble_size: int,
    batch_ids_produce: list[int],
    batch_size: int,
) -> list:
    """
    Pair initial conditions with model packages. In parallel setting, distribute among
    ranks.

    Parameters
    ----------
    ics : list
        Hydra config object
    model_packages : list
        List of available model packages.
    ensemble_size : int
        number of members in ensemble.
    batch_ids_produce: list[int]
        List of batch_ids that will be processed
    batch_size : int
        Size of each batch.

    Returns
    -------
    list
        List of configurations containing IC - model package pairs along with ensemble size offsets and batch IDs.
    """
    configs = []
    num_batch_per_ic = int(np.ceil(ensemble_size / batch_size))
    batch_ids_complete = list(range(0, num_batch_per_ic * len(model_packages)))
    for ic in ics:
        for ii, pkg in enumerate(model_packages):
            # Determine the batch IDs for the current package and initial condition
            batch_ids_model = batch_ids_complete[
                ii * num_batch_per_ic : (ii + 1) * num_batch_per_ic
            ]
            # Find the intersection with batch_ids_produce to filter relevant batches
            batch_ids_produce_model = list(
                set(batch_ids_model).intersection(set(batch_ids_produce))
            )
            # If there are batch IDs to process, add the configuration to the list
            if batch_ids_produce_model:
                configs.append((pkg, ic, ii * ensemble_size, batch_ids_produce_model))

    dist = DistributedManager()
    if dist.world_size > 1:
        if len(configs) % dist.world_size == 0:
            nconfigs_proc = len(configs) // dist.world_size
        else:
            nconfigs_proc = len(configs) // dist.world_size + 1

        idx = dist.rank * nconfigs_proc
        configs = configs[idx : min(idx + nconfigs_proc, len(configs))]

        if not len(configs) > 0:
            logger.warning(f"nothing to do for rank {dist.rank}. exiting.")
            exit()

    logger.info(
        f"rank {dist.rank}: predicting from following models/initial times: {configs}"
    )

    return configs


def initialise(cfg: DictConfig) -> tuple[list, dict, DataSource, OrderedDict]:
    """
    Set initial conditions, load models, and set up file output based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the settings for initial conditions, models, file output, and other relevant parameters.

    Returns
    -------
    tuple[list, dict, DataSource, dict, dict, dict, int]
        A tuple containing the following elements:
        - ensemble_configs: list of tuples containing model package configurations.
        - model_dict: dictionary containing the model, model class, and package name.
        - dx_model_dict: dictionary containing diagnostic models.
        - cyclone_tracking: instance of TCTrackerWuDuan if enabled, otherwise None.
        - data: DataSource object for obtaining initial conditions.
        - output_coords_dict: dictionary of output coordinates for different cropbox areas.
        - base_random_seed: base random seed for reproducibility.

    Raises
    ------
    ValueError
        If a project name is not specified in the config.
        If neither file_output nor cyclone_tracking is specified in the config.

    Notes
    -----
    - The function initializes initial conditions based on either a list of start times or a block of ICs.
    - It loads available model packages and sets up a default model for coordinate initialization.
    - It handles cropbox configurations for output coordinates if specified.
    - It retrieves reproducibility settings from the configuration.
    - It pairs initial conditions with model packages for distributed processing.
    - It initializes diagnostic models if specified in the configuration.
    """
    if "project" not in cfg:
        raise ValueError("specify a project name in the config: project: project_name")

    if "file_output" not in cfg and "cyclone_tracking" not in cfg:
        raise ValueError(
            "specify a file_output too store the results in the config: file_output"
        )

    ics = set_initial_times(cfg)

    model_dict, model_packages = get_model(cfg)
    default_model = update_model_dict(model_dict, model_packages[0])["model"]
    lon_coords = default_model.output_coords(default_model.input_coords())["lon"]
    lat_coords = default_model.output_coords(default_model.input_coords())["lat"]

    coords_dict = initialize_cropbox(
        cfg=cfg, lon_coords=lon_coords, lat_coords=lat_coords
    )

    # initialize output coordinates
    output_coords_dict = {}
    for name, value in coords_dict.items():
        lat_coords, lon_coords = value
        output_coords = initialize_output_coords(cfg, lon_coords, lat_coords)
        output_coords_dict[name] = OrderedDict(output_coords)

    # get reproducibility settings
    (
        base_random_seed,
        batch_ids_produce,
        torch_use_deterministic_algorithms,
    ) = get_reproducibility_settings(cfg)
    # set torch determinisic algorithms
    if torch_use_deterministic_algorithms:
        torch.use_deterministic_algorithms(torch_use_deterministic_algorithms)

    # get ensemble configs
    ensemble_configs = pair_packages_ics(
        ics, model_packages, cfg.nensemble, batch_ids_produce, cfg.batch_size
    )
    # get data source
    data = hydra.utils.instantiate(cfg.data_source)

    # initialize cyclone tracking
    if "cyclone_tracking" in cfg:
        cyclone_tracking = (
            TCTrackerWuDuan()
        )  # TODO choose and configure TC tracker in config

    else:
        cyclone_tracking = None

    # initialize diagnostic models
    dx_model_dict = initialize_diagnostic_models(cfg)

    # initialize output structures
    all_tracks_dict, writer_executor, writer_threads = initialize_output_structures(cfg)

    # make sure that all the seeds are unique
    ensure_all_torch_seeds_are_unique(ensemble_configs, base_random_seed)

    return (
        ensemble_configs,
        model_dict,
        dx_model_dict,
        cyclone_tracking,
        data,
        output_coords_dict,
        base_random_seed,
        all_tracks_dict,
        writer_executor,
        writer_threads,
    )


def initialize_diagnostic_models(cfg: DictConfig) -> dict:
    """
    Initialize diagnostic models based on the provided configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the settings for diagnostic models.

    Returns
    -------
    dx_model_dict : dict
        A dictionary containing the diagnostic models, where the keys are the model names and the values are the models.
    """
    dx_model_dict = {}
    if "diagnostic_models" in cfg:
        for k in cfg["diagnostic_models"]:
            cfg_dx_model = cfg["diagnostic_models"][k]
            if "architecture" in cfg_dx_model:
                dx_model = hydra.utils.get_class(cfg_dx_model.architecture)
                if "package" in cfg_dx_model:
                    if cfg_dx_model.package == "default":
                        package = dx_model.load_default_package()
                    else:
                        package = Package(cfg_dx_model.package)
                else:
                    package = dx_model.load_default_package()
                dx_model = dx_model.load_model(package=package)
            elif "_target_" in cfg["diagnostic_models"][k]:
                dx_model = hydra.utils.instantiate(cfg_dx_model)
            else:
                raise NotImplementedError(
                    f"diagnostic model {k} not configured correctly. Either 'architecture' or '_target_' must be specified. See examples/hens/configs/diagnostic_models.yaml for an example. "
                )
            dx_model_dict[k] = dx_model
    return dx_model_dict


def initialize_output_coords(
    cfg: DictConfig,
    lon_coords: np.ndarray[np.float64],
    lat_coords: np.ndarray[np.float64],
) -> dict:
    """
    initialize output coordinates

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    lon_coords: np.ndarray[np.float64]
        a 1d array containing the longitude values of the grid. ordered ascending
    lat_coords: np.ndarray[np.float64]
        a 1d array containing the latitude values of the grid. ordered descending

    Returns
    -------
    output_coords: dict
        output coordinates
    """
    output_vars = np.array(
        list(dict.fromkeys(cfg.file_output.output_vars))
    )  # this loads the variables from the config and discards duplicates
    output_coords = (
        {"variable": np.array(output_vars)}
        if "file_output" in cfg and "output_vars" in cfg.file_output
        else {}
    )
    output_coords["lon"] = lon_coords
    output_coords["lat"] = lat_coords
    return output_coords


def initialize_cropbox(
    cfg: DictConfig,
    lon_coords: np.ndarray[np.float64],
    lat_coords: np.ndarray[np.float64],
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """
    If cropbox is defined in config this function will assert that ranges are plausible.
    Then it returns reduced versions of the longitude and latitude coordinate arrays that align with the area of interest
    Otherwise lon_coords and lat_coords will not be changed

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    lon_coords: np.ndarray[np.float64]
        a 1d array containing the longitude values of the grid. ordered ascending
    lat_coords: np.ndarray[np.float64]
        a 1d array containing the latitude values of the grid. ordered descending
    Returns
    -------
    lon_coords: np.ndarray[np.float64]
        a 1d array containing the longitude values of the subgrid. ordered ascending
    lat_coords: np.ndarray[np.float64]
        a 1d array containing the latitude values of the subgrid. ordered descending
    """
    coords_dict = {}
    if "file_output" in cfg and "cropboxes" in cfg["file_output"]:
        cbs = cfg["file_output"]["cropboxes"]
        for k, cb in cbs.items():
            lon_range_model, lon_range_out = determine_lon_ranges(cb, lon_coords)
            lat_coords_sub, lon_coords_sub = crop_area(
                cb, lon_coords, lat_coords, lon_range_model, lon_range_out
            )
            coords_dict[k] = (lat_coords_sub, lon_coords_sub)
    else:
        coords_dict["global"] = (lat_coords, lon_coords)

    return coords_dict


def check_extent(cfg_cropbox: DictConfig) -> None:
    """
    Validate the extent of the cropbox configuration to ensure it is within plausible ranges.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lat_min`, `lat_max`, `lon_min`, and `lon_max`.

    Raises
    ------
    ValueError
        If any of the cropbox settings are out of the valid range.

    Notes
    -----
    - The function checks that `lat_min` is greater than or equal to -90 and `lat_max` is less than or equal to 90.
    - It also ensures that `lat_min` is less than `lat_max`.
    - For `lon_min` and `lon_max`, the function checks that `lon_min` is less than `lon_max`.
    - Additionally, it verifies that `lon_min` and `lon_max` are within the valid range of -180 to 180 or 0 to 360, depending on whether `lon_min` is negative or positive.
    """
    if not (cfg_cropbox.lat_min >= -90):
        raise ValueError("lat_min needs to be >=-90")
    if not (cfg_cropbox.lat_max <= 90):
        raise ValueError("lat_max needs to be <=90")
    if not (cfg_cropbox.lat_min < cfg_cropbox.lat_max):
        raise ValueError("lat_min needs to be smaller than lat_max")
    if not (cfg_cropbox.lon_min < cfg_cropbox.lon_max):
        raise ValueError("lon_max needs to be larger than lon_min")
    if cfg_cropbox.lon_min < 0:
        if not (cfg_cropbox.lon_min >= -180):
            raise ValueError("lon_min needs to be >= -180")
        if not (cfg_cropbox.lon_min < 180):
            raise ValueError("lon_min needs to be < 180")
        if not (cfg_cropbox.lon_max <= 180):
            raise ValueError("lon_max needs to be <= 180")
    else:
        if not (cfg_cropbox.lon_min >= 0):
            raise ValueError("lon_min needs to be >= 0")
        if not (cfg_cropbox.lon_min < 360):
            raise ValueError("lon_min needs to be < 360")
        if not (cfg_cropbox.lon_max <= 360):
            raise ValueError("lon_max needs to be <= 360")


def determine_lon_ranges(
    cfg_cropbox: DictConfig, lon_coords: np.ndarray[np.float64]
) -> tuple[int, int]:
    """
    Determine the longitude ranges of the model and output based on the provided cropbox configuration and longitude coordinates.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lon_min` and `lon_max`.
    lon_coords : np.ndarray[np.float64]
        A 1D array containing the longitude values of the grid, ordered ascending.

    Returns
    -------
    lon_range_model : int
        The range of the input longitude coordinates (either 180 or 360).
    lon_range_out : int
        The range of the desired output longitude coordinates (either 180 or 360).

    Notes
    -----
    - The function determines whether the longitude coordinates of the model and the desired output range span from 0 to 180
      or from -180 to 180 and converts accordingly.
    - This is important for ensuring consistency in longitude ranges particularly when transitioning between different coordinate
      systems or models.
    """
    # determine longitude ranges of input and output
    lon_range_model = (
        360 if ((lon_coords.max() > 180) and (lon_coords.min() >= 0)) else 180
    )
    lon_range_out = (
        360 if ((cfg_cropbox.lon_max > 180) and (cfg_cropbox.lon_min >= 0)) else 180
    )
    return lon_range_model, lon_range_out


def crop_area(
    cfg_cropbox: DictConfig,
    lon_coords: np.ndarray[np.float64],
    lat_coords: np.ndarray[np.float64],
    lon_range_model: int,
    lon_range_out: int,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """
    Crop the area based on the specified configuration and coordinate ranges.

    This function adjusts the longitude coordinates if necessary to align with the
    desired output range and then filters the latitude and longitude coordinates
    to match the cropbox defined in the configuration.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lat_min`,
        `lat_max`, `lon_min`, and `lon_max`.

    lon_coords : np.ndarray[np.float64]
        A 1D array containing the longitude values of the grid, ordered ascending.

    lat_coords : np.ndarray[np.float64]
        A 1D array containing the latitude values of the grid, ordered descending.

    lon_range_model : int
        The range of the input longitude coordinates (either 180 or 360).

    lon_range_out : int
        The range of the desired output longitude coordinates (either 180 or 360).

    Returns
    -------
    lat_coords_sub : np.ndarray[np.float64]
        A 1D array containing the latitude values of the subgrid, ordered descending.

    lon_coords_sub : np.ndarray[np.float64]
        A 1D array containing the longitude values of the subgrid, ordered ascending.

    Raises
    ------
    ValueError
        If the longitude range conversion is not supported.

    Notes
    -----
    - The function handles the conversion between 180-degree and 360-degree longitude ranges.
    - The function filters the latitude and longitude coordinates to fit within the specified cropbox.
    """
    if lon_range_model != lon_range_out:
        # determine where to split for conversion
        if lon_range_model == 180 and lon_range_out == 360:
            split_longitude = 0
        elif lon_range_model == 360 and lon_range_out == 180:
            split_longitude = 180

        ind_below = (lon_coords < split_longitude).nonzero()[0]
        ind_above = (lon_coords >= split_longitude).nonzero()[0]
        if lon_range_model == 180 and lon_range_out == 360:
            lon_coords = np.concatenate(
                [lon_coords[ind_above], lon_coords[ind_below] + 360]
            )
        elif lon_range_model == 360 and lon_range_out == 180:
            lon_coords = np.concatenate(
                [lon_coords[ind_above] - 360, lon_coords[ind_below]]
            )

    # filter/crop area
    idx_lon = np.where(
        (lon_coords >= cfg_cropbox.lon_min) & (lon_coords <= cfg_cropbox.lon_max)
    )
    idx_lat = np.where(
        (lat_coords >= cfg_cropbox.lat_min) & (lat_coords <= cfg_cropbox.lat_max)
    )
    lat_coords_sub = lat_coords[min(idx_lat[0]) : max(idx_lat[0]) + 1]
    lon_coords_sub = lon_coords[min(idx_lon[0]) : max(idx_lon[0]) + 1]
    return lat_coords_sub, lon_coords_sub


def update_model_dict(model_dict: dict, package: Package) -> dict:
    """
    check if model on GPU is same as needed for next inference.
    If not, load new model package and update model dict.

    Parameters
    ----------
    model_dict : dict
        dictionary specifying model, model class, and model package
    package : Package
        model package to be used in next inference

    Returns
    -------
    dict
        model dict.
    """
    if package != model_dict["package"]:
        model_dict["package"] = package
        package = (
            model_dict["class"].load_default_package()
            if package == "default"
            else Package(package)
        )
        model_dict["model"] = model_dict["class"].load_model(package=package)

    return model_dict


def store_tracks(area_name: str, tracks: list[pd.DataFrame], cfg: DictConfig) -> None:
    """
    method which writes cyclone tracks to file.

    Parameters
    ----------
    area_name : str
        The name of the area for which cyclone tracks are being stored.
    tracks : list[pd.DataFrame]
        list of tabular data of cyclone tracks
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    None

    Notes
    -----
    - The function concatenates all cyclone track data into a single DataFrame.
    - It assigns unique global track IDs to each track.
    - The tracks are then written to a CSV file in the specified output directory.
    """
    tracks = pd.concat(tracks)
    map_global = (
        tracks.groupby(["ic", "ens_member", "track_id"])
        .count()
        .reset_index()
        .reset_index()
        .rename(columns={"index": "track_id_global"})[
            ["track_id_global", "ic", "track_id", "ens_member"]
        ]
    )

    tracks = tracks.merge(map_global, on=["ic", "track_id", "ens_member"])[
        [
            "ic",
            "ens_member",
            "track_id_global",
            "vt",
            "point_number",
            "tc_lat",
            "tc_lon",
            "tc_msl",
            "tc_speed",
            "batch_id",
            "batch_size",
            "random_seed",
            "model_package",
        ]
    ]
    cols_uint16 = ["point_number", "ens_member"]
    tracks[cols_uint16] = tracks[cols_uint16].astype("uint16")
    tracks = tracks.rename(columns={"track_id_global": "track_id"})
    dir_path = os.path.join(cfg.cyclone_tracking.out_dir, area_name)
    if DistributedManager().rank == 0:
        os.makedirs(dir_path, exist_ok=True)
    tracks_file = (
        os.path.join(dir_path, cfg.project)
        + f"_tracks_rank_{str(DistributedManager().rank).zfill(3)}.csv"
    )
    tracks.to_csv(tracks_file, index=False)

    return


def write_to_disk(
    cfg: DictConfig,
    ic: str,
    model_dict: dict,
    io_dict: IOBackend,
    writer_threads: list[Future],
    writer_executor: ThreadPoolExecutor | None,
) -> tuple[list[Future], ThreadPoolExecutor | None]:
    """
    method which writes in-memory backends to file.

    Parameters
    ----------
    cfg : DictConfig
        config.
    ic : str
        initial condition.
    model_dict : dict
        dictionary containing loaded model, its class and its package
    io : IOBackend
        object for data output
    writer_threads : list[Future]
        threads for parallel file output
    writer_executor : ThreadPoolExecutor
        executor for parallel file output
    ensemble_idx_base : int
        initial value for counting ensemble members
    Returns
    -------
    writer_threads: list[Future]
    writer_executor: ThreadPoolExecutor
    """

    pkg = model_dict["package"]
    if pkg != "vanilla":
        pkg = "_pkg_" + pkg.split("_")[-1]

    file_name = cfg.project + "_" + str(ic)[:13] + pkg

    for k, io in io_dict.items():
        out_path = os.path.join(cfg.file_output.path, k, file_name)

        kw_args = {"path": out_path + ".nc", "format": "NETCDF4"}

        if writer_executor is not None:
            if isinstance(io, XarrayBackend):
                tmp = io.root
            elif isinstance(io, KVBackend):
                tmp = io.to_xarray()
            tmp = extend_xarray_for_reproducibility(tmp, io, cfg, model_dict)
            writer_threads.append(writer_executor.submit(tmp.to_netcdf, **kw_args))
        else:
            if isinstance(io, XarrayBackend):
                tmp = io.root
                tmp = extend_xarray_for_reproducibility(tmp, io, cfg, model_dict)
                tmp.to_netcdf(**kw_args)
            elif isinstance(io, KVBackend):
                tmp = io.to_xarray()
                tmp = extend_xarray_for_reproducibility(tmp, io, cfg, model_dict)
                tmp.to_netcdf(**kw_args)

    return writer_threads, writer_executor


def extend_xarray_for_reproducibility(
    x: xr.Dataset,
    io: IOBackend,
    cfg: DictConfig,
    model_dict: dict,
):
    """
    adds meta data to netcdf attributes

    Parameters
    ----------
    x : xarray.DataSet | xarray.Dataset
        the array that that we want to augment with metadata
    io : IOBackend
        object for data output
    cfg : DictConfig
        config.
    model_dict : dict
        dictionary containing loaded model, its class and its package
    Returns
    -------
    xarray.DataSet | xarray.Dataset
        the augmented array

    """
    batch_ids = [
        get_batchid_from_ensid(cfg.nensemble, cfg.batch_size, ensid)
        for ensid in io.coords["ensemble"]
    ]
    x = x.assign_attrs(batch_ids=batch_ids)
    x = x.assign_attrs(torch_version=torch.__version__)
    x = x.assign_attrs(model_package=model_dict["package"])
    x = x.assign_attrs(batch_size=cfg.batch_size)
    x = x.assign_attrs(nensemble=cfg.nensemble)
    x = x.assign_attrs(random_seed=cfg.random_seed)
    return x


def initialize_output_structures(cfg: DictConfig):
    """
    Initialize data structures for cyclone tracking and file output.

    This function sets up the necessary data structures based on the provided
    configuration. It initializes a dictionary to store cyclone track data
    and, if configured, a thread pool executor for parallel file I/O operations.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the settings for cyclone tracking
        and file output.

    Returns
    -------
    all_tracks_dict : Dict[str, List[Any]]
        A dictionary to store cyclone track data. The keys correspond to different
        cropbox areas (if specified), and the values are lists that will store
        the cyclone track data for each area.

    writer_executor : ThreadPoolExecutor | None
        An instance of `ThreadPoolExecutor` for parallel file I/O operations,
        or `None` if parallel I/O is not configured.

    writer_threads : List[Future]
        A list to hold future objects for the parallel file I/O tasks.
        Initially empty.

    Notes
    -----
    - If the configuration includes cyclone tracking and specifies cropboxes,
        `all_tracks_dict` will have keys corresponding to each cropbox area.
        Otherwise, it will have a single key, 'Global'.
    - If file output parallelism is enabled in the configuration, `writer_executor`
        will be an instance of `ThreadPoolExecutor`. Otherwise, it will be `None`.
    """

    if "file_output" in cfg and "cropboxes" in cfg["file_output"]:
        all_tracks_dict = {}
        for k in cfg["file_output"]["cropboxes"].keys():
            all_tracks_dict[k] = []
    else:
        all_tracks_dict = {"global": []}

    # Initialize threadpool for writers
    writer_executor = (
        ThreadPoolExecutor(max_workers=8)
        if ("file_output" in cfg and cfg.file_output.thread_io)
        else None
    )
    writer_threads: list[Future] = []

    return all_tracks_dict, writer_executor, writer_threads


def get_batchid_from_ensid(nensemble_per_package, batch_size, ensid):
    """
    Calculate the gloabl batch ID from a global ensemble member ID (ensid).

    Parameters
    ----------
    nensemble_per_package : int
        Number of ensemble members per model package (0-based index).
    batch_size : int
        Number of ensemble members per batch.
    ensid : int
        Global ensemble member ID (0-based index).
    num_packages : int
        Number of model packages.

    Returns
    -------
    int
        The global batch ID corresponding to the given global ensemble member ID.

    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if nensemble_per_package <= 0:
        raise ValueError("nensemble_per_package must be greater than zero.")

    num_batches = int(np.ceil(nensemble_per_package / batch_size))
    batch_id = int(
        np.floor((ensid % nensemble_per_package) / batch_size)
        + (np.floor(ensid / nensemble_per_package)) * num_batches
    )

    return batch_id
