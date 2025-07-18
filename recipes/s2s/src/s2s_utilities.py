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

import hashlib
import logging
import os
import secrets
import shutil
import sys
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, open_dict
from physicsnemo.distributed import DistributedManager

from earth2studio.data import DataSource
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, handshake_coords, map_coords
from earth2studio.utils.time import to_time_array


def initialize_perturbation(
    model: PrognosticModel,
    data_source: DataSource,
    start_time: np.ndarray[np.datetime64],
    cfg: DictConfig,
) -> Perturbation:
    """Initialize perturbation method. Some methods need to be initialized with model,
    data, start_time, etc. which can not always be defined in the config requireing
    partial instantiation.

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
            perturbation = perturbation(
                model=model, start_time=start_time, data_source=data_source
            )
        elif perturbation.func.__name__ == "BredVector":
            perturbation = perturbation(model=model)
        else:
            raise ValueError(
                f"perturbation method {perturbation.func.__name__} not implemented for partial instantiation"
            )

    return perturbation


def build_package_list(cfg: DictConfig) -> list[str]:
    """Find all available model packages.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    list[str]
        Available model packages.
    """
    if (
        "registry" in cfg.forecast_model
    ):  # pointing to single package; used to load HENS-SFNO models
        if cfg.forecast_model.registry == "default":
            return ["default"]

        elif os.path.isfile(os.path.join(cfg.forecast_model.registry, "config.json")):
            return [cfg.forecast_model.registry]

        else:  # pointing to directory of packages
            max_num_ckpts = 29
            if "max_num_checkpoints" in cfg.forecast_model:
                max_num_ckpts = cfg.forecast_model.max_num_checkpoints
            packages = []
            for pkg in os.listdir(cfg.forecast_model.registry):
                pth = os.path.abspath(os.path.join(cfg.forecast_model.registry, pkg))
                if os.path.isdir(pth) and os.path.isfile(
                    os.path.join(pth, "config.json")
                ):
                    packages.append(pth)
            if len(packages) == 0:
                raise ValueError(
                    f"Found no valid model packages under {cfg.forecast_model.registry}."
                )
            return (sorted(packages))[:max_num_ckpts]

    elif "DLESyM" in cfg.forecast_model.architecture:
        # DLESyM models are not stored in a registry, and the default package contains all atmos/ocean checkpoints
        # Loading specific checkpoint pairs is controlled by passing the model indices to `DLESyMLatLon.load_model`
        natmos, nocean = cfg.forecast_model.natmos, cfg.forecast_model.nocean
        pkg_list = [
            f"dlesym_atmos{i:02d}_ocean{j:02d}"
            for i in range(natmos)
            for j in range(nocean)
        ]

        if "max_num_checkpoint_pairs" in cfg.forecast_model:
            max_num_ckpts = cfg.forecast_model.max_num_checkpoint_pairs
        else:
            max_num_ckpts = len(pkg_list)

        return pkg_list[:max_num_ckpts]
    else:
        return ["default"]


def build_model_dict(cfg: DictConfig) -> dict:
    """Build a dictionary of loaded model, model class and package name.

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
    """Get a model dictionary and a list of available model packages.

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
    """Build list of IC times.

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


def initialize_output(
    cfg: DictConfig,
    times: list[np.datetime64],
    model_dict: dict,
    output_coords_dict: dict,
    add_arrays: bool = False,
) -> dict[str, ZarrBackend]:
    """Initialize data output.

    This function sets up the data output based on the provided configuration. It creates an IO handler for storing the
    forecast data either in memory or on disk. If file output is enabled in the configuration, it constructs the file path
    and name, creates the necessary directories, and initializes the IO backend according to the specified format.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object containing settings for the file output.
    times : list[np.datetime64]
        Initial Condition (IC) times for which the output is being generated.
    model_dict : dict
        Dictionary containing the prognostic model, its class, and the name of its package.
    output_coords_dict : dict
        Dictionary containing the output coordinates for different cropbox areas.
    add_arrays : bool
        Whether to add arrays to the IO backend. If False, the function will check if the variables are already in the IO backend.
        Used when initializing the same output file IO in a distributed setting.

    Returns
    -------
    dict[str, ZarrBackend]
        A dictionary where the keys are the names of different cropbox areas (e.g., 'Global', 'North', 'South'),
        and the values are the corresponding ZarrBackend objects for storing the data.

    Notes
    -----
    - If the configuration does not include file output settings (`file_output`), the function returns an empty dictionary.
    - The function constructs a file name based on the project name and run id.
    - This currently only supports ZarrBackend and .zarr file format.
    - The function ensures that the output directory exists before attempting to write data.
    """
    if "file_output" not in cfg:
        return {}

    io_dict = build_io_dict(
        cfg, list(output_coords_dict.keys()), create_store=add_arrays
    )

    # Populate with expected coords, dims
    ens_members = np.array(np.arange(cfg.nperturbed * cfg.ncheckpoints))
    total_coords = (
        OrderedDict({"ensemble": ens_members}) | model_dict["model"].input_coords()
    )
    total_coords.pop("batch")  # batch dimension not needed for output zarr
    total_coords["time"] = times

    input_coords = model_dict["model"].input_coords()
    output_coords = model_dict["model"].output_coords(total_coords)
    inp_lead_time = input_coords["lead_time"]
    out_lead_times = [
        output_coords["lead_time"] + output_coords["lead_time"][-1] * i
        for i in range(cfg.nsteps)
    ]
    total_coords["lead_time"] = np.concatenate([inp_lead_time, *out_lead_times]).astype(
        "timedelta64[ns]"
    )

    for i, (k, oc) in enumerate(output_coords_dict.items()):

        # augment and overwrite total coords with dimensions of output coords
        for key, value in total_coords.items():
            total_coords[key] = oc.get(key, value)

        if i == 0:
            # initialize place for variables in io backend
            variables_to_save = total_coords.pop("variable")

        if io_dict[k] is not None:
            if add_arrays:
                # Add the array to the IO backend, overwriting any existing arrays
                io_dict[k].add_array(total_coords, variables_to_save)
            else:
                # Output file exists, check for required variables
                for v in variables_to_save:
                    if v not in io_dict[k]:
                        raise ValueError(
                            f"Variable {v} not found in initialized {k} IO backend"
                        )

                # Verify expected coords
                for c in total_coords.keys():
                    handshake_coords(io_dict[k].coords, total_coords, required_dim=c)

    return io_dict


def build_io_dict(
    cfg: DictConfig,
    coords: list[str],
    create_store: bool = False,
    file_name: str = "forecast",
) -> dict[str, ZarrBackend]:
    """Build a dictionary of ZarrBackends for each cropbox area.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    coords : list[str]
        List of cropbox areas to build IO backends for.
    create_store : bool
        Whether to create a new store for the IO backend (overwriting existing store depending on cfg.file_output.overwrite_store).
    file_name : str
        Name of the file to store the data in.

    Returns
    -------
    dict[str, ZarrBackend]
        A dictionary where the keys are the names of different cropbox areas (e.g., 'Global', 'North', 'South'),
        and the values are the corresponding ZarrBackend objects for storing the data.
    """

    if "path" not in cfg.file_output:
        with open_dict(cfg):
            cfg["file_output"]["path"] = "outputs/"

    io_dict = {}
    for k in coords:
        out_path_base = os.path.join(cfg.file_output.path, k)
        out_path = os.path.join(out_path_base, file_name)
        os.makedirs(out_path_base, exist_ok=True)

        io = hydra.utils.instantiate(cfg.file_output.format)
        if isinstance(io, partial):  # add out file names
            file_name = out_path + ".zarr"
            if (
                os.path.exists(file_name)
                and not cfg.file_output.overwrite_store
                and create_store
            ):
                raise ValueError(
                    f"File {file_name} already exists. Set overwrite_store to True to overwrite."
                )
            elif (
                os.path.exists(file_name)
                and cfg.file_output.overwrite_store
                and create_store
            ):
                logging.warning(
                    f"Overwriting existing file {file_name} with new arrays."
                )
                shutil.rmtree(file_name)

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
    """Pair initial conditions with model packages. In parallel setting, distribute
    among ranks.

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

    for ii, pkg in enumerate(model_packages):
        for ic in ics:
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

        # Currently needed to prevent deadlock in distributed setting with uneven batch amounts per GPU
        if len(configs) % dist.world_size != 0:
            raise ValueError(
                f"Number of runs to make {len(configs)} is not divisible by number of ranks {dist.world_size}. Exiting."
            )

        nconfigs_proc = len(configs) // dist.world_size

        idx = dist.rank * nconfigs_proc
        configs = configs[idx : min(idx + nconfigs_proc, len(configs))]

        if not len(configs) > 0:
            logger.warning(f"nothing to do for rank {dist.rank}. exiting.")
            exit()

    # logger.info(
    #     f"rank {dist.rank}: predicting from following models/initial times: {configs}"
    # )

    return configs


def initialize(
    cfg: DictConfig,
) -> tuple[
    list[Any],
    dict[Any, Any],
    dict[Any, Any],
    DataSource,
    dict[str, OrderedDict[Any, Any]],
    str | int,
]:
    """Set initial conditions, load models, and set up file output based on the provided
    configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the settings for initial conditions,
        models, file output, and other relevant parameters.

    Returns
    -------
    tuple[list, dict, DataSource, dict, dict, dict, int]
        A tuple containing the following elements:
        - ensemble_configs: list of tuples containing model package configurations.
        - model_dict: dictionary containing the model, model class, and package name.
        - dx_model_dict: dictionary containing diagnostic models.
        - data: DataSource object for obtaining initial conditions.
        - output_coords_dict: dictionary of output coordinates for different cropbox areas.
        - base_random_seed: base random seed for reproducibility.

    Raises
    ------
    ValueError
        If a project name is not specified in the config.
        If file_output is not specified in the config.

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

    if "file_output" not in cfg:
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

    # get random seeds
    (
        base_random_seed,
        batch_ids_produce,
    ) = get_batch_seeds(cfg)

    # get ensemble configs
    ensemble_configs = pair_packages_ics(
        ics, model_packages, cfg.nperturbed, batch_ids_produce, cfg.batch_size
    )

    # get data source
    data_source = hydra.utils.instantiate(cfg.data_source)

    # initialize diagnostic models
    dx_model_dict = initialize_diagnostic_models(cfg)

    # make sure that all the seeds are unique
    ensure_all_torch_seeds_are_unique(ensemble_configs, str(base_random_seed))

    return (
        ensemble_configs,
        model_dict,
        dx_model_dict,
        data_source,
        output_coords_dict,
        base_random_seed,
    )


def initialize_diagnostic_models(cfg: DictConfig) -> dict:
    """Initialize diagnostic models based on the provided configuration.

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
                package = run_with_rank_ordered_execution(dx_model.load_default_package)
                dx_model = dx_model.load_model(package=package)
            elif "_target_" in cfg["diagnostic_models"][k]:
                dx_model = hydra.utils.instantiate(cfg_dx_model)
            else:
                raise NotImplementedError(
                    f"diagnostic model {k} not configured correctly. Either 'architecture' or '_target_' must be specified. See recipes/s2s/configs/pnw_sfno_precip.yaml for an example. "
                )
            dx_model_dict[k] = dx_model
    return dx_model_dict


def initialize_output_coords(
    cfg: DictConfig,
    lon_coords: np.ndarray[np.float32],
    lat_coords: np.ndarray[np.float32],
) -> dict:
    """Initialize output coordinates

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    lon_coords: np.ndarray[np.float32]
        a 1d array containing the longitude values of the grid. ordered ascending
    lat_coords: np.ndarray[np.float32]
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
    lon_coords: np.ndarray[np.float32],
    lat_coords: np.ndarray[np.float32],
) -> dict[str, tuple[np.ndarray[np.float32], np.ndarray[np.float32]]]:
    """If cropbox is defined in config this function will assert that ranges are plausible.
    Then it returns reduced versions of the longitude and latitude coordinate arrays that align with the area of interest
    Otherwise lon_coords and lat_coords will not be changed

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    lon_coords: np.ndarray[np.float32]
        a 1d array containing the longitude values of the grid. ordered ascending
    lat_coords: np.ndarray[np.float32]
        a 1d array containing the latitude values of the grid. ordered descending

    Returns
    -------
    dict[str, tuple[np.ndarray[np.float32], np.ndarray[np.float32]]]:
        Dictionary crop region lon and lat coords values of the requested sub grid, lon
        ordered ascending, lat ordered descending
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
        if cfg["file_output"]["resolution"] == "latlon121x240":
            coarse_lats = np.linspace(90, -90, 121, endpoint=True)
            coarse_lons = np.linspace(0, 360, 240, endpoint=False)
            coords_dict["global"] = (coarse_lats, coarse_lons)
        elif cfg["file_output"]["resolution"] == "latlon721x1440":
            coords_dict["global"] = (lat_coords, lon_coords)
        else:
            raise ValueError(
                f"Resolution {cfg['file_output']['resolution']} not supported"
            )

    return coords_dict


def check_extent(cfg_cropbox: DictConfig) -> None:
    """Validate the extent of the cropbox configuration to ensure it is within plausible
    ranges.

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
    cfg_cropbox: DictConfig, lon_coords: np.ndarray[np.float32]
) -> tuple[int, int]:
    """Determine the longitude ranges of the model and output based on the provided
    cropbox configuration and longitude coordinates.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lon_min` and `lon_max`.
    lon_coords : np.ndarray[np.float32]
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
    lon_coords: np.ndarray[np.float32],
    lat_coords: np.ndarray[np.float32],
    lon_range_model: int,
    lon_range_out: int,
) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """Crop the area based on the specified configuration and coordinate ranges.

    This function adjusts the longitude coordinates if necessary to align with the
    desired output range and then filters the latitude and longitude coordinates
    to match the cropbox defined in the configuration.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lat_min`,
        `lat_max`, `lon_min`, and `lon_max`.

    lon_coords : np.ndarray[np.float32]
        A 1D array containing the longitude values of the grid, ordered ascending.

    lat_coords : np.ndarray[np.float32]
        A 1D array containing the latitude values of the grid, ordered descending.

    lon_range_model : int
        The range of the input longitude coordinates (either 180 or 360).

    lon_range_out : int
        The range of the desired output longitude coordinates (either 180 or 360).

    Returns
    -------
    lat_coords_sub : np.ndarray[np.float32]
        A 1D array containing the latitude values of the subgrid, ordered descending.

    lon_coords_sub : np.ndarray[np.float32]
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


def update_model_dict(model_dict: dict, root: str) -> dict:
    """Check if model on GPU is same as needed for next inference.
    If not, load new model package and update model dict.

    Parameters
    ----------
    model_dict : dict
        dictionary specifying model, model class, and model package
    package : str
        model package location to be used in next inference

    Returns
    -------
    dict
        model dict.
    """

    if "dlesym" in root or root == "default":
        # Download model package safely across ranks so cache is not corrupted
        # Make sure the package download routine is called on all ranks to prevent deadlock
        pkg = run_with_rank_ordered_execution(model_dict["class"].load_default_package)

    if root != model_dict["package"]:
        model_dict["package"] = root

        # move to cpu to free GPU memory
        # TODO find other references and delete properly
        if model_dict["model"] is not None:
            model_dict["model"].to("cpu")

        if "dlesym" in root or root == "default":
            package = pkg
        else:
            package = Package(root)

        if "dlesym" in root:
            # Select appropriate model checkpoint pair for DLESyM
            atmos_model_idx = int(root.split("_")[-2].replace("atmos", ""))
            ocean_model_idx = int(root.split("_")[-1].replace("ocean", ""))
            model_dict["model"] = model_dict["class"].load_model(
                package=package,
                atmos_model_idx=atmos_model_idx,
                ocean_model_idx=ocean_model_idx,
            )
        else:
            model_dict["model"] = model_dict["class"].load_model(package=package)

    return model_dict


def initialize_output_structures(
    cfg: DictConfig,
) -> tuple[ThreadPoolExecutor | None, list[Future]]:
    """Initialize writer thread pool and threads

    This function initializes a thread pool executor for parallel file I/O
    operations, if configured.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing the settings for file output.

    Returns
    -------
    writer_executor : ThreadPoolExecutor | None
        An instance of `ThreadPoolExecutor` for parallel file I/O operations,
        or `None` if parallel I/O is not configured.

    writer_threads : List[Future]
        A list to hold future objects for the parallel file I/O tasks.
        Initially empty.

    Notes
    -----
    - If file output parallelism is enabled in the configuration, `writer_executor`
        will be an instance of `ThreadPoolExecutor`. Otherwise, it will be `None`.
    """

    # Initialize threadpool for writers
    writer_executor = (
        ThreadPoolExecutor(max_workers=8)
        if ("file_output" in cfg and cfg.file_output.thread_io)
        else None
    )
    writer_threads: list[Future] = []

    return writer_executor, writer_threads


def get_batchid_from_ensid(
    nperturbed_per_package: int, batch_size: int, ensid: int
) -> int:
    """Calculate the gloabl batch ID from a global ensemble member ID (ensid).

    Parameters
    ----------
    nperturbed_per_package : int
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
    if nperturbed_per_package <= 0:
        raise ValueError("nperturbed_per_package must be greater than zero.")

    num_batches = int(np.ceil(nperturbed_per_package / batch_size))
    batch_id = int(
        np.floor((ensid % nperturbed_per_package) / batch_size)
        + (np.floor(ensid / nperturbed_per_package)) * num_batches
    )

    return batch_id


def cat_coords(
    xx: torch.Tensor,
    cox: CoordSystem,
    yy: torch.Tensor,
    coy: CoordSystem,
    dim: str = "variable",
) -> tuple[torch.Tensor, CoordSystem]:
    """
    concatenate data along coordinate dimension.

    Parameters
    ----------
    xx : torch.Tensor
        First input tensor which to concatenate
    cox : CoordSystem
        Ordered dict representing coordinate system that describes xx
    yy : torch.Tensor
        Second input tensor which to concatenate
    coy : CoordSystem
        Ordered dict representing coordinate system that describes yy
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.
    """

    if dim not in cox:
        raise ValueError(f"dim {dim} is not in coords: {list(cox)}.")
    if dim not in coy:
        raise ValueError(f"dim {dim} is not in coords: {list(coy)}.")

    # fix difference in latitude
    _cox = cox.copy()
    _cox["lat"] = coy["lat"]
    xx, cox = map_coords(xx, cox, _cox)

    coords = cox.copy()
    dim_index = list(coords).index(dim)

    zz = torch.cat((xx, yy), dim=dim_index)
    coords[dim] = np.append(cox[dim], coy[dim])

    return zz, coords


def calculate_torch_seed(s: str) -> int:
    """Calculates torch seed based on a given string.
    String s is used as input to sha256 hash algorithm.
    Output is converted to integer by taking the maximum integer size of torch seed
    into account.

    Parameters
    ----------
    s : str
        seed string

    Returns
    -------
    torch: np.int64
        integer value that can be used as random seed in torch

    """
    torch_seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**64) - 1
    return torch_seed


def create_base_seed_string(pkg: str, ic: np.datetime64, base_random_seed: str) -> str:
    """Concatenates information of model package name, initial condition time and and
    base_random seed into one base seed string.

    Parameters
    ----------
    pkg : str
        Model package name
    ic : np.datetime64
        Initial condition time
    base_random_seed : str
        Base seed string

    Returns
    -------
    base_seed_string: str
        string that can be used as random seed

    """
    s0 = str(base_random_seed)
    s1 = "".join(
        e for e in pkg if e.isalnum()
    )  # remove all special characters from package name
    s2 = str(ic.astype("datetime64[s]"))
    base_seed_string = "_".join([s0, s1, s2])
    return base_seed_string


def get_batch_seeds(cfg: DictConfig) -> tuple[str | int, list[int]]:
    """Retrieve random seed cfg elements or their default values

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    base_random_seed: str|int
        a base random seed specfied in the config. If it is an integer it will be converted to a string later
    batch_ids_produce: list[int]
        a list of the batch ids that shall be produced in this run
    """
    try:
        batch_ids_produce = cfg["batch_ids_reproduce"]
    except KeyError:
        batch_ids_produce = list(
            range(
                0,
                int(np.ceil(cfg.nperturbed / cfg.batch_size) * cfg.ncheckpoints),
            )
        )
    try:
        base_random_seed = cfg["random_seed"]
    except KeyError:
        base_random_seed = secrets.randbelow(1_000_000)

    return base_random_seed, batch_ids_produce


def calculate_all_torch_seeds(
    base_seed_string: str, batch_ids: list[int]
) -> tuple[np.array, np.array]:
    """
    calculates all torch random seeds that will be used based on the base_seed_string
    and the batch_ids

    Parameters
    ----------
    base_seed_string : str
        base seed
    batch_ids : list[int]
        list of batch_ids that will be calculated

    Returns
    -------
    full_seed_strings: np.array
        contains all seed strings that will be used to calculate torch seeds
    torch_seeds: np.array
        contains all torch random seeds that will be used

    """
    sall = np.char.add(
        np.array(base_seed_string + "_"), np.array([str(x) for x in batch_ids])
    )
    torch_seeds = np.zeros((len(sall), 1), dtype=np.uint64)
    full_seed_strings = np.empty(np.shape(torch_seeds), dtype=object)
    for i, s in enumerate(sall):
        full_seed_strings[i] = s
        torch_seeds[i] = calculate_torch_seed(s)
    return full_seed_strings, torch_seeds


def check_uniquness_of_torch_seeds(torch_seeds: np.array) -> bool:
    """Checks if all torch seeds are unique

    Parameters
    ----------
    torch_seeds : np.array
        Array of torch seeds

    Returns
    -------
    bool:
        True if no duplicates of torch seeds were found

    """
    num_runs = len(torch_seeds)
    num_unique_seeds = len(np.unique(torch_seeds))
    if num_unique_seeds == num_runs:
        all_unique = True
    else:
        all_unique = False
        raise ValueError(
            "Calculated torch seeds for every run must be unique! num_unique_seeds = %s, num_runs = %s"
            % (num_unique_seeds, num_runs)
        )
    return all_unique


def ensure_all_torch_seeds_are_unique(
    ensemble_configs: list[tuple], base_random_seed: str
) -> None:
    """Checks if all torch seeds based on ensemble_configs and base_random_seed are
    unique

    Parameters
    ----------
    ensemble_configs : list[tuple]
        List of ensemble config objects
    base_random_seed : str
        Base seed string

    Raises
    ------
    ValueError
        If the random seeds of all ensembles are not fully unique (duplicates)
    """
    torch_seeds_list = []
    full_seed_string_list = []
    for pkg, ic, _, batch_ids_produce in ensemble_configs:
        base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)
        full_seed_strings, torch_seeds = calculate_all_torch_seeds(
            base_seed_string, batch_ids_produce
        )
        if check_uniquness_of_torch_seeds(torch_seeds):
            torch_seeds_list.append(torch_seeds)
            full_seed_string_list.append(full_seed_strings)
    if torch_seeds_list:
        check_uniquness_of_torch_seeds(np.concatenate(torch_seeds_list, axis=0))
    else:
        raise ValueError("Torch seeds could not be calculated.")


def configure_logging() -> None:
    """Configure logging to suppress noisy packages"""

    def noisy_packages(record: Any) -> bool:
        if record["name"] == "makani.models.model_package":
            return False
        elif record["name"] == "numba.core.transforms":
            return False
        return True

    class InterceptHandler(logging.Handler):
        def emit(self, record: Any) -> None:
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Log the message through loguru
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO", filter=noisy_packages)


def run_with_rank_ordered_execution(
    func: Callable, *args: Any, first_rank: int = 0, **kwargs: Any
) -> Any:
    """Executes `func(*args, **kwargs)` safely in a distributed setting:
    - First on the specified `rank`
    - Then, after synchronization, on the other ranks

    Args:
        func (Callable): Function to execute
        args (tuple, optional): Positional arguments for the function. Defaults to ().
        first_rank (int, optional): Rank to run the function first. Defaults to 0.
        kwargs (dict, optional): Keyword arguments for the function. Defaults to None.

    Returns:
        The return value of func(*args, **kwargs)
    """
    if kwargs is None:
        kwargs = {}

    dist = DistributedManager()
    current_rank = dist.rank

    if current_rank == first_rank:
        result = func(*args, **kwargs)
    else:
        result = None

    # Synchronize all processes after the first rank runs the function
    # Skip the barrier if single-process (no distributed process group)
    if dist.distributed:
        torch.distributed.barrier()

    if current_rank != first_rank:
        result = func(*args, **kwargs)

    if dist.distributed:
        torch.distributed.barrier()

    return result
