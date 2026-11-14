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

import hashlib
import logging
import os
import secrets
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, open_dict
from physicsnemo.distributed import DistributedManager

from earth2studio.grids import LatLonGrid
from earth2studio.io import ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import coord_array
from earth2studio.utils.time import to_time_array

# -- models -------------------------------------------------------------------


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

    else:
        return ["default"]


def model_loader(cfg: DictConfig, packages: Sequence[str]) -> Callable[[str], Any]:
    """Build a function loading the forecast model of one package.

    The default package, if any package uses it, is downloaded here, rank 0
    first, so later loads during the run need no collective.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    packages : Sequence[str]
        Package names from :func:`build_package_list`.

    Returns
    -------
    Callable[[str], PrognosticModel]
        Loads the model of a package name.
    """
    model_class = hydra.utils.get_class(cfg.forecast_model.architecture)
    default = None
    if "default" in packages:
        default = run_with_rank_ordered_execution(model_class.load_default_package)
    # Optional load_model arguments, e.g. the DLESyM checkpoint version
    kwargs = (
        {"version": cfg.forecast_model.version}
        if "version" in cfg.forecast_model
        else {}
    )

    def load(pkg: str) -> PrognosticModel:
        package = default if pkg == "default" else Package(pkg)
        return model_class.load_model(package=package, **kwargs)

    return load


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


# -- initial conditions and outputs ------------------------------------------------


def set_initial_times(cfg: DictConfig) -> np.ndarray:
    """Build list of IC times.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    np.ndarray
        Initial condition times.
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


def output_variables(cfg: DictConfig) -> list[str]:
    """Variables to save, in config order without duplicates."""
    return list(dict.fromkeys(cfg.file_output.output_vars))


@dataclass(frozen=True)
class OutputRegion:
    """Lat/lon subset of the model grid saved to one output store.

    ``lat_index``/``lon_index`` select the region from the model grid, and
    ``lat``/``lon`` are the labels it is stored under (longitudes may be shifted
    to the cropbox's convention).
    """

    lat_index: slice | np.ndarray
    lon_index: slice | np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def grid_attrs(self) -> dict[Any, Any]:
        """Grid metadata attributes for fields on this region."""
        grid = LatLonGrid(self.lat, self.lon)
        return dict(coord_array(("lat", "lon"), grid=grid).attrs)

    def crop(self, x: xr.DataArray, attrs: Mapping[Any, Any]) -> xr.DataArray:
        """Select this region from a field on the model grid."""
        y = x.isel(lat=self.lat_index, lon=self.lon_index).assign_coords(
            lat=self.lat, lon=self.lon
        )
        y.attrs = dict(attrs)
        return y


def _as_index(index: np.ndarray) -> slice | np.ndarray:
    """Use a (strided) slice for evenly spaced indices, so selections are views."""
    index = np.asarray(index, dtype=np.int64)
    if index.size == 0:
        raise ValueError("Output region selects no grid points")
    if index.size == 1:
        return slice(int(index[0]), int(index[0]) + 1)
    steps = np.diff(index)
    if steps[0] > 0 and (steps == steps[0]).all():
        return slice(int(index[0]), int(index[-1]) + 1, int(steps[0]))
    return index


def _nearest_index(source: np.ndarray, target: np.ndarray, name: str) -> np.ndarray:
    """Positions of ``target`` labels on ``source``, which must contain them."""
    index = np.abs(source[:, None] - target[None, :]).argmin(axis=0)
    if not np.allclose(source[index], target, atol=1e-6):
        raise ValueError(f"Output {name} grid is not a subset of the model grid")
    return index


def build_output_regions(
    cfg: DictConfig, lat_coords: np.ndarray, lon_coords: np.ndarray
) -> dict[str, OutputRegion]:
    """Build the output regions of the config on the model's lat/lon grid.

    With ``file_output.cropboxes`` each cropbox is a region; otherwise a single
    ``global`` region at ``file_output.resolution``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    lat_coords : np.ndarray
        Model latitudes, descending.
    lon_coords : np.ndarray
        Model longitudes, ascending.

    Returns
    -------
    dict[str, OutputRegion]
        Output region per name.
    """
    regions = {}
    if "cropboxes" in cfg.file_output:
        for name, cb in cfg.file_output.cropboxes.items():
            check_extent(cb)
            lon_range_model, lon_range_out = determine_lon_ranges(cb, lon_coords)
            regions[name] = crop_area(
                cb, lon_coords, lat_coords, lon_range_model, lon_range_out
            )
    elif cfg.file_output.resolution == "latlon721x1440":
        regions["global"] = OutputRegion(
            slice(None), slice(None), np.asarray(lat_coords), np.asarray(lon_coords)
        )
    elif cfg.file_output.resolution == "latlon121x240":
        lats = np.linspace(90, -90, 121, endpoint=True)
        lons = np.linspace(0, 360, 240, endpoint=False)
        regions["global"] = OutputRegion(
            _as_index(_nearest_index(lat_coords, lats, "lat")),
            _as_index(_nearest_index(lon_coords, lons, "lon")),
            lats,
            lons,
        )
    else:
        raise ValueError(f"Resolution {cfg.file_output.resolution} not supported")
    return regions


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
    cfg_cropbox: DictConfig, lon_coords: np.ndarray
) -> tuple[int, int]:
    """Determine the longitude ranges of the model and output based on the provided
    cropbox configuration and longitude coordinates.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lon_min` and `lon_max`.
    lon_coords : np.ndarray
        A 1D array containing the longitude values of the grid, ordered ascending.

    Returns
    -------
    lon_range_model : int
        The range of the input longitude coordinates (either 180 or 360).
    lon_range_out : int
        The range of the desired output longitude coordinates (either 180 or 360).
    """
    lon_range_model = (
        360 if ((lon_coords.max() > 180) and (lon_coords.min() >= 0)) else 180
    )
    lon_range_out = (
        360 if ((cfg_cropbox.lon_max > 180) and (cfg_cropbox.lon_min >= 0)) else 180
    )
    return lon_range_model, lon_range_out


def crop_area(
    cfg_cropbox: DictConfig,
    lon_coords: np.ndarray,
    lat_coords: np.ndarray,
    lon_range_model: int,
    lon_range_out: int,
) -> OutputRegion:
    """Select a cropbox from the model grid.

    Longitudes are relabeled to the cropbox's convention (0-360 or -180-180)
    and stored ascending, latitudes keep the model's order.

    Parameters
    ----------
    cfg_cropbox : DictConfig
        Configuration object containing the cropbox settings, including `lat_min`,
        `lat_max`, `lon_min`, and `lon_max`.
    lon_coords : np.ndarray
        A 1D array containing the longitude values of the grid, ordered ascending.
    lat_coords : np.ndarray
        A 1D array containing the latitude values of the grid, ordered descending.
    lon_range_model : int
        The range of the input longitude coordinates (either 180 or 360).
    lon_range_out : int
        The range of the desired output longitude coordinates (either 180 or 360).

    Returns
    -------
    OutputRegion
        The cropbox region.
    """
    lon_coords = np.asarray(lon_coords)
    lat_coords = np.asarray(lat_coords)
    labels = lon_coords
    if lon_range_model == 180 and lon_range_out == 360:
        labels = np.where(lon_coords < 0, lon_coords + 360, lon_coords)
    elif lon_range_model == 360 and lon_range_out == 180:
        labels = np.where(lon_coords >= 180, lon_coords - 360, lon_coords)
    order = np.argsort(labels, kind="stable")
    keep = (labels[order] >= cfg_cropbox.lon_min) & (
        labels[order] <= cfg_cropbox.lon_max
    )
    lon_index = order[keep]
    lat_index = np.nonzero(
        (lat_coords >= cfg_cropbox.lat_min) & (lat_coords <= cfg_cropbox.lat_max)
    )[0]
    return OutputRegion(
        _as_index(lat_index),
        _as_index(lon_index),
        lat_coords[lat_index],
        labels[lon_index],
    )


def build_io_dict(
    cfg: DictConfig,
    regions: Sequence[str],
    create_store: bool = False,
    file_name: str = "forecast",
    keep_existing: bool = False,
) -> dict[str, ZarrBackend]:
    """Build a dictionary of ZarrBackends, one per output region.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object
    regions : Sequence[str]
        Output regions to build IO backends for.
    create_store : bool
        Whether this process creates the stores, removing existing ones if
        ``cfg.file_output.overwrite_store`` is set.
    file_name : str
        Name of the file to store the data in.
    keep_existing : bool
        Reuse existing stores (to resume a run) instead of removing or
        rejecting them.

    Returns
    -------
    dict[str, ZarrBackend]
        Backend per region, stored at ``{file_output.path}/{region}/{file_name}.zarr``.
    """

    if "path" not in cfg.file_output:
        with open_dict(cfg):
            cfg["file_output"]["path"] = "outputs/"

    io_dict = {}
    for k in regions:
        out_path_base = os.path.join(cfg.file_output.path, k)
        os.makedirs(out_path_base, exist_ok=True)
        out_file = os.path.join(out_path_base, file_name) + ".zarr"

        io = hydra.utils.instantiate(cfg.file_output.format)
        if isinstance(io, partial):  # add out file names
            if os.path.exists(out_file) and create_store and not keep_existing:
                if not cfg.file_output.overwrite_store:
                    raise ValueError(
                        f"File {out_file} already exists. Set overwrite_store to True to overwrite."
                    )
                logger.warning(f"Overwriting existing file {out_file}")
                shutil.rmtree(out_file)
            io = io(file_name=out_file)
        io_dict[k] = io

    return io_dict


# -- seeds ------------------------------------------------------------------------


def get_batchid_from_ensid(
    nperturbed_per_package: int, batch_size: int, ensid: int
) -> int:
    """Calculate the global batch ID from a global ensemble member ID (ensid).

    Parameters
    ----------
    nperturbed_per_package : int
        Number of ensemble members per model package.
    batch_size : int
        Number of ensemble members per batch.
    ensid : int
        Global ensemble member ID (0-based index).

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
    int
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
    s2 = str(np.datetime64(ic).astype("datetime64[s]"))
    base_seed_string = "_".join([s0, s1, s2])
    return base_seed_string


def get_batch_seeds(cfg: DictConfig) -> tuple[str | int, list[int] | None]:
    """Retrieve random seed cfg elements or their default values

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    base_random_seed: str|int
        a base random seed specfied in the config. If it is an integer it will be converted to a string later
    batch_ids_produce: list[int] | None
        the batch ids that shall be produced in this run, or None for all
    """
    batch_ids_produce = cfg.get("batch_ids_reproduce", None)
    if batch_ids_produce is not None:
        batch_ids_produce = list(batch_ids_produce)
    base_random_seed = cfg.get("random_seed", None)
    if base_random_seed is None:
        base_random_seed = secrets.randbelow(1_000_000)

    return base_random_seed, batch_ids_produce


# -- distributed and logging --------------------------------------------------


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

    Every rank must call this the same number of times, so never call it from
    inside a work item.

    Args:
        func (Callable): Function to execute
        args (tuple, optional): Positional arguments for the function. Defaults to ().
        first_rank (int, optional): Rank to run the function first. Defaults to 0.
        kwargs (dict, optional): Keyword arguments for the function. Defaults to None.

    Returns:
        The return value of func(*args, **kwargs)
    """
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
