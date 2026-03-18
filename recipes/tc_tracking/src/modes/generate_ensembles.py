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

import os
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm
from zarr import consolidate_metadata

from earth2studio.data import fetch_data
from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import SphericalGaussian
from earth2studio.utils.coords import map_coords
from src.data.file_output import initialise_netcdf_output, setup_output, write_to_store
from src.data.utils import DataSourceManager, load_heights
from src.tempest_extremes import AsyncTempestExtremes, TempestExtremes
from src.utils import (
    InstabilityDetection,
    get_set_of_random_seeds,
    run_with_rank_ordered_execution,
    set_initial_times,
)


def initialise(cfg: DictConfig) -> None:
    """Set up the runtime environment for ensemble generation.

    Configures CUDA memory allocation, initialises the distributed manager,
    and seeds all random number generators if a seed is provided in the config.
    Note that this does not affect the internal random state of forecast models.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. May contain a ``random_seed`` key
    """
    # make pytorch occupy only GPU memory it really needs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # initialise ditributed manager
    DistributedManager.initialize()

    # intialise random seeds
    # NOTE: DOES NOT affect the internal random state of the forecast models)
    if "random_seed" in cfg:
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        torch.cuda.manual_seed(cfg.random_seed)

    return


def load_model(cfg: DictConfig) -> PrognosticModel:
    """Load a prognostic model to device.

    Model weights are loaded from the default package or from a
    custom path specified via ``cfg.model_package``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. May contain ``model`` (str) and
        ``model_package`` (str) keys

    Returns
    -------
    PrognosticModel
        Loaded prognostic model on the current device

    Raises
    ------
    ValueError
        If the requested model name is not supported
    """
    model_name = "fcn3"
    if "model" in cfg:
        model_name = cfg.model

    if model_name[:4] == "aifs":
        from earth2studio.models.px import AIFSENS

        model_cls = AIFSENS
    elif model_name == "fcn3":
        from earth2studio.models.px import FCN3

        model_cls = FCN3
    elif model_name == "sfno":
        from earth2studio.models.px import SFNO

        model_cls = SFNO
    else:
        raise ValueError(f"model {model_name} not supported")

    # load weights
    pkg = (
        Package(cfg.model_package)
        if "model_package" in cfg
        else model_cls.load_default_package()
    )

    model = model_cls.load_model(pkg).to(DistributedManager().device)

    return model


def run_inference(
    model: PrognosticModel,
    cfg: DictConfig,
    store: ZarrBackend | NetCDF4Backend | None,
    out_coords: OrderedDict,
    ic_mems: list[tuple[np.datetime64, np.ndarray, int]],
) -> ZarrBackend | NetCDF4Backend | None:
    """Run ensemble inference and optionally track tropical cyclones.

    Iterates over initial-condition / ensemble-member batches, fetches the
    corresponding data, rolls out the prognostic model, writes fields to the
    output store, and feeds predictions to TempestExtremes for cyclone tracking
    if configured. If stability check is enabled, unstable members are detected
    and re-queued with a new seed.

    Parameters
    ----------
    model : PrognosticModel
        Prognostic model
    cfg : DictConfig
        Hydra configuration object
    store : ZarrBackend | NetCDF4Backend | None
        Output store (Zarr, NetCDF, or None)
    out_coords : OrderedDict
        Output coordinate system for sub-selecting written fields
    ic_mems : list[tuple[np.datetime64, np.ndarray, int]]
        List of (initial_condition, member_indices, random_seed) tuples

    Returns
    -------
    ZarrBackend | NetCDF4Backend | None
        The output store after all data has been written
    """
    dist = DistributedManager()

    # data_source = hydra.utils.instantiate(cfg.data_source)
    data_source_mngr = DataSourceManager(cfg)

    # iterate over initial conditions
    ic_prev = None

    cyclone_tracking = None
    if "cyclone_tracking" in cfg:
        oco = model.output_coords(model.input_coords())

        heights, height_coords = (
            load_heights(cfg.cyclone_tracking.orography_path)
            if "orography_path" in cfg.cyclone_tracking
            else (None, None)
        )

        tracker = (
            AsyncTempestExtremes
            if cfg.cyclone_tracking.asynchronous
            else TempestExtremes
        )
        cyclone_tracking = tracker(
            detect_cmd=cfg.cyclone_tracking.detect_cmd,
            stitch_cmd=cfg.cyclone_tracking.stitch_cmd,
            input_vars=cfg.cyclone_tracking.vars,
            batch_size=cfg.batch_size,
            n_steps=cfg.n_steps,
            time_step=oco["lead_time"][0],
            lats=oco["lat"],
            lons=oco["lon"],
            static_vars=heights,
            static_coords=height_coords,
            store_dir=cfg.store_dir,
            keep_raw_data=cfg.cyclone_tracking.keep_raw_data,
            print_te_output=cfg.cyclone_tracking.print_te_output,
            use_ram=cfg.cyclone_tracking.use_ram,
            timeout=cfg.cyclone_tracking.task_timeout_seconds,
        )

    stability_check = None
    if "stability_check" in cfg:
        stability_check = InstabilityDetection(
            vars=np.array(cfg.stability_check.variables),
            thresholds=np.array(cfg.stability_check.thresholds),
        )

    for ic, mems, seed in ic_mems:
        mini_batch_size = len(mems)

        data_source = data_source_mngr.select_data_source(ic)

        # if new IC, fetch data, create iterator
        if ic != ic_prev:
            if cfg.store_type == "netcdf":
                store = initialise_netcdf_output(cfg, out_coords, ic, ic_mems)
            x0, coords0 = fetch_data(
                data_source,
                time=[np.datetime64(ic)],
                lead_time=model.input_coords()["lead_time"],
                variable=model.input_coords()["variable"],
                device=dist.device,
            )
            ic_prev = ic

        coords = {"ensemble": np.array(mems)} | coords0.copy()
        xx = x0.unsqueeze(0).repeat(mini_batch_size, *([1] * x0.ndim))

        if stability_check:
            stability_check.reset(deepcopy(coords))
        # print(stability_check.input_coords)
        # exit()

        # set random state or apply perturbation
        if ("model" not in cfg) or (cfg.model == "fcn3"):
            model.set_rng(seed=seed)
        elif (
            cfg.model[:4] == "aifs"
        ):  # no need for perturbation, but also cannot set internal noise state
            pass
        else:
            sg = SphericalGaussian(noise_amplitude=0.0005)
            xx, coords = sg(xx, coords)

        iterator = model.create_iterator(xx, coords)

        # roll out the model and record data as desired
        for _, (xx, coords) in tqdm(
            zip(range(cfg.n_steps + 1), iterator), total=cfg.n_steps + 1
        ):
            write_to_store(store, xx, coords, out_coords)
            if cyclone_tracking:
                cyclone_tracking.record_state(xx, coords)

            if stability_check:
                yy, coy = map_coords(xx, coords, stability_check.input_coords)
                stab, _ = stability_check(yy, coy)
                if not stab.all():
                    ic_mems.append((ic, mems, seed + 1))
                    print(
                        f"CAUTION: one of members {mems} became unstable. will re-create with new seed."
                    )
                    break

        if cyclone_tracking and (not stability_check or stab.all()):
            cyclone_tracking(
                out_file_names=[  # TODO add seed only for FCN3 members, as the seed in AIFS is not exposed. also check for netcdf output
                    f"tracks_{np.datetime_as_string(ic, unit='s')}_mem_{mem:04d}_seed_{seed}_bs_{cfg.batch_size}.csv"
                    for mem in mems
                ]
            )

    # Wait for all async cyclone tracking tasks to complete before continuing
    if cyclone_tracking and isinstance(cyclone_tracking, AsyncTempestExtremes):
        cyclone_tracking.cleanup()

    # Consolidate metadata in zarr files
    if dist.rank == 0 and cfg.store_type == "zarr" and store is not None:
        # TODO add barrier such that rank 0 finishes last
        consolidate_metadata(store.store)

    return store


def distribute_runs(
    ic_mems: list[tuple[np.datetime64, np.ndarray, int]],
) -> list[tuple[np.datetime64, np.ndarray, int]] | None:
    """Partition work items across distributed ranks.

    Splits the list of initial-condition / member batches evenly across all
    ranks. Returns ``None`` for ranks that receive no work.

    Parameters
    ----------
    ic_mems : list[tuple[np.datetime64, np.ndarray, int]]
        List of (initial_condition, member_indices, random_seed) tuples

    Returns
    -------
    list[tuple[np.datetime64, np.ndarray, int]] | None
        Subset of work items assigned to this rank, or None if idle
    """
    dist = DistributedManager()

    # get the number of initial conditions
    ic_mems_per_rank = len(ic_mems) // dist.world_size
    if len(ic_mems) % dist.world_size != 0:
        ic_mems_per_rank += 1

    # get the initial conditions for this rank
    ic_mems = ic_mems[dist.rank * ic_mems_per_rank : (dist.rank + 1) * ic_mems_per_rank]

    if len(ic_mems) == 0:
        print(f"nothing to do for rank {dist.rank}, exiting")
        return None

    return ic_mems


def configure_runs(
    cfg: DictConfig,
) -> tuple[list[tuple[np.datetime64, np.ndarray, int]] | None, list[np.datetime64]]:
    """Build and distribute the list of ensemble runs from the configuration.

    Generates all (initial_condition, member_batch, random_seed) combinations
    based on the configured ICs, ensemble size, and batch size, then
    distributes them across all ranks.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    -------
    tuple[list[tuple[np.datetime64, np.ndarray, int]] | None, list[np.datetime64]]
        Tuple of (work items for this rank, list of all initial-condition times)
    """
    ic_mems = []

    ics = set_initial_times(cfg)

    seeds = get_set_of_random_seeds(
        n_ics=len(ics),
        ensemble_size=cfg.ensemble_size,
        batch_size=cfg.batch_size,
        seed=cfg.random_seed if "random_seed" in cfg else None,
    )

    ii = 0
    for ic in ics:
        for mem in range(0, cfg.ensemble_size, cfg.batch_size):
            mems = np.arange(mem, min(mem + cfg.batch_size, cfg.ensemble_size))
            ic_mems.append((ic, mems, int(seeds[ii])))
            ii += 1

    if not DistributedManager().distributed:
        return ic_mems, ics

    ic_mems_rank = distribute_runs(ic_mems)

    return ic_mems_rank, ics


def generate_ensemble(cfg: DictConfig) -> None:
    """Generate an ensemble forecast with optional cyclone tracking.

    Entry point that initialises the environment, configures runs, loads
    the model, sets up the output store, and runs distributed ensemble
    inference.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    initialise(cfg)

    ic_mems, ics = configure_runs(cfg)

    model = load_model(cfg)

    # store, out_coords = setup_output(cfg, model, add_arrays=DistributedManager().rank == 0)
    store, out_coords = (
        run_with_rank_ordered_execution(  # TODO: wrap only zarr store in that loop
            setup_output,
            cfg=cfg,
            model=model,
            ics=ics,
            add_arrays=DistributedManager().rank == 0,
        )
    )

    if ic_mems is None:
        DistributedManager().cleanup()
        exit()

    store = run_inference(model, cfg, store, out_coords, ic_mems)

    return
