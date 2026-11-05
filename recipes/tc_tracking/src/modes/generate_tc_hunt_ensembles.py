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

import gc
import os
import random
import sys
from itertools import chain, islice

import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm
from zarr import consolidate_metadata

from earth2studio.data import fetch_data
from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from src.data.tc_hunt_data_utils import DataSourceManager, load_heights
from src.data.tc_hunt_file_output import initialise_netcdf_output, setup_output
from src.tc_hunt_utils import (
    InstabilityDetection,
    assign_runs_to_rank,
    get_set_of_random_seeds,
    remove_duplicates,
    run_with_rank_ordered_execution,
    set_initial_times,
    share_gpu_memory_with_cupy,
)
from src.tempest_extremes import AsyncTempestExtremes, TempestExtremes


def initialise(cfg: DictConfig) -> None:
    """Set up the runtime environment for ensemble generation.

    Selects deterministic GPU kernels if ``deterministic`` is ``True``,
    initialises the distributed manager, and seeds all random number
    generators if a seed is provided in the config. Note that this does not
    affect the internal random state of forecast models.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. May contain ``deterministic`` and
        ``random_seed`` keys
    """
    # Order-preserving GPU kernels make runs bitwise reproducible, which e.g.
    # the scatter operations in AIFS-ENS' graph layers otherwise are not.
    # Costs about 10% runtime. cuBLAS reads its workspace setting when it
    # creates its first handle, so this has to happen before any computation
    deterministic = cfg.get("deterministic", False)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)

    # initialise distributed manager
    DistributedManager.initialize()
    if torch.cuda.is_available():
        share_gpu_memory_with_cupy()

    # intialise random seeds
    # NOTE: DOES NOT affect the internal random state of the forecast models)
    if "random_seed" in cfg:
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)
        torch.cuda.manual_seed(cfg.random_seed)


def load_model(cfg: DictConfig) -> PrognosticModel:
    """Load a prognostic model to device.

    Model weights are loaded from the default package or from a
    custom path specified via ``cfg.model_package``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. Must contain ``model`` (str) and may
        contain ``model_package`` (str). Supported models are ``fcn3``,
        ``aifs`` (AIFS-ENS v1) and ``aifs-v2`` (AIFS-ENS v2)

    Returns
    -------
    PrognosticModel
        Loaded prognostic model on the current device

    Raises
    ------
    ValueError
        If ``cfg.model`` is not set or the requested model name is not
        supported
    """
    if "model" not in cfg:
        raise ValueError(
            "cfg.model must be specified in the YAML config (e.g. 'fcn3' or 'aifs')"
        )

    # imported lazily, so that only the installed models' dependencies are needed
    model_cls: type[AutoModelMixin]
    if cfg.model == "fcn3":
        from earth2studio.models.px import FCN3

        model_cls = FCN3
    elif cfg.model == "aifs":
        from earth2studio.models.px import AIFSENS

        model_cls = AIFSENS
    elif cfg.model == "aifs-v2":
        from earth2studio.models.px import AIFS2ENS

        model_cls = AIFS2ENS
    else:
        raise ValueError(
            f"model {cfg.model} not supported, choose from 'fcn3', 'aifs', 'aifs-v2'"
        )

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
    store: ZarrBackend | None,
    template: xr.DataArray | None,
    ic_mems: list[tuple[np.datetime64, np.ndarray, int]],
) -> None:
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
    store : ZarrBackend | None
        Zarr output store, or None if fields are written to NetCDF or not at all
    template : xr.DataArray | None
        Output template of the run, or None if no fields are written
    ic_mems : list[tuple[np.datetime64, np.ndarray, int]]
        List of (initial_condition, member_indices, random_seed) tuples
    """

    dist = DistributedManager()
    data_source_mngr = DataSourceManager(cfg)
    inputs = model.input_coords()

    # NetCDF stores are created per IC and stay open, as unstable members can
    # be re-queued to the end of the work list
    nc_stores: dict[np.datetime64, NetCDF4Backend] = {}
    x0, x0_ic = None, None

    # TODO: add the native WeatherNext 2 cyclone tracker as alternative to
    # TempestExtremes once WeatherNext 2 is supported by the recipe
    cyclone_tracking = None
    if "cyclone_tracking" in cfg:
        tracker = (
            AsyncTempestExtremes
            if cfg.cyclone_tracking.asynchronous
            else TempestExtremes
        )
        cyclone_tracking = tracker(
            detect_cmd=cfg.cyclone_tracking.detect_cmd,
            stitch_cmd=cfg.cyclone_tracking.stitch_cmd,
            input_vars=cfg.cyclone_tracking.vars,
            store_dir=cfg.store_dir,
            heights=(
                load_heights(cfg.cyclone_tracking.orography_path)
                if "orography_path" in cfg.cyclone_tracking
                else None
            ),
            keep_raw_data=cfg.cyclone_tracking.get("keep_raw_data", False),
            print_te_output=cfg.cyclone_tracking.get("print_te_output", False),
            scratch_dir=cfg.cyclone_tracking.get("scratch_dir", None),
            timeout=cfg.cyclone_tracking.get("task_timeout_seconds", 120),
            max_workers_per_rank=cfg.cyclone_tracking.get("max_workers_per_rank", None),
        )

    stability_check = None
    if "stability_check" in cfg:
        stability_check = InstabilityDetection(
            vars=cfg.stability_check.variables,
            thresholds=cfg.stability_check.thresholds,
        )

    for ic, mems, seed in ic_mems:
        if cfg.store_type == "netcdf" and ic not in nc_stores:
            nc_stores[ic] = initialise_netcdf_output(cfg, model, ic, ic_mems)
        if cfg.store_type == "netcdf":
            store = nc_stores[ic]

        # initial condition, shared by all batches of an IC
        if ic != x0_ic:
            x0 = fetch_data(
                data_source_mngr.select_data_source(ic),
                time=[ic],
                variable=inputs.coords["variable"].values,
                lead_time=inputs.coords["lead_time"].values,
                device=dist.device,
            )
            x0_ic = ic
        x = x0.expand_dims(ensemble=mems)

        if stability_check:
            stability_check.reset()

        # each batch draws its noise from its own seed, which makes the batch
        # reproducible, see reproduce_members
        model.set_rng(seed=seed)

        # the iterator yields forecasts only, so the IC is prepended
        rollout = chain(
            [initial_condition(x)], islice(model.create_iterator(x), cfg.n_steps)
        )
        stable = True
        for y in tqdm(rollout, total=cfg.n_steps + 1):
            # fields of previous steps can sit in reference cycles that still
            # hold GPU memory, collect them before the model allocates again
            gc.collect()
            if store is not None:
                store.write(y.sel(variable=template.coords["variable"].values))
            if cyclone_tracking:
                cyclone_tracking.record_state(y)

            if stability_check:
                stable = stability_check(y).all()
                if not stable:
                    if cyclone_tracking:
                        cyclone_tracking.reset()
                    ic_mems.append((ic, mems, seed + 1))
                    logger.warning(
                        f"CAUTION: one of members {mems} became unstable. will re-create with new seed."
                    )
                    break

        if cyclone_tracking and stable:
            cyclone_tracking(
                out_file_names=[
                    f"tracks_{np.datetime_as_string(ic, unit='s')}_mem_{mem:04d}_seed_{seed}_bs_{cfg.batch_size}.csv"
                    for mem in mems
                ]
            )

    # Wait for all async cyclone tracking tasks to complete before continuing
    if cyclone_tracking and isinstance(cyclone_tracking, AsyncTempestExtremes):
        cyclone_tracking.cleanup()

    for nc_store in nc_stores.values():
        nc_store.close()

    # Consolidate metadata in zarr files
    if isinstance(store, ZarrBackend):
        store.close()
        if dist.rank == 0:
            # TODO add barrier such that rank 0 finishes last
            consolidate_metadata(store.store)


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

    ic_mems_rank = assign_runs_to_rank(ic_mems)

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

    store, template = (
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
        sys.exit()

    run_inference(model, cfg, store, template, ic_mems)


def set_reproduction_configs(
    cfg: DictConfig,
) -> tuple[list[tuple[np.datetime64, np.ndarray, int]] | None, list[np.datetime64]]:
    """Build and distribute work items for reproducing specific ensemble members.

    Parses the ``reproduce_members`` config list, expands each requested member
    to its full batch (since the internal random state can only be set per
    batch), de-duplicates, and distributes across ranks.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.  Must contain ``reproduce_members`` (list
        of ``[ic, member_id, seed]`` triples), ``batch_size``, and
        ``ensemble_size``.

    Returns
    -------
    tuple[list[tuple[np.datetime64, np.ndarray, int]] | None, list[np.datetime64]]
        Tuple of (work items for this rank, list of all initial-condition times)
    """
    ic_mems = OmegaConf.to_container(cfg.reproduce_members)

    ics = []
    for ii in range(len(ic_mems)):
        ic_mems[ii][0] = np.datetime64(ic_mems[ii][0])
        ics.append(ic_mems[ii][0])

        # expand to the full batch that contains the requested member
        batch_id = ic_mems[ii][1] // cfg.batch_size
        ic_mems[ii][1] = np.arange(
            batch_id * cfg.batch_size,
            min((batch_id + 1) * cfg.batch_size, cfg.ensemble_size),
        )

    ic_mems = remove_duplicates(ic_mems)
    ic_mems.sort(key=lambda x: x[0])
    ics = list(set(ics))

    if not DistributedManager().distributed:
        return ic_mems, ics

    ic_mems_rank = assign_runs_to_rank(ic_mems)

    return ic_mems_rank, ics


def reproduce_members(cfg: DictConfig) -> None:
    """Reproduce specific ensemble members to extract atmospheric fields.

    Entry point that validates the configuration, sets up reproduction work
    items, loads the model, and re-runs inference for the requested members
    with the original random seeds so that results are bit-identical.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Raises
    ------
    ValueError
        If ``store_type`` is ``"zarr"`` (not supported for reproduction)
    """
    if cfg.store_type == "zarr":
        raise ValueError("Zarr output not supported for reproducing ensemble members")

    initialise(cfg)

    ic_mems, ics = set_reproduction_configs(cfg)

    model = load_model(cfg)

    store, template = (
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
        sys.exit()

    run_inference(model, cfg, store, template, ic_mems)
