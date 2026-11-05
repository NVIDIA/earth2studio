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
from itertools import chain, islice

import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm
from zarr import consolidate_metadata

from earth2studio.data import fetch_data
from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from src.data.tc_hunt_data_utils import DataSourceManager
from src.data.tc_hunt_file_output import initialise_netcdf_output, setup_output
from src.tc_hunt_utils import (
    InstabilityDetection,
    assign_runs_to_rank,
    get_set_of_random_seeds,
    run_with_rank_ordered_execution,
    set_initial_times,
    share_gpu_memory_with_cupy,
)
from src.tempest_extremes import TempestExtremes

# one model rollout: initial condition, indices of the members in the batch and
# the batch's random seed
Run = tuple[np.datetime64, np.ndarray, int]


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

    # seed global random number generators, which does not affect the
    # models' own random state
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
    runs: list[Run],
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
    runs : list[Run]
        List of (initial_condition, member_indices, random_seed) tuples
    """

    dist = DistributedManager()
    data_source_mngr = DataSourceManager(cfg)
    inputs = model.input_coords()

    # NetCDF stores are created per IC and stay open, as unstable members can
    # be re-queued to the end of the work list
    nc_stores: dict[np.datetime64, NetCDF4Backend] = {}
    out_vars = None if template is None else template.coords["variable"].values
    # initial condition, shared by all batches of an IC
    x0: xr.DataArray | None = None

    # TODO: add the native WeatherNext 2 cyclone tracker as alternative to
    # TempestExtremes once WeatherNext 2 is supported by the recipe
    cyclone_tracking = (
        TempestExtremes.from_config(cfg.cyclone_tracking, cfg.store_dir)
        if "cyclone_tracking" in cfg
        else None
    )

    stability_check = None
    if "stability_check" in cfg:
        stability_check = InstabilityDetection(
            vars=cfg.stability_check.variables,
            thresholds=cfg.stability_check.thresholds,
        )

    for ic, mems, seed in runs:
        if cfg.store_type == "netcdf" and ic not in nc_stores:
            nc_stores[ic] = initialise_netcdf_output(cfg, model, ic, runs)
        if cfg.store_type == "netcdf":
            store = nc_stores[ic]

        if x0 is None or x0.coords["time"].values[0] != ic:
            x0 = fetch_data(
                data_source_mngr.select_data_source(ic),
                time=[ic],
                variable=inputs.coords["variable"].values,
                lead_time=inputs.coords["lead_time"].values,
                device=dist.device,
            )
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
                store.write(y.sel(variable=out_vars))
            if cyclone_tracking:
                cyclone_tracking.record_state(y)

            if stability_check:
                stable = stability_check(y).all()
                if not stable:
                    if cyclone_tracking:
                        cyclone_tracking.reset()
                    runs.append((ic, mems, seed + 1))
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

    # wait for tracking running in the background
    if cyclone_tracking:
        cyclone_tracking.close()

    for nc_store in nc_stores.values():
        nc_store.close()

    # Consolidate metadata in zarr files
    if isinstance(store, ZarrBackend):
        store.close()
        if dist.rank == 0:
            # TODO add barrier such that rank 0 finishes last
            consolidate_metadata(store.store)


def ensemble_runs(cfg: DictConfig) -> tuple[list[Run], np.ndarray]:
    """All runs of an ensemble: every batch of members for every IC.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    -------
    tuple[list[Run], np.ndarray]
        Runs as (initial_condition, member_indices, random_seed) and all ICs
    """
    ics = set_initial_times(cfg)
    first_members = range(0, cfg.ensemble_size, cfg.batch_size)
    seeds = iter(
        get_set_of_random_seeds(
            n_ics=len(ics),
            ensemble_size=cfg.ensemble_size,
            batch_size=cfg.batch_size,
            seed=cfg.get("random_seed", None),
        )
    )

    runs = [
        (
            ic,
            np.arange(first, min(first + cfg.batch_size, cfg.ensemble_size)),
            int(next(seeds)),
        )
        for ic in ics
        for first in first_members
    ]
    return runs, ics


def reproduction_runs(cfg: DictConfig) -> tuple[list[Run], np.ndarray]:
    """Runs reproducing the members listed in ``cfg.reproduce_members``.

    A model's random state can only be set per batch, so each requested member
    is reproduced together with the other members of its original batch, which
    requires the same ``batch_size`` as in the original run.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. Must contain ``reproduce_members`` (list
        of ``[ic, member_id, seed]`` triples), ``batch_size``, and
        ``ensemble_size``.

    Returns
    -------
    tuple[list[Run], np.ndarray]
        Runs as (initial_condition, member_indices, random_seed), sorted by IC,
        and all ICs
    """
    # keyed by batch, so that members of the same batch are run only once
    runs: dict[tuple, Run] = {}
    for ic, member, seed in cfg.reproduce_members:
        ic = np.datetime64(ic)
        first = member // cfg.batch_size * cfg.batch_size
        mems = np.arange(first, min(first + cfg.batch_size, cfg.ensemble_size))
        runs[(ic, first, seed)] = (ic, mems, int(seed))

    ordered = sorted(runs.values(), key=lambda run: run[0])
    return ordered, np.unique([run[0] for run in ordered])


def run_ensemble(cfg: DictConfig) -> None:
    """Generate an ensemble, or reproduce some of its members, and track cyclones.

    Entry point of the modes ``generate_ensemble`` and ``reproduce_members``,
    which differ only in the runs they perform. Initialises the environment,
    distributes the runs across ranks, loads the model, sets up the output
    store and runs the inference.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Raises
    ------
    ValueError
        If members are reproduced into a Zarr store, which is not supported
    """
    reproduce = cfg.mode == "reproduce_members"
    if reproduce and cfg.store_type == "zarr":
        raise ValueError("Zarr output not supported for reproducing ensemble members")

    initialise(cfg)

    runs, ics = reproduction_runs(cfg) if reproduce else ensemble_runs(cfg)
    runs = assign_runs_to_rank(runs)

    model = load_model(cfg)

    store, template = run_with_rank_ordered_execution(
        setup_output,
        cfg=cfg,
        model=model,
        ics=ics,
        add_arrays=DistributedManager().rank == 0,
    )

    if runs is None:
        DistributedManager().cleanup()
        return

    run_inference(model, cfg, store, template, runs)
