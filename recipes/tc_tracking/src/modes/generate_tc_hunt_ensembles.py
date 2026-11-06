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
from typing import Any

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
from src.data.tc_hunt_file_output import (
    initialise_netcdf_output,
    setup_output,
    write_seed,
)
from src.tc_hunt_utils import (
    InstabilityDetection,
    RetryQueue,
    assign_runs_to_rank,
    get_set_of_random_seeds,
    run_with_rank_ordered_execution,
    set_initial_times,
    share_gpu_memory_with_cupy,
    shared_store,
)
from src.tempest_extremes import TempestExtremes

# one batch: initial condition, indices of its members and the random seeds to
# try in order, the original seed first, then the retry seeds
Run = tuple[np.datetime64, np.ndarray, tuple[int, ...]]


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
        # TODO: find the remaining source of non-determinism in AIFS-ENS v2.
        # Same seed and deterministic=True still give one of two slightly
        # different results (a few Pa in msl after 18 h, growing to hPa), seemingly
        # depending on GPU memory state. Not the seeding, not the edge sort of
        # anemoi's Triton graph attention, not gc.freeze; untested: PyG backend
        # (does not fit into 48 GB), allocation-dependent kernel selection.
        if cfg.get("deterministic", False) or cfg.mode == "reproduce_members":
            logger.warning(
                "\n" + "!" * 78 + "\n"
                "AIFS-ENS v2 is NOT bitwise reproducible, even with deterministic=True.\n"
                "Reproduced members can differ slightly from the original run, and\n"
                "their tracks may not match exactly. See README, section 2.2.\n"
                + "!" * 78
            )
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
    retries: RetryQueue,
) -> None:
    """Run ensemble inference and optionally track tropical cyclones.

    Iterates over the batches of this rank, fetches their initial condition,
    rolls out the prognostic model, writes fields to the output store, and
    feeds predictions to TempestExtremes for cyclone tracking if configured.
    If the stability check is enabled, a batch that becomes unstable is put
    into the retry queue with its next seed, where any rank can take it. A
    batch without seeds left is logged as failed. Afterwards, the rank takes
    retries until no rank has work left.

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
        Batches of this rank as (initial_condition, member_indices, seeds)
    retries : RetryQueue
        Queue of batches to retry, shared by all ranks
    """

    dist = DistributedManager()
    data_source_mngr = DataSourceManager(cfg)
    inputs = model.input_coords()
    if isinstance(inputs, tuple):
        raise TypeError("models with several inputs are not supported")

    out_vars = None if template is None else template.coords["variable"].values
    # initial condition of the last batch, shared by all batches of an IC
    x0: xr.DataArray | None = None
    # members that stayed unstable with all their seeds
    failed: list[tuple[np.datetime64, np.ndarray, tuple[int, ...]]] = []

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

    def rollout(x: xr.DataArray, seed: int, out: Any) -> bool:
        """Roll out one batch, write and record its steps, True if stable."""
        if stability_check:
            stability_check.reset()

        # each batch draws its noise from its own seed, which makes the batch
        # reproducible, see reproduce_members
        model.set_rng(seed=seed)

        # the iterator yields forecasts only, so the IC is prepended
        steps = chain(
            [initial_condition(x)], islice(model.create_iterator(x), cfg.n_steps)
        )
        for y in tqdm(steps, total=cfg.n_steps + 1):
            # fields of previous steps can sit in reference cycles that still
            # hold GPU memory, collect them before the model allocates again
            gc.collect()
            if out is not None:
                out.write(y.sel(variable=out_vars))
            if cyclone_tracking:
                cyclone_tracking.record_state(y)
            if stability_check and not stability_check(y).all():
                return False
        return True

    def run_batch(
        ic: np.datetime64, mems: np.ndarray, seeds: tuple[int, ...], attempt: int
    ) -> None:
        """Run attempt ``attempt`` of a batch, queue a retry if it is unstable."""
        nonlocal x0
        seed = seeds[attempt]
        last_attempt = attempt + 1 == len(seeds)

        if x0 is None or x0.coords["time"].values[0] != ic:
            x0 = fetch_data(
                data_source_mngr.select_data_source(ic),
                time=[ic],
                variable=inputs.coords["variable"].values,
                lead_time=inputs.coords["lead_time"].values,
                device=dist.device,
            )
        x = x0.expand_dims(ensemble=mems)

        out = store
        if cfg.store_type == "netcdf":
            out = initialise_netcdf_output(cfg, model, ic, mems, seed)
        try:
            stable = rollout(x, seed, out)
            if not stable and last_attempt and cfg.store_type == "netcdf":
                write_seed(out, mems, -1)
        finally:
            if cfg.store_type == "netcdf":
                out.close()

        ic_str = np.datetime_as_string(ic, unit="s")
        if stable:
            if cyclone_tracking:
                cyclone_tracking(
                    out_file_names=[
                        f"tracks_{ic_str}_mem_{mem:04d}_seed_{seed}_bs_{cfg.batch_size}.csv"
                        for mem in mems
                    ]
                )
            return

        if cyclone_tracking:
            cyclone_tracking.reset()
        if last_attempt:
            logger.error(
                f"rank {dist.rank}: members {mems.tolist()} of {ic_str} unstable with all "
                f"{len(seeds)} seeds {list(seeds)}, giving up"
            )
            failed.append((ic, mems, seeds))
            retries.add_failed(len(mems))
        else:
            logger.warning(
                f"rank {dist.rank}: members {mems.tolist()} of {ic_str} unstable with seed {seed}, "
                f"queued retry {attempt + 1} of {len(seeds) - 1}"
            )
            retries.put((ic, mems, seeds, attempt + 1))

    # exclude everything alive now (model, libraries) from garbage collection,
    # so the per-step gc.collect() below only scans new objects (ms, not 0.5 s)
    gc.freeze()

    try:
        for ic, mems, seeds in runs:
            run_batch(ic, mems, seeds, 0)
        retries.finish_initial()

        # retries of any rank, until no rank has work left
        while (item := retries.take()) is not None:
            try:
                run_batch(*item)
            finally:
                retries.task_done()
    finally:
        # also on failure, so outputs are flushed and no scratch files are
        # left in /dev/shm
        if cyclone_tracking:
            cyclone_tracking.close()

    for ic, mems, seeds in failed:
        logger.error(
            f"FAILED: members {mems.tolist()} of {np.datetime_as_string(ic, unit='s')}, "
            f"seeds tried {list(seeds)}"
        )
    if dist.rank == 0 and retries.failed():
        logger.error(
            f"{retries.failed()} members of all ranks stayed unstable, see the log "
            "of each rank"
        )

    # Consolidate metadata in zarr files
    if isinstance(store, ZarrBackend):
        store.close()
        if dist.rank == 0:
            # TODO add barrier such that rank 0 finishes last
            consolidate_metadata(store.store)


def max_retries(cfg: DictConfig) -> int:
    """Number of retries of an unstable batch, 0 without stability check."""
    if "stability_check" not in cfg:
        return 0
    retries = cfg.stability_check.get("max_retries", 3)
    if not isinstance(retries, int) or retries < 0:
        raise ValueError(f"stability_check.max_retries must be >= 0, got {retries}")
    return retries


def base_seed(cfg: DictConfig) -> int:
    """The configured ``random_seed``, or one drawn by rank 0 for all ranks.

    All ranks need the same seed to build the same seeds for all batches.
    """
    if cfg.get("random_seed", None) is not None:
        return int(cfg.random_seed)

    store = shared_store()
    if DistributedManager().rank == 0:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
        store.set("base_seed", str(seed))
        logger.info(f"random_seed not set, using {seed}")
    # waits until rank 0 has set it
    return int(store.get("base_seed"))


def ensemble_runs(cfg: DictConfig) -> tuple[list[Run], np.ndarray]:
    """All runs of an ensemble: every batch of members for every IC.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Returns
    -------
    tuple[list[Run], np.ndarray]
        Runs as (initial_condition, member_indices, seeds) and all ICs
    """
    ics = set_initial_times(cfg)
    first_members = range(0, cfg.ensemble_size, cfg.batch_size)
    seeds = iter(
        get_set_of_random_seeds(
            n_ics=len(ics),
            ensemble_size=cfg.ensemble_size,
            batch_size=cfg.batch_size,
            seed=base_seed(cfg),
            n_retries=max_retries(cfg),
        )
    )

    runs = [
        (
            ic,
            np.arange(first, min(first + cfg.batch_size, cfg.ensemble_size)),
            tuple(int(seed) for seed in next(seeds)),
        )
        for ic in ics
        for first in first_members
    ]
    return runs, ics


def reproduction_runs(cfg: DictConfig) -> tuple[list[Run], np.ndarray]:
    """Runs reproducing the members listed in ``cfg.reproduce_members``.

    A model's random state can only be set per batch, so each requested member
    is reproduced together with the other members of its original batch, which
    requires the same ``batch_size`` and ``ensemble_size`` as in the original
    run (the last batch can be smaller than ``batch_size``). A reproduced batch
    is never retried, as another seed would not reproduce it.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. Must contain ``reproduce_members`` (list
        of ``[ic, member_id, seed]`` triples), ``batch_size``, and
        ``ensemble_size``.

    Returns
    -------
    tuple[list[Run], np.ndarray]
        Runs as (initial_condition, member_indices, seeds), sorted by IC, and
        all ICs
    """
    # keyed by batch, so that members of the same batch are run only once
    runs: dict[tuple, Run] = {}
    for ic, member, seed in cfg.reproduce_members:
        ic = np.datetime64(ic)
        first = member // cfg.batch_size * cfg.batch_size
        mems = np.arange(first, min(first + cfg.batch_size, cfg.ensemble_size))
        runs[(ic, first, seed)] = (ic, mems, (int(seed),))

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

    # every rank runs the inference, also without runs: its setup contains
    # collectives (e.g. loading the orography), which all ranks have to reach.
    # No collectives after the setup, as idle ranks finish early
    retries = RetryQueue(shared_store(), DistributedManager().world_size)
    run_inference(model, cfg, store, template, runs, retries)
