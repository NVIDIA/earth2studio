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

"""Generate ensembles with cyclone tracking, or reproduce some of their members.

The run is built from Earth2Studio's pipeline components:

- the work items are batches of ``batch_size`` members of one initial
  condition (IC), see :func:`earth2studio.run.work_items`;
- :class:`~src.ensemble.runner.StableRunner` rolls out one attempt of a batch
  and stops it if it becomes unstable;
- :class:`~src.ensemble.schedule.RetryPipeline`, an Earth2Studio
  :class:`~earth2studio.run.Pipeline`, distributes the batches over the ranks
  and runs the retries of unstable ones, with the next seed, on whichever rank
  is free; it writes the fields of stream ``forecast`` and records finished
  batches, so an interrupted run can resume;
- :class:`~src.ensemble.tracking.TrackingScorer` tracks cyclones in every batch.
"""

import os
import shutil
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager

from earth2studio.io.base import IOBackend
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px import PrognosticModel
from earth2studio.run import Pipeline, WorkItem, work_items
from earth2studio.run.work import barrier, broadcast, rank0_first, rank_info
from src.data.tc_hunt_data_utils import CachedSource, data_source_from_config
from src.ensemble.output import (
    AttemptFlush,
    BatchNetCDFBackend,
    SeedRecorder,
    zarr_backend,
)
from src.ensemble.progress import BatchProgress
from src.ensemble.runner import BatchObserver, InstabilityDetection, StableRunner
from src.ensemble.schedule import RetryPipeline, RetryQueue, RetryScheduler
from src.ensemble.tracking import TRACKING_STREAM, TrackerInputs, TrackingScorer
from src.tc_hunt_utils import set_initial_times
from src.tempest_extremes import TempestExtremes

# attempts searched for the seed of a reproduced member
MAX_REPRODUCED_ATTEMPT = 1000


def initialise(cfg: DictConfig) -> None:
    """Set up the runtime environment for ensemble generation.

    Selects deterministic GPU kernels if ``deterministic`` is ``True`` and
    initialises the distributed manager. Random seeds are not set here: the
    runner seeds the model of every batch, see
    :class:`~src.ensemble.runner.StableRunner`.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object. May contain ``deterministic``
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


def max_retries(cfg: DictConfig) -> int:
    """Number of retries of an unstable batch, 0 without stability check."""
    if "stability_check" not in cfg:
        return 0
    retries = cfg.stability_check.get("max_retries", 3)
    if not isinstance(retries, int) or retries < 0:
        raise ValueError(f"stability_check.max_retries must be >= 0, got {retries}")
    return retries


def base_seed(cfg: DictConfig, progress_dir: str) -> int:
    """Base seed of the run, from which the seed of every batch is derived.

    ``random_seed`` of the config, or else the seed of the run being resumed,
    or else a seed drawn by rank 0. All ranks get the same seed.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    progress_dir : str
        Directory of the progress records of the run

    Returns
    -------
    int
        The base seed
    """
    if cfg.get("random_seed") is not None:
        return int(cfg.random_seed)
    seed = None
    if rank_info()[0] == 0:
        if cfg.get("resume", False):
            seed = BatchProgress.stored_seed(progress_dir)
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
            logger.info(f"random_seed not set, using {seed}")
    return broadcast(seed)


def progress_tags(cfg: DictConfig) -> dict:
    """Settings a resumed run must keep, besides those the pipeline checks.

    The pipeline itself checks the seed, the batches, the lead times and the
    output variables.
    """
    tags = {
        "model": cfg.model,
        "model_package": cfg.get("model_package"),
        "deterministic": cfg.get("deterministic", False),
        "data_source": OmegaConf.to_container(cfg.data_source, resolve=True),
        "store_type": cfg.store_type,
    }
    if "cyclone_tracking" in cfg:
        tags["cyclone_tracking"] = {
            key: cfg.cyclone_tracking.get(key)
            for key in ("vars", "detect_cmd", "stitch_cmd", "orography_path")
        }
    return tags


def reproduction_attempts(
    cfg: DictConfig, items: list[WorkItem], runner: StableRunner
) -> dict[WorkItem, int]:
    """Batches holding the members to reproduce, and the attempt to rerun.

    The model's random state is seeded per batch, so a member is reproduced
    together with the other members of its batch, which requires the
    ``batch_size`` and ``ensemble_size`` of the original run. The seed of a
    member, part of its track file names, identifies the attempt it was run
    with, given the original ``random_seed``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object, with ``reproduce_members``, a list of
        ``[ic, member, seed]``
    items : list[WorkItem]
        All batches of the original run
    runner : StableRunner
        Runner seeded with the original ``random_seed``

    Returns
    -------
    dict[WorkItem, int]
        Attempt of each batch to rerun

    Raises
    ------
    ValueError
        If a member is not part of the run, or its seed is none of its batch's
    """
    attempts: dict[WorkItem, int] = {}
    for ic, member, seed in cfg.reproduce_members:
        time = np.datetime64(ic, "s")
        batch = [i for i in items if i.time == time and member in i.member_ids]
        if not batch:
            raise ValueError(f"member {member} of {ic} is not part of the ensemble")
        item = batch[0]
        attempt = next(
            (
                attempt
                for attempt in range(MAX_REPRODUCED_ATTEMPT + 1)
                if runner.batch_seed(item, attempt) == int(seed)
            ),
            None,
        )
        if attempt is None or attempts.get(item, attempt) != attempt:
            raise ValueError(
                f"seed {seed} of member {member} of {ic} is not a seed of its "
                "batch: use the random_seed, batch_size and ensemble_size of "
                "the original run, and the same seed for all members of a batch"
            )
        attempts[item] = attempt
    return attempts


def field_store(
    cfg: DictConfig, items: list[WorkItem], lead_times: np.ndarray, fresh: bool
) -> IOBackend | None:
    """Store of the fields, None if no fields are written.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    items : list[WorkItem]
        All batches of the run
    lead_times : np.ndarray
        Lead times of the run
    fresh : bool
        Start from scratch, deleting the Zarr store of an earlier run

    Returns
    -------
    IOBackend | None
        A Zarr store for all batches, a NetCDF file per batch, or None

    Raises
    ------
    ValueError
        If ``store_type`` is unknown, or fields are stored without ``out_vars``
    """
    if cfg.store_type == "none":
        return None
    if cfg.store_type not in ("zarr", "netcdf"):
        raise ValueError(
            f"invalid store_type {cfg.store_type}, choose from zarr, netcdf, none"
        )
    if "out_vars" not in cfg:
        raise ValueError("out_vars must be specified if store_type is not none")

    os.makedirs(cfg.store_dir, exist_ok=True)
    prefix = os.path.join(cfg.store_dir, cfg.project.removesuffix(".nc"))
    if cfg.store_type == "netcdf":
        return BatchNetCDFBackend(prefix)

    path = prefix.removesuffix(".zarr") + ".zarr"
    if fresh and rank_info()[0] == 0:
        shutil.rmtree(path, ignore_errors=True)
    barrier()
    shard = cfg.get("zarr_shard_per_member", True)
    # rank 0 creates the store before the other ranks open it
    return rank0_first(lambda: zarr_backend(path, items, lead_times, shard))


def run_ensemble(cfg: DictConfig) -> None:
    """Generate an ensemble, or reproduce some of its members, and track cyclones.

    Entry point of the modes ``generate_ensemble`` and ``reproduce_members``,
    which differ only in the batches they run and their attempts.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Raises
    ------
    ValueError
        If members are reproduced with ``resume``, which is not supported, or
        the run neither writes fields nor tracks cyclones
    """
    reproduce = cfg.mode == "reproduce_members"
    resume = cfg.get("resume", False)
    if reproduce and resume:
        raise ValueError("resume is not supported for reproducing ensemble members")
    if reproduce and cfg.get("random_seed") is None:
        raise ValueError("reproducing members needs the random_seed of the run")

    initialise(cfg)
    progress_dir = os.path.join(cfg.store_dir, f"{cfg.project}_progress")
    seed = base_seed(cfg, progress_dir)

    # to make sure tracker.close() runs even if the run fails
    with ExitStack() as cleanup:
        # cyclone tracking: the tracker's input fields become stream "tracking"
        tracker = None
        diagnostics = {}
        if "cyclone_tracking" in cfg:
            tracker = TempestExtremes.from_config(cfg.cyclone_tracking, cfg.store_dir)
            # also on failure, so no scratch files are left in /dev/shm
            cleanup.callback(tracker.close)
            diagnostics[TRACKING_STREAM] = TrackerInputs(tracker)

        stability_check = None
        if "stability_check" in cfg:
            stability_check = InstabilityDetection(
                cfg.stability_check.variables, cfg.stability_check.thresholds
            )

        # initial conditions are fetched once for all batches and retries of an IC
        runner = StableRunner(
            load_model(cfg),
            stability_check,
            source=CachedSource(data_source_from_config(cfg.data_source)),
            diagnostics=diagnostics,
            seed=seed,
        )

        final_lead_time = runner.step * cfg.n_steps
        if reproduce:
            ics = np.unique(
                [np.datetime64(ic, "s") for ic, _, _ in cfg.reproduce_members]
            )
        else:
            ics = set_initial_times(cfg)
        items = work_items(
            ics,
            final_lead_time,
            members=cfg.ensemble_size,
            member_batch=cfg.batch_size,
        )

        # the attempts of each batch and the runner's observers depend on the
        # work items, so they are set once the items are known; reproduced
        # members rerun their attempt only, so they are not retried
        progress = None
        scheduler = None
        if reproduce:
            attempts = reproduction_attempts(cfg, items, runner)
            items = [item for item in items if item in attempts]
            runner.attempt = attempts.__getitem__
        else:
            progress = BatchProgress(
                progress_dir, progress_tags(cfg), overwrite=not resume
            )
            scheduler = RetryScheduler(
                RetryQueue.shared(), max_retries(cfg), progress.attempts_made
            )
            runner.attempt = scheduler.attempt
            runner.max_retries = scheduler.max_retries

        forecast = runner.output_coords(final_lead_time)["forecast"]
        store = field_store(cfg, items, forecast["lead_time"].values, not resume)

        scorer = None if tracker is None else TrackingScorer(tracker, cfg.batch_size)
        if store is None and scorer is None:
            raise ValueError("the run neither stores fields nor tracks cyclones")

        observers: list[BatchObserver] = []
        if scheduler is not None:
            observers.append(scheduler)
        if scorer is not None:
            observers.append(scorer)
        if store is not None:
            seeds = SeedRecorder(store, items)
            rank0_first(seeds.add_array)
            observers += [AttemptFlush(store), seeds]
        if progress is not None:
            observers.append(progress)
        runner.observers = observers

        pipeline_kwargs: dict[str, Any] = {
            "output": None if store is None else {"forecast": store},
            "scorer": scorer,
            "output_variables": (
                None if store is None else {"forecast": list(cfg.out_vars)}
            ),
            "progress": progress,
            "device": DistributedManager().device,
        }
        if scheduler is None:
            Pipeline(runner, **pipeline_kwargs).run(items)
        else:
            RetryPipeline(scheduler, runner, **pipeline_kwargs).run(items)
