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
from functools import partial

import hydra
import numpy as np
import zarr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager

from earth2studio.io import IOBackend
from earth2studio.run import Pipeline, ProgressStore, WorkItem
from src.s2s_ensemble import S2SRunner, build_work_items
from src.s2s_io import BufferedIOBackend
from src.s2s_perturbation import initialize_perturbation
from src.s2s_utilities import (
    build_io_dict,
    build_output_regions,
    build_package_list,
    get_batch_seeds,
    initialize_diagnostic_models,
    model_loader,
    output_variables,
    run_with_rank_ordered_execution,
    set_initial_times,
)


def initialize(
    cfg: DictConfig,
) -> tuple[S2SRunner, list[WorkItem], dict[str, IOBackend], ProgressStore | None]:
    """Load models and data, and plan the work items and outputs of a run.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    tuple[S2SRunner, list[WorkItem], dict[str, IOBackend], ProgressStore | None]
        The runner, every work item of the run, the output backend per output
        region, and the progress store when ``file_output.resume`` is set.
    """
    if "project" not in cfg:
        raise ValueError("specify a project name in the config: project: project_name")
    if "file_output" not in cfg:
        raise ValueError(
            "specify a file_output too store the results in the config: file_output"
        )

    if cfg.perturbation.type == "HENSPerturbation" and (
        cfg.nperturbed % 2 or min(cfg.batch_size, cfg.nperturbed) % 2
    ):
        raise ValueError(
            "Bred-vector perturbations perturb members in +/- pairs; "
            "nperturbed and batch_size must be even"
        )

    resume = cfg.file_output.get("resume", False)
    if resume:
        output_format = cfg.file_output.format
        if (
            output_format._target_ != "earth2studio.io.ZarrBackend"
            or not output_format.get("_partial_", False)
            or output_format.get("backend_kwargs", {}).get("overwrite", False)
        ):
            raise ValueError(
                "Resume requires a partial ZarrBackend with overwrite=False"
            )
        if cfg.get("random_seed", None) is None:
            raise ValueError("Resume requires an explicit random_seed")

    ics = set_initial_times(cfg)
    packages = build_package_list(cfg)
    if len(packages) < cfg.ncheckpoints:
        logger.warning(
            f"Found {len(packages)} checkpoints, fewer than ncheckpoints={cfg.ncheckpoints}"
        )
    packages = packages[: cfg.ncheckpoints]
    load_model = model_loader(cfg, packages)
    model = load_model(packages[0])
    signature = model.output_coords(model.input_coords())
    regions = build_output_regions(
        cfg, signature["lat"].values, signature["lon"].values
    )

    base_random_seed, batch_ids = get_batch_seeds(cfg)
    data_source = hydra.utils.instantiate(cfg.data_source)
    runner = S2SRunner(
        model=model,
        load_model=load_model,
        packages=packages,
        data=data_source,
        perturbation_factory=partial(initialize_perturbation, cfg),
        regions=regions,
        output_variables=output_variables(cfg),
        nperturbed=cfg.nperturbed,
        batch_size=cfg.batch_size,
        base_random_seed=base_random_seed,
        diagnostics=initialize_diagnostic_models(cfg),
    )

    items = build_work_items(
        ics,
        cfg.nsteps * runner.step,
        len(packages),
        cfg.nperturbed,
        cfg.batch_size,
        batch_ids,
    )
    if not items:
        raise ValueError("No work items: check batch_ids_reproduce")
    seeds = [runner.batch_seed(item) for item in items]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Calculated torch seeds for every batch must be unique")

    # Rank 0 creates (or replaces) the stores before the other ranks open them
    dist = DistributedManager()
    io_dict = run_with_rank_ordered_execution(
        build_io_dict,
        cfg,
        list(regions),
        create_store=(dist.rank == 0),
        keep_existing=resume,
    )
    if cfg.file_output.get("thread_io", False):
        buffer_size = cfg.file_output.get("buffer_size", 2)
        io_dict = {k: BufferedIOBackend(io, buffer_size) for k, io in io_dict.items()}

    # Finished items are recorded under the output path and skipped on rerun;
    # the tags reject resuming with a different configuration
    progress = None
    if resume:
        progress = ProgressStore(
            os.path.join(cfg.file_output.path, "progress"),
            tags={"config": OmegaConf.to_container(cfg, resolve=True)},
        )
    return runner, items, io_dict, progress


def run_inference(
    runner: S2SRunner,
    items: list[WorkItem],
    io_dict: dict[str, IOBackend],
    progress: ProgressStore | None = None,
) -> None:
    """Run every work item over all ranks and write one store per output region.

    Parameters
    ----------
    runner : S2SRunner
        Runner of one perturbation batch.
    items : list[WorkItem]
        Every work item of the run, identical on every rank.
    io_dict : dict[str, IOBackend]
        Output backend per output region.
    progress : ProgressStore, optional
        Records finished items so a rerun resumes, by default None
    """
    Pipeline(runner, output=io_dict, progress=progress).run(items)

    if DistributedManager().rank == 0:
        logger.info("Consolidating metadata in zarr files")
        for io in io_dict.values():
            store = getattr(io, "store", None)
            if store is not None:
                zarr.consolidate_metadata(store)
    logger.success(
        f"Inference complete: {len(items)} batches, "
        f"{len(np.unique([m for item in items for m in item.member_ids]))} members"
    )
