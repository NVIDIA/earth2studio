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

from typing import Any

import numpy as np
import zarr
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.data.base import DataSource
from src.s2s_ensemble import S2SEnsembleRunner
from src.s2s_perturbation import initialize_perturbation
from src.s2s_utilities import (
    create_base_seed_string,
    initialize_output,
    initialize_output_structures,
    run_with_rank_ordered_execution,
    update_model_dict,
)


def run_inference(
    cfg: DictConfig,
    ensemble_configs: list[Any],
    model_dict: dict,
    dx_model_dict: dict,
    data_source: DataSource,
    output_coords_dict: dict,
    base_random_seed: str | int,
) -> None:
    """S2S run inference function for subseasonal forecasting

    This function coordinates the ensemble forecast process for seasonal-to-subseasonal
    forecasting.
    """
    # We iterate through each ensemble configuration to generate ensemble members
    # The process applies appropriate perturbations to initial conditions and runs
    # the forecast model to evaluate ensemble mean predictions against ERA5 data

    # Before running the ensemble, initialize the full output zarr file across all ranks
    # Done in a rank-ordered manner to avoid race conditions: rank 0 creates the zarr first
    dist = DistributedManager()
    times = np.array(sorted(list({ic for (_, ic, _, _) in ensemble_configs})))

    io_dict = run_with_rank_ordered_execution(
        initialize_output,
        cfg,
        times,
        model_dict,
        output_coords_dict,
        add_arrays=(dist.rank == 0),
    )

    for pkg, ic, ens_idx, batch_ids_produce in ensemble_configs:
        # Create seed base string for reproducibility
        base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)

        # Load new weights if necessary
        model_dict = update_model_dict(model_dict, pkg)

        # Create new IO write threads, if multiprocessing IO is enabled
        writer_executor, writer_threads = initialize_output_structures(cfg)

        # Initialize the S2S perturbation method
        perturbation = initialize_perturbation(
            cfg=cfg, model=model_dict["model"], data_source=data_source
        )

        # Initialize ensemble runner for S2S forecasting
        ensemble_runner = S2SEnsembleRunner(
            time=[ic],
            nsteps=cfg.nsteps,
            nperturbed=cfg.nperturbed,
            ncheckpoints=cfg.ncheckpoints,
            prognostic=model_dict["model"],
            data=data_source,
            io_dict=io_dict,
            perturbation=perturbation,
            output_coords_dict=output_coords_dict,
            dx_model_dict=dx_model_dict,
            batch_size=cfg.batch_size,
            ensemble_idx_base=ens_idx,
            batch_ids_produce=batch_ids_produce,
            base_seed_string=base_seed_string,
            pkg=pkg,
            writer_executor=writer_executor,
            writer_threads=writer_threads,
        )

        # Run the ensemble forecast
        ensemble_runner()

    # Consolidate metadata in zarr files
    if dist.rank == 0:
        logger.info("Consolidating metadata in zarr files")
        for k in io_dict.keys():
            zarr.consolidate_metadata(io_dict[k].store)
