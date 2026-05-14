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

"""Scoring entry point — compare predictions against verification data.

Usage
-----
Single process::

    python score.py

With a campaign config::

    python score.py campaign=fcn3_2024_monthly

Distributed (multi-GPU)::

    torchrun --nproc_per_node=$NGPU --standalone score.py \\
        campaign=fcn3_2024_monthly

Prerequisites
-------------
1. ``predownload.py`` with ``predownload.verification.enabled=true``
2. ``main.py`` (inference must have completed)
"""

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.distributed import configure_logging, run_on_rank0_first
from src.output import OutputManager
from src.scoring import (
    add_score_arrays,
    build_input_coords_template,
    build_lead_time_chunks,
    build_superset_score_coords,
    group_score_arrays_by_dims,
    instantiate_metrics,
    open_prediction_store,
    open_verification_source,
    run_scoring,
    spatial_coords_from_dataset,
    validate_lead_time_chunking,
)
from src.work import (
    build_work_items,
    clear_scoring_progress,
    distribute_work,
    filter_scoring_completed,
)


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe scoring entry point — distributed metric computation."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()
    device = dist.device

    # --- Build and distribute work -----------------------------------------
    all_items = build_work_items(cfg)
    all_times = sorted({item.time for item in all_items})

    resume = cfg.scoring.get("resume", False)
    if resume:
        remaining_times = filter_scoring_completed(all_times, cfg)
        if not remaining_times:
            logger.success("All times already scored — nothing to do.")
            if dist.distributed:
                torch.distributed.barrier()
            return
    else:
        remaining_times = list(all_times)
        if cfg.scoring.output.get("overwrite", False):
            clear_scoring_progress(cfg)

    my_times = distribute_work(remaining_times, dist.rank, dist.world_size)

    # --- Open data stores --------------------------------------------------
    prediction_ds = open_prediction_store(cfg)
    verif_source = open_verification_source(cfg)

    # --- Extract spatial coords and build template -------------------------
    spatial_coords = spatial_coords_from_dataset(prediction_ds)
    variables = list(cfg.scoring.variables)
    lead_times = prediction_ds.lead_time.values

    input_template = build_input_coords_template(prediction_ds, lead_times, variables)

    # --- Instantiate and validate metrics ----------------------------------
    metrics = instantiate_metrics(cfg, spatial_coords)

    all_times_arr = np.array(all_times)
    chunk_size = cfg.scoring.get("lead_time_chunk_size", None)
    validate_lead_time_chunking(metrics, chunk_size, len(lead_times))
    lt_chunks = build_lead_time_chunks(lead_times, chunk_size)

    # --- Build score store coordinates -------------------------------------
    # Superset coords define the store's coordinate axes; individual metric
    # arrays may use any subset of these dimensions.
    superset_coords = build_superset_score_coords(
        metrics, input_template, all_times_arr
    )
    array_groups = group_score_arrays_by_dims(metrics, input_template, all_times_arr)

    # --- Run scoring -------------------------------------------------------
    store_name = cfg.scoring.output.get("store_name", "scores.zarr")
    overwrite = cfg.scoring.output.get("overwrite", False)

    with OutputManager(
        cfg, store_name=store_name, overwrite=overwrite, resume=resume
    ) as output_mgr:
        # Create store with superset coordinate axes (no data arrays yet).
        output_mgr.validate_output_store(superset_coords, [])
        # Add data arrays per dimension group — metrics that reduce
        # different dims get arrays with different dimension sets.
        run_on_rank0_first(add_score_arrays, output_mgr.io, array_groups)
        if my_times:
            run_scoring(
                my_times,
                prediction_ds,
                verif_source,
                metrics,
                output_mgr,
                variables,
                lead_times,
                lt_chunks,
                spatial_coords,
                device,
                cfg,
            )
        else:
            logger.info(f"Rank {dist.rank}: no times assigned, waiting at barrier.")

    logger.success("Scoring finished.")


if __name__ == "__main__":
    main()
