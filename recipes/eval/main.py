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

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.distributed import configure_logging
from src.inference import run_inference
from src.models import load_diagnostics, load_prognostic
from src.output import OutputManager
from src.work import build_work_items, distribute_work


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe entry point — distributed model inference."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()

    # --- Build and distribute work ------------------------------------------
    all_items = build_work_items(cfg)
    my_items = distribute_work(all_items, dist.rank, dist.world_size)

    # --- Load models --------------------------------------------------------
    # All ranks must participate so barriers inside run_on_rank0_first are met.
    prognostic = load_prognostic(cfg)
    diagnostics = load_diagnostics(cfg)

    # --- Instantiate perturbation if running ensembles ----------------------
    perturbation = None
    if cfg.get("ensemble_size", 1) > 1 and "perturbation" in cfg:
        perturbation = hydra.utils.instantiate(cfg.perturbation)

    # --- Instantiate data source --------------------------------------------
    data_source = hydra.utils.instantiate(cfg.data_source)

    # --- Collect all IC times for output coordinate setup -------------------
    all_times = np.array(sorted({item.time for item in all_items}))

    # --- Run inference with managed output ----------------------------------
    # OutputManager.__enter__/__exit__ contain barriers, so every rank must
    # enter the context manager even if it has no work items.
    with OutputManager(
        cfg,
        prognostic=prognostic,
        times=all_times,
        nsteps=cfg.nsteps,
        ensemble_size=cfg.get("ensemble_size", 1),
    ) as output_mgr:
        if my_items:
            run_inference(
                work_items=my_items,
                prognostic=prognostic,
                data_source=data_source,
                output_mgr=output_mgr,
                nsteps=cfg.nsteps,
                perturbation=perturbation,
                diagnostics=diagnostics,
            )
        else:
            logger.info(f"Rank {dist.rank}: no work items, waiting at barrier.")

    logger.success("Eval recipe finished.")


if __name__ == "__main__":
    main()
