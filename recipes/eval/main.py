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
from src.models import load_diagnostics, load_prognostic
from src.output import (
    OutputManager,
    build_diagnostic_coords,
    build_forecast_coords,
    sentinel_path,
)
from src.work import build_work_items, distribute_work


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe entry point — distributed model inference."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()
    device = dist.device

    # --- Pre-download check -------------------------------------------------
    if cfg.get("require_predownload", True):
        sp = sentinel_path(cfg)
        if not sp.exists():
            raise RuntimeError(
                f"Pre-download sentinel not found at '{sp}'.\n"
                "Run 'python predownload.py' with the same config before inference.\n"
                "To skip this check, set require_predownload=false."
            )
        logger.info(f"Pre-download sentinel found: {sp}")

    # --- Build and distribute work ------------------------------------------
    all_items = build_work_items(cfg)
    my_items = distribute_work(all_items, dist.rank, dist.world_size)

    pipeline = cfg.get("pipeline", "forecast")

    if pipeline == "forecast":
        _run_forecast_pipeline(cfg, all_items, my_items, dist, device)
    elif pipeline == "diagnostic":
        _run_diagnostic_pipeline(cfg, all_items, my_items, dist, device)
    else:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. Expected 'forecast' or 'diagnostic'."
        )

    logger.success("Eval recipe finished.")


def _run_forecast_pipeline(
    cfg: DictConfig,
    all_items: list,
    my_items: list,
    dist: DistributedManager,
    device: object,
) -> None:
    """Standard prognostic forecast pipeline (with optional diagnostics)."""
    from src.inference import run_inference

    # All ranks must participate in model loading for barrier correctness.
    prognostic = load_prognostic(cfg)
    diagnostics = load_diagnostics(cfg)

    perturbation = None
    if cfg.get("ensemble_size", 1) > 1 and "perturbation" in cfg:
        perturbation = hydra.utils.instantiate(cfg.perturbation)

    data_source = hydra.utils.instantiate(cfg.data_source)

    all_times = np.array(sorted({item.time for item in all_items}))
    output_variables = list(cfg.output.variables)
    total_coords = build_forecast_coords(
        prognostic, all_times, cfg.nsteps, cfg.get("ensemble_size", 1)
    )

    with OutputManager(cfg) as output_mgr:
        output_mgr.validate_output_store(total_coords, output_variables)
        if my_items:
            run_inference(
                work_items=my_items,
                prognostic=prognostic,
                data_source=data_source,
                output_mgr=output_mgr,
                output_variables=output_variables,
                nsteps=cfg.nsteps,
                perturbation=perturbation,
                diagnostics=diagnostics,
                device=device,
            )
        else:
            logger.info(f"Rank {dist.rank}: no work items, waiting at barrier.")


def _run_diagnostic_pipeline(
    cfg: DictConfig,
    all_items: list,
    my_items: list,
    dist: DistributedManager,
    device: object,
) -> None:
    """Diagnostic-only pipeline (no prognostic rollout)."""
    from src.diagnostic_inference import run_diagnostic_inference

    diagnostics = load_diagnostics(cfg)
    if not diagnostics:
        raise ValueError(
            "Diagnostic pipeline requires at least one entry in 'diagnostics'."
        )

    data_source = hydra.utils.instantiate(cfg.data_source)

    all_times = np.array(sorted({item.time for item in all_items}))
    output_variables = list(cfg.output.variables)
    total_coords = build_diagnostic_coords(
        diagnostics, all_times, cfg.get("ensemble_size", 1)
    )

    with OutputManager(cfg) as output_mgr:
        output_mgr.validate_output_store(total_coords, output_variables)
        if my_items:
            run_diagnostic_inference(
                work_items=my_items,
                diagnostics=diagnostics,
                data_source=data_source,
                output_mgr=output_mgr,
                output_variables=output_variables,
                device=device,
            )
        else:
            logger.info(f"Rank {dist.rank}: no work items, waiting at barrier.")


if __name__ == "__main__":
    main()
