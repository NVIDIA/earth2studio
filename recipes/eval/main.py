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
from src.output import OutputManager, sentinel_path
from src.pipeline import build_pipeline
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

    # --- Pipeline setup -----------------------------------------------------
    pipeline = build_pipeline(cfg)
    pipeline.setup(cfg, device)

    all_times = np.array(sorted({item.time for item in all_items}))
    output_variables = list(cfg.output.variables)
    total_coords = pipeline.build_total_coords(all_times, cfg.get("ensemble_size", 1))
    data_source = hydra.utils.instantiate(cfg.data_source)

    # --- Run ----------------------------------------------------------------
    with OutputManager(cfg) as output_mgr:
        output_mgr.validate_output_store(total_coords, output_variables)
        if my_items:
            pipeline.run(my_items, data_source, output_mgr, output_variables, device)
        else:
            logger.info(f"Rank {dist.rank}: no work items, waiting at barrier.")

    logger.success("Eval recipe finished.")


if __name__ == "__main__":
    main()
