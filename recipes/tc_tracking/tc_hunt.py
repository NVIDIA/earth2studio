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

# Set MKL/OMP threading environment variables BEFORE any other imports
# This prevents MKL initialization race conditions that can cause
# divide-by-zero crashes when running FCN3 with NCCL/UCX
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import hydra  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from src.modes.generate_tc_hunt_ensembles import (  # noqa: E402
    generate_ensemble,
    reproduce_members,
)


@hydra.main(version_base=None, config_path="cfg", config_name="none")
def tc_hunt(cfg: DictConfig) -> None:
    """main function with initialisation."""

    if cfg.mode == "generate_ensemble":
        generate_ensemble(cfg)

    elif cfg.mode == "reproduce_members":
        reproduce_members(cfg)

    else:
        raise ValueError(
            f'invalid mode: {cfg.mode}, choose from "generate_ensemble", "reproduce_members"'
        )

    logger.success("finished **yaaayyyy**")

    return


if __name__ == "__main__":
    tc_hunt()
