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
import signal
import sys

# Set MKL/OMP threading environment variables BEFORE any other imports
# This prevents MKL initialization race conditions that can cause
# divide-by-zero crashes when running FCN3 with NCCL/UCX
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
# Let PyTorch grow its memory segments instead of fragmenting the GPU memory,
# must be set before CUDA is initialised
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from src.data.zarr_merge import merge_batch_stores  # noqa: E402
from src.modes.baseline_extraction import extract_baseline  # noqa: E402
from src.modes.generate_tc_hunt_ensembles import run_ensemble  # noqa: E402


def _exit_on_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into a normal exit, so that cleanup code runs."""
    sys.exit(128 + signum)


@hydra.main(version_base=None, config_path="cfg", config_name="none")
def tc_hunt(cfg: DictConfig) -> None:
    """main function with initialisation."""
    # SIGTERM (e.g. kill, or torchrun stopping its workers) would otherwise end
    # the process without running finally blocks, leaving scratch files in
    # /dev/shm. Ctrl-C does not work, as intake_esgf replaces the SIGINT handler
    signal.signal(signal.SIGTERM, _exit_on_sigterm)

    if cfg.mode in ("generate_ensemble", "reproduce_members"):
        run_ensemble(cfg)

    elif cfg.mode == "extract_baseline":
        extract_baseline(cfg)

    # merge the batch stores of an interrupted Zarr run
    elif cfg.mode == "merge_zarr":
        merge_batch_stores(cfg)

    else:
        raise ValueError(
            f"invalid mode: {cfg.mode}, choose from "
            f'"generate_ensemble", "reproduce_members", "extract_baseline", "merge_zarr"'
        )

    logger.success("finished **yaaayyyy**")

    return


if __name__ == "__main__":
    tc_hunt()
