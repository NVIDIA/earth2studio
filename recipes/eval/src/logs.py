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

"""Process setup shared by the entry points: logging and the process group."""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
from loguru import logger

_NOISY_LOGGERS = frozenset({"makani.models.model_package", "numba.core.transforms"})


def configure_logging() -> None:
    """Route stdlib logging through loguru and quiet noisy packages."""

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level: str | int = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            logger.opt(depth=6, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.DEBUG, force=True)
    logger.remove()
    rank = int(os.environ.get("RANK", 0))
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            f"rank {rank} | <level>{{message}}</level>"
        ),
        filter=lambda r: r["name"] not in _NOISY_LOGGERS,
    )


def init_distributed(timeout_s: float | None = None) -> torch.device:
    """Initialize ``torch.distributed`` under ``torchrun`` and pick the device.

    Without launcher environment variables the process stays single-rank.
    The library's :class:`~earth2studio.run.Pipeline` reads rank and world
    size from ``torch.distributed`` directly.

    Parameters
    ----------
    timeout_s : float, optional
        Collective timeout in seconds, by default the backend default

    Returns
    -------
    torch.device
        ``cuda:{LOCAL_RANK}`` when CUDA is available, otherwise CPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        kwargs = {}
        if timeout_s is not None:
            kwargs["timeout"] = timedelta(seconds=float(timeout_s))
        dist.init_process_group(backend=backend, **kwargs)
        logger.info(
            f"Initialized {backend} process group: rank {dist.get_rank()} of "
            f"{dist.get_world_size()}"
        )
    return device


def finalize_distributed() -> None:
    """Tear down the process group if this process created one."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
