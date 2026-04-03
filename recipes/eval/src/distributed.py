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

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from loguru import logger
from physicsnemo.distributed import DistributedManager

T = TypeVar("T")


def run_on_rank0_first(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute a function on rank 0 first, barrier, then on remaining ranks.

    This ensures that rank 0 can create filesystem objects (caches, zarr stores,
    etc.) before other ranks attempt to access them.  In a single-process
    setting the function is called once with no barriers.

    Parameters
    ----------
    func : Callable[..., T]
        Function to execute.
    *args : Any
        Positional arguments forwarded to *func*.
    **kwargs : Any
        Keyword arguments forwarded to *func*.

    Returns
    -------
    T
        Return value of *func*.
    """
    dist = DistributedManager()

    if not dist.distributed:
        return func(*args, **kwargs)

    if dist.rank == 0:
        result = func(*args, **kwargs)
        torch.distributed.barrier()  # Others let rank 0 go first, then sync
        torch.distributed.barrier()  # Other ranks execute, then sync again
        return result

    torch.distributed.barrier()  # Others wait for rank 0 to finish first, then sync
    result = func(*args, **kwargs)
    torch.distributed.barrier()  # Others execute, then sync again
    return result


def configure_logging() -> None:
    """Set up loguru as the unified logging sink and suppress noisy packages."""

    _NOISY_LOGGERS = frozenset({"makani.models.model_package", "numba.core.transforms"})

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
    logger.add(
        sys.stdout,
        level="INFO",
        filter=lambda r: r["name"] not in _NOISY_LOGGERS,
    )
