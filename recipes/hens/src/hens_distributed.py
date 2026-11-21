# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher ranks without process-group collectives for independent HENS jobs."""

import os

import torch


def worker_rank() -> tuple[int, int]:
    """Read torchrun or Open MPI's global rank and worker count."""
    return (
        int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))),
        int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))),
    )


def local_device() -> torch.device:
    """Select this worker's GPU, or CPU when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    )
    return torch.device("cuda", rank)
