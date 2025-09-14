# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from pathlib import Path
from typing import Any

import torch

from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import CoordSystem


def save_checkpoint(
    step: int,
    state: torch.Tensor,
    coords: CoordSystem,
    checkpoint_path: str,
    workflow_type: str = "deterministic",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save workflow checkpoint to disk.
    Parameters
    ----------

    step : int
        Current simulation step number
    state : torch.Tensor
        Current atmospheric state tensor on GPU
    coords : CoordSystem
        Current coordinate system (OrderedDict or coordinate arrays)
    checkpoint_path : str
        File path where checkpoint will be saved
    workflow_type : str, optional
        Type of workflow being checkpointed, be default "deterministic"
    metadata : dict[str, Any], optional
        Additional metadata to store with checkpoint, by default None
    """

    checkpoint = {
        "step": step,
        "state": state,
        "coords": coords,
        "workflow_type": workflow_type,
        "torch_rng_state": torch.get_rng_state(),
        "metadata": metadata or {},
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """Load workflow checkpoint from disk."""

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu())

    return checkpoint


def validate_checkpoint_compatibility(
    checkpoint_coords: CoordSystem, prognostic: PrognosticModel
) -> bool:
    """Validate that checkpoint is compatible with prognostic model."""
    try:
        expected_coords = prognostic.input_coords()

        for key in expected_coords:
            if key not in checkpoint_coords:
                return False
            if key == "batch":
                continue
            if expected_coords[key].shape != checkpoint_coords[key].shape:
                return False

        return True
    except Exception:
        return False


def should_checkpoint(
    step: int,
    checkpoint_interval: int | None,
    checkpoint_path: str | None,
) -> bool:
    """Determine if checkpoint should be saved at current step."""
    return (
        checkpoint_path is not None
        and checkpoint_interval is not None
        and step % checkpoint_interval == 0
    )
