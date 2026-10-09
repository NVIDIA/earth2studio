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

"""Validated schedules and cached deployment plans for Flash inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from earth2studio.models.nn.stormscope_flash import (
    flow_time_to_sigma,
    sigma_to_flow_time,
)

FLASH_EXPERT_ORDER = ("high", "middle", "low")
FLASH_REGION_INTERVALS = {"high": 16, "middle": 32, "low": 80}
FLASH_ROUTING_BOUNDARY_INDICES = (2, 36)
FLASH_GLOBAL_NUM_INTERVALS = 128
FLASH_SIGMA_MAX = 800.0


class FlashExpert(Protocol):
    """Inference surface needed to build a Flash chain plan."""

    sigma_grid: torch.Tensor

    def prepare_inference_block(
        self, start_index: int, end_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the block's sigma, fused weight, fused bias, and time increment."""
        ...

    def prepare_condition_patch(self, condition: torch.Tensor) -> torch.Tensor | None:
        """Optionally cache the patch projection of static conditioning channels."""
        ...

    def __call__(
        self,
        state: torch.Tensor,
        sigma: torch.Tensor,
        condition: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class FlashInferenceBlock:
    """Device-resident, pre-fused data for one Flash model invocation."""

    start: int
    end: int
    sigma: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    delta: torch.Tensor


@dataclass(frozen=True)
class FlashExpertPlan:
    """Cached inference blocks for one regional expert."""

    name: str
    sigmas: torch.Tensor
    blocks: tuple[FlashInferenceBlock, ...]


@dataclass(frozen=True)
class FlashChainPlan:
    """Validated immutable high-to-low plan reused by forecast steps."""

    total_nfe: int
    experts: Mapping[str, FlashExpertPlan]
    first_time: torch.Tensor


def uniform_flow_time_sigma_grid(
    *,
    num_intervals: int,
    sigma_max: float,
    sigma_min: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a decreasing EDM sigma grid with uniform bounded-flow spacing."""

    if isinstance(num_intervals, bool) or not isinstance(num_intervals, int):
        raise TypeError("num_intervals must be an integer")
    if num_intervals < 1:
        raise ValueError("num_intervals must be positive")
    if sigma_min < 0 or sigma_max <= sigma_min:
        raise ValueError("require 0 <= sigma_min < sigma_max")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    sigma_max_tensor = torch.tensor(float(sigma_max), device=device, dtype=dtype)
    sigma_min_tensor = torch.tensor(float(sigma_min), device=device, dtype=dtype)
    time_grid = torch.linspace(
        sigma_to_flow_time(sigma_max_tensor),
        sigma_to_flow_time(sigma_min_tensor),
        num_intervals + 1,
        device=device,
        dtype=dtype,
    )
    sigma_grid = flow_time_to_sigma(time_grid)
    sigma_grid[0] = sigma_max_tensor
    sigma_grid[-1] = sigma_min_tensor
    if not bool(torch.all(sigma_grid[:-1] > sigma_grid[1:])):
        raise RuntimeError("uniform flow-time sigma grid is not strictly decreasing")
    return sigma_grid.contiguous()


def regional_sigma_grids(
    *,
    num_intervals: int = FLASH_GLOBAL_NUM_INTERVALS,
    sigma_max: float = FLASH_SIGMA_MAX,
    boundary_indices: tuple[int, int] = FLASH_ROUTING_BOUNDARY_INDICES,
    region_intervals: Mapping[str, int] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build the three regional grids from one global routing reference."""

    interval_counts = region_intervals or FLASH_REGION_INTERVALS
    if set(interval_counts) != set(FLASH_EXPERT_ORDER):
        raise ValueError("region_intervals must define high, middle, and low")
    global_grid = uniform_flow_time_sigma_grid(
        num_intervals=num_intervals,
        sigma_max=sigma_max,
        sigma_min=0.0,
        device=device,
        dtype=dtype,
    )
    high_middle_index, middle_low_index = boundary_indices
    if not 0 < high_middle_index < middle_low_index < num_intervals:
        raise ValueError("invalid Flash routing boundary indices")
    high_middle = float(global_grid[high_middle_index].detach().cpu())
    middle_low = float(global_grid[middle_low_index].detach().cpu())
    return {
        "high": uniform_flow_time_sigma_grid(
            num_intervals=int(interval_counts["high"]),
            sigma_max=sigma_max,
            sigma_min=high_middle,
            device=device,
            dtype=dtype,
        ),
        "middle": uniform_flow_time_sigma_grid(
            num_intervals=int(interval_counts["middle"]),
            sigma_max=high_middle,
            sigma_min=middle_low,
            device=device,
            dtype=dtype,
        ),
        "low": uniform_flow_time_sigma_grid(
            num_intervals=int(interval_counts["low"]),
            sigma_max=middle_low,
            sigma_min=0.0,
            device=device,
            dtype=dtype,
        ),
    }


def inference_block_boundaries(
    num_intervals: int,
    nfe: int,
    *,
    alignment: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return balanced Flash block boundaries for a regional call budget."""

    if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 1:
        raise ValueError("nfe must be a positive integer")
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment < 1:
        raise ValueError("alignment must be a positive integer")
    if num_intervals % alignment or nfe > num_intervals // alignment:
        raise ValueError("call budget must fit aligned training intervals")
    base, remainder = divmod(num_intervals // alignment, nfe)
    boundaries = [0]
    cursor = 0
    for block_index in range(nfe):
        cursor += alignment * (base + int(block_index < remainder))
        boundaries.append(cursor)
    return torch.tensor(boundaries, device=device, dtype=torch.long)


def _unwrap_expert(expert: FlashExpert) -> FlashExpert:
    module = getattr(expert, "module", expert)
    return getattr(module, "_orig_mod", module)


def build_flash_chain_plan(
    experts: Mapping[str, FlashExpert],
    *,
    total_nfe: int,
    region_calls: Mapping[str, int],
    alignment: int = 16,
) -> FlashChainPlan:
    """Validate handoffs and pre-fuse a high/middle/low deployment schedule."""

    if set(experts) != set(FLASH_EXPERT_ORDER):
        raise ValueError(
            "Flash chain must contain exactly high, middle, and low experts"
        )

    calls = region_calls
    if set(calls) != set(FLASH_EXPERT_ORDER) or sum(calls.values()) != total_nfe:
        raise ValueError("regional calls must sum to the requested total NFE")
    plans: dict[str, FlashExpertPlan] = {}
    for name in FLASH_EXPERT_ORDER:
        module = _unwrap_expert(experts[name])
        sigmas = torch.as_tensor(
            module.sigma_grid,
            device=module.sigma_grid.device,
            dtype=torch.float64,
        )
        boundaries = inference_block_boundaries(
            sigmas.numel() - 1, calls[name], alignment=alignment
        ).tolist()
        blocks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            sigma, weight, bias, delta = module.prepare_inference_block(start, end)
            blocks.append(
                FlashInferenceBlock(
                    start=start,
                    end=end,
                    sigma=sigma,
                    weight=weight,
                    bias=bias,
                    delta=delta,
                )
            )
        plans[name] = FlashExpertPlan(name, sigmas, tuple(blocks))

    if not torch.isclose(
        plans["high"].sigmas[0],
        plans["high"].sigmas.new_tensor(FLASH_SIGMA_MAX),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(f"high Flash expert must start at sigma={FLASH_SIGMA_MAX}")
    for left, right in zip(FLASH_EXPERT_ORDER[:-1], FLASH_EXPERT_ORDER[1:]):
        if not torch.isclose(
            plans[left].sigmas[-1],
            plans[right].sigmas[0],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"invalid Flash handoff between {left} and {right}")
    if float(plans["low"].sigmas[-1].detach().cpu()) != 0.0:
        raise ValueError("low Flash expert must terminate at sigma=0")
    if sum(len(plan.blocks) for plan in plans.values()) != total_nfe:
        raise RuntimeError("Flash chain plan does not match the requested NFE budget")

    return FlashChainPlan(
        total_nfe=total_nfe,
        experts=plans,
        first_time=sigma_to_flow_time(plans["high"].sigmas[0]),
    )


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{label} contains non-finite values")


def _advance_blocks(
    expert: FlashExpert,
    state: torch.Tensor,
    condition: torch.Tensor | None,
    plan: FlashExpertPlan,
) -> torch.Tensor:
    module = _unwrap_expert(expert)
    condition_patch = (
        module.prepare_condition_patch(condition)
        if condition is not None and hasattr(module, "prepare_condition_patch")
        else None
    )
    for block in plan.blocks:
        sigma = block.sigma.to(
            device=state.device,
            dtype=torch.float64 if state.dtype == torch.float64 else torch.float32,
        )
        sigma = sigma.reshape(1).expand(state.shape[0])
        state = expert(
            state,
            sigma,
            condition if condition_patch is None else None,
            inference_weight=block.weight,
            inference_bias=block.bias,
            inference_delta=block.delta,
            condition_patch=condition_patch,
            training=False,
        )
        if not isinstance(state, torch.Tensor):
            raise TypeError("each Flash block must return a tensor")
        if hasattr(expert, "_orig_mod"):
            state = state.clone()
    return state


@torch.no_grad()
def flash_sampler_chain(
    experts: Mapping[str, FlashExpert],
    latents: torch.Tensor,
    condition: torch.Tensor | None = None,
    *,
    total_nfe: int,
    plan: FlashChainPlan,
) -> torch.Tensor:
    """Run the prepared high-to-low Flash chain without intermediate noise injection."""

    if latents.ndim != 4 or not latents.dtype.is_floating_point:
        raise ValueError(
            "latents must be a floating [batch, channel, height, width] tensor"
        )
    _require_finite(latents, "Flash latents")
    if condition is not None:
        if condition.ndim != 4:
            raise ValueError(
                "condition must be a [batch, channel, height, width] tensor"
            )
        _require_finite(condition, "Flash condition")

    if plan.total_nfe != total_nfe:
        raise ValueError(
            f"Flash plan NFE={plan.total_nfe} does not match requested {total_nfe}"
        )

    state = latents.float() * (1 - plan.first_time).to(
        device=latents.device, dtype=latents.dtype
    )
    for name in FLASH_EXPERT_ORDER:
        state = _advance_blocks(experts[name], state, condition, plan.experts[name])
        if state.shape != latents.shape:
            raise ValueError(
                f"Flash {name} expert returned {tuple(state.shape)}, "
                f"expected {tuple(latents.shape)}"
            )
    return state
