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

"""Packed-head EDM-to-flow preconditioner for PDD inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from earth2studio.models.px._stormscope_flash.dit import DiT
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from physicsnemo import Module
except ImportError:
    OptionalDependencyFailure("stormscope-flash")
    # Permit optional-dependency discovery; the decorated class cannot instantiate.
    Module = nn.Module


def sigma_to_flow_time(sigma: Any) -> torch.Tensor:
    """Map a non-negative EDM noise level to data-time ``t`` in ``(0, 1]``."""

    sigma_tensor = torch.as_tensor(sigma)
    if not torch.compiler.is_compiling() and bool((sigma_tensor < 0).any()):
        raise ValueError("sigma cannot be negative")
    return (1 + sigma_tensor).reciprocal()


def flow_time_to_sigma(time: Any) -> torch.Tensor:
    """Map linear-flow data-time ``t`` in ``(0, 1]`` to EDM sigma."""

    time_tensor = torch.as_tensor(time)
    if not torch.compiler.is_compiling() and bool(
        ((time_tensor <= 0) | (time_tensor > 1)).any()
    ):
        raise ValueError("flow time must lie in (0, 1]")
    return (1 - time_tensor) / time_tensor


def edm_state_to_flow_state(x_sigma: torch.Tensor, sigma: Any) -> torch.Tensor:
    """Convert an EDM state ``x_sigma`` to the bounded flow state ``y_t``."""

    time = sigma_to_flow_time(sigma).to(device=x_sigma.device, dtype=x_sigma.dtype)
    while time.ndim < x_sigma.ndim:
        time = time.unsqueeze(-1)
    return time * x_sigma


def flow_state_to_edm_state(y_t: torch.Tensor, time: Any) -> torch.Tensor:
    """Convert a bounded flow state ``y_t`` to the corresponding EDM state."""

    time_tensor = torch.as_tensor(time, device=y_t.device, dtype=y_t.dtype)
    if not torch.compiler.is_compiling() and bool(
        ((time_tensor <= 0) | (time_tensor > 1)).any()
    ):
        raise ValueError("flow time must lie in (0, 1]")
    while time_tensor.ndim < y_t.ndim:
        time_tensor = time_tensor.unsqueeze(-1)
    return y_t / time_tensor


class PDDPrecond(nn.Module):
    """Inference-only regional PDD expert backed by a DiT trunk."""

    sigma_grid: torch.Tensor
    time_grid: torch.Tensor
    block_boundaries: torch.Tensor

    def __init__(
        self,
        model: DiT,
        sigma_grid: torch.Tensor | Sequence[float],
        block_boundaries: torch.Tensor | Sequence[int],
        *,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        sigmas = torch.as_tensor(sigma_grid, dtype=torch.float32)
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("sigma_grid must have at least two nodes")
        if not bool(torch.isfinite(sigmas).all()):
            raise ValueError("sigma_grid must contain only finite values")
        if not bool((sigmas[:-1] > sigmas[1:]).all()) or bool((sigmas < 0).any()):
            raise ValueError("sigma_grid must be non-negative and strictly decreasing")

        boundaries = torch.as_tensor(block_boundaries, dtype=torch.long)
        num_intervals = sigmas.numel() - 1
        if (
            boundaries.ndim != 1
            or boundaries.numel() < 2
            or int(boundaries[0]) != 0
            or int(boundaries[-1]) != num_intervals
            or not bool((boundaries[:-1] < boundaries[1:]).all())
        ):
            raise ValueError(
                "block_boundaries must increase from 0 to len(sigma_grid) - 1"
            )
        if sigma_data != 0.5:
            raise ValueError("StormScope Flash experts require sigma_data=0.5")
        if not hasattr(model, "forward_features") or not hasattr(model, "unpatchify"):
            raise TypeError("PDD model must expose DiT feature and unpatchify methods")
        final_layer = getattr(model, "_final_layer", None)
        if final_layer is None or not hasattr(final_layer, "linear"):
            raise TypeError("PDD model must expose _final_layer.linear")

        self.model = model
        self.sigma_data = float(sigma_data)
        self.use_fp16 = False
        self.num_intervals = num_intervals
        self.register_buffer("sigma_grid", sigmas.contiguous())
        self.register_buffer("time_grid", sigma_to_flow_time(sigmas).contiguous())
        self.register_buffer("block_boundaries", boundaries.contiguous())

        projection = self.model._final_layer.linear
        weight = projection.weight.detach()
        bias = (
            projection.bias.detach()
            if projection.bias is not None
            else weight.new_zeros(weight.shape[0])
        )
        self.head_weight = nn.Parameter(weight.unsqueeze(0).repeat(num_intervals, 1, 1))
        self.head_bias = nn.Parameter(bias.unsqueeze(0).repeat(num_intervals, 1))
        projection.requires_grad_(False)

    @property
    def sigma_min(self) -> float:
        terminal = float(self.sigma_grid[-1])
        return float(self.sigma_grid[-2]) if terminal == 0.0 else terminal

    @property
    def sigma_max(self) -> float:
        return float(self.sigma_grid[0])

    def round_sigma(self, sigma: Any) -> torch.Tensor:
        """Expose the compatibility method provided by EDM preconditioners."""

        return torch.as_tensor(sigma)

    @staticmethod
    def _batch_sigma(sigma: Any, batch_size: int, device: torch.device) -> torch.Tensor:
        values = torch.as_tensor(sigma, device=device, dtype=torch.float32)
        if values.numel() == 1:
            values = values.reshape(1).expand(batch_size)
        elif values.numel() == batch_size:
            values = values.reshape(batch_size)
        else:
            raise ValueError("sigma must be scalar or contain one value per sample")
        if not torch.compiler.is_compiling() and bool((values <= 0).any()):
            raise ValueError("PDD block-start sigma must be positive")
        return values

    def _trunk_dtype(self) -> torch.dtype:
        try:
            dtype = next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32
        return dtype if dtype in (torch.float16, torch.bfloat16) else torch.float32

    @torch.no_grad()
    def prepare_inference_block(
        self, start_index: int, end_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-fuse all interval heads used by one deployment call."""

        if (
            start_index < 0
            or end_index <= start_index
            or end_index > self.num_intervals
        ):
            raise IndexError("invalid PDD inference block")
        coefficients = (
            self.time_grid[start_index + 1 : end_index + 1]
            - self.time_grid[start_index:end_index]
        ).to(device=self.head_weight.device, dtype=self.head_weight.dtype)
        weight = torch.einsum(
            "k,koh->oh", coefficients, self.head_weight[start_index:end_index]
        )
        bias = torch.einsum(
            "k,ko->o", coefficients, self.head_bias[start_index:end_index]
        )
        sigma = self.sigma_grid[start_index]
        delta = self.time_grid[end_index] - self.time_grid[start_index]
        return (
            sigma.detach(),
            weight.detach().contiguous(),
            bias.detach().contiguous(),
            delta.detach(),
        )

    @torch.no_grad()
    def prepare_condition_patch(self, condition: torch.Tensor) -> torch.Tensor | None:
        """Cache the condition contribution to the DiT patch projection."""

        if not hasattr(self.model, "prepare_condition_patch"):
            return None
        return self.model.prepare_condition_patch(
            condition.to(device=self.head_weight.device, dtype=self._trunk_dtype()),
            state_channels=int(getattr(self.model, "_out_chans", 0)),
            output_dtype=self._trunk_dtype(),
        )

    def _features_and_coefficients(
        self,
        state: torch.Tensor,
        sigma: Any,
        condition: torch.Tensor | None,
        condition_patch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 4:
            raise ValueError("state must have shape [batch, channel, height, width]")
        if condition is not None and condition_patch is not None:
            raise ValueError("provide condition or condition_patch, not both")

        state = state.to(torch.float32)
        sigma_batch = self._batch_sigma(sigma, state.shape[0], state.device)
        sigma_4d = sigma_batch.reshape(-1, 1, 1, 1)
        time_4d = sigma_to_flow_time(sigma_4d)
        # Data-time t=1/(1+sigma); avoid subtracting nearly equal EDM terms.
        noise_time = sigma_4d * time_4d
        denominator = noise_time.square() + self.sigma_data**2 * time_4d.square()
        velocity_skip = (self.sigma_data**2 * time_4d - noise_time) / denominator
        velocity_out = self.sigma_data * denominator.rsqrt()
        model_input = denominator.rsqrt() * state
        c_noise = sigma_batch.log() / 4
        if condition is not None:
            if (
                condition.ndim != 4
                or condition.shape[0] != state.shape[0]
                or condition.shape[-2:] != state.shape[-2:]
            ):
                raise ValueError(
                    "condition batch and spatial dimensions must match state"
                )
            model_input = torch.cat(
                (model_input, condition.to(device=state.device, dtype=torch.float32)),
                dim=1,
            )

        try:
            time_condition = c_noise.to(self.model._time_step_emb.freqs.dtype)
        except AttributeError:
            time_condition = c_noise
        if condition_patch is None:
            features = self.model.forward_features(
                model_input.to(self._trunk_dtype()),
                time_condition,
                training=False,
            )
        else:
            features = self.model.forward_features_with_condition_patch(
                model_input.to(self._trunk_dtype()),
                condition_patch.to(device=state.device, dtype=self._trunk_dtype()),
                time_condition,
                p_dropout=None,
                training=False,
            )
        return features, state, velocity_skip, velocity_out

    def _project(
        self,
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=features.device.type, enabled=False):
            projected = torch.einsum(
                "bnh,oh->bno",
                features.to(torch.float32),
                weight.to(device=features.device, dtype=torch.float32),
            )
            projected = (
                projected
                + bias.to(device=features.device, dtype=torch.float32)[None, None]
            )
        return self.model.unpatchify(projected, height, width)

    def forward(
        self,
        state: torch.Tensor,
        sigma: Any,
        condition: torch.Tensor | None,
        *,
        inference_weight: torch.Tensor,
        inference_bias: torch.Tensor,
        inference_delta: torch.Tensor,
        condition_patch: torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        """Evaluate one pre-fused PDD block."""

        if training:
            raise ValueError("packaged PDD experts support inference only")
        height, width = state.shape[-2:]
        features, state, velocity_skip, velocity_out = self._features_and_coefficients(
            state, sigma, condition, condition_patch
        )
        residual = self._project(
            features, inference_weight, inference_bias, height, width
        )
        return (
            state
            + inference_delta.to(device=state.device, dtype=torch.float32)
            * velocity_skip
            * state
            + velocity_out * residual.to(torch.float32)
        )


@check_optional_dependencies()
class PDDModel(PDDPrecond, Module):
    """Reconstructible PhysicsNeMo deployment model for a packed PDD expert.

    Parameters
    ----------
    model_config : dict[str, Any]
        JSON-serializable constructor arguments for the checkpoint-compatible DiT.
    pdd_metadata : dict[str, Any]
        Trained sigma grid, block boundaries, and deployment compatibility metadata.

    Notes
    -----
    Large parameters are constructed on the meta device and assigned when loading
    the checkpoint. Load weights before moving or calling this inference-only model.
    """

    def __init__(
        self, model_config: dict[str, Any], pdd_metadata: dict[str, Any]
    ) -> None:
        with torch.device("meta"):
            trunk = DiT(**model_config)
        super().__init__(
            trunk,
            pdd_metadata["sigma_grid"],
            pdd_metadata["block_boundaries"],
            sigma_data=pdd_metadata["config"]["student"]["sigma_data"],
        )
        self.model_config = deepcopy(model_config)
        self.pdd_metadata = deepcopy(pdd_metadata)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True
    ) -> Any:
        """Assign checkpoint tensors without allocating a second initialized trunk."""
        if not assign:
            raise ValueError("PDD deployment weights require assign=True")
        result = super().load_state_dict(state_dict, strict=strict, assign=True)
        if any(t.is_meta for t in self.parameters()):
            raise ValueError("PDD checkpoint left uninitialized model parameters")
        self.eval().requires_grad_(False)
        return result
