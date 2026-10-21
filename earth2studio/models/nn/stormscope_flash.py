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

import hashlib
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from earth2studio.models.auto import Package
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

if TYPE_CHECKING:
    from earth2studio.models.px.stormscope import StormScopeBase

try:
    from physicsnemo import Module
    from physicsnemo.models.dit import DiT as PhysicsNeMoDiT
    from physicsnemo.nn.module.dit_layers import DiTBlock as PhysicsNeMoDiTBlock
    from physicsnemo.nn.module.dit_layers import get_layer_norm
    from physicsnemo.nn.module.rope import build_axial_rope_cos_sin_2d
except (ImportError, OSError):
    OptionalDependencyFailure("stormscope")
    Module = PhysicsNeMoDiT = PhysicsNeMoDiTBlock = nn.Module


# TODO: Replace adapters with upstream PhysicsNeMo support as it becomes available.
# Remove numerical overrides only after upstream support and parity validation.
# Upstream candidates: fused interval heads, EDM-to-flow preconditioning, cached
# condition projection, NaN-safe mask-token replacement, and affine Q/K options.
# Compatibility: versioned .mdlus metadata with native parameter names, explicit
# projection/Q/K dtypes, and unfused residual products.


def _restore_input_dtype(
    module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    return output.to(inputs[0].dtype)


class _FlashDiTBlock(PhysicsNeMoDiTBlock):
    def __init__(
        self,
        block: PhysicsNeMoDiTBlock,
        *,
        qk_norm: bool,
    ) -> None:
        # Reuse native modules without allocating a second block.
        nn.Module.__init__(self)
        self.attention = block.attention
        self.pre_attention_norm = block.pre_attention_norm
        self.pre_mlp_norm = block.pre_mlp_norm
        self.interdrop = block.interdrop
        self.linear = block.linear
        self.adaptive_modulation = block.adaptive_modulation
        self.modulation = block.modulation
        self.drop_path = block.drop_path
        # Trained affine Q/K uses eps=1e-5; upstream defaults to non-affine 1e-6.
        if qk_norm:
            self.attention.q_norm = get_layer_norm(
                self.attention.head_dim, "torch", elementwise_affine=True, eps=1e-5
            )
            self.attention.k_norm = get_layer_norm(
                self.attention.head_dim, "torch", elementwise_affine=True, eps=1e-5
            )
        # NATTEN requires Q/K/V dtypes to match under AMP.
        self.attention.q_norm.register_forward_hook(_restore_input_dtype)
        self.attention.k_norm.register_forward_hook(_restore_input_dtype)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_kwargs: dict[str, Any] | None = None,
        p_dropout: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive attention and MLP residual updates for inference."""
        if p_dropout not in (None, 0, 0.0):
            raise ValueError("the packaged Flash DiT supports inference only")

        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.adaptive_modulation(
            c
        ).chunk(6, dim=1)
        attention_input = self.modulation(self.pre_attention_norm(x), scale, shift)
        attention_kwargs = dict(attn_kwargs or {})
        invalid_token_mask = attention_kwargs.pop("invalid_token_mask", None)
        mask_token = self.attention.mask_token
        if invalid_token_mask is not None and mask_token is not None:
            # The trained splice replaces NaNs; upstream's arithmetic mask cannot.
            attention_input = torch.where(
                invalid_token_mask[..., None],
                mask_token.to(attention_input),
                attention_input,
            )
        attended = self.attention(attention_input, **attention_kwargs)
        # Retain the checkpoint's reduced-precision product before residual addition.
        # The upstream block instead uses a fused addcmul update.
        gate = gate.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        x = x + self.drop_path(gate * attended)
        mlp = self.linear(self.modulation(self.pre_mlp_norm(x), scale_mlp, shift_mlp))
        return x + self.drop_path(gate_mlp * mlp)


class FlashDiT(PhysicsNeMoDiT):
    """PhysicsNeMo DiT for rotary Flash inference.

    Preserves the trained architecture with native checkpoint parameter names.
    """

    def __init__(
        self,
        *,
        height: int,
        width: int,
        patch_size: int,
        in_chans: int,
        base_out_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        frequency_embed_dim: int = 256,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        pos_embedding_type: str = "rotary",
        attn_kernel: int = 49,
        rope_theta: float = 10000.0,
        use_nan_mask_tokens: bool = False,
    ) -> None:
        if attn_kernel < 2:
            raise ValueError("StormScope Flash requires neighborhood attention")
        if pos_embedding_type != "rotary":
            raise ValueError("StormScope Flash requires a conditional rotary FlashDiT")

        if patch_size < 1 or height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        super().__init__(
            input_size=(height, width),
            in_channels=in_chans,
            out_channels=base_out_chans,
            patch_size=patch_size,
            hidden_size=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attention_backend="natten2d_rope",
            layernorm_backend="torch",
            conditioning_embedder_kwargs={
                "freq_embed_dim": frequency_embed_dim,
                "mlp_hidden_dim": embed_dim,
                "embed_fn": "np_sin_cos",
            },
            tokenizer_kwargs={"pos_embed": "none"},
            attn_kwargs={
                "qkv_bias": qkv_bias,
                "qk_norm": False,
                "attn_kernel": attn_kernel,
                "rope_theta": rope_theta,
            },
            dit_initialization=False,
            force_tokenization_fp32=True,
            use_nan_mask_tokens=use_nan_mask_tokens,
        )
        self._out_chans = base_out_chans
        self._in_chans = in_chans
        self._patch_size = patch_size
        self._height, self._width = height, width
        self.register_buffer("invalid_token_mask_flat", None, persistent=False)
        self.blocks = nn.ModuleList(
            _FlashDiTBlock(block, qk_norm=qk_norm) for block in self.blocks
        )
        if self.rope.rope_cos.is_meta:
            # Nonpersistent RoPE tables need materializing before .to().
            self._materialize_rope(torch.device("cpu"))

    invalid_token_mask_flat: torch.Tensor | None

    def _materialize_rope(self, device: torch.device) -> None:
        self.rope.rope_cos, self.rope.rope_sin = build_axial_rope_cos_sin_2d(
            *self.rope._latent_hw,
            self.rope.head_dim,
            theta=self.rope.theta,
            device=device,
        )

    def set_image_size(self, height: int, width: int) -> None:
        """Set this instance's rotary grid before forecasting or compiling."""
        if min(height, width) < 200 or height % 4 or width % 4:
            raise ValueError(
                "Flash image dimensions must be >=200 and divisible by four"
            )
        self._height, self._width = height, width
        self.input_size = (height, width)
        self._latent_h = height // self._patch_size
        self._latent_w = width // self._patch_size
        self.attn_kwargs_forward["latent_hw"] = (self._latent_h, self._latent_w)
        for module in (self.tokenizer, self.detokenizer):
            module.input_size = self.input_size
            module.h_patches, module.w_patches = self._latent_h, self._latent_w
        self.tokenizer.num_patches = self._latent_h * self._latent_w
        self.tokenizer.x_embedder.img_size = self.input_size
        self.invalid_token_mask_flat = None

    def set_nan_pixel_mask(self, pixel_mask: torch.Tensor) -> None:
        """Set missing conditioning pixels, marking every touched patch invalid.

        Parameters
        ----------
        pixel_mask : torch.Tensor
            Boolean mask with shape [height, width] or [batch, height, width].
        """
        if not self._use_nan_mask_tokens:
            raise RuntimeError("this FlashDiT does not use NaN mask tokens")
        if pixel_mask.ndim not in (2, 3) or pixel_mask.shape[-2:] != (
            self._height,
            self._width,
        ):
            raise ValueError("pixel mask must match the model image dimensions")
        if pixel_mask.ndim == 2:
            pixel_mask = pixel_mask[None]
        self.invalid_token_mask_flat = self._pixel_mask_to_token_mask(
            pixel_mask.to(device=self.tokenizer.x_embedder.proj.weight.device)
        )

    def prepare_patch_tokens(
        self, patch: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Flatten a rotary model's projected image without additive positions."""
        return patch.flatten(2).transpose(1, 2), patch.shape[-2:]

    def prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Embed an image and return flattened tokens with their patch-grid shape."""
        # Preserve trained FP32 tokenization under AMP.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            tokens = self.tokenizer(x.to(torch.float32))
        return tokens.to(x.dtype), (
            x.shape[-2] // self._patch_size,
            x.shape[-1] // self._patch_size,
        )

    def _split_patch_projection(
        self,
        x: torch.Tensor,
        *,
        channel_start: int,
        channel_stop: int,
        include_bias: bool,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        projection = self.tokenizer.x_embedder.proj
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            projected = F.conv2d(
                x.to(torch.float32),
                projection.weight[:, channel_start:channel_stop].to(torch.float32),
                (
                    projection.bias.to(torch.float32)
                    if include_bias and projection.bias is not None
                    else None
                ),
                stride=projection.stride,
                padding=projection.padding,
                dilation=projection.dilation,
                groups=projection.groups,
            )
        return projected.to(output_dtype)

    def prepare_condition_patch(
        self,
        condition: torch.Tensor,
        *,
        state_channels: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project static condition channels once for a Flash trajectory."""

        if condition.ndim != 4:
            raise ValueError(
                "condition must have shape [batch, channel, height, width]"
            )
        expected = self._in_chans - state_channels
        if (
            state_channels <= 0
            or state_channels >= self._in_chans
            or condition.shape[1] != expected
        ):
            raise ValueError(
                f"condition has {condition.shape[1]} channels, expected {expected}"
            )
        return self._split_patch_projection(
            condition,
            channel_start=state_channels,
            channel_stop=self._in_chans,
            include_bias=False,
            output_dtype=output_dtype,
        )

    def forward_features_from_tokens(
        self,
        tokens: torch.Tensor,
        latent_hw: tuple[int, int],
        time_step_cond: torch.Tensor | None,
        *,
        training: bool = False,
    ) -> torch.Tensor:
        """Run the shared transformer on prepared tokens at the given noise level."""
        if training:
            raise ValueError("the packaged Flash DiT supports inference only")
        if time_step_cond is None:
            time_step_cond = torch.zeros(
                tokens.shape[0], device=tokens.device, dtype=tokens.dtype
            )
        condition = self.conditioning_embedder(time_step_cond)
        # Keep RoPE construction in the graph as in the deployed model. Reusing
        # eager cached tables changes compiled trigonometric rounding. A compiler
        # flag is insufficient here because attention may introduce graph breaks.
        cos, sin = build_axial_rope_cos_sin_2d(
            *latent_hw,
            self.rope.head_dim,
            theta=self.rope.theta,
            device=tokens.device,
        )
        attention_kwargs = {
            "latent_hw": latent_hw,
            "rope_cos": cos,
            "rope_sin": sin,
            "invalid_token_mask": self.invalid_token_mask_flat,
        }
        for block in self.blocks:
            tokens = block(tokens, condition, attn_kwargs=attention_kwargs)
        projection = self.detokenizer.proj_layer
        shift, scale = projection.adaptive_modulation(condition).chunk(2, dim=1)
        return projection.modulation(projection.proj_layer_norm(tokens), scale, shift)

    def forward_features(
        self,
        x: torch.Tensor,
        time_step_cond: torch.Tensor | None = None,
        *,
        label_cond: torch.Tensor | None = None,
        points: Any | None = None,
        p_dropout: float | torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        """Return shared features for the parallel interval heads."""
        if (
            label_cond is not None
            or points is not None
            or p_dropout not in (None, 0, 0.0)
        ):
            raise ValueError(
                "the StormScope Flash DiT does not use labels, points, or dropout"
            )
        tokens, latent_hw = self.prepare_tokens(x)
        return self.forward_features_from_tokens(
            tokens, latent_hw, time_step_cond, training=training
        )

    def forward_features_with_condition_patch(
        self,
        state_input: torch.Tensor,
        condition_patch: torch.Tensor,
        time_step_cond: torch.Tensor | None = None,
        *,
        training: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        """Run trunk features with a cached condition patch contribution."""

        state_patch = self._split_patch_projection(
            state_input,
            channel_start=0,
            channel_stop=state_input.shape[1],
            include_bias=True,
            output_dtype=state_input.dtype,
        )
        if state_patch.shape != condition_patch.shape:
            raise ValueError("state and condition patch projections must match")
        tokens, latent_hw = self.prepare_patch_tokens(state_patch + condition_patch)
        return self.forward_features_from_tokens(
            tokens, latent_hw, time_step_cond, training=training
        )

    def unpatchify(
        self, tokens: torch.Tensor, height: int | None = None, width: int | None = None
    ) -> torch.Tensor:
        """Restore patch tokens to a channel-first image."""

        height = height or self._height
        width = width or self._width
        patch_height = height // self._patch_size
        patch_width = width // self._patch_size
        batch = tokens.shape[0]
        tokens = tokens.reshape(
            batch,
            patch_height,
            patch_width,
            self._patch_size,
            self._patch_size,
            self._out_chans,
        )
        return tokens.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self._out_chans, height, width
        )

    def forward(
        self, x: torch.Tensor, time_step_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the checkpoint-compatible image prediction before preconditioning."""
        height, width = x.shape[-2:]
        features = self.forward_features(x, time_step_cond)
        with torch.amp.autocast(device_type=features.device.type, enabled=False):
            projected = self.detokenizer.proj_layer.output_projection(
                features.to(torch.float32)
            )
        return self.unpatchify(projected.to(features.dtype), height, width)


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


@check_optional_dependencies()
class FlashModel(Module):
    """Reconstructible PhysicsNeMo deployment model for a packed Flash expert.

    Parameters
    ----------
    model_config : dict[str, Any]
        JSON-serializable constructor arguments for the checkpoint-compatible FlashDiT.
    flash_metadata : dict[str, Any]
        Trained sigma grid, block boundaries, and deployment compatibility metadata.

    Notes
    -----
    Large parameters are constructed on the meta device and assigned when loading
    the checkpoint. Load weights before moving or calling this inference-only model.
    """

    sigma_grid: torch.Tensor
    time_grid: torch.Tensor
    block_boundaries: torch.Tensor

    def __init__(
        self, model_config: dict[str, Any], flash_metadata: dict[str, Any]
    ) -> None:
        if flash_metadata.get("checkpoint_layout") != "physicsnemo-dit-v1":
            raise ValueError(
                "Flash checkpoint_layout must be physicsnemo-dit-v1; "
                "migrate the legacy package first"
            )
        with torch.device("meta"):
            model = FlashDiT(**model_config)
        super().__init__()
        sigma_grid = flash_metadata["sigma_grid"]
        block_boundaries = flash_metadata["block_boundaries"]
        sigma_data = flash_metadata["config"]["student"]["sigma_data"]
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
        self.model = model
        self.sigma_data = float(sigma_data)
        self.num_intervals = num_intervals
        self.register_buffer("sigma_grid", sigmas.contiguous())
        self.register_buffer("time_grid", sigma_to_flow_time(sigmas).contiguous())
        self.register_buffer("block_boundaries", boundaries.contiguous())

        projection = self.model.detokenizer.proj_layer.output_projection
        weight = projection.weight.detach()
        bias = (
            projection.bias.detach()
            if projection.bias is not None
            else weight.new_zeros(weight.shape[0])
        )
        self.head_weight = nn.Parameter(weight.unsqueeze(0).repeat(num_intervals, 1, 1))
        self.head_bias = nn.Parameter(bias.unsqueeze(0).repeat(num_intervals, 1))
        projection.requires_grad_(False)
        self.model_config = deepcopy(model_config)
        self.flash_metadata = deepcopy(flash_metadata)

    @staticmethod
    def _batch_sigma(
        sigma: Any,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.as_tensor(sigma, device=device, dtype=dtype)
        if values.numel() == 1:
            values = values.reshape(1).expand(batch_size)
        elif values.numel() == batch_size:
            values = values.reshape(batch_size)
        else:
            raise ValueError("sigma must be scalar or contain one value per sample")
        if not torch.compiler.is_compiling() and bool((values <= 0).any()):
            raise ValueError("Flash block-start sigma must be positive")
        return values

    def _trunk_dtype(self) -> torch.dtype:
        dtype = next(self.model.parameters()).dtype
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
            raise IndexError("invalid Flash inference block")
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
    def prepare_condition_patch(self, condition: torch.Tensor) -> torch.Tensor:
        """Cache the condition contribution to the FlashDiT patch projection."""

        return self.model.prepare_condition_patch(
            condition.to(device=self.head_weight.device, dtype=self._trunk_dtype()),
            state_channels=self.model._out_chans,
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

        state = state.to(
            torch.float64 if state.dtype == torch.float64 else torch.float32
        )
        sigma_batch = self._batch_sigma(
            sigma, state.shape[0], state.device, state.dtype
        )
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
                (model_input, condition.to(device=state.device, dtype=state.dtype)),
                dim=1,
            )

        time_condition = c_noise.to(
            self.model.conditioning_embedder.t_embedder.freqs.dtype
        )
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
        """Evaluate one pre-fused Flash block."""

        if training:
            raise ValueError("packaged Flash experts support inference only")
        height, width = state.shape[-2:]
        features, state, velocity_skip, velocity_out = self._features_and_coefficients(
            state, sigma, condition, condition_patch
        )
        residual = self._project(
            features, inference_weight, inference_bias, height, width
        )
        return (
            state
            + inference_delta.to(device=state.device, dtype=state.dtype)
            * velocity_skip
            * state
            + velocity_out * residual.to(state.dtype)
        )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True
    ) -> Any:
        """Assign checkpoint tensors without allocating a second initialized trunk."""
        if not assign:
            raise ValueError("Flash deployment weights require assign=True")
        result = super().load_state_dict(state_dict, strict=strict, assign=True)
        if self.model.rope.rope_cos.device != self.head_weight.device:
            self.model._materialize_rope(self.head_weight.device)
        # Nonpersistent device markers have no checkpoint values to assign.
        for module in self.modules():
            if isinstance(module, Module):
                marker = getattr(module, "device_buffer", None)
                if marker is not None and marker.is_meta and marker.numel() == 0:
                    module.device_buffer = torch.empty(
                        0, device=self.head_weight.device
                    )
        if any(t.is_meta for t in (*self.parameters(), *self.buffers())):
            raise ValueError("Flash checkpoint left uninitialized model tensors")
        self.eval().requires_grad_(False)
        return result


@dataclass(frozen=True)
class RegionInfo:
    """Resolved native input, requested rectangle, and exact geographic mask.

    Index bounds are half-open and ordered ``(y0, y1, x0, x1)``. Padding is
    ``(top, bottom, left, right)`` in array order and includes size/alignment
    expansion. ``requested_mask`` is relative to the requested output rectangle.
    """

    requested_bounds: tuple[float, float, float, float] | None
    input_bounds: tuple[int, int, int, int]
    output_bounds: tuple[int, int, int, int]
    requested_mask: np.ndarray
    padding: tuple[int, int, int, int]

    @property
    def input_shape(self) -> tuple[int, int]:
        """Actual model input dimensions, including available context."""
        y0, y1, x0, x1 = self.input_bounds
        return y1 - y0, x1 - x0

    @property
    def output_slices(self) -> tuple[slice, slice]:
        """Output rectangle relative to the retained model input."""
        y0, _, x0, _ = self.input_bounds
        a, b, c, d = self.output_bounds
        return slice(a - y0, b - y0), slice(c - x0, d - x0)

    def crop_output(
        self, forecast: torch.Tensor, coords: CoordSystem, *, mask: bool = True
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Crop presentation/output fields without modifying rollout histories."""
        if forecast.shape[-2:] != self.input_shape:
            raise ValueError("Forecast spatial dimensions do not match this region")
        ys, xs = self.output_slices
        result = forecast[..., ys, xs].clone()
        output_coords = coords.copy()
        for name, index in (("y", ys), ("x", xs)):
            if name in output_coords:
                output_coords[name] = output_coords[name][index].copy()
        for name in ("lat", "lon"):
            if name in output_coords:
                value = output_coords[name]
                output_coords[name] = (
                    value[ys, xs].copy() if value.ndim == 2 else value.copy()
                )
        if mask:
            valid = torch.as_tensor(self.requested_mask.copy(), device=result.device)
            result = torch.where(valid, result, torch.nan)
        return result, output_coords


def _axis_context(start: int, stop: int, size: int, padding: int) -> tuple[int, int]:
    start, stop = max(0, start - padding), min(size, stop + padding)
    missing = max(0, 200 - (stop - start))
    start = max(0, start - missing // 2)
    stop = min(size, max(stop, start + 200))
    start = max(0, min(start, stop - 200))
    return (start // 4) * 4, min(size, ((stop + 3) // 4) * 4)


def _inside_grid(
    corners: np.ndarray, latitude: np.ndarray, longitude: np.ndarray
) -> bool:
    # Ray casting on the native curvilinear perimeter, including its boundary.
    lat = np.concatenate(
        (latitude[0], latitude[1:, -1], latitude[-1, -2::-1], latitude[-2:0:-1, 0])
    )
    lon = np.concatenate(
        (longitude[0], longitude[1:, -1], longitude[-1, -2::-1], longitude[-2:0:-1, 0])
    )
    x1, y1, x2, y2 = lon, lat, np.roll(lon, -1), np.roll(lat, -1)
    for x, y in corners:
        dx, dy = x2 - x1, y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx
        on = (
            (np.abs(cross) <= 1e-6 * np.maximum(np.hypot(dx, dy), 1e-6))
            & (x >= np.minimum(x1, x2) - 1e-6)
            & (x <= np.maximum(x1, x2) + 1e-6)
            & (y >= np.minimum(y1, y2) - 1e-6)
            & (y <= np.maximum(y1, y2) + 1e-6)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = ((y1 > y) != (y2 > y)) & (x < dx * (y - y1) / dy + x1)
        if not bool(on.any()) and not bool(np.count_nonzero(crossings) % 2):
            return False
    return True


def resolve_region(
    latitude: np.ndarray,
    longitude: np.ndarray,
    region: Mapping[str, Sequence[float]] | None,
    padding: int = 25,
) -> RegionInfo:
    """Resolve an in-domain lat/lon box with clipped real context and a 200px floor."""
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    if latitude.ndim != 2 or latitude.shape != longitude.shape:
        raise ValueError(
            "Latitude and longitude must be matching two-dimensional grids"
        )
    h, w = latitude.shape
    if min(h, w) < 200 or h % 4 or w % 4:
        raise ValueError("Native grid dimensions must be >=200 and divisible by four")
    bounds = None
    if region is None:
        mask = np.ones((h, w), dtype=bool)
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        if set(region) != {"lat", "lon"}:
            raise ValueError("region must contain exactly lat and lon bounds")
        if len(region["lat"]) != 2 or len(region["lon"]) != 2:
            raise ValueError("Each region bound must contain two values")
        south, north = map(float, region["lat"])
        raw_west, raw_east = map(float, region["lon"])
        west, east = (raw_west + 180) % 360 - 180, (raw_east + 180) % 360 - 180
        if (
            not np.isfinite([south, north, raw_west, raw_east]).all()
            or not -90 <= south < north <= 90
            or not west < east
            or abs(raw_east - raw_west) >= 360
        ):
            raise ValueError(
                "Use finite increasing latitude and longitude bounds within the native CONUS domain"
            )
        signed: np.ndarray = (longitude.astype(np.float64) + 180) % 360 - 180
        if not _inside_grid(
            np.array([(west, south), (west, north), (east, south), (east, north)]),
            latitude,
            signed,
        ):
            raise ValueError(
                "Requested region extends outside the native model footprint"
            )
        mask = (
            (latitude >= south)
            & (latitude <= north)
            & (signed >= west)
            & (signed <= east)
        )
        rows, cols = np.nonzero(mask)
        if not rows.size:
            raise ValueError("Requested region contains no native grid points")
        y0, y1, x0, x1 = (
            int(rows.min()),
            int(rows.max()) + 1,
            int(cols.min()),
            int(cols.max()) + 1,
        )
        bounds = south, north, west, east
    iy0, iy1 = _axis_context(y0, y1, h, padding)
    ix0, ix1 = _axis_context(x0, x1, w, padding)
    roi_mask = mask[y0:y1, x0:x1].copy()
    roi_mask.flags.writeable = False
    return RegionInfo(
        bounds,
        (iy0, iy1, ix0, ix1),
        (y0, y1, x0, x1),
        roi_mask,
        (y0 - iy0, iy1 - y1, x0 - ix0, ix1 - x1),
    )


FLASH_EXPERT_ORDER = ("high", "middle", "low")
FLASH_REGION_INTERVALS = {"high": 16, "middle": 32, "low": 80}
FLASH_ROUTING_BOUNDARY_INDICES = (2, 36)
FLASH_GLOBAL_NUM_INTERVALS = 128
FLASH_SIGMA_MAX = 800.0
FLASH_CALLS = {
    "goes": {"high": 1, "middle": 1, "low": 3},
    "mrms": {"high": 1, "middle": 1, "low": 5},
}

# One cached invocation: expert name, sigma, fused weight/bias, and time increment.
_FlashStep = tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
    """Build the three expert grids from one global routing reference."""
    interval_counts = region_intervals or FLASH_REGION_INTERVALS
    if set(interval_counts) != set(FLASH_EXPERT_ORDER):
        raise ValueError("region_intervals must define high, middle, and low")
    high_middle, middle_low = boundary_indices
    if not 0 < high_middle < middle_low < num_intervals:
        raise ValueError("invalid Flash routing boundary indices")
    global_grid = uniform_flow_time_sigma_grid(
        num_intervals=num_intervals,
        sigma_max=sigma_max,
        sigma_min=0.0,
        device=device,
        dtype=dtype,
    )
    limits = (
        sigma_max,
        float(global_grid[high_middle].detach().cpu()),
        float(global_grid[middle_low].detach().cpu()),
        0.0,
    )
    return {
        name: uniform_flow_time_sigma_grid(
            num_intervals=int(interval_counts[name]),
            sigma_max=start,
            sigma_min=end,
            device=device,
            dtype=dtype,
        )
        for name, start, end in zip(FLASH_EXPERT_ORDER, limits[:-1], limits[1:])
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


def _unwrap_expert(expert: torch.nn.Module) -> torch.nn.Module:
    module = getattr(expert, "module", expert)
    return getattr(module, "_orig_mod", module)


def build_flash_chain_plan(
    experts: Mapping[str, torch.nn.Module],
    *,
    total_nfe: int,
    region_calls: Mapping[str, int],
    alignment: int = 16,
) -> list[_FlashStep]:
    """Validate the expert handoffs and prepare a flat list of fused invocations."""
    if set(experts) != set(FLASH_EXPERT_ORDER):
        raise ValueError(
            "Flash chain must contain exactly high, middle, and low experts"
        )
    if (
        set(region_calls) != set(FLASH_EXPERT_ORDER)
        or sum(region_calls.values()) != total_nfe
    ):
        raise ValueError("regional calls must sum to the requested total NFE")
    grids = {
        name: _unwrap_expert(experts[name]).sigma_grid.double()
        for name in FLASH_EXPERT_ORDER
    }
    if not torch.isclose(
        grids["high"][0],
        grids["high"].new_tensor(FLASH_SIGMA_MAX),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError(f"high Flash expert must start at sigma={FLASH_SIGMA_MAX}")
    for left, right in zip(FLASH_EXPERT_ORDER[:-1], FLASH_EXPERT_ORDER[1:]):
        if not torch.isclose(grids[left][-1], grids[right][0], rtol=1e-6, atol=1e-6):
            raise ValueError(f"invalid Flash handoff between {left} and {right}")
    if float(grids["low"][-1].detach().cpu()) != 0.0:
        raise ValueError("low Flash expert must terminate at sigma=0")
    plan = []
    for name in FLASH_EXPERT_ORDER:
        module = _unwrap_expert(experts[name])
        boundaries = inference_block_boundaries(
            grids[name].numel() - 1, region_calls[name], alignment=alignment
        ).tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            sigma, weight, bias, delta = module.prepare_inference_block(start, end)
            plan.append((name, sigma, weight, bias, delta))
    return plan


def _require_finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{label} contains non-finite values")


@torch.no_grad()
def flash_sampler_chain(
    experts: Mapping[str, torch.nn.Module],
    latents: torch.Tensor,
    condition: torch.Tensor | None = None,
    *,
    total_nfe: int,
    plan: list[_FlashStep],
) -> torch.Tensor:
    """Apply the fused expert calls without intermediate noise injection."""
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
    if len(plan) != total_nfe:
        raise ValueError(
            f"Flash plan NFE={len(plan)} does not match requested {total_nfe}"
        )
    first_time = sigma_to_flow_time(
        _unwrap_expert(experts["high"]).sigma_grid[0].double()
    )
    state = latents * (1 - first_time).to(device=latents.device, dtype=latents.dtype)
    previous = None
    condition_patch = None
    for name, sigma, weight, bias, delta in plan:
        expert = experts[name]
        if name != previous:
            module = _unwrap_expert(expert)
            condition_patch = (
                module.prepare_condition_patch(condition)
                if condition is not None and hasattr(module, "prepare_condition_patch")
                else None
            )
            previous = name
        sigma = sigma.to(
            device=state.device,
            dtype=torch.float64 if state.dtype == torch.float64 else torch.float32,
        )
        state = expert(
            state,
            sigma.reshape(1).expand(state.shape[0]),
            condition if condition_patch is None else None,
            inference_weight=weight,
            inference_bias=bias,
            inference_delta=delta,
            condition_patch=condition_patch,
            training=False,
        )
        if not isinstance(state, torch.Tensor):
            raise TypeError("each Flash block must return a tensor")
        if hasattr(expert, "_orig_mod"):
            state = state.clone()
        if state.shape != latents.shape:
            raise ValueError(
                f"Flash {name} expert returned {tuple(state.shape)}, expected {tuple(latents.shape)}"
            )
    return state


def bind_flash_domain(model: StormScopeBase) -> None:
    """Bind shared experts to this wrapper's grid before each sequential forecast.

    Call before preparing GOES missing-patch masks: changing the expert grid clears
    those masks. Wrappers sharing experts cannot forecast concurrently.
    """
    for expert in model.stage_models:
        _unwrap_expert(expert).model.set_image_size(*model.latitudes.shape)


def prepare_flash_goes_state(
    model: StormScopeBase, x: torch.Tensor, normalized: torch.Tensor
) -> torch.Tensor:
    """Apply trained missing-patch tokens and normalized zeros to GOES observations."""
    missing = torch.isnan(x).flatten(0, 1).any(dim=(1, 2)) | ~model.valid_mask
    for expert in model.stage_models:
        _unwrap_expert(expert).model.set_nan_pixel_mask(missing)
    return torch.where(torch.isnan(normalized), 0.0, normalized)


def sample_flash(
    model: StormScopeBase,
    latents: torch.Tensor,
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Flash with cached FP32 fused heads and the model's network AMP policy."""
    region_calls = FLASH_CALLS[cast(str, model._REGISTRY_KEY)]
    total_nfe = sum(region_calls.values())
    experts = dict(zip(FLASH_EXPERT_ORDER, model.stage_models))
    sources = tuple(
        getattr(_unwrap_expert(expert), name)
        for expert in experts.values()
        for name in ("head_weight", "head_bias", "time_grid", "sigma_grid")
    )
    key = tuple(
        (
            source.data_ptr(),
            None if source.is_inference() else source._version,
            source.device,
            source.dtype,
        )
        for source in sources
    )
    if (
        model._flash_plan is None
        or key != getattr(model, "_flash_plan_key", None)
        or any(version is None for _, version, _, _ in key)
    ):
        with torch.autocast(device_type=latents.device.type, enabled=False):
            model._flash_plan = build_flash_chain_plan(
                experts, total_nfe=total_nfe, region_calls=region_calls
            )
        setattr(model, "_flash_plan_key", key)
        # Keep the old storage alive so a shared expert's dtype round-trip cannot
        # reuse its address and accidentally validate stale fused heads.
        setattr(
            model, "_flash_plan_sources", tuple(source.detach() for source in sources)
        )
    with torch.autocast(device_type=latents.device.type, enabled=model.amp):
        result = flash_sampler_chain(
            experts, latents, condition, total_nfe=total_nfe, plan=model._flash_plan
        )
    _require_finite(result, "Flash forecast")
    return result


GOES_VARIABLES = [
    "abi01c",
    "abi02c",
    "abi03c",
    "abi07c",
    "abi08c",
    "abi09c",
    "abi10c",
    "abi13c",
]


def validate_flash_entry(entry: Mapping[str, Any], kind: str) -> None:
    """Validate the physical-input and sampling contract of a flash package."""
    if kind not in FLASH_CALLS:
        raise ValueError("flash kind must be goes or mrms")
    expected = {
        "flash": True,
        "n_steps": 6,
        "image_size": [1024, 1792],
        "step_interval": 10,
        "spatial_downsample": 1,
        "sliding_window": True,
        "topo": True,
        "variables": (
            GOES_VARIABLES if kind == "goes" else ["refc", "refc_base", "glm_density"]
        ),
        "conditioning_vars": [] if kind == "goes" else GOES_VARIABLES,
        "nexrad_proximity": kind == "mrms",
        "region_calls": FLASH_CALLS[kind],
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError(
                f"Invalid flash {kind} package field {key}: expected {value!r}"
            )
    if kind == "mrms" and not entry.get("mrms_coverage_mask"):
        raise ValueError("flash MRMS requires the training coverage mask")


def _validate_flash_expert(model: FlashModel, kind: str, region: str) -> None:
    meta = model.flash_metadata
    config = meta["model_config"]
    expected_grids = regional_sigma_grids()
    shared_config = {
        "patch_size": 4,
        "embed_dim": 768,
        "depth": 16,
        "num_heads": 6,
        "attn_kernel": 49,
        "rope_theta": 10000.0,
        "use_nan_mask_tokens": kind == "goes",
    }
    expected_model = {
        **shared_config,
        "architecture": "dit",
        "alternate_attn": False,
        "use_transformer_engine": False,
        "use_fused_layernorm": False,
        "qk_norm": True,
        "pos_embed": "rotary",
        "num_register_tokens": 0,
        "n_goes_channels": 8,
        "n_mrms_channels": 0 if kind == "goes" else 2,
        "n_glm_channels": 0 if kind == "goes" else 1,
        "latlon_input": True,
        "cos_zenith_input": True,
    }
    for key, value in expected_model.items():
        if config.get(key) != value:
            raise ValueError(
                f"{region}: incompatible model_config.{key}: {config.get(key)!r}"
            )
    if config.get("n_nldn_channels", 0) != 0:
        raise ValueError("MRMS Flash requires GLM checkpoints, not NLDN checkpoints")
    if (
        meta.get("version") != 2
        or meta.get("coordinate") != "edm_linear_t"
        or meta.get("region") != region
        or meta.get("inference_block_alignment") != 16
        or meta["config"]["student"]["sigma_data"] != 0.5
    ):
        raise ValueError(f"{region}: incompatible Flash metadata")
    state = model.state_dict()
    sigmas = torch.as_tensor(meta["sigma_grid"], dtype=torch.float32)
    if sigmas.shape != expected_grids[region].shape or not torch.allclose(
        sigmas, expected_grids[region], rtol=2e-5, atol=1e-6
    ):
        raise ValueError(f"{region}: unexpected trained sigma grid")
    for key, expected in (
        ("sigma_grid", sigmas),
        ("time_grid", sigma_to_flow_time(sigmas)),
        ("block_boundaries", torch.tensor(meta["block_boundaries"])),
    ):
        if (
            key not in state
            or state[key].shape != expected.shape
            or not torch.allclose(state[key], expected, rtol=1e-6, atol=1e-7)
        ):
            raise ValueError(f"{region}: metadata and saved {key} disagree")
    boundaries = inference_block_boundaries(
        sigmas.numel() - 1, FLASH_CALLS[kind][region], alignment=16
    ).tolist()
    ranges = {int(k): int(v) for k, v in meta["training_ranges"].items()}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start not in ranges or end > ranges[start]:
            raise ValueError(
                f"{region}: flash block {start}:{end} exceeds trained support"
            )
    expected_constructor = {
        **shared_config,
        "height": 1024,
        "width": 1792,
        "in_chans": 61 if kind == "goes" else 75,
        "base_out_chans": 8 if kind == "goes" else 3,
        "pos_embedding_type": "rotary",
    }
    if model.model_config != expected_constructor:
        raise ValueError(f"{region}: incompatible serialized DiT constructor")


def _validate_checkpoint_file(path: Path, spec: Mapping[str, Any]) -> None:
    # Cached files may have hashed names without the original extension.
    if (
        Path(spec["path"]).suffix != ".mdlus"
        or spec.get("deployment_format") != "flash-mdlus-v1"
    ):
        raise ValueError("Flash requires a .mdlus deployment package")
    if path.stat().st_size != spec["deployment_size_bytes"]:
        raise ValueError(f"Packaged checkpoint size mismatch: {path}")
    expected_hash = spec["deployment_sha256"]
    if os.environ.get("EARTH2STUDIO_VERIFY_CHECKPOINT_HASH", "0").lower() in (
        "1",
        "true",
        "yes",
    ):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Packaged checkpoint hash mismatch: {path}")


def load_flash_experts(
    package: Package, entry: Mapping[str, Any], kind: str
) -> list[dict[str, Any]]:
    """Load three .mdlus experts and validate architecture and trained block support."""
    validate_flash_entry(entry, kind)
    specs = entry["checkpoints"]
    if len(specs) != 3 or {s["name"] for s in specs} != {"high", "middle", "low"}:
        raise ValueError("flash requires exactly high, middle, and low checkpoints")
    stages = []
    for spec in sorted(
        specs, key=lambda item: ("high", "middle", "low").index(item["name"])
    ):
        region = spec["name"]
        path = Path(package.resolve(spec["path"]))
        _validate_checkpoint_file(path, spec)
        model = FlashModel.from_checkpoint(str(path), strict=True)
        if not isinstance(model, FlashModel):
            raise TypeError("Flash checkpoint must contain a FlashModel")
        _validate_flash_expert(model, kind, region)
        model.eval().requires_grad_(False)
        stages.append(
            {
                "name": region,
                "model": model,
                "sigma_min": float(model.sigma_grid[-1]),
                "sigma_max": float(model.sigma_grid[0]),
            }
        )
    return stages


def configure_region(
    model: StormScopeBase, region: Mapping[str, Sequence[float]] | None, padding: int
) -> None:
    """Slice wrapper geometry without mutating the shared Flash experts."""
    model.region_info = resolve_region(
        model._lat_cpu_copy, model._lon_cpu_copy, region, padding
    )
    y0, y1, x0, x1 = model.region_info.input_bounds
    for name in (
        "latitudes",
        "longitudes",
        "valid_mask",
        "conditioning_valid_mask",
        "topo",
        "nexrad_proximity",
        "mrms_coverage_mask",
    ):
        value = model._buffers.get(name)
        if value is not None:
            model.register_buffer(name, value[y0:y1, x0:x1].clone())
    model.y = model.y[y0:y1].copy()
    model.x = model.x[x0:x1].copy()
    model._lat_cpu_copy = model.latitudes.detach().cpu().numpy()
    model._lon_cpu_copy = model.longitudes.detach().cpu().numpy()
