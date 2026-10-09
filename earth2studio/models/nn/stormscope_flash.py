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

"""StormScope Flash checkpoint adapters built from PhysicsNeMo components.

The adapters retain trained normalization, residual precision, packed interval
heads, and cached conditioning. Generic attention, RoPE, MLP, patch projection,
and timestep embedding come from PhysicsNeMo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from physicsnemo import Module
    from physicsnemo.nn.module.dit_layers import DiTBlock as PhysicsNeMoDiTBlock
    from physicsnemo.nn.module.dit_layers import ProjLayer as PhysicsNeMoProjLayer
    from physicsnemo.nn.module.dit_layers import get_layer_norm
    from physicsnemo.nn.module.embedding_layers import PositionalEmbedding
    from physicsnemo.nn.module.rope import build_axial_rope_cos_sin_2d
    from physicsnemo.nn.module.utils import PatchEmbed2D as PhysicsNeMoPatchEmbed
except (ImportError, OSError):
    OptionalDependencyFailure("stormscope-flash")
    Module = PhysicsNeMoDiTBlock = PhysicsNeMoProjLayer = PhysicsNeMoPatchEmbed = (
        nn.Module
    )


def _modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    """Apply the trained adaptive normalization arithmetic."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _restore_input_dtype(
    module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    """Retain the projection dtype after upstream Q/K normalization under AMP."""
    return output.to(inputs[0].dtype)


class _FlashDiTBlock(PhysicsNeMoDiTBlock):
    """Checkpoint-compatible adaptive-LayerNorm DiT block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        attn_kernel: int,
    ) -> None:
        super().__init__(
            hidden_size=dim,
            num_heads=num_heads,
            attention_backend="natten2d_rope",
            layernorm_backend="torch",
            norm_eps=1e-6,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=False,
            attn_kernel=attn_kernel,
        )
        # Preserve deployed state-dict paths while using PhysicsNeMo components.
        for upstream, deployed in (
            ("attention", "attn"),
            ("pre_attention_norm", "norm1"),
            ("pre_mlp_norm", "norm2"),
            ("linear", "mlp"),
            ("adaptive_modulation", "adaLN_modulation"),
        ):
            self.add_module(deployed, self._modules.pop(upstream))
        # These checkpoints trained affine Q/K LayerNorm at eps=1e-5.
        # PhysicsNeMo's attention currently defaults to non-affine eps=1e-6.
        if qk_norm:
            self.attn.q_norm = get_layer_norm(
                dim // num_heads, "torch", elementwise_affine=True, eps=1e-5
            )
            self.attn.k_norm = get_layer_norm(
                dim // num_heads, "torch", elementwise_affine=True, eps=1e-5
            )
        # NATTEN requires Q/K/V to share a dtype under AMP. Preserve the
        # deployed attention's explicit cast after affine Q/K normalization.
        self.attn.q_norm.register_forward_hook(_restore_input_dtype)
        self.attn.k_norm.register_forward_hook(_restore_input_dtype)
        self.num_ada_ln = 6
        self.register_load_state_dict_pre_hook(self._load_legacy_mlp)

    @staticmethod
    def _load_legacy_mlp(module, state_dict, prefix, *args):
        """Translate deployed MLP parameter names to PhysicsNeMo's layout."""
        for old, new in (("0", "0"), ("3", "2")):
            for field in ("weight", "bias"):
                old_key = f"{prefix}mlp.fwd.{old}.{field}"
                new_key = f"{prefix}mlp.layers.{new}.{field}"
                if old_key in state_dict:
                    if new_key in state_dict:
                        raise ValueError(f"Duplicate MLP parameter: {new_key}")
                    state_dict[new_key] = state_dict.pop(old_key)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        kv: torch.Tensor | None = None,
        latent_hw: tuple[int, int] | None = None,
        rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
        p_dropout: float | torch.Tensor | None = None,
        training: bool = False,
        invalid_token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive attention and MLP residual updates for inference."""
        if kv is not None:
            raise ValueError("the StormScope Flash DiT does not use cross attention")
        if training or p_dropout not in (None, 0, 0.0):
            raise ValueError("the packaged Flash DiT supports inference only")

        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            cond
        ).chunk(self.num_ada_ln, dim=1)
        if latent_hw is None or rope_tables is None:
            raise ValueError("rotary attention requires latent_hw and RoPE tables")
        attention_input = _modulate(self.norm1(x), scale, shift)
        if invalid_token_mask is not None and mask_token is not None:
            attention_input = torch.where(
                invalid_token_mask[..., None],
                mask_token.to(attention_input),
                attention_input,
            )
        attended = self.attn(
            attention_input,
            latent_hw=latent_hw,
            rope_cos=rope_tables[0],
            rope_sin=rope_tables[1],
        )
        # Retain the checkpoint's reduced-precision product before residual addition.
        # The upstream block instead uses a fused addcmul update.
        gate = gate.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        x = x + self.drop_path(gate * attended)
        mlp = self.mlp(_modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x + self.drop_path(gate_mlp * mlp)


class _FlashProjection(PhysicsNeMoProjLayer):
    """Adaptive normalization and final patch projection."""

    def __init__(
        self, hidden_size: int, patch_size: tuple[int, int], out_chans: int
    ) -> None:
        super().__init__(hidden_size, patch_size[0] * patch_size[1] * out_chans)
        for upstream, deployed in (
            ("proj_layer_norm", "norm"),
            ("output_projection", "linear"),
            ("adaptive_modulation", "adaLN_modulation"),
        ):
            self.add_module(deployed, self._modules.pop(upstream))

    def forward_features(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the final adaptive normalization before interval-head projection."""
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return _modulate(self.norm(x), scale, shift)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project patches in FP32, then restore the incoming feature dtype."""
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.linear(x.to(torch.float32))
        return x.to(input_dtype)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Normalize and project the final patch features."""
        return self.project(self.forward_features(x, cond))


class _FlashPatchEmbed(PhysicsNeMoPatchEmbed):
    """Convolutional non-overlapping patch projection."""

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        super().__init__((height, width), (patch_size, patch_size), in_chans, embed_dim)
        self.height, self.width = height, width
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches in FP32 and restore the input dtype."""
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.proj(x.to(torch.float32))
        return x.to(input_dtype)


class FlashDiT(nn.Module):
    """Lean inference implementation matching the validated StormScope Flash DiT."""

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
        learn_sigma: bool = False,
        num_classes: int = 0,
        class_dropout_prob: float = 0.1,
        pos_embedding_type: str = "rotary",
        is_conditional: bool = True,
        use_skip_connection: bool = False,
        use_concat_skip_connection: bool = False,
        num_input_time_steps: int = 1,
        num_output_time_steps: int = 1,
        attn_mask_type: str | None = None,
        attn_kernel: int = 49,
        point_channels: int = 0,
        use_fused_layernorm: bool = False,
        p_dropout: float | None = None,
        grad_checkpoint_blocks: int = 0,
        alternate_attn: bool = False,
        use_transformer_engine: bool = False,
        num_register_tokens: int = 0,
        rope_theta: float = 10000.0,
        use_nan_mask_tokens: bool = False,
    ) -> None:
        super().__init__()
        unsupported = {
            "alternate_attn": alternate_attn,
            "learn_sigma": learn_sigma,
            "num_classes": num_classes,
            "use_skip_connection": use_skip_connection,
            "use_concat_skip_connection": use_concat_skip_connection,
            "attn_mask_type": attn_mask_type,
            "point_channels": point_channels,
            "use_fused_layernorm": use_fused_layernorm,
            "p_dropout": p_dropout,
            "grad_checkpoint_blocks": grad_checkpoint_blocks,
            "use_transformer_engine": use_transformer_engine,
            "num_register_tokens": num_register_tokens,
        }
        enabled = [
            name for name, value in unsupported.items() if value not in (0, None, False)
        ]
        if enabled:
            raise ValueError(
                "unsupported StormScope Flash DiT options: " + ", ".join(enabled)
            )
        if attn_kernel < 2:
            raise ValueError("StormScope Flash requires neighborhood attention")
        if not is_conditional or pos_embedding_type != "rotary":
            raise ValueError("StormScope Flash requires a conditional rotary FlashDiT")

        self._is_conditional = is_conditional
        self._learn_sigma = False
        self._out_chans = base_out_chans
        self._in_chans = in_chans
        self._patch_size = patch_size
        self._height = height
        self._width = width
        self._embed_dim = embed_dim
        self._depth = depth
        self._num_heads = num_heads
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._qk_norm = qk_norm
        self._num_classes = num_classes
        self._class_dropout_prob = class_dropout_prob
        self._frequency_embed_dim = frequency_embed_dim
        self._pos_embedding_type = pos_embedding_type
        self._use_fused_layernorm = False
        self._grad_checkpoint_blocks = 0
        self._alternate_attn = alternate_attn
        self._use_transformer_engine = False
        self._num_register_tokens = 0
        self._register_tokens = None
        self._use_skip_connection = False
        self._use_concat_skip_connection = False
        self._num_input_time_steps = num_input_time_steps
        self._num_output_time_steps = num_output_time_steps
        self._embed_points = None

        self._patch_emb = _FlashPatchEmbed(
            height, width, patch_size, in_chans, embed_dim
        )
        self._pos_emb = None
        self._rope_theta = rope_theta
        self._use_nan_mask_tokens = use_nan_mask_tokens
        self.register_buffer("invalid_token_mask_flat", None, persistent=False)
        self._label_emb = 0.0
        self._time_step_emb = PositionalEmbedding(
            num_channels=embed_dim,
            learnable=True,
            freq_embed_dim=frequency_embed_dim,
            mlp_hidden_dim=embed_dim,
            embed_fn="np_sin_cos",
        )
        self.attn_mask = None
        self._blocks = nn.ModuleList(
            [
                _FlashDiTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_kernel=attn_kernel,
                )
                for _ in range(depth)
            ]
        )
        self._nan_mask_tokens = nn.ParameterDict(
            {
                str(i): nn.Parameter(torch.zeros(1, 1, embed_dim))
                for i, block in enumerate(self._blocks)
                if use_nan_mask_tokens
                and cast(_FlashDiTBlock, block).attn.attn_kernel > 0
            }
        )
        if use_nan_mask_tokens and not self._nan_mask_tokens:
            raise ValueError("NaN mask tokens require neighborhood attention")
        self._final_layer = _FlashProjection(
            embed_dim, (patch_size, patch_size), base_out_chans
        )

    invalid_token_mask_flat: torch.Tensor | None

    def set_image_size(self, height: int, width: int) -> None:
        """Set this instance's rotary grid before forecasting or compiling."""
        if min(height, width) < 200 or height % 4 or width % 4:
            raise ValueError(
                "Flash image dimensions must be >=200 and divisible by four"
            )
        self._height, self._width = height, width
        self._patch_emb.height, self._patch_emb.width = height, width
        self._patch_emb.num_patches = (height // self._patch_size) * (
            width // self._patch_size
        )
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
        self.invalid_token_mask_flat = (
            F.max_pool2d(
                pixel_mask.to(
                    device=self._patch_emb.proj.weight.device, dtype=torch.float32
                )[:, None],
                self._patch_size,
                self._patch_size,
            )[:, 0]
            .flatten(1)
            .bool()
        )

    def prepare_patch_tokens(
        self, patch: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Flatten a projected image and add the learned positional embedding."""

        height, width = patch.shape[-2:]
        tokens = patch.flatten(2).transpose(1, 2)
        if self._pos_emb is not None:
            if tokens.shape[1] != self._pos_emb.shape[1]:
                raise ValueError(
                    "input image does not match the trained positional grid"
                )
            tokens = tokens + self._pos_emb
        return tokens, (height, width)

    def prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Embed an image and return flattened tokens with their patch-grid shape."""
        return self.prepare_patch_tokens(self._patch_emb(x))

    def _split_patch_projection(
        self,
        x: torch.Tensor,
        *,
        channel_start: int,
        channel_stop: int,
        include_bias: bool,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        projection = self._patch_emb.proj
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
        condition = self._time_step_emb(time_step_cond) + self._label_emb
        cos, sin = build_axial_rope_cos_sin_2d(
            *latent_hw,
            self._embed_dim // self._num_heads,
            theta=self._rope_theta,
            device=tokens.device,
        )
        rope_tables = (cos, sin)
        for index, block in enumerate(self._blocks):
            tokens = block(
                tokens,
                condition,
                latent_hw=latent_hw,
                rope_tables=rope_tables,
                p_dropout=None,
                training=False,
                invalid_token_mask=self.invalid_token_mask_flat,
                mask_token=(
                    self._nan_mask_tokens[str(index)]
                    if str(index) in self._nan_mask_tokens
                    else None
                ),
            )
        return self._final_layer.forward_features(tokens, condition)

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
        return self.unpatchify(self._final_layer.project(features), height, width)


__all__ = ["FlashDiT"]


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


class FlashPrecond(nn.Module):
    """Inference-only regional Flash expert backed by a FlashDiT trunk."""

    sigma_grid: torch.Tensor
    time_grid: torch.Tensor
    block_boundaries: torch.Tensor

    def __init__(
        self,
        model: FlashDiT,
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
            raise TypeError(
                "Flash model must expose FlashDiT feature and unpatchify methods"
            )
        final_layer = getattr(model, "_final_layer", None)
        if final_layer is None or not hasattr(final_layer, "linear"):
            raise TypeError("Flash model must expose _final_layer.linear")

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
            raise ValueError("Flash block-start sigma must be positive")
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
    def prepare_condition_patch(self, condition: torch.Tensor) -> torch.Tensor | None:
        """Cache the condition contribution to the FlashDiT patch projection."""

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
            + inference_delta.to(device=state.device, dtype=torch.float32)
            * velocity_skip
            * state
            + velocity_out * residual.to(torch.float32)
        )


@check_optional_dependencies()
class FlashModel(FlashPrecond, Module):
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

    def __init__(
        self, model_config: dict[str, Any], flash_metadata: dict[str, Any]
    ) -> None:
        with torch.device("meta"):
            trunk = FlashDiT(**model_config)
        super().__init__(
            trunk,
            flash_metadata["sigma_grid"],
            flash_metadata["block_boundaries"],
            sigma_data=flash_metadata["config"]["student"]["sigma_data"],
        )
        self.model_config = deepcopy(model_config)
        self.flash_metadata = deepcopy(flash_metadata)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = True
    ) -> Any:
        """Assign checkpoint tensors without allocating a second initialized trunk."""
        if not assign:
            raise ValueError("Flash deployment weights require assign=True")
        result = super().load_state_dict(state_dict, strict=strict, assign=True)
        # Nested PhysicsNeMo attention modules create a nonpersistent, empty
        # device marker. It has no checkpoint value to assign after meta init.
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
